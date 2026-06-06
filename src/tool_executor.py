import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import duckdb
import requests
from dotenv import load_dotenv


load_dotenv(override=True)


def clean_env(name: str) -> Optional[str]:
    value = os.getenv(name)

    if value is None:
        return None

    return value.strip().strip('"').strip("'")


def make_json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]

    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}

    return str(value)


def register_snapshot_views(con: duckdb.DuckDBPyConnection, snapshot_dir: str | Path) -> List[str]:
    snapshot_dir = Path(snapshot_dir)
    warnings: List[str] = []

    if not snapshot_dir.exists():
        warnings.append(f"Snapshot directory not found: {snapshot_dir}")
        return warnings

    for fname in os.listdir(snapshot_dir):
        if not fname.endswith(".parquet"):
            continue

        table_name = fname[:-len(".parquet")]
        parquet_path = snapshot_dir / fname

        try:
            con.execute(
                f"""
                CREATE OR REPLACE VIEW {table_name} AS
                SELECT * FROM read_parquet('{parquet_path.as_posix()}')
                """
            )
        except Exception as e:
            warnings.append(f"Skipped {fname}: {str(e).splitlines()[0]}")

    return warnings


class ToolExecutor:
    def __init__(
        self,
        snapshot_dir: str | Path = "DBSnapshot",
        api_base_url: Optional[str] = None,
        mock_api: bool = True,
    ):
        self.snapshot_dir = Path(snapshot_dir)
        self.api_base_url = (
            api_base_url
            or os.getenv("ADOBE_API_BASE_URL")
            or "https://platform.adobe.io"
        ).rstrip("/")

        self.mock_api = mock_api

        self.client_id = clean_env("ADOBE_CLIENT_ID") or clean_env("CLIENT_ID")
        self.client_secret = (
            clean_env("ADOBE_CLIENT_SECRET")
            or clean_env("CLIENTSECRET")
            or clean_env("CLIENT_SECRET")
        )
        self.org_id = clean_env("ADOBE_ORG_ID") or clean_env("IMS_ORG")
        self.sandbox_name = clean_env("ADOBE_SANDBOX_NAME") or clean_env("SANDBOX")
        self.scopes = os.getenv(
            "ADOBE_SCOPES",
            "openid,AdobeID,read_organizations,additional_info.projectedProductContext,session",
        )
        self.token_url = os.getenv(
            "ADOBE_IMS_TOKEN_URL",
            "https://ims-na1.adobelogin.com/ims/token/v3",
        )

        self._access_token: Optional[str] = None
        self._access_token_expires_at: float = 0.0

    def execute_sql(self, sql: str, max_preview_rows: int = 20) -> Dict[str, Any]:
        con = duckdb.connect(database=":memory:")
        warnings = register_snapshot_views(con, self.snapshot_dir)

        try:
            cursor = con.execute(sql)
            rows = cursor.fetchmany(max_preview_rows)
            columns = [desc[0] for desc in cursor.description]

            result_rows = [
                {
                    columns[i]: make_json_safe(row[i])
                    for i in range(len(columns))
                }
                for row in rows
            ]

            return {
                "status": "success",
                "rows": result_rows,
                "row_preview_count": len(result_rows),
                "warnings": warnings,
            }

        except Exception as e:
            return {
                "status": "error",
                "rows": [],
                "error": str(e).splitlines()[0],
                "warnings": warnings,
            }

        finally:
            con.close()

    def credentials_ready(self) -> bool:
        return bool(
            self.client_id
            and self.client_secret
            and self.org_id
            and self.sandbox_name
        )

    def get_access_token(self, force_refresh: bool = False) -> str:
        now = time.time()

        if (
            not force_refresh
            and self._access_token
            and now < self._access_token_expires_at - 60
        ):
            return self._access_token

        if not self.credentials_ready():
            missing = []

            if not self.client_id:
                missing.append("ADOBE_CLIENT_ID")
            if not self.client_secret:
                missing.append("ADOBE_CLIENT_SECRET")
            if not self.org_id:
                missing.append("ADOBE_ORG_ID")
            if not self.sandbox_name:
                missing.append("ADOBE_SANDBOX_NAME")

            raise RuntimeError(
                "Missing Adobe credentials in .env: " + ", ".join(missing)
            )

        response = requests.post(
            self.token_url,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": self.scopes,
            },
            timeout=30,
        )

        if not response.ok:
            raise RuntimeError(
                f"Adobe token request failed: HTTP {response.status_code} - {response.text[:500]}"
            )

        payload = response.json()

        access_token = payload.get("access_token")
        expires_in = int(payload.get("expires_in", 3600))

        if not access_token:
            raise RuntimeError(
                f"Adobe token response did not contain access_token: {payload}"
            )

        self._access_token = access_token
        self._access_token_expires_at = now + expires_in

        return access_token

    def build_adobe_headers(
        self,
        method: str = "GET",
        has_body: bool = False,
        include_sandbox: bool = True,
    ) -> Dict[str, str]:
        token = self.get_access_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "x-api-key": self.client_id,
            "x-gw-ims-org-id": self.org_id,
            "Accept": "application/json",
        }

        if include_sandbox and self.sandbox_name:
            headers["x-sandbox-name"] = self.sandbox_name

        if method.upper() in {"POST", "PUT", "PATCH"} or has_body:
            headers["Content-Type"] = "application/json"

        return headers

    def call_api(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Dict[str, Any]] = None,
        max_preview_chars: int = 4000,
    ) -> Dict[str, Any]:
        method = method.upper()
        params = params or {}

        if self.mock_api:
            return {
                "status": "mock_success",
                "method": method,
                "url": url,
                "params": params,
                "result_preview": [
                    {
                        "mock": True,
                        "message": "API execution is mocked. Set mock_api=False for real Adobe API calls.",
                    }
                ],
            }

        if url.startswith("http://") or url.startswith("https://"):
            full_url = url
        else:
            full_url = urljoin(self.api_base_url + "/", url.lstrip("/"))

        try:
            request_headers = self.build_adobe_headers(
                method=method,
                has_body=body is not None,
            )

            if headers:
                request_headers.update(headers)

            response = requests.request(
                method=method,
                url=full_url,
                params=params,
                headers=request_headers,
                json=body,
                timeout=30,
            )

            content_type = response.headers.get("content-type", "")

            if "application/json" in content_type:
                preview = make_json_safe(response.json())
            else:
                preview = response.text[:max_preview_chars]

            return {
                "status": "success" if response.ok else "error",
                "status_code": response.status_code,
                "method": method,
                "url": url,
                "full_url": full_url,
                "params": params,
                "result_preview": preview,
            }

        except Exception as e:
            return {
                "status": "error",
                "method": method,
                "url": url,
                "params": params,
                "error": str(e),
                "result_preview": [],
            }