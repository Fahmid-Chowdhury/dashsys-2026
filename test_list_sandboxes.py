import json
import requests

from src.tool_executor import ToolExecutor


def find_sandbox_items(payload):
    """
    Tries common Adobe response shapes.
    """
    if isinstance(payload, dict):
        if isinstance(payload.get("sandboxes"), list):
            return payload["sandboxes"]
        if isinstance(payload.get("children"), list):
            return payload["children"]
        if isinstance(payload.get("items"), list):
            return payload["items"]
        if isinstance(payload.get("_embedded"), dict):
            for value in payload["_embedded"].values():
                if isinstance(value, list):
                    return value
    return []


def main():
    executor = ToolExecutor(mock_api=False)
    token = executor.get_access_token()

    headers = {
        "Authorization": f"Bearer {token}",
        "x-api-key": executor.client_id,
        "x-gw-ims-org-id": executor.org_id,
        # "Accept": "application/json",
    }

    url = "https://platform.adobe.io/data/foundation/sandbox-management/"

    response = requests.get(
        url,
        headers=headers,
        params={"limit": 100, "offset": 100},
        timeout=30,
    )

    print("Status:", response.status_code)

    try:
        payload = response.json()
    except Exception:
        print(response.text)
        return

    print(json.dumps(payload, indent=2, ensure_ascii=False)[:5000])

    items = find_sandbox_items(payload)

    print("\nSandbox summary")
    print("===============")

    if not items:
        print("No sandbox list found in response. Check full response above.")
        return

    for sandbox in items:
        print(
            f"name={sandbox.get('name')} | "
            f"title={sandbox.get('title')} | "
            f"state={sandbox.get('state')} | "
            f"type={sandbox.get('type')}"
        )


if __name__ == "__main__":
    main()