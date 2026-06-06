import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

TOKEN_URL = "https://ims-na1.adobelogin.com/ims/token/v3"

client_id = os.getenv("ADOBE_CLIENT_ID", "").strip().strip('"').strip("'")
client_secret = os.getenv("ADOBE_CLIENT_SECRET", "").strip().strip('"').strip("'")
scopes = os.getenv(
    "ADOBE_SCOPES",
    "openid,AdobeID,read_organizations,additional_info.projectedProductContext,session",
).strip()

print("Direct Adobe token test")
print("=======================")
print(f"client_id length: {len(client_id)}")
print(f"client_secret length: {len(client_secret)}")
print(f"scopes: {scopes}")
print()

response = requests.post(
    TOKEN_URL,
    headers={
        "Content-Type": "application/x-www-form-urlencoded",
    },
    data={
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials",
        "scope": scopes,
    },
    timeout=30,
)

print("Status:", response.status_code)
print("Response:")
print(response.text)