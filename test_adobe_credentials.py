from src.tool_executor import ToolExecutor


def main():
    executor = ToolExecutor(mock_api=False)

    print("Checking credential variables...")

    if not executor.credentials_ready():
        print("Credentials are missing. Check your .env file.")
        return

    print("Credentials found.")
    print("Requesting Adobe access token...")

    try:
        token = executor.get_access_token()
        print("Access token generated successfully.")
        print(f"Token preview: {token[:12]}...{token[-8:]}")
    except Exception as e:
        print("Token generation failed.")
        print(str(e))


if __name__ == "__main__":
    main()