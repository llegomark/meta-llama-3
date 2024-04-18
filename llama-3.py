import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Cloudflare account ID and authentication token
ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
AUTH_TOKEN = os.getenv("CLOUDFLARE_AUTH_TOKEN")

# Base URL for the Cloudflare AI API
BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run"


def chat_with_llama(messages):
    url = f"{BASE_URL}/@cf/meta/llama-3-8b-instruct"
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "messages": messages,
        "stream": True
    }

    response = requests.post(url, headers=headers, json=data, stream=True)

    if response.status_code == 200:
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                chunk = chunk.decode("utf-8")
                if chunk == "[DONE]":
                    break
                data = chunk.strip().split("data: ")[1]
                yield data
    else:
        print(f"Error: {response.status_code} - {response.text}")


def main():
    messages = [
        {"role": "system", "content": "You are a friendly assistant."}
    ]

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        messages.append({"role": "user", "content": user_input})

        print("Assistant: ", end="")
        assistant_response = ""
        for response in chat_with_llama(messages):
            assistant_response += response
            print(response, end="")
        print()

        messages.append({"role": "assistant", "content": assistant_response})


if __name__ == "__main__":
    main()
