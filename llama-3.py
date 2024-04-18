import os
import requests
import json
import logging
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Cloudflare account ID and authentication token
ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
AUTH_TOKEN = os.getenv("CLOUDFLARE_AUTH_TOKEN")

# Base URL for the Cloudflare AI API
BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run"

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def chat_with_llama(messages, timeout=30, max_tokens=4096):
    url = f"{BASE_URL}/@cf/meta/llama-3-8b-instruct"
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(url, headers=headers,
                                 json=data, stream=True, timeout=timeout)
        response.raise_for_status()
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    chunk = chunk.decode("utf-8")
                    if chunk == "[DONE]":
                        break
                    try:
                        chunk_parts = chunk.strip().split("data: ")
                        if len(chunk_parts) == 2:
                            data = json.loads(chunk_parts[1])
                            yield data["response"]
                    except (json.JSONDecodeError, KeyError):
                        continue
        else:
            logging.error(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error occurred during API request: {str(e)}")
        raise


def save_conversation(conversation, filename):
    with open(filename, "w") as file:
        for message in conversation:
            role = message["role"]
            content = message["content"]
            file.write(f"{role.capitalize()}: {content}\n")


def main():
    print("Welcome to the Llama-3 Chatbot!")
    print("Type 'new' to start a new conversation.")
    print("Type 'quit' to exit the program.")

    try:
        system_prompt = "You are a friendly assistant."
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        while True:
            user_input = input("User: ")
            if user_input.lower() == "new":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"conversation_{timestamp}.txt"
                save_conversation(messages, filename)
                print(f"Previous conversation saved to {filename}")
                messages = [
                    {"role": "system", "content": system_prompt}
                ]
                print("Starting a new conversation.")
                continue
            elif user_input.lower() == "quit":
                break
            messages.append({"role": "user", "content": user_input})
            print("Assistant: ", end="")
            assistant_response = ""
            for response in chat_with_llama(messages, max_tokens=4096):
                assistant_response += response
                print(response, end="", flush=True)
            print()
            messages.append(
                {"role": "assistant", "content": assistant_response})
    except KeyboardInterrupt:
        print("\nExiting the program...")
    finally:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.txt"
        save_conversation(messages, filename)
        print(f"Conversation saved to {filename}")


if __name__ == "__main__":
    main()
