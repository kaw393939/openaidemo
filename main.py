from dotenv import load_dotenv
from pydantic_settings import BaseSettings
import openai
import os
from pprint import pprint

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    openai_api_key: str = os.getenv("openai_api_key")
    openai_model: str = os.getenv("openai_model", "gpt-4o-mini")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __str__(self):
        return f"Settings(openai_api_key={'*' * 8}, openai_model='{self.openai_model}')"

settings = Settings()

# Initialize OpenAI API with the loaded API key
openai.api_key = settings.openai_api_key

def print_debug(label, obj):
    print(f"\n{'=' * 40}")
    print(f"{label}:")
    print(f"{'=' * 40}")
    pprint(obj, width=80, depth=None, compact=False)
    print(f"{'=' * 40}\n")

def main():
    client = openai.OpenAI(api_key=openai.api_key)

    messages = [
        {"role": "system", "content": "You are a spongebob and you are responding to squidward."},
        {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
    ]

    print_debug("Settings", settings)
    print_debug("Messages", messages)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    print_debug("Completion", completion)
    print_debug("Completion Message", completion.choices[0].message)

if __name__ == "__main__":
    print("Start")
    main()
    print("End")