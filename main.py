from abc import ABC, abstractmethod
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from openai import OpenAI
import os
from pprint import pprint
import json

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    openai_api_key: str = os.getenv("openai_api_key")
    openai_model: str = os.getenv("openai_model", "gpt-3.5-turbo")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __str__(self):
        return f"Settings(openai_api_key={'*' * 8}, openai_model='{self.openai_model}')"

settings = Settings()
client = OpenAI(api_key=settings.openai_api_key)

def print_debug(label, obj):
    print(f"\n{'=' * 40}")
    print(f"{label}:")
    print(f"{'=' * 40}")
    pprint(obj, width=80, depth=None, compact=False)
    print(f"{'=' * 40}\n")

# Command interface
class MathOperation(ABC):
    @abstractmethod
    def execute(self, a, b):
        pass

    @abstractmethod
    def get_name(self):
        pass

# Concrete command classes
class AddOperation(MathOperation):
    def execute(self, a, b):
        return a + b

    def get_name(self):
        return "Addition"

class SubtractOperation(MathOperation):
    def execute(self, a, b):
        return a - b

    def get_name(self):
        return "Subtraction"

class MultiplyOperation(MathOperation):
    def execute(self, a, b):
        return a * b

    def get_name(self):
        return "Multiplication"

class DivideOperation(MathOperation):
    def execute(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    def get_name(self):
        return "Division"

# AI operation class
class AIOperation(MathOperation):
    def execute(self, question):
        system_message = """
        You are a helpful assistant that responds to math questions. 
        Your response should be in the following JSON format:
        {
            "operation": "add|subtract|multiply|divide",
            "num1": <first_number>,
            "num2": <second_number>
        }
        """

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]

        completion = client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            response_format={"type": "json_object"}
        )

        response = json.loads(completion.choices[0].message.content)
        print_debug("AI Response", response)
        
        operation = OPERATIONS.get(response["operation"])
        if not operation:
            raise ValueError(f"Invalid operation: {response['operation']}")
        
        return operation.execute(response["num1"], response["num2"])

    def get_name(self):
        return "AI-assisted calculation"

# Operation registry
OPERATIONS = {
    "add": AddOperation(),
    "subtract": SubtractOperation(),
    "multiply": MultiplyOperation(),
    "divide": DivideOperation(),
    "ai": AIOperation()
}

class Calculator:
    def __init__(self, operations):
        self.operations = operations

    def execute_operation(self, operation_key, *args):
        operation = self.operations.get(operation_key)
        if not operation:
            raise ValueError(f"Invalid operation: {operation_key}")
        return operation.execute(*args)

    def get_menu_options(self):
        return [f"{i+1}. {op.get_name()}" for i, op in enumerate(self.operations.values())]

def main_menu(options):
    print("\nMath Operations Menu:")
    for option in options:
        print(option)
    print(f"{len(options) + 1}. Exit")
    return input(f"Choose an option (1-{len(options) + 1}): ")

def get_numbers():
    num1 = float(input("Enter the first number: "))
    num2 = float(input("Enter the second number: "))
    return num1, num2

def main():
    calculator = Calculator(OPERATIONS)
    menu_options = calculator.get_menu_options()

    print("Welcome to the Interactive Math REPL!")
    
    while True:
        choice = main_menu(menu_options)
        
        if choice == str(len(menu_options) + 1):
            print("Thank you for using the Math REPL. Goodbye!")
            break
        
        try:
            choice = int(choice) - 1
            if 0 <= choice < len(menu_options):
                operation_key = list(OPERATIONS.keys())[choice]
                if operation_key == "ai":
                    user_input = input("Enter your math question: ")
                    result = calculator.execute_operation(operation_key, user_input)
                else:
                    num1, num2 = get_numbers()
                    result = calculator.execute_operation(operation_key, num1, num2)
                print(f"\nResult: {result}")
            else:
                raise ValueError
        except ValueError:
            print("Invalid option. Please choose a valid number.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()