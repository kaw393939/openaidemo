import logging
from abc import ABC, abstractmethod
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
from openai import OpenAI
import os
import json
from typing import Literal, Union
import yaml
import unittest

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
def load_config(config_path: str = 'config.yaml') -> dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()

class Settings(BaseSettings):
    openai_api_key: str = Field(default=config['openai_api_key'])
    openai_model: str = Field(default=config['openai_model'])

    class ConfigDict:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
client = OpenAI(api_key=settings.openai_api_key)

class MathOperation(ABC):
    @abstractmethod
    def execute(self, a: float, b: float) -> float:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

class AddOperation(MathOperation):
    def execute(self, a: float, b: float) -> float:
        return a + b

    def get_name(self) -> str:
        return "Addition"

class SubtractOperation(MathOperation):
    def execute(self, a: float, b: float) -> float:
        return a - b

    def get_name(self) -> str:
        return "Subtraction"

class MultiplyOperation(MathOperation):
    def execute(self, a: float, b: float) -> float:
        return a * b

    def get_name(self) -> str:
        return "Multiplication"

class DivideOperation(MathOperation):
    def execute(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    def get_name(self) -> str:
        return "Division"

class AIResponse(BaseModel):
    operation: Literal["add", "subtract", "multiply", "divide"]
    num1: float
    num2: float

class AIOperation(MathOperation):
    def execute(self, question: str) -> float:
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

        try:
            completion = client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                response_format={"type": "json_object"}
            )

            response = AIResponse.parse_raw(completion.choices[0].message.content)
            logger.info(f"AI Response: {response}")
            
            operation = OPERATIONS.get(response.operation)
            if not operation:
                raise ValueError(f"Invalid operation: {response.operation}")
            
            return operation.execute(response.num1, response.num2)
        except Exception as e:
            logger.error(f"Error in AI operation: {str(e)}")
            raise

    def get_name(self) -> str:
        return "AI-assisted calculation"

OPERATIONS = {
    "add": AddOperation(),
    "subtract": SubtractOperation(),
    "multiply": MultiplyOperation(),
    "divide": DivideOperation(),
    "ai": AIOperation()
}

class OperationRequest(BaseModel):
    operation: Literal["add", "subtract", "multiply", "divide", "ai"]
    num1: Union[float, None] = None
    num2: Union[float, None] = None
    question: Union[str, None] = None

class Calculator:
    def __init__(self, operations: dict[str, MathOperation]):
        self.operations = operations

    def execute_operation(self, request: OperationRequest) -> float:
        operation = self.operations.get(request.operation)
        if not operation:
            raise ValueError(f"Invalid operation: {request.operation}")
        
        if request.operation == "ai":
            if request.question is None:
                raise ValueError("Question is required for AI operation")
            return operation.execute(request.question)
        else:
            if request.num1 is None or request.num2 is None:
                raise ValueError("Both numbers are required for non-AI operations")
            return operation.execute(request.num1, request.num2)

    def get_menu_options(self) -> list[str]:
        return [f"{i+1}. {op.get_name()}" for i, op in enumerate(self.operations.values())]

def main_menu(options: list[str]) -> str:
    print("\nMath Operations Menu:")
    for option in options:
        print(option)
    print(f"{len(options) + 1}. Exit")
    return input(f"Choose an option (1-{len(options) + 1}): ")

def get_numbers() -> tuple[float, float]:
    num1 = float(input("Enter the first number: "))
    num2 = float(input("Enter the second number: "))
    return num1, num2

def main():
    calculator = Calculator(OPERATIONS)
    menu_options = calculator.get_menu_options()

    logger.info("Starting the Interactive Math REPL")
    print("Welcome to the Interactive Math REPL!")
    
    while True:
        choice = main_menu(menu_options)
        
        if choice == str(len(menu_options) + 1):
            logger.info("User chose to exit the application")
            print("Thank you for using the Math REPL. Goodbye!")
            break
        
        try:
            choice = int(choice) - 1
            if 0 <= choice < len(menu_options):
                operation_key = list(OPERATIONS.keys())[choice]
                if operation_key == "ai":
                    question = input("Enter your math question: ")
                    request = OperationRequest(operation=operation_key, question=question)
                else:
                    num1, num2 = get_numbers()
                    request = OperationRequest(operation=operation_key, num1=num1, num2=num2)
                
                result = calculator.execute_operation(request)
                logger.info(f"Operation: {operation_key}, Result: {result}")
                print(f"\nResult: {result}")
            else:
                raise ValueError("Invalid menu option")
        except ValueError as e:
            logger.warning(f"Invalid input: {str(e)}")
            print(f"Invalid input: {str(e)}")
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            print(f"An error occurred: {str(e)}")

# Unit tests
class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.calculator = Calculator(OPERATIONS)

    def test_addition(self):
        request = OperationRequest(operation="add", num1=2, num2=3)
        self.assertEqual(self.calculator.execute_operation(request), 5)

    def test_subtraction(self):
        request = OperationRequest(operation="subtract", num1=5, num2=3)
        self.assertEqual(self.calculator.execute_operation(request), 2)

    def test_multiplication(self):
        request = OperationRequest(operation="multiply", num1=2, num2=3)
        self.assertEqual(self.calculator.execute_operation(request), 6)

    def test_division(self):
        request = OperationRequest(operation="divide", num1=6, num2=3)
        self.assertEqual(self.calculator.execute_operation(request), 2)

    def test_division_by_zero(self):
        request = OperationRequest(operation="divide", num1=6, num2=0)
        with self.assertRaises(ValueError):
            self.calculator.execute_operation(request)

if __name__ == "__main__":
    main()