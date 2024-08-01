import logging
from abc import ABC, abstractmethod
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
from openai import OpenAI
import os
import json
from typing import Literal, Union, List
import yaml
import unittest
import math
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
def load_config(config_path: str = 'config.yaml') -> dict:
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise

config = load_config()

class Settings(BaseSettings):
    openai_api_key: str = Field(default=config['openai_api_key'])
    openai_model: str = Field(default=config['openai_model'])

    class ConfigDict:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
client = OpenAI(api_key=settings.openai_api_key)

# Set up SQLAlchemy
Base = declarative_base()
engine = create_engine('sqlite:///:memory:', echo=True)
Session = sessionmaker(bind=engine)

class Calculation(Base):
    __tablename__ = 'calculations'

    id = Column(Integer, primary_key=True)
    operation = Column(String)
    num1 = Column(Float, nullable=True)
    num2 = Column(Float, nullable=True)
    question = Column(String, nullable=True)
    result = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

class MathOperation(ABC):
    @abstractmethod
    def execute(self, a: float, b: float) -> float:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_help(self) -> str:
        pass

class AddOperation(MathOperation):
    def execute(self, a: float, b: float) -> float:
        return a + b

    def get_name(self) -> str:
        return "Addition"

    def get_help(self) -> str:
        return "Adds two numbers: a + b"

class SubtractOperation(MathOperation):
    def execute(self, a: float, b: float) -> float:
        return a - b

    def get_name(self) -> str:
        return "Subtraction"

    def get_help(self) -> str:
        return "Subtracts the second number from the first: a - b"

class MultiplyOperation(MathOperation):
    def execute(self, a: float, b: float) -> float:
        return a * b

    def get_name(self) -> str:
        return "Multiplication"

    def get_help(self) -> str:
        return "Multiplies two numbers: a * b"

class DivideOperation(MathOperation):
    def execute(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    def get_name(self) -> str:
        return "Division"

    def get_help(self) -> str:
        return "Divides the first number by the second: a / b (b ≠ 0)"

class PowerOperation(MathOperation):
    def execute(self, a: float, b: float) -> float:
        return math.pow(a, b)

    def get_name(self) -> str:
        return "Exponentiation"

    def get_help(self) -> str:
        return "Raises the first number to the power of the second: a ^ b"

class SquareRootOperation(MathOperation):
    def execute(self, a: float, b: float = None) -> float:
        if a < 0:
            raise ValueError("Cannot calculate square root of a negative number")
        return math.sqrt(a)

    def get_name(self) -> str:
        return "Square Root"

    def get_help(self) -> str:
        return "Calculates the square root of a number: √a (a ≥ 0)"

class AIResponse(BaseModel):
    operation: Literal["add", "subtract", "multiply", "divide", "power", "sqrt"]
    num1: float
    num2: Union[float, None] = None

class AIOperation(MathOperation):
    def execute(self, question: str) -> float:
        system_message = """
        You are a helpful assistant that responds to math questions. 
        Your response should be in the following JSON format:
        {
            "operation": "add|subtract|multiply|divide|power|sqrt",
            "num1": <first_number>,
            "num2": <second_number>  // Optional for sqrt
        }
        For square root, only num1 is required.
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
            
            if response.operation == "sqrt":
                return operation.execute(response.num1)
            else:
                return operation.execute(response.num1, response.num2)
        except Exception as e:
            logger.error(f"Error in AI operation: {str(e)}")
            raise

    def get_name(self) -> str:
        return "AI-assisted calculation"

    def get_help(self) -> str:
        return "Ask a math question in natural language, and the AI will interpret and solve it."

OPERATIONS = {
    "add": AddOperation(),
    "subtract": SubtractOperation(),
    "multiply": MultiplyOperation(),
    "divide": DivideOperation(),
    "power": PowerOperation(),
    "sqrt": SquareRootOperation(),
    "ai": AIOperation()
}

class OperationRequest(BaseModel):
    operation: Literal["add", "subtract", "multiply", "divide", "power", "sqrt", "ai"]
    num1: Union[float, None] = None
    num2: Union[float, None] = None
    question: Union[str, None] = None

class Calculator:
    def __init__(self, operations: dict[str, MathOperation]):
        self.operations = operations
        self.session = Session()

    def execute_operation(self, request: OperationRequest) -> float:
        operation = self.operations.get(request.operation)
        if not operation:
            raise ValueError(f"Invalid operation: {request.operation}")
        
        result = None
        if request.operation == "ai":
            if request.question is None:
                raise ValueError("Question is required for AI operation")
            result = operation.execute(request.question)
        elif request.operation == "sqrt":
            if request.num1 is None:
                raise ValueError("Number is required for square root operation")
            result = operation.execute(request.num1)
        else:
            if request.num1 is None or request.num2 is None:
                raise ValueError("Both numbers are required for this operation")
            result = operation.execute(request.num1, request.num2)

        # Save calculation to database
        calculation = Calculation(
            operation=request.operation,
            num1=request.num1,
            num2=request.num2,
            question=request.question,
            result=result
        )
        self.session.add(calculation)
        self.session.commit()

        return result

    def get_menu_options(self) -> List[str]:
        return [f"{i+1}. {op.get_name()}" for i, op in enumerate(self.operations.values())]

    def get_calculation_history(self) -> List[Calculation]:
        return self.session.query(Calculation).all()

def main_menu(options: List[str]) -> str:
    print("\nMath Operations Menu:")
    for option in options:
        print(option)
    print(f"{len(options) + 1}. View Calculation History")
    print(f"{len(options) + 2}. Help")
    print(f"{len(options) + 3}. Exit")
    return input(f"Choose an option (1-{len(options) + 3}): ")

def get_numbers(operation: str) -> Union[tuple[float, float], float]:
    try:
        if operation == "sqrt":
            return float(input("Enter the number: "))
        else:
            num1 = float(input("Enter the first number: "))
            num2 = float(input("Enter the second number: "))
            return num1, num2
    except ValueError:
        logger.warning("Invalid number input")
        raise ValueError("Please enter valid numbers")

def display_help(operations: dict[str, MathOperation]):
    print("\nHelp Menu:")
    for op in operations.values():
        print(f"{op.get_name()}: {op.get_help()}")

def main():
    calculator = Calculator(OPERATIONS)
    menu_options = calculator.get_menu_options()

    logger.info("Starting the Interactive Math REPL")
    print("Welcome to the Interactive Math REPL!")
    
    while True:
        try:
            choice = main_menu(menu_options)
            
            if choice == str(len(menu_options) + 3):
                logger.info("User chose to exit the application")
                print("Thank you for using the Math REPL. Goodbye!")
                break
            
            if choice == str(len(menu_options) + 1):
                history = calculator.get_calculation_history()
                print("\nCalculation History:")
                for calc in history:
                    print(f"{calc.timestamp}: {calc.operation} - Result: {calc.result}")
                continue

            if choice == str(len(menu_options) + 2):
                display_help(OPERATIONS)
                continue

            choice = int(choice) - 1
            if 0 <= choice < len(menu_options):
                operation_key = list(OPERATIONS.keys())[choice]
                if operation_key == "ai":
                    question = input("Enter your math question: ")
                    request = OperationRequest(operation=operation_key, question=question)
                else:
                    numbers = get_numbers(operation_key)
                    if operation_key == "sqrt":
                        request = OperationRequest(operation=operation_key, num1=numbers)
                    else:
                        request = OperationRequest(operation=operation_key, num1=numbers[0], num2=numbers[1])
                
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

    def test_power(self):
        request = OperationRequest(operation="power", num1=2, num2=3)
        self.assertEqual(self.calculator.execute_operation(request), 8)

    def test_square_root(self):
        request = OperationRequest(operation="sqrt", num1=9)
        self.assertEqual(self.calculator.execute_operation(request), 3)

    def test_square_root_negative(self):
        request = OperationRequest(operation="sqrt", num1=-1)
        with self.assertRaises(ValueError):
            self.calculator.execute_operation(request)

    def test_calculation_history(self):
        # Clear the calculation history
        self.calculator.session.query(Calculation).delete()
        self.calculator.session.commit()

        # Add two calculations
        self.calculator.execute_operation(OperationRequest(operation="add", num1=2, num2=3))
        self.calculator.execute_operation(OperationRequest(operation="multiply", num1=4, num2=5))

        # Check the history
        history = self.calculator.get_calculation_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].operation, "add")
        self.assertEqual(history[1].operation, "multiply")

if __name__ == "__main__":
    main()