# =========================
# Docstrings
# Docs: https://peps.python.org/pep-0257/
#
# Docstrings are Python’s version of Javadocs.
# They describe what a module, class, function, or method does.
# They are written using triple quotes: ''' or """
# =========================


def square(num: float) -> float:
    """Return the square of a number.

    Args:
        num: A numeric value to be squared.

    Returns:
        The squared value of num.
    """
    return num * num


# Viewing a docstring
print(square.__doc__)


# =========================
# Design by Contract
#
# A function defines a contract:
# - Preconditions: what must be true before calling it
# - Postconditions: what is guaranteed after it returns
# =========================


def is_even(num: int) -> bool:
    """Return True if num is even.

    Preconditions:
        num must be an integer (not bool).

    Postconditions:
        Returns True if num is even, False otherwise.

    Raises:
        TypeError: If num is not a valid integer.
    """
    if not isinstance(num, int) or isinstance(num, bool):
        raise TypeError("num must be an integer")

    return num % 2 == 0


# =========================
# Typing
# Docs: https://docs.python.org/3/library/typing.html
#
# Type hints improve readability and tooling support.
# They do NOT enforce types at runtime.
# =========================


from typing import List

Vector = List[float]


def scale(scalar: float, vector: Vector) -> Vector:
    """Scale a vector by a scalar.

    Args:
        scalar: Multiplier value.
        vector: List of numeric values.

    Returns:
        A new scaled vector.
    """
    return [scalar * num for num in vector]


# Valid usage
new_vector = scale(2.0, [1.0, -4.2, 1.0])
print(new_vector)


# =========================
# Packages
#
# A package is a folder that contains an __init__.py file.
# This allows Python to treat the folder as importable code.
# =========================


# Example package import (will only work if the package exists)
import some_package.math_utils

print("test")
print(some_package.math_utils.__doc__)

from some_package.math_utils import square
print(square(5))


# =========================
# Your Turn: Build a Small Python Package (Tip Calculator)
#
# Goal:
# Create a Python package with a function that calculates a tip,
# then import and test it from a separate script.
#
# Function contract:
# - Accepts bill and tip_percent (numbers)
# - Validates both are >= 0
# - Returns the tip as a float
# - Optional: round to 2 decimals
#
# Package structure:
# tip_calculator/
#   tip_calculator/
#     __init__.py
#     tip.py
#   test_tip.py
#
# Implementation steps:
# - Write calculate_tip(bill: float, tip_percent: float) -> float
# - Export it in __init__.py
# - Import and test it in test_tip.py
#
# Test cases:
# - bill=50, tip_percent=20  -> 10.0
# - bill=0, tip_percent=15   -> 0.0
# - negative bill            -> error
# - non-numeric bill         -> error
# =========================

#AI tools were used to assist with formatting and grammar. All content and technical decisions are my own.