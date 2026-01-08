"""
math_utils.py

Common math helper functions.
Each function is small, focused, and reusable.
"""

def square(num: float) -> float:
    """Return the square of a number."""
    return num * num


def cube(num: float) -> float:
    """Return the cube of a number."""
    return num * num * num


def is_even(num: int) -> bool:
    """Return True if num is even.

    Raises:
        TypeError: If num is not an integer.
    """
    if not isinstance(num, int) or isinstance(num, bool):
        raise TypeError("num must be an integer")

    return num % 2 == 0


