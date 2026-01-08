# =========================
# Docstrings
# Docs: https://peps.python.org/pep-0257/
#
# Docstrings are Python’s version of Javadocs.
# They describe what a module, class, function, or method does.
# They are written using triple quotes: ''' or """
# =========================




# =========================
# Design by Contract
#
# A function defines a contract:
# - Preconditions: what must be true before calling it
# - Postconditions: what is guaranteed after it returns
# =========================



# =========================
# Typing
# Docs: https://docs.python.org/3/library/typing.html
#
# Type hints improve readability and tooling support.
# They do NOT enforce types at runtime.
# =========================


# =========================
# Packages
#
# A package is a folder that contains an __init__.py file.
# This allows Python to treat the folder as importable code.
# =========================




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