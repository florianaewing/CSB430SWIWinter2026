# Decorators and Closures
# In Python, decorators let us extend or modify a function’s behavior
# without changing the function itself.
# A decorator takes a function as input and returns a new version of that function.

# Decorators commonly use closures.
# A closure is a function that remembers values from the environment
# where it was created.

# A closure typically has:
# - A function defined inside another function (a nested function)
# - The inner function referencing variables from the outer function
# - The outer function returning the inner function

# A closure becomes a decorator when it takes a function as an argument
# and returns a modified version of that function.

# This decorator transforms categorical data into vectors
def to_vectors(func):
    def wrapper(data):
        print(data)
        categories = sorted(set(data))
        vectors = []

        for item in data:
            row = []
            for category in categories:
                row.append(1 if item == category else 0)
            vectors.append(row)

        return func(vectors)
    return wrapper


# Count the frequency of our categories
@to_vectors
def count_by_category(vectors):
    print(vectors)
    totals = [0] * len(vectors[0])

    for row in vectors:
        for i in range(len(row)):
            totals[i] += row[i]

    return totals


animals = ["cat", "dog", "cat", "cat"]

result = count_by_category(animals)
print(result)