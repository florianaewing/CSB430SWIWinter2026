#Lambda 
#A lambda is a small anonomous function. 
#It's best to use when you have a simple bit of logic that may not need a full funciton
# They are prefict for being used in another function or funciton call

data = ["cat", None, "dog", None, "cat"]

cleaned = list(filter(lambda x: x is not None, ata))

print(cleaned)
