from abc import ABC, abstractmethod

#Python OOP 
class DataTools(ABC):
    def __init__(self, data):
        #Encapulaiton Note
        # Private attributes have a __
        # Protected attributes have a _ 
        self.data = data 

    def describe(self):
        return {"count": len(self.data)}

    def remove_none_value(self):
        cleaned = []
        for item in self.data:
            if item is not None:
                cleaned.append(item)
        self.data = cleaned
    #Abstraction Note
    @abstractmethod
    def mean(self):
        pass

# Inheritance
class NumericDataTools(DataTools):
    def mean(self):
        return sum(self.data) / len(self.data)
    #Override and Polymorphism 
    def describe(self):
       
        result = super().describe()
        result["mean"] = self.mean()
        return result
   

nums = NumericDataTools([10, 20, 30, None])
print(nums.remove_none_value())
print(nums.describe())
print(nums.data)

print(nums.mean())