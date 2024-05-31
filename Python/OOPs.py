class Car:

    total_car = 0

    def __init__(self, brand, model):
        self.__brand = brand
        self.model = model
        Car.total_car += 1

    def get_brand(self):
        return self.__brand + " !"

    def full_name(self):
        return f"{self.__brand} {self.model}"
    
    def fuel_type(self):
        return "Diesel"


class ElectricCar(Car):
    def __init__(self, brand, model, battery_size):
        super().__init__(brand, model)
        self.battery_size = battery_size

    def fuel_type(self):
        return "Electric Charge"

# my_tesla = ElectricCar("Tesla", "Model S", "85kWh")
# print(my_tesla.__brand)
# print(my_tesla.fuel_type()) 

Car("TATA", "Safari")
Car("TATA", "Nexon")
Car("Hyundai", "i20")

print(Car.total_car)
# print(safari.fuel_type())ML

# car1 = Car("Toyota", "Etios")
# print(car1.brand)
# print(car1.model)
# print(car1.full_name())

