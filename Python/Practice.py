class Car:
    total_car = 0

    def __init__(self, brand, model):
        self.__brand = brand
        self.model = model
        Car.total_car += 1
        
    def full_name(self):
        return f"{self.__brand} {self.model}"
    
    def get_brand(self):
        return self.__brand
    
    def fuel_type(self):
        return "Petrol & Diesel"
    
    @staticmethod
    def general_description():
        return "Cars are means of transport"
    
class ElectricCar(Car):
    def __init__(self, brand, model, battery_size):
        super().__init__(brand, model)
        self.battery_size = battery_size

    def fuel_type(self):
        return "Electric Charge"


# car = Car("TATA", "Safari")
# ev = ElectricCar("Telsa", "Model S", "85kWh")
# print(ev.fuel_type())

car = Car("TATA", "Nexon")
car = Car("Hyundai", "i20")
car = Car("Mahindra", "Thar")
# print(car.fuel_type())
# print(car.get_brand())
# print(car.total_car)

print(Car.general_description())

