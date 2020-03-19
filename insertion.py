class Insertion:
    def __init__(self, city_id, position_in_solution, cost):
        self.city_id = city_id
        self.position_in_solution = position_in_solution
        self.cost = cost

    def __str__(self):
        return f'Insertion(City {self.city_id}, In {self.position_in_solution}, Cost {self.cost})'

    def __repr__(self):
        return self.__str__()