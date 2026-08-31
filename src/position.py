# Object = Data + Behaviour

class Position:

    def __init__(self,symbol,quantity, average_cost):

        if symbol == "":
            raise ValueError("symbol cannot be empty")

        if isinstance(quantity, (int, float)):
            pass
        else:
            raise TypeError("quantity must be a number")

        if quantity <= 0:
            raise ValueError("quantity must be positive")

        if not isinstance(average_cost, (int, float)):
            raise TypeError("average cost must be a number")

        if average_cost <= 0:
            raise ValueError("average cost must be positive")
        
        self.symbol = symbol
        self.quantity = quantity
        self.average_cost = average_cost

    def market_value(self,price):
        return self.quantity * price

    def update_quantity(self,new_quantity):

        if not isinstance(new_quantity, (int, float)):
            raise TypeError("quantity must be a number")

        self.quantity = new_quantity

    def is_long(self):
        if self.quantity > 0:
            return True
        else:
            return False

    def add_quantity(self, quantity, price):

        if not isinstance(quantity, (int, float)):
            raise TypeError("quantity must be a number")

        if quantity <= 0:
            raise ValueError("quantity must be positive")

        if not isinstance(price, (int, float)):
            raise TypeError("price must be a number")

        if price <= 0:
            raise ValueError("price must be positive")

        total_cost = self.average_cost * self.quantity + quantity * price

        self.average_cost = total_cost / (self.quantity + quantity)

        self.quantity = quantity + self.quantity

    def reduce_quantity(self, quantity, price):

        if not isinstance(quantity, (int, float)):
            raise TypeError("quantity must be a number")

        if not isinstance(price, (int, float)):
            raise TypeError("price must be a number")

        if price <= 0:
            raise ValueError("price must be positive")

        if quantity <= 0:
            raise ValueError("quantity must be positive")

        if quantity > self.quantity:
            raise ValueError("we could not selling more than what we have")

        realised_pnl = (price - self.average_cost) * quantity

        new_quantity = self.quantity - quantity

        self.update_quantity(new_quantity)

        return realised_pnl

    def unrealised_pnl(self, price):

        total = (price - self.average_cost) * self.quantity

        return total

        


