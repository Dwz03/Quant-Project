# Object = Data + Behaviour

class Position:

    def __init__(self,symbol,quantity):

        if symbol == "":
            raise ValueError("symbol cannot be empty")

        if isinstance(quantity, (int, float)):
            pass
        else:
            raise TypeError("quantity must be a number")
        
        self.symbol = symbol
        self.quantity = quantity

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

if __name__ == "__main__":
    apple = Position("AAPL",100)
    print(apple.symbol)
    print(apple.quantity)
    apple_value = apple.market_value(200)
    print(apple_value)

    microsoft = Position("MSFT",50)
    print(microsoft.symbol)
    print(microsoft.quantity)
    microsoft_value = microsoft.market_value(300)
    print(microsoft_value)

    apple.update_quantity(150)
    print(apple.quantity)

    Tesla = Position("TSLA",30)
    print(Tesla.quantity)
    print(Tesla.symbol)
    tesla_value = Tesla.market_value(250)
    print(tesla_value)

    print(apple.is_long())

    short_position = Position("TSLA", -50)
    print(short_position.is_long())

    a = Position("AAPL", 100)
    b = Position("", 100)
    c = Position("MSFT", "50")
