import json
from pathlib import Path


class OrderStore:

    def __init__(self, filepath):

        self.filepath = Path(filepath)

        self.orders = {}

        self.load()


    def save(self):

        self.filepath.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(self.filepath, "w") as file:

            json.dump(
                self.orders,
                file,
                indent=4
            )


    def load(self):

        if not self.filepath.exists():

            self.orders = {}
            return

        with open(self.filepath, "r") as file:

            self.orders = json.load(file)


    def record_order(
        self,
        client_order_id,
        symbol,
        side,
        quantity,
        state,
        broker_order_id=None,
        filled_quantity=0
    ):

        self.orders[client_order_id] = {

            "client_order_id": client_order_id,

            "broker_order_id": broker_order_id,

            "symbol": symbol,

            "side": side,

            "quantity": quantity,

            "filled_quantity": filled_quantity,

            "state": state
        }

        self.save()


    def get_order(self, client_order_id):

        return self.orders.get(client_order_id)


    def update_state(
        self,
        client_order_id,
        state,
        filled_quantity=None
    ):

        if client_order_id not in self.orders:

            raise KeyError("order does not exist")

        self.orders[client_order_id]["state"] = state

        if filled_quantity is not None:

            self.orders[
                client_order_id
            ]["filled_quantity"] = filled_quantity

        self.save()

