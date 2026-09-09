import time
import uuid


class OrderSubmissionError(Exception):
    pass


class SafeOrderSubmitter:

    def __init__(
        self,
        client,
        max_retries=3,
        retry_delay_seconds=0
    ):

        self.client = client
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds


    def generate_client_order_id(self):

        return str(uuid.uuid4())


    def submit(
        self,
        create_order_request,
        client_order_id
    ):

        last_error = None

        for attempt in range(self.max_retries):

            try:

                return self.client.submit_order(
                    create_order_request(client_order_id)
                )

            except (TimeoutError, ConnectionError) as error:

                last_error = error

                # Important:
                # maybe Alpaca already received the order
                try:

                    existing_order = (
                        self.client.get_order_by_client_id(
                            client_order_id
                        )
                    )

                    return existing_order

                except Exception:

                    pass

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay_seconds)

        raise OrderSubmissionError(
            f"failed to submit order after "
            f"{self.max_retries} attempts"
        ) from last_error