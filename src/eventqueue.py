from collections import deque

class EventQueue:

    def __init__(self):
        self.queue = deque()

    def add_event(self, event):
        self.queue.append(event)

    def get_event(self):
        if len(self.queue) == 0:
            return None
        else:
            event = self.queue.popleft()
            return event

if __name__ == "__main__":
    event_queue = EventQueue()

    event_queue.add_event("Market Data")
    event_queue.add_event("Signal")
    event_queue.add_event("Order")

    print(event_queue.get_event())
    print(event_queue.get_event())
    print(event_queue.get_event())
    print(event_queue.get_event())


