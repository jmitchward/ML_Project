import time


class function_timer:
    def __init__(self):
        self.start_time = None

    def begin_time(self):
        self.start_time = time.perf_counter()

    def end_time(self):
        self.stop_time = time.perf_counter()

        elapsed_time = self.stop_time - self.start_time
        self.start_time = None

        print(elapsed_time)
