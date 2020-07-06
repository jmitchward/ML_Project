import time
import logging


def function_timer(function):
    logging.basicConfig(level=logging.DEBUG, format='\n %(asctime)s - %(levelname)s - %(message)s')

    def run_function():
        start_time = time.time()

        function()

        stop_time = time.time()

        logging.DEBUG(function.__name__ + " : " + stop_time - start_time)

    return run_function()



