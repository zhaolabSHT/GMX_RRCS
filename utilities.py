"""
The `utilities.py` module integrates essential logging, performance profiling, 
and exception handling features aimed at enhancing code robustness and efficiency.

- Logging: Offers `log_error` and `log_warning` functions for recording error and 
  warning messages respectively, color-coded for easy issue identification.
  
- Performance Profiling: The `timing_decorator` decorator automatically measures 
  and logs function execution times, aiding in the optimization of performance 
  bottlenecks.
  
- Exception Handling: Defines custom exceptions `InputFileError`, 
  `ParameterWrongError`, and `ResidueIndexError` for precise capture and handling 
  of specific error scenarios.

This module streamlines error management, boosts self-diagnostic capabilities of 
programs, and assists in optimizing code execution through performance monitoring.
"""


import logging
import timeit
import numpy as np
from numba import jit

from functools import wraps
from termcolor import colored

def log_error(error_type: str, message: str):
    """Log and print the error message."""
    logging.error(message)
    print(colored(f"Error: {error_type}", "red"), message)


def log_warning(warning_type: str, message: str):
    """Log and print the warning message."""
    logging.warning(message)
    print(colored(f"Warning: {warning_type}", "yellow"), message)

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Define an inner function to call the original function with its arguments
        def wrapped_func():
            return func(*args, **kwargs)
        
        # Measure the execution time of the wrapped function using timeit.timeit
        elapsed_time = timeit.timeit(wrapped_func, number=1)
        logging.info(
            f"Function '{func.__name__}' took "
            + colored(f"{elapsed_time:.6f} ", "green")
            + "seconds to complete."
            )
        return wrapped_func()
    return wrapper


class InputFileError(FileNotFoundError):
    pass


class ParameterWrongError(Exception):
    pass


class ResidueIndexError(Exception):
    def __init__(self):
        super().__init__(
        colored(f"Error: ", "red"),
        + "The ResidueIndex file format is incorrect."
    )
        