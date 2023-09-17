import logging
import time
from functools import wraps


def timeit(func):
    """
    Decorator to time the execution of a function and log the time taken.

    Usage:
        @timeit
        def my_function():
            pass

    The time taken for 'my_function' will be logged.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Dynamically grab the logger based on the module where `func` is defined
        logger = logging.getLogger(func.__module__)

        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        logger.debug(
            f"[{func.__name__}] finished. Time taken: {end - start:.4f} seconds"
        )

        return result

    return wrapper


class ImmutableAttributeError(Exception):
    def __init__(self, message, source=None):
        if source is not None:
            message += f" (Source: {repr(source)})"
        super().__init__(message)


class ImmutableMixin:
    def __setattr__(self, name, value):
        raise ImmutableAttributeError("Instances are immutable")
