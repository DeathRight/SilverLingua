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


class MatchingNameEnum:
    """
    A custom Enum that ensures that all derived Enums have the same names.
    They can have different values.

    This allows grandchild classes to be used interchangeably with their
    parent classes. (e.g., `ChatRole` and `OpenAIChatRole`)

    Raises:
        TypeError: If a subclass of a child does not have the same names as
        the child.
    """

    @classmethod
    def keys(cls):
        return [key for key in cls.__dict__ if not key.startswith("_")]

    @classmethod
    def values(cls):
        return [value for key, value in cls.__dict__.items() if not key.startswith("_")]

    @classmethod
    def items(cls):
        return [
            (key, value)
            for key, value in cls.__dict__.items()
            if not key.startswith("_")
        ]

    @classmethod
    def __iter__(cls):
        for key in cls:
            yield key

    @classmethod
    def __init_subclass__(cls):
        # Check if it's a subclass of a subclass of MatchingNameEnum (i.e., a grandchild)
        if any(
            issubclass(base, MatchingNameEnum) and base != MatchingNameEnum
            for base in cls.__bases__
        ):
            base_cls = next(
                base for base in cls.__bases__ if issubclass(base, MatchingNameEnum)
            )
            base_keys = set(base_cls.keys())
            derived_keys = set(cls.keys())

            extra_keys = derived_keys - base_keys
            missing_keys = base_keys - derived_keys

            if extra_keys or missing_keys:
                error_message = []
                if missing_keys:
                    error_message.append(f"\nMissing keys: {', '.join(missing_keys)}")
                if extra_keys:
                    error_message.append(f"\nExtra keys: {', '.join(extra_keys)}")
                raise TypeError(
                    f"{cls.__name__} must have the same keys as {base_cls.__name__}. {' '.join(error_message)}"
                )
