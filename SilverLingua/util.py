import logging
import time
from enum import Enum, EnumMeta
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
        f"[{func.__name__}] finished. Time taken: {end - start:.4f} seconds")

    return result

  return wrapper


class MatchingNameEnum(Enum):
  """
    A base Enum class that uses __init_subclass__ to enforce that 
    all derived Enums have the same names as itself.
    
    Raises:
        TypeError: If the derived enum does not have the same names as the base enum.
    """
  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)

    base = cls.__bases__[0]
    base_keys = set(getattr(base, '__members__', {}).keys())
    derived_keys = set(getattr(cls, '__members__', {}).keys())

    missing_keys = base_keys - derived_keys
    extra_keys = derived_keys - base_keys

    if missing_keys or extra_keys:
      error_message = (
          f"Enum {cls.__name__} must have the same names as its base enum {base.__name__}."
      )

      if missing_keys:
        error_message += f" Missing keys: {', '.join(missing_keys)}. "

      if extra_keys:
        error_message += f" Extra keys: {', '.join(extra_keys)}."

      raise TypeError(error_message)
