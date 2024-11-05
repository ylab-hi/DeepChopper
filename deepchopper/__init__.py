"""DeepChopper package."""

from . import cli, data, eval, models, train, ui, utils
from .deepchopper import *  # noqa: F403
from .models import DeepChopper

__version__ = "1.2.6"

__all__ = ["models", "utils", "data", "train", "eval", "DeepChopper", "ui", "cli", "__version__"]
