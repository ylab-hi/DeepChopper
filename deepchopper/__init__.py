"""DeepChopper package."""

from . import cli, data, eval, models, train, ui, utils
from .deepchopper import *  # noqa: F403
from .models import DeepChopper

__version__ = "1.2.9"

__all__ = ["DeepChopper", "__version__", "cli", "data", "eval", "models", "train", "ui", "utils"]
