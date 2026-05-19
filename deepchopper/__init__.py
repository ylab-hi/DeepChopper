"""DeepChopper package."""

from . import cli, data, eval, models, train, ui, utils
from .deepchopper import *  # noqa: F403
from .models import DeepChopper
from .utils.suppress_warnings import suppress_third_party_warnings

__version__ = "1.3.4.dev0"

__all__ = ["DeepChopper", "__version__", "cli", "data", "eval", "models", "train", "ui", "utils"]

# Suppress noisy third-party warnings by default.
# Use deepchopper.utils.restore_third_party_warnings() or --verbose to re-enable.
suppress_third_party_warnings()
