"""Suppress noisy third-party warnings that are not actionable by users.

This module provides idempotent functions to suppress and restore
third-party warnings from HuggingFace Hub, Transformers, and Lightning.
Suppression is applied automatically when `deepchopper` is imported.
Use `restore_third_party_warnings()` to re-enable all warnings for
debugging (e.g., via the ``--verbose`` CLI flag).

Note: Lightning's ``rank_zero_info`` / ``rank_zero_warn`` already only
fire on rank 0 via the ``@rank_zero_only`` decorator, so no additional
multi-GPU deduplication is needed here.
"""

import logging
import warnings

_suppressed = False
_saved_warning_filters: list | None = None


class _LightningTipFilter(logging.Filter):
    """Filter out Lightning promotional tip messages while keeping useful info.

    This preserves GPU/TPU availability and LOCAL_RANK messages but
    suppresses promotional tips about litlogger, litmodels, etc.
    """

    _SUPPRESSED_PATTERNS = (
        "litlogger",
        "litmodels",
        "LitLogger",
        "LitModelCheckpoint",
        "seamless cloud",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(pattern in msg for pattern in self._SUPPRESSED_PATTERNS)


_tip_filter = _LightningTipFilter()


def suppress_third_party_warnings():
    """Suppress noisy third-party warnings.

    This function is idempotent -- safe to call multiple times.
    Only the first call takes effect; subsequent calls are no-ops.
    """
    global _suppressed, _saved_warning_filters
    if _suppressed:
        return
    _suppressed = True

    # HuggingFace Hub unauthenticated request warning (uses logging, not warnings)
    import huggingface_hub.utils.logging as hf_logging

    hf_logging.set_verbosity_error()

    # Suppress transformers logging and progress bars
    import transformers

    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()

    # Suppress Lightning promotional tips while keeping GPU/device info
    logging.getLogger("lightning.pytorch.utilities.rank_zero").addFilter(_tip_filter)

    # Lightning/PyTorch _pytree LeafSpec deprecation
    _saved_warning_filters = warnings.filters[:]
    warnings.filterwarnings("ignore", message=".*LeafSpec.*deprecated.*")


def restore_third_party_warnings():
    """Restore third-party warnings for verbose/debug mode.

    Reverses all suppressions applied by :func:`suppress_third_party_warnings`.
    """
    global _suppressed, _saved_warning_filters

    import huggingface_hub.utils.logging as hf_logging

    hf_logging.set_verbosity_warning()

    import transformers

    transformers.logging.set_verbosity_warning()
    transformers.logging.enable_progress_bar()

    logging.getLogger("lightning.pytorch.utilities.rank_zero").removeFilter(_tip_filter)

    # Restore only the warning filters we changed (not warnings.resetwarnings())
    if _saved_warning_filters is not None:
        warnings.filters[:] = _saved_warning_filters
        _saved_warning_filters = None

    _suppressed = False
