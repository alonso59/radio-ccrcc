"""Backward-compatible imports for metric history utilities."""

from .metric_history import EpochScalarRecorder
from .metric_plots import save_metric_plots

__all__ = ["EpochScalarRecorder", "save_metric_plots"]
