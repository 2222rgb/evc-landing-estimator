"""Estimate which last-click channels absorb Google Ads Engaged-View Conversions."""

from evc_landing_estimator.config import ChannelResult, ColumnOverrides, Config, Result, TOOL_VERSION

__all__ = ["Config", "ColumnOverrides", "Result", "ChannelResult", "TOOL_VERSION", "run"]


def run(*args, **kwargs) -> Result:
    from evc_landing_estimator.pipeline import run as _run

    return _run(*args, **kwargs)
