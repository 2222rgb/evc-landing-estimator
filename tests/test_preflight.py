"""Tests for pre-flight checks, error gating, force bypass, and planted-signal power test."""

import numpy as np
import pandas as pd
import pytest

from evc_landing_estimator.adapters import load_conversions, load_evc
from evc_landing_estimator.channels import map_channels
from evc_landing_estimator.config import Config
from evc_landing_estimator.model import fit_model
from evc_landing_estimator.preflight import PreflightError, check_preflight_errors, run_power_test, run_preflight
from evc_landing_estimator.synthetic import generate


def _make_clean_series(days: int = 120):
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    rng = np.random.default_rng(1)
    conv = pd.DataFrame({"Direct": rng.poisson(60, size=days), "Affiliate": rng.poisson(30, size=days)}, index=dates, dtype=float)
    evc = pd.Series(rng.uniform(15, 65, size=days), index=dates, name="evc")
    return conv, evc


def _by_id(items):
    return {it.id: it for it in items}


def test_preflight_warn_and_error_thresholds():
    # Clean 120-day baseline -> all ok
    conv, evc = _make_clean_series(120)
    res = _by_id(run_preflight(conv, evc, Config()))
    assert res["days"].level == "ok"
    assert res["overlap"].level == "ok"
    assert res["evc_cv"].level == "ok"
    assert res["evc_zero_days"].level == "ok"
    assert res["evc_vs_conv"].level == "ok"
    assert res["unmapped"].level == "ok"
    assert res["date_gaps"].level == "ok"

    # Short window: warn (<90) and error (<45)
    c60, e60 = _make_clean_series(60)
    assert _by_id(run_preflight(c60, e60, Config()))["days"].level == "warn"
    c30, e30 = _make_clean_series(30)
    items_err = run_preflight(c30, e30, Config())
    assert _by_id(items_err)["days"].level == "error"
    with pytest.raises(PreflightError):
        check_preflight_errors(items_err, force=False)
    check_preflight_errors(items_err, force=True)  # force bypasses error

    # Flat EVC (CV < 0.05) -> error; low CV (< 0.20) -> warn
    evc_flat = pd.Series(40.0, index=conv.index)
    assert _by_id(run_preflight(conv, evc_flat, Config()))["evc_cv"].level == "error"

    # Zero days (>30% warn, >70% error)
    evc_z40 = evc.copy()
    evc_z40.iloc[:50] = 0.0
    assert _by_id(run_preflight(conv, evc_z40, Config()))["evc_zero_days"].level == "warn"
    evc_z80 = evc.copy()
    evc_z80.iloc[:100] = 0.0
    assert _by_id(run_preflight(conv, evc_z80, Config()))["evc_zero_days"].level == "error"

    # EVC > total conversions (>1.2 error, >0.9 warn)
    assert _by_id(run_preflight(conv, evc * 5.0, Config()))["evc_vs_conv"].level == "error"

    # Unmapped > 5% -> warn; duplicates > 0 -> info; brand_split=False -> info
    misc = _by_id(
        run_preflight(conv, evc, Config(), unmapped_pct=0.08, duplicates_removed=12, brand_split_determinable=False)
    )
    assert misc["unmapped"].level == "warn"
    assert misc["duplicates"].level == "info"
    assert misc["brand_split"].level == "info"

    # Zero overlap -> error
    evc_shifted = evc.copy()
    evc_shifted.index = evc_shifted.index + pd.Timedelta(days=400)
    assert _by_id(run_preflight(conv, evc_shifted, Config()))["overlap"].level == "error"


def test_power_test_low_signal_detects_underpowered():
    data = generate("low_signal", seed=42)
    cfg = Config(ads_tz="America/New_York", source_tz="UTC", brand_terms=["acme"], bootstrap=100, power_bootstrap=60, seed=42)
    mapped = map_channels(load_conversions(data["crm_rowlevel"], cfg).df, cfg)
    evc = load_evc(data["ads_evc"], cfg)
    base_fit = fit_model(mapped.conv_daily, evc, cfg)
    pwr = run_power_test(mapped.conv_daily, evc, base_fit, cfg)
    assert any(p.powered is False for p in pwr), f"Expected underpowered channel on low_signal: {pwr}"


def test_power_test_known_split_recovers_planted_signal():
    data = generate("known_split", seed=42)
    cfg = Config(ads_tz="America/New_York", source_tz="UTC", brand_terms=["acme"], bootstrap=100, power_bootstrap=60, seed=42)
    mapped = map_channels(load_conversions(data["crm_rowlevel"], cfg).df, cfg)
    evc = load_evc(data["ads_evc"], cfg)
    base_fit = fit_model(mapped.conv_daily, evc, cfg)
    pwr = run_power_test(mapped.conv_daily, evc, base_fit, cfg)
    powered_true = [p for p in pwr if p.powered is True]
    assert len(powered_true) >= 5, f"Expected most channels powered on known_split, got: {pwr}"
