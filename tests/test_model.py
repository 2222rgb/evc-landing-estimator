"""Acceptance tests for projection, lag direction, known-split recovery, confounder, low-signal, and conversion-time."""

import numpy as np
import pandas as pd
import pytest

from evc_landing_estimator.adapters import load_conversions, load_evc
from evc_landing_estimator.channels import map_channels
from evc_landing_estimator.config import Config
from evc_landing_estimator.model import assign_confidence_label, fit_model, project_simplex_ball
from evc_landing_estimator.synthetic import KNOWN_SPLIT_TRUE, K_TRUE_IMPRESSION, generate


def test_projection():
    p1 = project_simplex_ball(np.array([0.7, 0.6, -0.1]))
    assert np.all(p1 >= 0.0)
    assert np.isclose(p1.sum(), 1.0, atol=1e-9)
    assert np.allclose(p1, [0.55, 0.45, 0.0], atol=1e-9)

    p2 = project_simplex_ball(np.array([0.2, -0.3, 0.1]))
    assert np.allclose(p2, [0.2, 0.0, 0.1], atol=1e-9)


def test_lag_direction_noise_free():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2026-01-01", periods=120, freq="D")
    evc = pd.Series(rng.uniform(10, 100, size=120), index=dates, name="evc")
    y_direct = 100.0 + 0.5 * evc.shift(1).fillna(0.0)
    y_null = pd.Series(50.0, index=dates)
    conv = pd.DataFrame({"Direct": y_direct, "Other": y_null}, index=dates)

    fit = fit_model(conv, evc, Config(max_lag=3, bootstrap=50, seed=42))
    idx_d = fit.channels.index("Direct")
    assert fit.k[1] >= 0.95
    assert abs(fit.shares[idx_d] - 0.5) <= 0.01


def test_reverse_direction_not_credited():
    rng = np.random.default_rng(11)
    dates = pd.date_range("2026-01-01", periods=160, freq="D")
    evc = pd.Series(rng.uniform(10, 120, size=160), index=dates, name="evc")
    # Future EVC (t+1) drives today's Y(t) -> physically impossible reverse direction
    y_direct = 100.0 + 0.5 * evc.shift(-1).fillna(0.0)
    y_null = pd.Series(50.0 + rng.normal(0, 2, size=160), index=dates)
    conv = pd.DataFrame({"Direct": y_direct, "Other": y_null}, index=dates)

    fit = fit_model(conv, evc, Config(max_lag=3, bootstrap=200, seed=42))
    idx_d = fit.channels.index("Direct")
    lbl = assign_confidence_label(fit.ci_low[idx_d], bool(fit.lead_flag[idx_d]), powered=True)
    assert bool(fit.lead_flag[idx_d]) or lbl in {"Inconclusive", "Confounded"}
    assert lbl != "Strong"


def test_known_split_recovery():
    data = generate("known_split", seed=42)
    cfg = Config(ads_tz="America/New_York", source_tz="UTC", brand_terms=["acme"], bootstrap=250, seed=42)
    loaded = load_conversions(data["crm_rowlevel"], cfg)
    mapped = map_channels(loaded.df, cfg)
    evc = load_evc(data["ads_evc"], cfg)

    fit = fit_model(mapped.conv_daily, evc, cfg)
    assert fit.converged is True
    assert fit.confounded is False

    by_ch = {ch: i for i, ch in enumerate(fit.channels)}
    for ch, true_w in KNOWN_SPLIT_TRUE.items():
        i = by_ch[ch]
        assert abs(fit.shares[i] - true_w) <= 0.08, f"{ch}: got {fit.shares[i]:.3f} vs true {true_w:.3f}"
        if true_w == 0.0:
            lbl = assign_confidence_label(fit.ci_low[i], bool(fit.lead_flag[i]), powered=True)
            assert lbl == "Inconclusive", f"Null channel {ch} got label {lbl} (ci_low={fit.ci_low[i]:.3f})"

    idx_direct = by_ch["Direct"]
    assert fit.channels[int(np.argmax(fit.shares))] == "Direct"
    assert fit.p_top[idx_direct] >= 0.80
    assert abs(fit.not_visible - 0.15) <= 0.10
    assert float(np.sum(np.abs(fit.k - np.array(K_TRUE_IMPRESSION)))) <= 0.30


def test_confounder_flagged():
    data = generate("confounder", seed=42)
    cfg = Config(ads_tz="America/New_York", source_tz="UTC", brand_terms=["acme"], bootstrap=250, seed=42)
    loaded = load_conversions(data["crm_rowlevel"], cfg)
    mapped = map_channels(loaded.df, cfg)
    evc = load_evc(data["ads_evc"], cfg)

    fit = fit_model(mapped.conv_daily, evc, cfg)
    labels = [
        assign_confidence_label(fit.ci_low[i], bool(fit.lead_flag[i]), powered=True)
        for i in range(len(fit.channels))
    ]
    assert "Strong" not in labels, f"Confounder falsely labeled Strong: {list(zip(fit.channels, fit.shares, fit.lead_flag, labels))}"
    assert fit.confounded is True or all(l == "Inconclusive" for l in labels)


def test_conversion_time_path():
    data = generate("conversion_time", seed=42)
    cfg = Config(
        ads_tz="America/New_York",
        source_tz="UTC",
        brand_terms=["acme"],
        date_basis="conversion_time",
        bootstrap=200,
        seed=42,
    )
    loaded = load_conversions(data["crm_rowlevel"], cfg)
    mapped = map_channels(loaded.df, cfg)
    evc = load_evc(data["ads_evc"], cfg)

    fit = fit_model(mapped.conv_daily, evc, cfg)
    assert cfg.max_lag == 0
    assert np.allclose(fit.k, [1.0])
    by_ch = {ch: i for i, ch in enumerate(fit.channels)}
    for ch, true_w in KNOWN_SPLIT_TRUE.items():
        i = by_ch[ch]
        assert abs(fit.shares[i] - true_w) <= 0.08, f"{ch}: got {fit.shares[i]:.3f} vs true {true_w:.3f}"
