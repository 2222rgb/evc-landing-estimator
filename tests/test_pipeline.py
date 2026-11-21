"""End-to-end pipeline, CLI exit-code, low-signal, and runtime acceptance tests."""

from pathlib import Path
import time

import pytest

from evc_landing_estimator import Config, run
from evc_landing_estimator.cli import main
from evc_landing_estimator.synthetic import KNOWN_SPLIT_TRUE, generate

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_low_signal_end_to_end_inconclusive(tmp_path):
    data = generate("low_signal", seed=42)
    cfg = Config(ads_tz="America/New_York", source_tz="UTC", brand_terms=["acme"], bootstrap=150, power_bootstrap=60, seed=42)
    res = run(data["crm_rowlevel"], data["ads_evc"], config=cfg, out_dir=tmp_path)
    assert all(c.label != "Strong" for c in res.channels)
    assert any(p.powered is False for p in res.power_test)


def test_cli_map_and_preflight_exit_code_2(tmp_path):
    rc_map = main(["map", "--conversions", str(EXAMPLES / "ga4_daily_export.csv")])
    assert rc_map == 0

    # Short 15-day file triggers preflight error -> exit code 2
    bad_conv = tmp_path / "short_conv.csv"
    bad_evc = tmp_path / "short_evc.csv"
    bad_conv.write_text(
        "Date,Channel,Conversions\n" + "\n".join(f"2026-01-{d:02d},Direct,20" for d in range(1, 16)),
        encoding="utf-8",
    )
    bad_evc.write_text(
        "Day,Engaged-view conversions\n" + "\n".join(f"2026-01-{d:02d},{5 + (d % 4)}" for d in range(1, 16)),
        encoding="utf-8",
    )
    rc_err = main(["run", "--conversions", str(bad_conv), "--evc", str(bad_evc), "--out-dir", str(tmp_path / "o")])
    assert rc_err == 2


@pytest.mark.slow
def test_full_bootstrap_500_runtime_and_accuracy(tmp_path):
    t0 = time.perf_counter()
    cfg = Config(
        ads_tz="America/New_York",
        source_tz="UTC",
        brand_terms=["acme"],
        bootstrap=500,
        power_bootstrap=100,
        seed=42,
    )
    res = run(
        EXAMPLES / "crm_leads_rowlevel.csv",
        EXAMPLES / "google_ads_evc_daily.csv",
        config=cfg,
        out_dir=tmp_path,
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, f"Full bootstrap=500 pipeline took {elapsed:.1f}s (limit 30s)"
    assert res.confounded is False
    by_ch = {c.bucket: c for c in res.channels}
    for ch, true_w in KNOWN_SPLIT_TRUE.items():
        assert abs(by_ch[ch].share - true_w) <= 0.08
    assert res.channels[0].bucket == "Direct"
    assert res.channels[0].p_top >= 0.80
