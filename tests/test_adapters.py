"""Tests for CRM, UTM, GA4, wide/pivot, XLSX, and Google Ads EVC adapters."""

from pathlib import Path

import pandas as pd

from evc_landing_estimator.adapters import load_conversions, load_evc
from evc_landing_estimator.config import ColumnOverrides, Config

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_ga4_comments_and_thousands():
    loaded = load_conversions(EXAMPLES / "ga4_daily_export.csv")
    assert loaded.shape == "aggregated_long"
    assert len(loaded.df) > 100
    assert loaded.df["conversions"].sum() > 50_000
    assert "Direct" in set(loaded.df["channel"])


def test_google_ads_title_rows_total_and_dashes(tmp_path):
    p = tmp_path / "ads.csv"
    p.write_text(
        "My YouTube Report\n"
        "Jan 1 - Jan 5, 2026\n"
        "Day,Engaged-view conversions\n"
        "2026-01-01,\"1,025\"\n"
        "2026-01-02,--\n"
        "2026-01-03,42\n"
        "Total,\"1,067\"\n",
        encoding="utf-8",
    )
    evc = load_evc(p)
    assert len(evc) == 3
    assert evc.loc[pd.Timestamp("2026-01-01")] == 1025.0
    assert evc.loc[pd.Timestamp("2026-01-02")] == 0.0
    assert evc.loc[pd.Timestamp("2026-01-03")] == 42.0


def test_crm_dedup_and_timezone_date_boundary(tmp_path):
    # America/New_York in January is UTC-5:
    # 2026-01-02T01:30:00Z -> 2026-01-01 20:30 EST (shifts date back to Jan 1)
    # 2026-01-02T04:59:00Z -> 2026-01-01 23:59 EST (shifts date back to Jan 1)
    # 2026-01-02T05:01:00Z -> 2026-01-02 00:01 EST (stays Jan 2)
    p = tmp_path / "crm.csv"
    p.write_text(
        "lead_id,created_at,utm_source,utm_medium\n"
        "L1,2026-01-02T01:30:00Z,(direct),(none)\n"
        "L2,2026-01-02T04:59:00Z,(direct),(none)\n"
        "L2,2026-01-02T04:59:00Z,(direct),(none)\n"
        "L3,2026-01-02T05:01:00Z,(direct),(none)\n",
        encoding="utf-8",
    )
    cfg = Config(source_tz="UTC", ads_tz="America/New_York")
    loaded = load_conversions(p, cfg)
    assert loaded.shape == "row_level"
    assert loaded.duplicates_removed == 1
    by_day = loaded.df.groupby("date")["conversions"].sum()
    assert by_day.loc[pd.Timestamp("2026-01-01")] == 2.0
    assert by_day.loc[pd.Timestamp("2026-01-02")] == 1.0


def test_wide_and_pivot_formats(tmp_path):
    wide = tmp_path / "wide.csv"
    wide.write_text(
        "Date,Direct,Affiliate\n"
        "2026-01-01,10,5\n"
        "2026-01-02,12,8\n",
        encoding="utf-8",
    )
    w_res = load_conversions(wide)
    assert w_res.shape == "aggregated_wide"
    assert w_res.df["conversions"].sum() == 35.0

    piv = tmp_path / "pivot.csv"
    piv.write_text(
        "Channel,2026-01-01,2026-01-02,2026-01-03\n"
        "Direct,10,12,14\n"
        "Affiliate,5,8,6\n",
        encoding="utf-8",
    )
    p_res = load_conversions(piv)
    assert p_res.shape == "aggregated_pivot"
    assert p_res.df["conversions"].sum() == 55.0


def test_xlsx_matches_csv(tmp_path):
    df = pd.DataFrame(
        {
            "Date": ["2026-01-01", "2026-01-02"],
            "Channel": ["Direct", "Affiliates"],
            "Conversions": [15, 7],
        }
    )
    csv_p = tmp_path / "c.csv"
    xlsx_p = tmp_path / "c.xlsx"
    df.to_csv(csv_p, index=False)
    df.to_excel(xlsx_p, index=False)

    r_csv = load_conversions(csv_p).df
    r_xlsx = load_conversions(xlsx_p).df
    pd.testing.assert_frame_equal(r_csv, r_xlsx)


def test_column_overrides_win(tmp_path):
    p = tmp_path / "custom.csv"
    p.write_text(
        "my_dt,my_src,my_cnt,ignore_date\n"
        "2026-01-01,rakuten,9,2020-01-01\n",
        encoding="utf-8",
    )
    cfg = Config(
        columns=ColumnOverrides(timestamp="my_dt", source="my_src", conversions="my_cnt")
    )
    res = load_conversions(p, cfg)
    assert res.df.loc[0, "date"] == pd.Timestamp("2026-01-01")
    assert res.df.loc[0, "source"] == "rakuten"
    assert res.df.loc[0, "conversions"] == 9.0
