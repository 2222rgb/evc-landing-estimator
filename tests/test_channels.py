"""Tests for deterministic channel mapper, brand split, unmapped reporting, and folding."""

import pandas as pd
import pytest

from evc_landing_estimator.channels import map_channels
from evc_landing_estimator.config import Config


CASES = [
    # Direct
    ("(direct)", "(none)", "", "", "", "Direct"),
    ("direct", "", "", "", "", "Direct"),
    ("", "", "", "", "", "Direct"),
    ("(not set)", "(none)", "", "", "", "Direct"),
    ("Typed", "", "", "", "", "Direct"),
    ("", "", "", "", "Direct", "Direct"),
    # Paid Video
    ("youtube", "cpv", "yt_awareness", "", "", "Paid Video"),
    ("youtube", "cpc", "yt_demand_gen", "", "", "Paid Video"),
    ("", "", "", "", "Paid Video", "Paid Video"),
    # Affiliate
    ("rakuten_partner", "affiliate", "spring", "", "", "Affiliate"),
    ("cj", "partner", "", "", "", "Affiliate"),
    ("honey_coupons", "referral", "", "", "", "Affiliate"),
    ("retailmenot", "referral", "", "", "", "Affiliate"),
    ("", "", "", "", "Affiliates", "Affiliate"),
    # Email / SMS
    ("klaviyo", "email", "weekly", "", "", "Email/SMS"),
    ("newsletter", "e-mail", "", "", "", "Email/SMS"),
    ("attentive", "sms", "", "", "", "Email/SMS"),
    ("", "", "", "", "Email", "Email/SMS"),
    ("", "", "", "", "SMS", "Email/SMS"),
    # Paid Social
    ("facebook", "paid_social", "prospecting", "", "", "Paid Social"),
    ("instagram", "cpc", "retargeting", "", "", "Paid Social"),
    ("tiktok", "cpm", "awareness", "", "", "Paid Social"),
    ("", "", "", "", "Paid Social", "Paid Social"),
    # Organic Social
    ("facebook", "referral", "", "", "", "Organic Social"),
    ("reddit", "social", "", "", "", "Organic Social"),
    ("", "", "", "", "Organic Social", "Organic Social"),
    # Paid Search & Organic Search (no brand_terms -> unsplit)
    ("google", "cpc", "generic_shoes", "", "", "Paid Search"),
    ("bing", "ppc", "search_nonbrand", "", "", "Paid Search"),
    ("", "", "", "", "Paid Search", "Paid Search"),
    ("google", "organic", "", "", "", "Organic Search"),
    ("", "", "", "", "Organic Search", "Organic Search"),
    # Display & Referral
    ("gdn", "display", "", "", "", "Display"),
    ("", "", "", "", "Display", "Display"),
    ("nytimes.com", "referral", "", "", "", "Referral"),
    ("", "", "", "", "Referral", "Referral"),
    # Unmapped -> Other
    ("mystery_partner_xyz", "weird_med", "", "", "", "Other"),
]


@pytest.mark.parametrize("source,medium,campaign,term,channel,expected", CASES)
def test_default_mapping_table(source, medium, campaign, term, channel, expected):
    df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-01"),
                "source": source,
                "medium": medium,
                "campaign": campaign,
                "term": term,
                "channel": channel,
                "conversions": 10.0,
            }
        ]
    )
    res = map_channels(df, Config(brand_terms=()))
    assert expected in res.conv_daily.columns
    assert res.conv_daily[expected].sum() == 10.0


def test_brand_split_and_fallback():
    df = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-01"), "source": "google", "medium": "cpc", "campaign": "acme_brand_exact", "term": "", "channel": "", "conversions": 20.0},
            {"date": pd.Timestamp("2026-01-01"), "source": "google", "medium": "cpc", "campaign": "generic_shoes", "term": "", "channel": "", "conversions": 30.0},
            {"date": pd.Timestamp("2026-01-01"), "source": "google", "medium": "organic", "campaign": "", "term": "acme official", "channel": "", "conversions": 15.0},
            {"date": pd.Timestamp("2026-01-01"), "source": "google", "medium": "organic", "campaign": "", "term": "running shoes", "channel": "", "conversions": 25.0},
        ]
    )
    with_brand = map_channels(df, Config(brand_terms=["acme"]))
    assert with_brand.brand_split_determinable is True
    assert with_brand.conv_daily["Paid Brand Search"].sum() == 20.0
    assert with_brand.conv_daily["Paid Non-Brand Search"].sum() == 30.0
    assert with_brand.conv_daily["Organic Brand Search"].sum() == 15.0
    assert with_brand.conv_daily["Organic Non-Brand Search"].sum() == 25.0

    no_brand = map_channels(df, Config(brand_terms=[]))
    assert no_brand.brand_split_determinable is False
    assert no_brand.conv_daily["Paid Search"].sum() == 50.0
    assert no_brand.conv_daily["Organic Search"].sum() == 40.0


def test_unmapped_reporting_and_folding():
    rows = [
        {"date": pd.Timestamp("2026-01-01"), "source": "(direct)", "medium": "(none)", "campaign": "", "term": "", "channel": "", "conversions": 5.0},
        {"date": pd.Timestamp("2026-01-01"), "source": "google", "medium": "cpc", "campaign": "", "term": "", "channel": "", "conversions": 100.0},
        {"date": pd.Timestamp("2026-01-01"), "source": "google", "medium": "organic", "campaign": "", "term": "", "channel": "", "conversions": 90.0},
        {"date": pd.Timestamp("2026-01-01"), "source": "rakuten", "medium": "affiliate", "campaign": "", "term": "", "channel": "", "conversions": 80.0},
        {"date": pd.Timestamp("2026-01-01"), "source": "klaviyo", "medium": "email", "campaign": "", "term": "", "channel": "", "conversions": 70.0},
        {"date": pd.Timestamp("2026-01-01"), "source": "unknown_xyz", "medium": "custom", "campaign": "", "term": "", "channel": "", "conversions": 12.0},
    ]
    res = map_channels(pd.DataFrame(rows), Config(max_channels=4))
    assert len(res.conv_daily.columns) <= 4
    # Direct must always be preserved even though its volume (5) is smallest
    assert "Direct" in res.conv_daily.columns
    assert "Other" in res.conv_daily.columns
    assert len(res.folded_buckets) >= 1
    assert len(res.unmapped) == 1
    assert res.unmapped.loc[0, "conversions"] == 12.0
    assert pytest.approx(res.unmapped.loc[0, "pct"], rel=1e-4) == 12.0 / 357.0
