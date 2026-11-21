"""Deterministic synthetic data generator for CRM, UTM, GA4, and Google Ads EVC exports."""

from __future__ import annotations

import argparse
from pathlib import Path
from uuid import UUID
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

DEFAULT_CHANNELS: tuple[str, ...] = (
    "Direct",
    "Paid Brand Search",
    "Paid Non-Brand Search",
    "Organic Brand Search",
    "Organic Non-Brand Search",
    "Affiliate",
    "Email/SMS",
    "Paid Social",
)

CHANNEL_BASE_MU: dict[str, float] = {
    "Direct": 60.0,
    "Paid Brand Search": 35.0,
    "Paid Non-Brand Search": 40.0,
    "Organic Brand Search": 30.0,
    "Organic Non-Brand Search": 50.0,
    "Affiliate": 20.0,
    "Email/SMS": 25.0,
    "Paid Social": 18.0,
}

CHANNEL_DOW_PROFILES: dict[str, list[float]] = {
    "Direct": [1.05, 1.08, 1.06, 1.04, 0.98, 0.88, 0.91],
    "Paid Brand Search": [1.12, 1.10, 1.06, 1.03, 0.95, 0.84, 0.90],
    "Paid Non-Brand Search": [1.08, 1.12, 1.10, 1.05, 0.95, 0.82, 0.88],
    "Organic Brand Search": [1.04, 1.06, 1.05, 1.02, 0.97, 0.92, 0.94],
    "Organic Non-Brand Search": [1.10, 1.14, 1.11, 1.07, 0.94, 0.80, 0.84],
    "Affiliate": [0.96, 0.98, 1.00, 1.02, 1.08, 1.02, 0.94],
    "Email/SMS": [0.85, 1.25, 0.90, 1.22, 1.05, 0.88, 0.85],
    "Paid Social": [0.92, 0.94, 0.96, 1.00, 1.06, 1.08, 1.04],
}

CHANNEL_TRENDS: dict[str, float] = {
    "Direct": 0.08,
    "Paid Brand Search": 0.05,
    "Paid Non-Brand Search": -0.04,
    "Organic Brand Search": 0.10,
    "Organic Non-Brand Search": 0.06,
    "Affiliate": 0.12,
    "Email/SMS": 0.03,
    "Paid Social": -0.05,
}

K_TRUE_IMPRESSION: tuple[float, ...] = (0.45, 0.30, 0.15, 0.10)

KNOWN_SPLIT_TRUE: dict[str, float] = {
    "Direct": 0.40,
    "Paid Brand Search": 0.20,
    "Paid Non-Brand Search": 0.0,
    "Organic Brand Search": 0.10,
    "Organic Non-Brand Search": 0.0,
    "Affiliate": 0.15,
    "Email/SMS": 0.0,
    "Paid Social": 0.0,
}

SCENARIO_SPECS: dict[str, dict] = {
    "known_split": {
        "w_true": dict(KNOWN_SPLIT_TRUE),
        "k_true": list(K_TRUE_IMPRESSION),
        "date_basis": "impression_time",
        "not_visible_true": 0.15,
    },
    "confounder": {
        "w_true": {c: 0.0 for c in DEFAULT_CHANNELS},
        "k_true": list(K_TRUE_IMPRESSION),
        "date_basis": "impression_time",
        "not_visible_true": 1.0,
    },
    "low_signal": {
        "w_true": dict(KNOWN_SPLIT_TRUE),
        "k_true": list(K_TRUE_IMPRESSION),
        "date_basis": "impression_time",
        "not_visible_true": 0.15,
    },
    "conversion_time": {
        "w_true": dict(KNOWN_SPLIT_TRUE),
        "k_true": [1.0],
        "date_basis": "conversion_time",
        "not_visible_true": 0.15,
    },
}

# Raw CRM/UTM tuples per bucket: (lead_source, utm_source, utm_medium, utm_campaign, utm_term, ga4_group)
BUCKET_TO_RAW: dict[str, tuple[str, str, str, str, str, str]] = {
    "Direct": ("Direct", "(direct)", "(none)", "", "", "Direct"),
    "Paid Brand Search": ("Paid Search", "google", "cpc", "acme_brand_exact", "acme shoes", "Paid Search"),
    "Paid Non-Brand Search": ("Paid Search", "google", "cpc", "generic_running_shoes", "trail running shoes", "Paid Search"),
    "Organic Brand Search": ("Organic Search", "google", "organic", "", "acme official", "Organic Search"),
    "Organic Non-Brand Search": ("Organic Search", "google", "organic", "", "best cushioned sneakers", "Organic Search"),
    "Affiliate": ("Affiliate", "rakuten_partner", "affiliate", "spring_coupon_deals", "", "Affiliates"),
    "Email/SMS": ("Email", "klaviyo", "email", "weekly_drop", "", "Email"),
    "Paid Social": ("Paid Social", "facebook", "paid_social", "prospecting_broad", "", "Paid Social"),
}


def _seeded_uuid(rng: np.random.Generator) -> str:
    return str(UUID(bytes=rng.bytes(16), version=4))


def generate(
    scenario: str = "known_split",
    seed: int = 42,
    days: int = 180,
    ads_tz: str = "America/New_York",
    start_date: str = "2026-01-01",
) -> dict[str, pd.DataFrame]:
    """Generate aligned synthetic CRM, UTM, GA4, and Google Ads EVC DataFrames."""
    if scenario not in SCENARIO_SPECS:
        raise ValueError(f"Unknown scenario {scenario!r}. Choose from {sorted(SCENARIO_SPECS)}")

    rng = np.random.default_rng(seed)
    spec = SCENARIO_SPECS[scenario]
    dates = pd.date_range(start_date, periods=days, freq="D")
    dows = dates.dayofweek.to_numpy()
    t_norm = np.arange(days, dtype=float) / max(days, 1)

    # 1. EVC series with 2-week on / 1-week low flighting + pulses
    dow_ads = np.array([1.04, 1.06, 1.05, 1.03, 0.97, 0.90, 0.95])[dows]
    cycle = np.arange(days) % 21
    if scenario == "low_signal":
        flight = np.where(cycle < 14, 1.05, 0.95) * rng.uniform(0.92, 1.08, size=days)
        lam = 4.0 * flight * dow_ads
    else:
        flight = np.where(cycle < 14, 1.0, 0.35) * rng.uniform(0.75, 1.25, size=days)
        lam = 40.0 * flight * dow_ads

    promo_mask = np.zeros(days, dtype=bool)
    if scenario == "confounder":
        # Six 5-day promo waves lifting EVCs and Direct + Paid Brand simultaneously
        for p_start, p_len in [(18, 5), (44, 6), (72, 4), (101, 5), (129, 7), (154, 4)]:
            if p_start + p_len <= days:
                promo_mask[p_start : p_start + p_len] = True
        lam = np.where(promo_mask, lam * 1.6, lam)

    evc_counts = rng.poisson(lam).astype(int)

    # 2. Channel baselines
    daily_counts: dict[str, np.ndarray] = {}
    for ch in DEFAULT_CHANNELS:
        dow_mult = np.array(CHANNEL_DOW_PROFILES[ch])[dows]
        mu = CHANNEL_BASE_MU[ch] * dow_mult * (1.0 + CHANNEL_TRENDS[ch] * t_norm)
        if scenario == "confounder" and ch in ("Direct", "Paid Brand Search"):
            mu = np.where(promo_mask, mu * 1.3, mu)
        daily_counts[ch] = rng.poisson(mu).astype(int)

    # 3. Distribute EVC landings across channels & forward delays
    w_true = spec["w_true"]
    k_true = np.array(spec["k_true"], dtype=float)
    probs = [w_true[c] for c in DEFAULT_CHANNELS]
    not_vis_p = max(0.0, 1.0 - sum(probs))
    alloc_probs = np.array(probs + [not_vis_p], dtype=float)
    alloc_probs /= alloc_probs.sum()

    # For conversion_time, we still generate views on d and stamp EVC on d+L (so conv_day == evc_day)
    evc_stamped = np.zeros(days, dtype=int)
    if scenario == "conversion_time":
        k_delay = np.array(K_TRUE_IMPRESSION, dtype=float)
        for d in range(days):
            n_evc = int(evc_counts[d])
            if n_evc == 0:
                continue
            lags = rng.choice(len(k_delay), size=n_evc, p=k_delay)
            dests = rng.choice(len(DEFAULT_CHANNELS) + 1, size=n_evc, p=alloc_probs)
            for lag, dest in zip(lags, dests):
                c_day = d + int(lag)
                if c_day < days:
                    evc_stamped[c_day] += 1
                    if dest < len(DEFAULT_CHANNELS):
                        daily_counts[DEFAULT_CHANNELS[dest]][c_day] += 1
        evc_reported = evc_stamped
    else:
        for d in range(days):
            n_evc = int(evc_counts[d])
            if n_evc == 0:
                continue
            lags = rng.choice(len(k_true), size=n_evc, p=k_true)
            dests = rng.choice(len(DEFAULT_CHANNELS) + 1, size=n_evc, p=alloc_probs)
            for lag, dest in zip(lags, dests):
                c_day = d + int(lag)
                if c_day < days and dest < len(DEFAULT_CHANNELS):
                    daily_counts[DEFAULT_CHANNELS[dest]][c_day] += 1
        evc_reported = evc_counts

    # 4. Build Ads EVC table
    ads_evc = pd.DataFrame(
        {
            "Day": dates.strftime("%Y-%m-%d"),
            "Engaged-view conversions": evc_reported,
        }
    )

    # 5. Build GA4 daily table (aggregated by GA4 default channel group)
    ga4_rows = []
    for i, d in enumerate(dates):
        d_str = d.strftime("%Y-%m-%d")
        group_totals: dict[str, int] = {}
        for ch in DEFAULT_CHANNELS:
            grp = BUCKET_TO_RAW[ch][5]
            group_totals[grp] = group_totals.get(grp, 0) + int(daily_counts[ch][i])
        for grp, cnt in group_totals.items():
            ga4_rows.append({"Date": d_str, "Session default channel group": grp, "Key events": cnt})
    ga4_daily = pd.DataFrame(ga4_rows)

    # 6. Build row-level CRM & UTM tables with UTC timestamps converted from ads_tz local times
    local_tz = ZoneInfo(ads_tz)
    utc_tz = ZoneInfo("UTC")

    crm_rows = []
    utm_rows = []
    for i, d in enumerate(dates):
        base_local = pd.Timestamp(d).tz_localize(local_tz)
        for ch in DEFAULT_CHANNELS:
            cnt = int(daily_counts[ch][i])
            if cnt == 0:
                continue
            lead_src, utm_src, utm_med, utm_camp, utm_term, _ = BUCKET_TO_RAW[ch]
            sec_offsets = rng.integers(0, 86400, size=cnt)
            messy_flags = rng.random(cnt) < 0.03
            for sec, is_messy in zip(sec_offsets, messy_flags):
                ts_utc = (base_local + pd.Timedelta(seconds=int(sec))).tz_convert(utc_tz).strftime("%Y-%m-%dT%H:%M:%SZ")
                lid = _seeded_uuid(rng)
                cur_lead_src, cur_src, cur_med = lead_src, utm_src, utm_med
                if is_messy:
                    if ch == "Paid Social":
                        cur_src = "Facebook "
                    elif ch == "Email/SMS":
                        cur_med = "EMAIL"
                    elif ch == "Direct":
                        cur_lead_src = "DIRECT"
                crm_rows.append(
                    {
                        "lead_id": lid,
                        "created_at": ts_utc,
                        "lead_source": cur_lead_src,
                        "utm_source": cur_src,
                        "utm_medium": cur_med,
                        "utm_campaign": utm_camp,
                        "utm_term": utm_term,
                    }
                )
                utm_rows.append(
                    {
                        "timestamp": ts_utc,
                        "utm_source": cur_src,
                        "utm_medium": cur_med,
                        "utm_campaign": utm_camp,
                        "utm_term": utm_term,
                        "conversions": 1,
                    }
                )

    crm_df = pd.DataFrame(crm_rows)
    # Inject 2% duplicate lead_ids
    n_dups = max(1, int(round(len(crm_df) * 0.02)))
    dup_idx = rng.choice(len(crm_df), size=n_dups, replace=False)
    crm_df = pd.concat([crm_df, crm_df.iloc[dup_idx]], ignore_index=True)

    utm_df = pd.DataFrame(utm_rows)
    return {
        "crm_rowlevel": crm_df,
        "utm_log": utm_df,
        "ga4_daily": ga4_daily,
        "ads_evc": ads_evc,
    }


def write_examples(out_dir: str | Path, scenario: str = "known_split", seed: int = 42) -> dict[str, Path]:
    """Write synthetic CRM, UTM, GA4 (with # headers), and Google Ads (with title/Total rows) CSVs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = generate(scenario=scenario, seed=seed)

    crm_path = out / "crm_leads_rowlevel.csv"
    data["crm_rowlevel"].to_csv(crm_path, index=False)

    utm_path = out / "utm_conversions_log.csv"
    data["utm_log"].to_csv(utm_path, index=False)

    # GA4 export with # comment header and thousands-formatted integers
    ga4_path = out / "ga4_daily_export.csv"
    ga4_df = data["ga4_daily"].copy()
    ga4_df["Key events"] = ga4_df["Key events"].map(lambda v: f"{int(v):,}")
    with ga4_path.open("w", encoding="utf-8", newline="") as f:
        f.write("# ----------------------------------------\n")
        f.write("# Acme Demo Property - GA4 Key Events by Default Channel Group\n")
        f.write("# Attribution: Paid and organic last click\n")
        f.write("# ----------------------------------------\n")
        ga4_df.to_csv(f, index=False)

    # Google Ads export with 2 title rows and a trailing Total row
    ads_path = out / "google_ads_evc_daily.csv"
    ads_df = data["ads_evc"].copy()
    total_evc = int(ads_df["Engaged-view conversions"].sum())
    ads_df["Engaged-view conversions"] = ads_df["Engaged-view conversions"].map(lambda v: f"{int(v):,}")
    total_row = pd.DataFrame([{"Day": "Total", "Engaged-view conversions": f"{total_evc:,}"}])
    ads_out = pd.concat([ads_df, total_row], ignore_index=True)
    with ads_path.open("w", encoding="utf-8", newline="") as f:
        f.write("Acme YouTube Engaged-View Conversions Report\n")
        f.write("2026-01-01 - 2026-06-29\n")
        ads_out.to_csv(f, index=False)

    return {
        "crm_rowlevel": crm_path,
        "utm_log": utm_path,
        "ga4_daily": ga4_path,
        "ads_evc": ads_path,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic EVC landing datasets.")
    parser.add_argument("--write-examples", type=str, required=True, help="Output directory for example CSVs.")
    parser.add_argument("--scenario", default="known_split", choices=sorted(SCENARIO_SPECS))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    paths = write_examples(args.write_examples, scenario=args.scenario, seed=args.seed)
    for k, p in paths.items():
        print(f"Wrote {k}: {p}")


if __name__ == "__main__":
    _cli()
