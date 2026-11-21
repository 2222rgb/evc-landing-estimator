"""Pre-flight data quality checks and planted-signal power test."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evc_landing_estimator.config import Config, PowerTestItem, PreflightItem
from evc_landing_estimator.model import FitOutput, fit_model


class PreflightError(RuntimeError):
    """Raised when one or more pre-flight checks fail with level='error'."""

    def __init__(self, items: list[PreflightItem]):
        self.items = items
        errs = [f"[{it.id}] {it.message}" for it in items if it.level == "error"]
        super().__init__("Pre-flight failed:\n" + "\n".join(errs))


def run_preflight(
    conv_daily: pd.DataFrame,
    evc_daily: pd.Series,
    config: Config,
    unmapped_pct: float = 0.0,
    duplicates_removed: int = 0,
    brand_split_determinable: bool = True,
    raw_conv_dates: pd.Series | None = None,
) -> list[PreflightItem]:
    """Run all pre-flight checks per spec §8.1."""
    items: list[PreflightItem] = [
        PreflightItem(
            id="conversion_action_reminder",
            level="info",
            message=(
                "Confirm the Google Ads EVC conversion action is the same event as your "
                "conversions file (e.g. lead submit vs lead submit)."
            ),
        )
    ]

    # 1. Overlap
    if conv_daily.empty or evc_daily.empty:
        overlap_days = 0
        overlap_pct = 0.0
        window_days = 0
    else:
        c_start, c_end = conv_daily.index.min(), conv_daily.index.max()
        e_start, e_end = evc_daily.index.min(), evc_daily.index.max()
        ov_start, ov_end = max(c_start, e_start), min(c_end, e_end)
        if ov_start > ov_end:
            overlap_days = 0
            overlap_pct = 0.0
            window_days = 0
        else:
            ov_dates = pd.date_range(ov_start, ov_end, freq="D")
            overlap_days = len(ov_dates)
            overlap_pct = float(evc_daily.index.isin(ov_dates).mean())
            window_days = max(0, overlap_days - config.max_lag - 2)

    if overlap_days == 0:
        items.append(
            PreflightItem(
                id="overlap",
                level="error",
                message="Zero overlapping dates between conversions and Google Ads EVC series.",
                value=0.0,
            )
        )
    elif overlap_pct < 0.90:
        items.append(
            PreflightItem(
                id="overlap",
                level="warn",
                message=f"Only {overlap_pct:.1%} of EVC dates overlap with conversions.",
                value=round(overlap_pct, 4),
            )
        )
    else:
        items.append(
            PreflightItem(
                id="overlap",
                level="ok",
                message=f"{overlap_pct:.1%} of EVC dates overlap with conversions.",
                value=round(overlap_pct, 4),
            )
        )

    # 2. Analysis window length
    if window_days < 45:
        items.append(
            PreflightItem(
                id="days",
                level="error",
                message=f"Analysis window is only {window_days} days (minimum 45 required).",
                value=window_days,
            )
        )
    elif window_days < 90:
        items.append(
            PreflightItem(
                id="days",
                level="warn",
                message=f"Analysis window is {window_days} days (< 90 recommended).",
                value=window_days,
            )
        )
    else:
        items.append(
            PreflightItem(
                id="days",
                level="ok",
                message=f"Analysis window has {window_days} usable days.",
                value=window_days,
            )
        )

    # Slice aligned series for EVC distribution checks
    if overlap_days > 0:
        full_ov = pd.date_range(ov_start, ov_end, freq="D")
        evc_al = evc_daily.reindex(full_ov, fill_value=0.0).astype(float)
        conv_al = conv_daily.reindex(full_ov, fill_value=0.0).astype(float)
        if window_days > 0:
            evc_win = evc_al.iloc[config.max_lag : len(evc_al) - 2]
            conv_win = conv_al.iloc[config.max_lag : len(conv_al) - 2]
        else:
            evc_win = evc_al
            conv_win = conv_al
    else:
        evc_win = pd.Series(dtype=float)
        conv_win = pd.DataFrame()

    # 3. EVC coefficient of variation
    evc_mean = float(evc_win.mean()) if len(evc_win) > 0 else 0.0
    evc_cv = float(evc_win.std(ddof=0) / evc_mean) if evc_mean > 0 else 0.0
    if evc_cv < 0.05:
        items.append(
            PreflightItem(
                id="evc_cv",
                level="error",
                message=f"EVC series has almost no day-to-day variation (CV={evc_cv:.3f} < 0.05).",
                value=round(evc_cv, 4),
            )
        )
    elif evc_cv < 0.20:
        items.append(
            PreflightItem(
                id="evc_cv",
                level="warn",
                message=f"Low day-to-day EVC variation (CV={evc_cv:.3f} < 0.20); shares may be noisy.",
                value=round(evc_cv, 4),
            )
        )
    else:
        items.append(
            PreflightItem(
                id="evc_cv",
                level="ok",
                message=f"Healthy EVC day-to-day variation (CV={evc_cv:.2f}).",
                value=round(evc_cv, 4),
            )
        )

    # 4. EVC zero days
    zero_pct = float((evc_win <= 0).mean()) if len(evc_win) > 0 else 1.0
    if zero_pct > 0.70:
        items.append(
            PreflightItem(
                id="evc_zero_days",
                level="error",
                message=f"{zero_pct:.1%} of days have zero EVCs (> 70% limit).",
                value=round(zero_pct, 4),
            )
        )
    elif zero_pct > 0.30:
        items.append(
            PreflightItem(
                id="evc_zero_days",
                level="warn",
                message=f"{zero_pct:.1%} of days have zero EVCs (> 30%).",
                value=round(zero_pct, 4),
            )
        )
    else:
        items.append(
            PreflightItem(
                id="evc_zero_days",
                level="ok",
                message=f"{zero_pct:.1%} of days have zero EVCs.",
                value=round(zero_pct, 4),
            )
        )

    # 5. EVC vs total conversions
    total_evc = float(evc_win.sum()) if len(evc_win) > 0 else 0.0
    total_conv = float(conv_win.to_numpy().sum()) if not conv_win.empty else 0.0
    ratio = (total_evc / total_conv) if total_conv > 0 else float("inf")
    if ratio > 1.2:
        items.append(
            PreflightItem(
                id="evc_vs_conv",
                level="error",
                message=f"Conversion action mismatch? EVCs exceed total conversions (ratio={ratio:.2f} > 1.2).",
                value=round(ratio, 4) if np.isfinite(ratio) else None,
            )
        )
    elif ratio > 0.9:
        items.append(
            PreflightItem(
                id="evc_vs_conv",
                level="warn",
                message=f"EVCs are {ratio:.0%} of total conversions (> 90%); verify conversion actions match.",
                value=round(ratio, 4),
            )
        )
    else:
        items.append(
            PreflightItem(
                id="evc_vs_conv",
                level="ok",
                message=f"EVC-to-total-conversion ratio is {ratio:.1%}.",
                value=round(ratio, 4),
            )
        )

    # 6. Unmapped %
    if unmapped_pct > 0.05:
        items.append(
            PreflightItem(
                id="unmapped",
                level="warn",
                message=f"{unmapped_pct:.1%} of conversions did not match any mapping rule (fell into Other).",
                value=round(unmapped_pct, 4),
            )
        )
    else:
        items.append(
            PreflightItem(
                id="unmapped",
                level="ok",
                message=f"Unmapped conversions: {unmapped_pct:.1%}.",
                value=round(unmapped_pct, 4),
            )
        )

    # 7. Duplicates removed
    items.append(
        PreflightItem(
            id="duplicates",
            level="info" if duplicates_removed > 0 else "ok",
            message=(
                f"Removed {duplicates_removed:,} duplicate conversion IDs."
                if duplicates_removed > 0
                else "No duplicate conversion IDs found."
            ),
            value=duplicates_removed,
        )
    )

    # 8. Date gaps in raw conversions
    if raw_conv_dates is not None and len(raw_conv_dates) > 0:
        uniq = pd.DatetimeIndex(raw_conv_dates.dropna().unique()).sort_values()
        span_days = int((uniq.max() - uniq.min()).days) + 1
        missing_days = max(0, span_days - len(uniq))
        gap_pct = missing_days / max(span_days, 1)
    elif not conv_daily.empty:
        row_sums = conv_daily.sum(axis=1)
        missing_days = int((row_sums <= 0).sum())
        gap_pct = missing_days / max(len(conv_daily), 1)
    else:
        missing_days = 0
        gap_pct = 0.0

    if gap_pct > 0.10:
        items.append(
            PreflightItem(
                id="date_gaps",
                level="error",
                message=f"{missing_days} days ({gap_pct:.1%}) missing from conversions (> 10% limit).",
                value=missing_days,
            )
        )
    elif missing_days > 0:
        items.append(
            PreflightItem(
                id="date_gaps",
                level="warn",
                message=f"{missing_days} days ({gap_pct:.1%}) missing from conversions (filled as 0).",
                value=missing_days,
            )
        )
    else:
        items.append(
            PreflightItem(
                id="date_gaps",
                level="ok",
                message="No missing dates in conversions.",
                value=0,
            )
        )

    # 9. Brand split determinable
    if not brand_split_determinable:
        items.append(
            PreflightItem(
                id="brand_split",
                level="info",
                message="Brand vs non-brand not determinable.",
                value=False,
            )
        )
    else:
        items.append(
            PreflightItem(
                id="brand_split",
                level="ok",
                message="Brand vs non-brand search split applied.",
                value=True,
            )
        )

    return items


def check_preflight_errors(items: list[PreflightItem], force: bool = False) -> None:
    if not force and any(it.level == "error" for it in items):
        raise PreflightError(items)


def run_power_test(
    conv_daily: pd.DataFrame,
    evc_daily: pd.Series,
    base_fit: FitOutput,
    config: Config,
    controls: pd.DataFrame | None = None,
) -> list[PowerTestItem]:
    """Planted-signal power test (+0.25 share per channel) on the user's actual data (§8.2)."""
    planted_share = 0.25
    k_base = np.array([0.45, 0.30, 0.15, 0.10], dtype=float)
    n_lags = config.max_lag + 1
    if n_lags <= len(k_base):
        k_plant = k_base[:n_lags].copy()
    else:
        k_plant = np.concatenate([k_base, np.zeros(n_lags - len(k_base))])
    k_plant /= k_plant.sum()

    # Align EVC onto conv_daily index for planting
    full_idx = conv_daily.index
    evc_al = evc_daily.reindex(full_idx, fill_value=0.0).astype(float)
    exposure = np.zeros(len(full_idx), dtype=float)
    for L, weight in enumerate(k_plant):
        exposure += weight * evc_al.shift(L).fillna(0.0).to_numpy(dtype=float)
    lam_plant = np.maximum(planted_share * exposure, 0.0)

    results: list[PowerTestItem] = []
    for c_idx, ch in enumerate(base_fit.channels):
        base_w = float(base_fit.shares[c_idx])
        if base_w > 0.70:
            results.append(
                PowerTestItem(
                    bucket=ch,
                    planted=planted_share,
                    recovered=None,
                    powered=None,
                    note="Skipped: base share > 0.70 (sum cap could bind).",
                )
            )
            continue

        rng = np.random.default_rng(config.seed + c_idx)
        planted_counts = rng.poisson(lam_plant).astype(float)
        y_planted = conv_daily.copy()
        y_planted[ch] = y_planted[ch] + planted_counts

        fit_p = fit_model(
            y_planted,
            evc_daily,
            config,
            controls=controls,
            bootstrap_override=config.power_bootstrap,
            seed_override=config.seed + c_idx,
        )
        delta = float(fit_p.shares[c_idx] - base_w)
        ci_diff_low = float(fit_p.ci_low[c_idx] - base_w)
        recovered_ok = (abs(delta - planted_share) <= 0.10) and (ci_diff_low > 0.0)
        results.append(
            PowerTestItem(
                bucket=ch,
                planted=planted_share,
                recovered=round(delta, 4),
                powered=bool(recovered_ok),
            )
        )

    return results
