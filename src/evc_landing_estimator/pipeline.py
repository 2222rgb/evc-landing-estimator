"""End-to-end pipeline wiring adapters, channel mapping, pre-flight, model, power test, and reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from evc_landing_estimator.adapters import load_controls, load_conversions, load_evc
from evc_landing_estimator.channels import MappedChannels, map_channels
from evc_landing_estimator.config import ChannelResult, Config, PreflightItem, Result, TOOL_VERSION
from evc_landing_estimator.model import assign_confidence_label, fit_model
from evc_landing_estimator.preflight import check_preflight_errors, run_power_test, run_preflight
from evc_landing_estimator.report import build_summary_text, write_charts, write_html, write_json, write_xlsx


def preview_map(conversions: str | Path | pd.DataFrame, config: Config | None = None) -> MappedChannels:
    """Load conversions and return channel mapping + unmapped preview without modeling."""
    cfg = config or Config()
    loaded = load_conversions(conversions, cfg)
    return map_channels(loaded.df, cfg)


def run(
    conversions: str | Path | pd.DataFrame,
    evc: str | Path | pd.Series | pd.DataFrame,
    config: Config | None = None,
    out_dir: str | Path | None = None,
) -> Result:
    """Run the full EVC landing estimation pipeline."""
    cfg = config or Config()

    loaded = load_conversions(conversions, cfg)
    mapped = map_channels(loaded.df, cfg)
    evc_daily = evc.copy() if isinstance(evc, pd.Series) else load_evc(evc, cfg)
    controls_df = load_controls(cfg.controls_path) if cfg.controls_path else None

    unmapped_pct = float(mapped.unmapped["pct"].sum()) if not mapped.unmapped.empty else 0.0
    preflight_items = run_preflight(
        conv_daily=mapped.conv_daily,
        evc_daily=evc_daily,
        config=cfg,
        unmapped_pct=unmapped_pct,
        duplicates_removed=loaded.duplicates_removed,
        brand_split_determinable=mapped.brand_split_determinable,
        raw_conv_dates=loaded.df["date"],
    )
    check_preflight_errors(preflight_items, force=cfg.force)

    fit = fit_model(mapped.conv_daily, evc_daily, config=cfg, controls=controls_df)
    preflight_items.append(
        PreflightItem(
            id="converged",
            level="ok" if fit.converged else "warn",
            message="Alternating estimator converged." if fit.converged else "Estimator reached max iterations without full convergence.",
            value=fit.converged,
        )
    )

    power_items = run_power_test(mapped.conv_daily, evc_daily, fit, cfg, controls=controls_df)
    power_by_bucket = {p.bucket: p.powered for p in power_items}

    total_evc_win = float(fit.evc_series.sum())
    total_conv_win = float(fit.actual_y.sum())

    ch_results: list[ChannelResult] = []
    for i, ch in enumerate(fit.channels):
        pwr = power_by_bucket.get(ch)
        lbl = assign_confidence_label(float(fit.ci_low[i]), bool(fit.lead_flag[i]), pwr)
        share_val = float(fit.shares[i])
        ch_results.append(
            ChannelResult(
                bucket=ch,
                share=share_val,
                ci_low=float(fit.ci_low[i]),
                ci_high=float(fit.ci_high[i]),
                p_top=float(fit.p_top[i]),
                p_material=float(fit.p_material[i]),
                lead_coef=float(fit.lead_coef[i]),
                lead_ci=[float(fit.lead_ci_low[i]), float(fit.lead_ci_high[i])],
                lead_flag=bool(fit.lead_flag[i]),
                powered=pwr,
                label=lbl,
                total_conversions=float(fit.actual_y[:, i].sum()),
                implied_evc_conversions=share_val * total_evc_win,
                unconstrained_share=float(fit.unconstrained_shares[i]),
                partial_r2=float(fit.partial_r2[i]),
            )
        )

    ch_results.sort(key=lambda c: c.share, reverse=True)
    summary_text = build_summary_text(ch_results, fit.not_visible, fit.confounded)

    result = Result(
        tool_version=TOOL_VERSION,
        config=cfg.to_report_dict(),
        window={
            "start": str(fit.dates[0].date()),
            "end": str(fit.dates[-1].date()),
            "days": int(len(fit.dates)),
        },
        totals={
            "evc": round(total_evc_win, 2),
            "conversions": round(total_conv_win, 2),
            "evc_to_conversion_ratio": round(total_evc_win / total_conv_win, 4) if total_conv_win > 0 else 0.0,
        },
        channels=ch_results,
        not_visible={
            "share": fit.not_visible,
            "ci_low": fit.not_visible_ci[0],
            "ci_high": fit.not_visible_ci[1],
        },
        delay_curve={
            "k": [float(v) for v in fit.k],
            "ci_low": [float(v) for v in fit.k_ci_low],
            "ci_high": [float(v) for v in fit.k_ci_high],
        },
        confounded=fit.confounded,
        preflight=preflight_items,
        power_test=power_items,
        summary_text=summary_text,
        converged=fit.converged,
        warnings=fit.warnings,
    )

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        chart_paths = write_charts(result, fit, out_path)
        json_p = write_json(result, out_path)
        xlsx_p = write_xlsx(result, fit, mapped, out_path)
        html_p = write_html(result, chart_paths, out_path)
        result.output_files = {
            "json": str(json_p),
            "xlsx": str(xlsx_p),
            "html": str(html_p),
            **{f"chart_{k}": str(v) for k, v in chart_paths.items()},
        }

    return result
