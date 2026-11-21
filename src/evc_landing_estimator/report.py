"""Reporting outputs: results.json, results.xlsx, PNG charts, and self-contained report.html."""

from __future__ import annotations

import base64
from dataclasses import asdict
import html
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evc_landing_estimator.channels import MappedChannels
from evc_landing_estimator.config import ChannelResult, Result
from evc_landing_estimator.model import FitOutput

LABEL_COLORS = {
    "Strong": "#10b981",
    "Directional": "#2563eb",
    "Inconclusive": "#9ca3af",
    "Confounded": "#ef4444",
}

plt.rcParams.update(
    {
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#d1d5db",
        "axes.labelcolor": "#111827",
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "text.color": "#111827",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "axes.grid": True,
        "grid.color": "#e5e7eb",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
    }
)


def _clean_float(val: float | None, ndigits: int = 4) -> float | None:
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return None
    return round(float(val), ndigits)


def build_summary_text(
    channels: list[ChannelResult],
    not_visible_share: float,
    confounded: bool,
) -> str:
    """Deterministic plain-English summary per spec §9.5 (zero LLM calls)."""
    prefix = ""
    if confounded:
        prefix = (
            "⚠ Results may be confounded by shared demand shocks (future EVCs predict current conversions). "
            "Add promo/holiday controls (--controls) and treat shares as upper bounds. "
        )

    reliable = [c for c in channels if c.label in ("Strong", "Directional")]
    inconclusive = [c for c in channels if c.label == "Inconclusive"]

    if not reliable:
        return prefix + "No channel shows a reliable EVC landing signal in this data."

    top = reliable[0]
    parts = [
        f"Most likely landing channel: {top.bucket} ({top.share:.0%}, 90% CI {top.ci_low:.0%}–{top.ci_high:.0%}, P(#1) = {top.p_top:.0%})."
    ]
    others = reliable[1:]
    if others:
        other_str = ", ".join(f"{c.bucket} ({c.share:.0%})" for c in others)
        parts.append(f"Also likely: {other_str}.")
    if inconclusive:
        inc_str = ", ".join(c.bucket for c in inconclusive)
        parts.append(f"Inconclusive: {inc_str}.")
    parts.append(f"About {not_visible_share:.0%} of EVCs are not visible in this data.")
    return prefix + " ".join(parts)


def result_to_dict(result: Result) -> dict:
    """Serialize Result to strict RFC-8259 JSON dict matching spec §9.1."""
    return {
        "tool_version": result.tool_version,
        "config": result.config,
        "window": result.window,
        "totals": result.totals,
        "channels": [
            {
                "bucket": c.bucket,
                "share": _clean_float(c.share),
                "ci_low": _clean_float(c.ci_low),
                "ci_high": _clean_float(c.ci_high),
                "p_top": _clean_float(c.p_top),
                "p_material": _clean_float(c.p_material),
                "lead_coef": _clean_float(c.lead_coef),
                "lead_ci": [_clean_float(c.lead_ci[0]), _clean_float(c.lead_ci[1])],
                "lead_flag": bool(c.lead_flag),
                "powered": c.powered,
                "label": c.label,
                "total_conversions": _clean_float(c.total_conversions, 2),
                "implied_evc_conversions": _clean_float(c.implied_evc_conversions, 1),
            }
            for c in result.channels
        ],
        "not_visible": {
            "share": _clean_float(result.not_visible["share"]),
            "ci_low": _clean_float(result.not_visible["ci_low"]),
            "ci_high": _clean_float(result.not_visible["ci_high"]),
        },
        "delay_curve": {
            "k": [_clean_float(v) for v in result.delay_curve["k"]],
            "ci_low": [_clean_float(v) for v in result.delay_curve["ci_low"]],
            "ci_high": [_clean_float(v) for v in result.delay_curve["ci_high"]],
        },
        "confounded": bool(result.confounded),
        "preflight": [asdict(p) for p in result.preflight],
        "power_test": [
            {
                "bucket": p.bucket,
                "planted": _clean_float(p.planted),
                "recovered": _clean_float(p.recovered),
                "powered": p.powered,
                **({"note": p.note} if p.note else {}),
            }
            for p in result.power_test
        ],
        "summary_text": result.summary_text,
    }


def write_json(result: Result, out_dir: Path) -> Path:
    p = out_dir / "results.json"
    p.write_text(json.dumps(result_to_dict(result), indent=2) + "\n", encoding="utf-8")
    return p


def write_charts(result: Result, fit: FitOutput, out_dir: Path) -> dict[str, Path]:
    """Render shares.png, p_top.png, delay_curve.png, and top_channel_fit.png."""
    paths: dict[str, Path] = {}

    # 1. shares.png
    fig, ax = plt.subplots(figsize=(9.5, max(4.2, 0.52 * (len(result.channels) + 2))))
    rows = list(result.channels)
    labels = [c.bucket for c in rows] + ["Not visible"]
    vals = [c.share * 100.0 for c in rows] + [result.not_visible["share"] * 100.0]
    lo_err = [max(0.0, (c.share - c.ci_low) * 100.0) for c in rows] + [
        max(0.0, (result.not_visible["share"] - result.not_visible["ci_low"]) * 100.0)
    ]
    hi_err = [max(0.0, (c.ci_high - c.share) * 100.0) for c in rows] + [
        max(0.0, (result.not_visible["ci_high"] - result.not_visible["share"]) * 100.0)
    ]
    colors = [LABEL_COLORS.get(c.label, "#9ca3af") for c in rows] + ["#e5e7eb"]
    y = np.arange(len(labels))[::-1]

    bars = ax.barh(y, vals, color=colors, height=0.62, edgecolor="#374151", linewidth=0.5)
    bars[-1].set_hatch("///")
    ax.errorbar(vals, y, xerr=[lo_err, hi_err], fmt="none", ecolor="#111827", elinewidth=1.2, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlim(0, max(50.0, max(v + h for v, h in zip(vals, hi_err)) + 12.0))
    ax.set_xlabel("Estimated Share of YouTube EVCs (% with 90% Block-Bootstrap CI)", fontsize=10, fontweight="bold")
    title = "Where YouTube Engaged-View Conversions Land (Last-Click Share)"
    if result.confounded:
        title += "  [CONFOUNDED]"
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
    ax.grid(axis="y", visible=False)
    for bar, v, h, idx in zip(bars, vals, hi_err, range(len(labels))):
        tag = f" ({rows[idx].label})" if idx < len(rows) else ""
        ax.text(v + h + 1.2, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%{tag}", va="center", fontsize=8.5)
    plt.tight_layout()
    p_shares = out_dir / "shares.png"
    fig.savefig(p_shares, dpi=160)
    plt.close(fig)
    paths["shares"] = p_shares

    # 2. p_top.png
    fig, ax = plt.subplots(figsize=(8.8, max(3.8, 0.45 * len(rows) + 1.2)))
    p_labels = [c.bucket for c in rows]
    p_vals = [c.p_top * 100.0 for c in rows]
    p_cols = [LABEL_COLORS.get(c.label, "#9ca3af") for c in rows]
    yp = np.arange(len(p_labels))[::-1]
    bars = ax.barh(yp, p_vals, color=p_cols, height=0.58)
    ax.set_yticks(yp)
    ax.set_yticklabels(p_labels, fontsize=9.5)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Bootstrap Probability of Being #1 Landing Channel (%)", fontsize=10, fontweight="bold")
    ax.set_title("Probability Each Channel Is the #1 EVC Landing Spot", fontsize=11.5, fontweight="bold", pad=10)
    ax.grid(axis="y", visible=False)
    for bar, v in zip(bars, p_vals):
        ax.text(v + 1.5, bar.get_y() + bar.get_height() / 2, f"{v:.0f}%", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    p_top = out_dir / "p_top.png"
    fig.savefig(p_top, dpi=160)
    plt.close(fig)
    paths["p_top"] = p_top

    # 3. delay_curve.png
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    k_vals = np.array(result.delay_curve["k"], dtype=float)
    lags = np.arange(len(k_vals))
    if np.isnan(k_vals).any():
        ax.text(0.5, 0.5, "Delay curve unavailable (no positive EVC signal detected)", ha="center", va="center", transform=ax.transAxes, fontsize=10)
    else:
        k_lo = np.array([v if v is not None else np.nan for v in result.delay_curve["ci_low"]], dtype=float)
        k_hi = np.array([v if v is not None else np.nan for v in result.delay_curve["ci_high"]], dtype=float)
        lo_e = np.maximum(0.0, (k_vals - k_lo) * 100.0)
        hi_e = np.maximum(0.0, (k_hi - k_vals) * 100.0)
        bars = ax.bar(lags, k_vals * 100.0, color="#2563eb", width=0.55)
        if not np.isnan(lo_e).any():
            ax.errorbar(lags, k_vals * 100.0, yerr=[lo_e, hi_e], fmt="none", ecolor="#111827", elinewidth=1.2, capsize=4)
        for bar, v, h in zip(bars, k_vals * 100.0, np.nan_to_num(hi_e, nan=0.0)):
            ax.text(bar.get_x() + bar.get_width() / 2, v + h + 2.0, f"{v:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(lags)
    ax.set_xticklabels([f"Day +{L}" for L in lags], fontsize=9.5)
    ax.set_ylim(0, 112)
    ax.set_xlabel("Days After Engaged View", fontsize=10, fontweight="bold")
    ax.set_ylabel("Share of Conversions (%)", fontsize=10, fontweight="bold")
    ax.set_title("Shared View-to-Conversion Delay Curve", fontsize=11.5, fontweight="bold", pad=10)
    ax.grid(axis="x", visible=False)
    plt.tight_layout()
    p_delay = out_dir / "delay_curve.png"
    fig.savefig(p_delay, dpi=160)
    plt.close(fig)
    paths["delay_curve"] = p_delay

    # 4. top_channel_fit.png
    top_bucket = result.channels[0].bucket
    c_idx = fit.channels.index(top_bucket)
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    actual = fit.actual_y[:, c_idx]
    base = fit.baseline_fit[:, c_idx]
    total_fit = base + fit.evc_component[:, c_idx]

    ax.plot(fit.dates, actual, label="Actual Conversions", color="#111827", alpha=0.55, linewidth=1.2)
    ax.plot(fit.dates, base, label="Baseline (Trend + Day-of-Week)", color="#9ca3af", linestyle="--", linewidth=1.4)
    ax.plot(fit.dates, total_fit, label="Baseline + EVC Landing Component", color="#10b981", linewidth=1.7)
    ax.fill_between(fit.dates, base, total_fit, color="#10b981", alpha=0.20)
    ax.set_ylabel("Daily Conversions", fontsize=10, fontweight="bold")
    ax.set_title(f"Daily Fit — Top Channel: {top_bucket} (Estimated Share {result.channels[0].share:.1%})", fontsize=11.5, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=8.8)
    plt.tight_layout()
    p_fit = out_dir / "top_channel_fit.png"
    fig.savefig(p_fit, dpi=160)
    plt.close(fig)
    paths["top_channel_fit"] = p_fit

    return paths


def write_xlsx(result: Result, fit: FitOutput, mapped: MappedChannels, out_dir: Path) -> Path:
    """Write 8-sheet workbook matching spec §9.2."""
    p = out_dir / "results.xlsx"

    summary_df = pd.DataFrame(
        [
            {"Metric": "Summary", "Value": result.summary_text},
            {"Metric": "Window", "Value": f"{result.window['start']} to {result.window['end']} ({result.window['days']} days)"},
            {"Metric": "Total EVCs", "Value": result.totals["evc"]},
            {"Metric": "Total Conversions", "Value": result.totals["conversions"]},
            {"Metric": "EVC / Conversion Ratio", "Value": result.totals["evc_to_conversion_ratio"]},
            {"Metric": "Not Visible Share", "Value": result.not_visible["share"]},
            {"Metric": "Confounded Flag", "Value": result.confounded},
            {"Metric": "Date Basis", "Value": result.config["date_basis"]},
            {"Metric": "Ads TZ / Source TZ", "Value": f"{result.config['ads_tz']} / {result.config['source_tz']}"},
        ]
    )

    ch_df = pd.DataFrame(
        [
            {
                "Bucket": c.bucket,
                "Share": c.share,
                "CI Low (5%)": c.ci_low,
                "CI High (95%)": c.ci_high,
                "P(#1)": c.p_top,
                "P(>=5%)": c.p_material,
                "Lead Coef": c.lead_coef,
                "Lead Flag": c.lead_flag,
                "Powered": c.powered,
                "Label": c.label,
                "Total Conversions": c.total_conversions,
                "Implied EVC Conversions": c.implied_evc_conversions,
            }
            for c in result.channels
        ]
    )

    k_vals = result.delay_curve["k"]
    delay_df = pd.DataFrame(
        {
            "Lag Day": list(range(len(k_vals))),
            "Weight k": k_vals,
            "CI Low (5%)": result.delay_curve["ci_low"],
            "CI High (95%)": result.delay_curve["ci_high"],
        }
    )

    preflight_df = pd.DataFrame([asdict(it) for it in result.preflight])
    power_df = pd.DataFrame([asdict(it) for it in result.power_test])

    daily_dict: dict[str, object] = {"date": fit.dates.strftime("%Y-%m-%d"), "evc": fit.evc_series}
    for i, ch in enumerate(fit.channels):
        daily_dict[f"{ch} | actual"] = fit.actual_y[:, i]
        daily_dict[f"{ch} | baseline"] = fit.baseline_fit[:, i]
        daily_dict[f"{ch} | evc_component"] = fit.evc_component[:, i]
    daily_df = pd.DataFrame(daily_dict)

    with pd.ExcelWriter(p, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        ch_df.to_excel(writer, sheet_name="Channel Shares", index=False)
        delay_df.to_excel(writer, sheet_name="Delay Curve", index=False)
        preflight_df.to_excel(writer, sheet_name="Preflight", index=False)
        power_df.to_excel(writer, sheet_name="Power Test", index=False)
        mapped.mapping_applied.to_excel(writer, sheet_name="Mapping Applied", index=False)
        mapped.unmapped.to_excel(writer, sheet_name="Unmapped Values", index=False)
        daily_df.to_excel(writer, sheet_name="Daily Fit", index=False)

    return p


def _img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def write_html(result: Result, chart_paths: dict[str, Path], out_dir: Path) -> Path:
    """Write self-contained report.html with inline CSS and base64 PNGs (zero external URLs)."""
    p = out_dir / "report.html"

    ch_rows = []
    for c in result.channels:
        col = LABEL_COLORS.get(c.label, "#6b7280")
        pwr_txt = "Yes" if c.powered is True else ("No" if c.powered is False else "Skipped")
        ch_rows.append(
            f"<tr><td><strong>{html.escape(c.bucket)}</strong></td>"
            f"<td>{c.share:.1%}</td>"
            f"<td>{c.ci_low:.1%} – {c.ci_high:.1%}</td>"
            f"<td>{c.p_top:.0%}</td>"
            f"<td><span class='pill' style='background:{col}'>{html.escape(c.label)}</span></td>"
            f"<td>{pwr_txt}</td>"
            f"<td>{'Yes' if c.lead_flag else 'No'} ({c.lead_coef:+.2f})</td>"
            f"<td>{c.implied_evc_conversions:,.0f}</td></tr>"
        )
    ch_rows.append(
        f"<tr class='muted'><td>Not visible in this data</td>"
        f"<td>{result.not_visible['share']:.1%}</td>"
        f"<td>{result.not_visible['ci_low']:.1%} – {result.not_visible['ci_high']:.1%}</td>"
        f"<td colspan='5'>Consent loss, cross-device, app, or excluded channels</td></tr>"
    )

    pf_rows = []
    pf_colors = {"ok": "#065f46", "info": "#1e40af", "warn": "#92400e", "error": "#991b1b"}
    pf_bg = {"ok": "#d1fae5", "info": "#dbeafe", "warn": "#fef3c7", "error": "#fee2e2"}
    for it in result.preflight:
        pf_rows.append(
            f"<tr><td><code>{html.escape(it.id)}</code></td>"
            f"<td><span class='pill' style='background:{pf_bg.get(it.level, '#eee')};color:{pf_colors.get(it.level, '#111')}'>{html.escape(it.level.upper())}</span></td>"
            f"<td>{html.escape(it.message)}</td></tr>"
        )

    pw_rows = []
    for it in result.power_test:
        rec = f"{it.recovered:.1%}" if it.recovered is not None else "—"
        st = "Pass" if it.powered is True else ("Underpowered" if it.powered is False else "Skipped")
        pw_rows.append(
            f"<tr><td>{html.escape(it.bucket)}</td><td>{it.planted:.0%}</td><td>{rec}</td>"
            f"<td>{st}</td><td>{html.escape(it.note)}</td></tr>"
        )

    banner_cls = "banner warn-banner" if result.confounded else "banner"
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EVC Landing Estimator Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #111827; background: #f9fafb; margin: 0; padding: 28px; line-height: 1.5; }}
  .wrap {{ max-width: 1020px; margin: 0 auto; }}
  h1 {{ margin: 0 0 6px 0; font-size: 24px; }}
  h2 {{ margin: 28px 0 10px 0; font-size: 17px; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; }}
  .sub {{ color: #4b5563; margin-bottom: 18px; font-size: 14px; }}
  .card {{ background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px 20px; margin-bottom: 18px; }}
  .banner {{ background: #ecfdf5; border: 1px solid #10b981; border-radius: 8px; padding: 14px 18px; margin-bottom: 18px; font-size: 15px; }}
  .warn-banner {{ background: #fef2f2; border-color: #ef4444; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e5e7eb; }}
  th {{ background: #f3f4f6; font-weight: 600; }}
  .muted td {{ color: #6b7280; background: #f9fafb; }}
  .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; color: #ffffff; font-size: 12px; font-weight: 600; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
  img {{ width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 6px; }}
  ul {{ margin: 8px 0 0 18px; padding: 0; }}
  li {{ margin-bottom: 5px; font-size: 13.5px; }}
  code {{ background: #f3f4f6; padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>EVC Landing Estimator Report</h1>
  <div class="sub">Window: {html.escape(result.window['start'])} to {html.escape(result.window['end'])} ({result.window['days']} days) · Total EVCs: {result.totals['evc']:,.0f} · Total Conversions: {result.totals['conversions']:,.0f}</div>

  <div class="{banner_cls}"><strong>Headline Summary:</strong> {html.escape(result.summary_text)}</div>

  <div class="card">
    <h2>1. Estimated EVC Landing Shares by Channel</h2>
    <table>
      <thead><tr><th>Channel</th><th>Share</th><th>90% CI</th><th>P(#1)</th><th>Confidence</th><th>Powered</th><th>Lead Check</th><th>Implied EVCs</th></tr></thead>
      <tbody>{''.join(ch_rows)}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>2. Diagnostic Charts</h2>
    <div class="grid">
      <img alt="Channel Shares" src="data:image/png;base64,{_img_b64(chart_paths['shares'])}">
      <img alt="Probability Top Channel" src="data:image/png;base64,{_img_b64(chart_paths['p_top'])}">
      <img alt="Delay Curve" src="data:image/png;base64,{_img_b64(chart_paths['delay_curve'])}">
      <img alt="Top Channel Daily Fit" src="data:image/png;base64,{_img_b64(chart_paths['top_channel_fit'])}">
    </div>
  </div>

  <div class="card">
    <h2>3. Pre-Flight Checks</h2>
    <table>
      <thead><tr><th>Check</th><th>Status</th><th>Detail</th></tr></thead>
      <tbody>{''.join(pf_rows)}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>4. Planted-Signal Power Test (+25% Share Recovery)</h2>
    <table>
      <thead><tr><th>Channel</th><th>Planted</th><th>Recovered</th><th>Status</th><th>Notes</th></tr></thead>
      <tbody>{''.join(pw_rows)}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>5. Assumptions</h2>
    <ul>
      <li><strong>Date basis:</strong> <code>{html.escape(str(result.config['date_basis']))}</code> (max lag: <code>{result.config['max_lag']}</code> days).</li>
      <li><strong>Time zones:</strong> Google Ads <code>{html.escape(str(result.config['ads_tz']))}</code>; conversions source <code>{html.escape(str(result.config['source_tz']))}</code>.</li>
      <li><strong>Conversion-action alignment:</strong> Confirm the Google Ads EVC conversion action is the same event as your conversions file (e.g. lead submit vs lead submit).</li>
      <li><strong>Resampling:</strong> {result.config['bootstrap']} moving-block bootstrap draws (block length {result.config['block_len']} days, seed {result.config['seed']}).</li>
    </ul>
  </div>

  <div class="card">
    <h2>6. How to Read This &amp; Limitations</h2>
    <ul>
      <li><strong>Not incrementality or ROI:</strong> EVCs are double-counted conversions. This report estimates where those conversions receive last-click credit, not whether YouTube caused them.</li>
      <li><strong>Not user-level matching:</strong> All estimates come from daily totals only.</li>
      <li><strong>Confounding caveat:</strong> Simultaneous promos or demand shocks lifting both EVCs and a channel can mimic overlap. The future-EVC lead check flags this when detectable, but only a holdout or geo experiment rules it out completely.</li>
    </ul>
  </div>
</div>
</body>
</html>
"""
    p.write_text(doc, encoding="utf-8")
    return p
