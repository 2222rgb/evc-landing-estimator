"""Deterministic rule-based channel mapper (CSV regex rules + brand-term split)."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
import re

import pandas as pd

from evc_landing_estimator.config import Config

VALID_FIELDS = frozenset({"source", "medium", "campaign", "term", "channel", "source_medium", "any"})


@dataclass(frozen=True)
class MappingRule:
    priority: int
    bucket: str
    field: str
    pattern: re.Pattern[str]


@dataclass
class MappedChannels:
    conv_daily: pd.DataFrame
    mapping_applied: pd.DataFrame
    unmapped: pd.DataFrame
    brand_split_determinable: bool
    folded_buckets: list[str] = field(default_factory=list)


def load_rules(path: str | Path | None = None) -> list[MappingRule]:
    """Load and compile channel mapping rules sorted by priority ascending."""
    if path is None:
        with resources.files("evc_landing_estimator").joinpath("channel_rules.csv").open("r", encoding="utf-8") as f:
            df = pd.read_csv(f)
    else:
        df = pd.read_csv(path)

    required = {"priority", "bucket", "field", "pattern"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"channel_rules.csv missing required columns: {sorted(missing)}")

    rules: list[MappingRule] = []
    for row in df.sort_values("priority", ascending=True).itertuples(index=False):
        fld = str(row.field).strip()
        if fld not in VALID_FIELDS:
            raise ValueError(f"Invalid rule field {fld!r}; must be one of {sorted(VALID_FIELDS)}")
        rules.append(
            MappingRule(
                priority=int(row.priority),
                bucket=str(row.bucket).strip(),
                field=fld,
                pattern=re.compile(str(row.pattern), re.IGNORECASE),
            )
        )
    return rules


def _compile_brand_pattern(brand_terms: tuple[str, ...] | list[str]) -> re.Pattern[str] | None:
    terms = [t.strip() for t in brand_terms if t and t.strip()]
    if not terms:
        return None
    escaped = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    # Match word or underscore-delimited tokens (e.g., "acme_brand_exact" and "acme shoes")
    return re.compile(rf"(?<![A-Za-z0-9])(?:{escaped})(?![A-Za-z0-9])", re.IGNORECASE)


def _match_row(
    source: str,
    medium: str,
    campaign: str,
    term: str,
    channel: str,
    rules: list[MappingRule],
) -> tuple[str, bool]:
    """Return (bucket, matched_rule_bool)."""
    s = source.strip()
    m = medium.strip()
    c = campaign.strip()
    t = term.strip()
    ch = channel.strip()
    sm = f"{s} / {m}"
    any_val = " | ".join(v for v in (s, m, c, t, ch) if v)

    fields = {
        "source": s,
        "medium": m,
        "campaign": c,
        "term": t,
        "channel": ch,
        "source_medium": sm,
        "any": any_val,
    }
    for rule in rules:
        target = fields[rule.field]
        if rule.pattern.search(target):
            return rule.bucket, True
    return "Other", False


def _format_raw_key(source: str, medium: str, channel: str) -> str:
    s, m, ch = source.strip(), medium.strip(), channel.strip()
    if s or m:
        return f"{s or '(empty)'} / {m or '(empty)'}"
    return ch or "(empty)"


def map_channels(df: pd.DataFrame, config: Config | None = None) -> MappedChannels:
    """Map canonical conversion rows into standardized daily channel buckets."""
    cfg = config or Config()
    rules = load_rules(cfg.rules_path)
    brand_pat = _compile_brand_pattern(cfg.brand_terms)

    work = df.copy()
    for col in ("source", "medium", "campaign", "term", "channel"):
        if col not in work.columns:
            work[col] = ""
        else:
            work[col] = work[col].fillna("").astype(str).str.strip()

    # Map unique combinations once for speed
    key_cols = ["source", "medium", "campaign", "term", "channel"]
    combos = work[key_cols].drop_duplicates().reset_index(drop=True)

    base_buckets: list[str] = []
    matched_flags: list[bool] = []
    for r in combos.itertuples(index=False):
        b, matched = _match_row(r.source, r.medium, r.campaign, r.term, r.channel, rules)
        base_buckets.append(b)
        matched_flags.append(matched)

    combos["_base_bucket"] = base_buckets
    combos["_matched"] = matched_flags

    # Determine whether brand vs non-brand split is possible for Paid Search and Organic Search
    paid_rows = combos[combos["_base_bucket"] == "Paid Search"]
    org_rows = combos[combos["_base_bucket"] == "Organic Search"]
    paid_has_text = bool(((paid_rows["campaign"] != "") | (paid_rows["term"] != "")).any())
    org_has_text = bool(((org_rows["campaign"] != "") | (org_rows["term"] != "")).any())

    can_split_paid = brand_pat is not None and paid_has_text
    can_split_org = brand_pat is not None and org_has_text
    brand_split_determinable = can_split_paid or can_split_org

    final_buckets: list[str] = []
    for base_b, camp_v, term_v in zip(base_buckets, combos["campaign"], combos["term"]):
        b = base_b
        if b == "Paid Search" and can_split_paid:
            text = f"{camp_v} {term_v}"
            b = "Paid Brand Search" if brand_pat.search(text) else "Paid Non-Brand Search"
        elif b == "Organic Search" and can_split_org:
            text = f"{camp_v} {term_v}"
            b = "Organic Brand Search" if brand_pat.search(text) else "Organic Non-Brand Search"
        final_buckets.append(b)

    combos["_bucket"] = final_buckets
    combos["_raw_key"] = [_format_raw_key(r.source, r.medium, r.channel) for r in combos.itertuples(index=False)]

    merged = work.merge(combos[key_cols + ["_bucket", "_matched", "_raw_key"]], on=key_cols, how="left")
    total_conv = float(merged["conversions"].sum()) or 1.0

    # Mapping applied & unmapped reports
    mapping_applied = (
        merged.groupby(["_raw_key", "_bucket"], as_index=False)["conversions"]
        .sum()
        .rename(columns={"_raw_key": "raw_key", "_bucket": "bucket"})
        .sort_values("conversions", ascending=False)
        .reset_index(drop=True)
    )
    mapping_applied["pct"] = mapping_applied["conversions"] / total_conv

    unmapped_rows = merged[~merged["_matched"]]
    if unmapped_rows.empty:
        unmapped = pd.DataFrame(columns=["raw_key", "conversions", "pct"])
    else:
        unmapped = (
            unmapped_rows.groupby("_raw_key", as_index=False)["conversions"]
            .sum()
            .rename(columns={"_raw_key": "raw_key"})
            .sort_values("conversions", ascending=False)
            .reset_index(drop=True)
        )
        unmapped["pct"] = unmapped["conversions"] / total_conv

    # Build continuous daily pivot
    daily = (
        merged.groupby(["date", "_bucket"])["conversions"]
        .sum()
        .unstack(fill_value=0.0)
        .sort_index()
        .astype(float)
    )
    if not daily.empty:
        full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        daily = daily.reindex(full_idx, fill_value=0.0)
        daily.index.name = "date"

    # Fold low-volume buckets if > max_channels (always preserve Direct if present)
    folded: list[str] = []
    if daily.shape[1] > cfg.max_channels:
        totals = daily.sum(axis=0).sort_values(ascending=False)
        non_other = [c for c in totals.index if c != "Other"]
        keep_slots = max(1, cfg.max_channels - 1)  # reserve 1 slot for Other
        keep: list[str] = []
        if "Direct" in non_other:
            keep.append("Direct")
        for c in non_other:
            if len(keep) >= keep_slots:
                break
            if c not in keep:
                keep.append(c)
        folded = [c for c in daily.columns if c not in keep and c != "Other"]
        other_series = daily["Other"].copy() if "Other" in daily.columns else pd.Series(0.0, index=daily.index)
        for c in folded:
            other_series = other_series + daily[c]
        daily = daily[keep].copy()
        daily["Other"] = other_series

    return MappedChannels(
        conv_daily=daily,
        mapping_applied=mapping_applied,
        unmapped=unmapped,
        brand_split_determinable=brand_split_determinable,
        folded_buckets=folded,
    )
