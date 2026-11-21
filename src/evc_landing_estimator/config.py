"""Configuration and result dataclasses for evc-landing-estimator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


TOOL_VERSION = "2.0.0"


@dataclass(frozen=True)
class ColumnOverrides:
    timestamp: str | None = None
    source: str | None = None
    medium: str | None = None
    campaign: str | None = None
    term: str | None = None
    channel: str | None = None
    id: str | None = None
    conversions: str | None = None
    evc_date: str | None = None
    evc_column: str | None = None


@dataclass(frozen=True)
class Config:
    date_basis: Literal["impression_time", "conversion_time"] = "impression_time"
    max_lag: int = 3
    ads_tz: str = "UTC"
    source_tz: str = "UTC"
    brand_terms: tuple[str, ...] | list[str] = field(default_factory=tuple)
    rules_path: str | None = None
    controls_path: str | None = None
    max_channels: int = 8
    bootstrap: int = 500
    power_bootstrap: int = 100
    block_len: int = 7
    seed: int = 42
    force: bool = False
    columns: ColumnOverrides = field(default_factory=ColumnOverrides)

    def __post_init__(self) -> None:
        if self.date_basis not in ("impression_time", "conversion_time"):
            raise ValueError(f"Unsupported date_basis: {self.date_basis!r}")
        if isinstance(self.brand_terms, list):
            object.__setattr__(self, "brand_terms", tuple(self.brand_terms))
        if self.date_basis == "conversion_time":
            object.__setattr__(self, "max_lag", 0)
        elif not (0 <= self.max_lag <= 14):
            raise ValueError(f"max_lag must be in 0..14, got {self.max_lag}")
        if self.max_channels < 1:
            raise ValueError("max_channels must be >= 1")
        if self.block_len < 1:
            raise ValueError("block_len must be >= 1")
        if self.bootstrap < 0 or self.power_bootstrap < 0:
            raise ValueError("bootstrap counts must be >= 0")

    def to_report_dict(self) -> dict:
        return {
            "date_basis": self.date_basis,
            "max_lag": self.max_lag,
            "ads_tz": self.ads_tz,
            "source_tz": self.source_tz,
            "bootstrap": self.bootstrap,
            "block_len": self.block_len,
            "seed": self.seed,
            "max_channels": self.max_channels,
        }


@dataclass
class PreflightItem:
    id: str
    level: Literal["ok", "warn", "error", "info"]
    message: str
    value: object = None


@dataclass
class PowerTestItem:
    bucket: str
    planted: float
    recovered: float | None
    powered: bool | None
    note: str = ""


@dataclass
class ChannelResult:
    bucket: str
    share: float
    ci_low: float
    ci_high: float
    p_top: float
    p_material: float
    lead_coef: float
    lead_ci: list[float]
    lead_flag: bool
    powered: bool | None
    label: str
    total_conversions: float
    implied_evc_conversions: float
    unconstrained_share: float = 0.0
    partial_r2: float = 0.0


@dataclass
class Result:
    tool_version: str
    config: dict
    window: dict
    totals: dict
    channels: list[ChannelResult]
    not_visible: dict
    delay_curve: dict
    confounded: bool
    preflight: list[PreflightItem]
    power_test: list[PowerTestItem]
    summary_text: str
    converged: bool = True
    warnings: list[str] = field(default_factory=list)
    output_files: dict[str, str] = field(default_factory=dict)
