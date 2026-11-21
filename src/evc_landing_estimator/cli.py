"""Command-line interface for evc-landing-estimator (run | map | demo)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evc_landing_estimator.config import ColumnOverrides, Config
from evc_landing_estimator.pipeline import preview_map, run
from evc_landing_estimator.preflight import PreflightError
from evc_landing_estimator.synthetic import SCENARIO_SPECS, generate


def _add_column_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--col-timestamp", default=None)
    parser.add_argument("--col-source", default=None)
    parser.add_argument("--col-medium", default=None)
    parser.add_argument("--col-campaign", default=None)
    parser.add_argument("--col-term", default=None)
    parser.add_argument("--col-channel", default=None)
    parser.add_argument("--col-id", default=None)
    parser.add_argument("--col-conversions", default=None)
    parser.add_argument("--evc-column", default=None)


def _parse_brand_terms(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(t.strip() for t in raw.split(",") if t.strip())


def _build_overrides(args: argparse.Namespace) -> ColumnOverrides:
    return ColumnOverrides(
        timestamp=args.col_timestamp,
        source=args.col_source,
        medium=args.col_medium,
        campaign=args.col_campaign,
        term=args.col_term,
        channel=args.col_channel,
        id=args.col_id,
        conversions=args.col_conversions,
        evc_column=args.evc_column,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evc-landing-estimator",
        description="Estimate which last-click channels absorb Google Ads Engaged-View Conversions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run EVC landing estimation on conversions + Google Ads EVC files.")
    p_run.add_argument("--conversions", required=True, help="Path to CRM, UTM, or GA4 CSV/XLSX.")
    p_run.add_argument("--evc", required=True, help="Path to Google Ads daily EVC CSV/XLSX.")
    p_run.add_argument("--ads-tz", default="UTC")
    p_run.add_argument("--source-tz", default="UTC")
    p_run.add_argument("--date-basis", default="impression_time", choices=["impression_time", "conversion_time"])
    p_run.add_argument("--max-lag", type=int, default=3)
    p_run.add_argument("--brand-terms", default=None, help="Comma-separated brand terms (e.g. 'acme,acme co').")
    p_run.add_argument("--rules", default=None, help="Path to custom channel_rules.csv.")
    p_run.add_argument("--controls", default=None, help="Path to optional daily controls CSV/XLSX.")
    p_run.add_argument("--max-channels", type=int, default=8)
    p_run.add_argument("--bootstrap", type=int, default=500)
    p_run.add_argument("--power-bootstrap", type=int, default=100)
    p_run.add_argument("--block-len", type=int, default=7)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--out-dir", default="out/")
    p_run.add_argument("--force", action="store_true", help="Proceed even if pre-flight errors occur.")
    _add_column_args(p_run)

    # map
    p_map = sub.add_parser("map", help="Preview channel mapping and unmapped values without modeling.")
    p_map.add_argument("--conversions", required=True)
    p_map.add_argument("--rules", default=None)
    p_map.add_argument("--brand-terms", default=None)
    p_map.add_argument("--ads-tz", default="UTC")
    p_map.add_argument("--source-tz", default="UTC")
    p_map.add_argument("--max-channels", type=int, default=8)
    _add_column_args(p_map)

    # demo
    p_demo = sub.add_parser("demo", help="Run on a built-in synthetic scenario.")
    p_demo.add_argument("--scenario", default="known_split", choices=sorted(SCENARIO_SPECS))
    p_demo.add_argument("--out-dir", default="out/")
    p_demo.add_argument("--bootstrap", type=int, default=500)
    p_demo.add_argument("--power-bootstrap", type=int, default=100)
    p_demo.add_argument("--seed", type=int, default=42)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "map":
            cfg = Config(
                ads_tz=args.ads_tz,
                source_tz=args.source_tz,
                brand_terms=_parse_brand_terms(args.brand_terms),
                rules_path=args.rules,
                max_channels=args.max_channels,
                columns=_build_overrides(args),
            )
            mapped = preview_map(args.conversions, cfg)
            print("=== Channel Mapping Applied ===")
            print(mapped.mapping_applied.to_string(index=False))
            print("\n=== Unmapped Values ===")
            if mapped.unmapped.empty:
                print("(none)")
            else:
                print(mapped.unmapped.to_string(index=False))
            return 0

        if args.command == "demo":
            data = generate(scenario=args.scenario, seed=args.seed)
            date_basis = "conversion_time" if args.scenario == "conversion_time" else "impression_time"
            cfg = Config(
                ads_tz="America/New_York",
                source_tz="UTC",
                date_basis=date_basis,
                brand_terms=("acme",),
                bootstrap=args.bootstrap,
                power_bootstrap=args.power_bootstrap,
                seed=args.seed,
            )
            res = run(data["crm_rowlevel"], data["ads_evc"], config=cfg, out_dir=args.out_dir)
        else:
            cfg = Config(
                ads_tz=args.ads_tz,
                source_tz=args.source_tz,
                date_basis=args.date_basis,
                max_lag=args.max_lag,
                brand_terms=_parse_brand_terms(args.brand_terms),
                rules_path=args.rules,
                controls_path=args.controls,
                max_channels=args.max_channels,
                bootstrap=args.bootstrap,
                power_bootstrap=args.power_bootstrap,
                block_len=args.block_len,
                seed=args.seed,
                force=args.force,
                columns=_build_overrides(args),
            )
            res = run(args.conversions, args.evc, config=cfg, out_dir=args.out_dir)

        print(res.summary_text)
        print("\nOutput files:")
        for k, v in res.output_files.items():
            print(f"  {k}: {Path(v)}")
        return 0

    except PreflightError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
