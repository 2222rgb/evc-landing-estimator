"""Tests for JSON/XLSX/HTML/PNG report generation, determinism, and examples hygiene."""

import json
from pathlib import Path
import re

import openpyxl

from evc_landing_estimator.cli import main

EXPECTED_SHEETS = [
    "Summary",
    "Channel Shares",
    "Delay Curve",
    "Preflight",
    "Power Test",
    "Mapping Applied",
    "Unmapped Values",
    "Daily Fit",
]


def test_demo_outputs_sheets_html_and_determinism(tmp_path):
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    rc1 = main(["demo", "--scenario", "known_split", "--bootstrap", "120", "--power-bootstrap", "40", "--seed", "42", "--out-dir", str(out1)])
    rc2 = main(["demo", "--scenario", "known_split", "--bootstrap", "120", "--power-bootstrap", "40", "--seed", "42", "--out-dir", str(out2)])
    assert rc1 == 0 and rc2 == 0

    # All required artifacts exist
    for name in ("results.json", "results.xlsx", "report.html", "shares.png", "p_top.png", "delay_curve.png", "top_channel_fit.png"):
        assert (out1 / name).exists(), f"Missing {name}"

    # Byte-identical results.json determinism
    txt1 = (out1 / "results.json").read_text(encoding="utf-8")
    txt2 = (out2 / "results.json").read_text(encoding="utf-8")
    assert txt1 == txt2

    # Valid JSON schema
    payload = json.loads(txt1)
    assert payload["tool_version"] == "2.0.0"
    assert payload["confounded"] is False
    assert payload["channels"][0]["bucket"] == "Direct"

    # All 8 XLSX sheets present
    wb = openpyxl.load_workbook(out1 / "results.xlsx", read_only=True)
    assert wb.sheetnames == EXPECTED_SHEETS
    wb.close()

    # Self-contained HTML: zero external URLs
    html_txt = (out1 / "report.html").read_text(encoding="utf-8")
    assert "http://" not in html_txt
    assert "https://" not in html_txt
    assert "data:image/png;base64," in html_txt


def test_examples_hygiene():
    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    files = list(examples_dir.glob("*.csv"))
    assert len(files) == 4
    private_ip = re.compile(r"\b(?:192\.168|10\.\d{1,3})\.\d{1,3}\.\d{1,3}\b")
    for p in files:
        txt = p.read_text(encoding="utf-8")
        assert "@" not in txt, f"Email-like '@' found in {p.name}"
        assert not private_ip.search(txt), f"Private LAN IP found in {p.name}"
