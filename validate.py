"""
Export utilities — CSV and Excel.

The Excel workbook is the artifact a Community Partnerships analyst can
hand off:
  - "Summary"     : run metadata, confidence histogram, extractor health
  - "Benchmark"   : wide companies × metrics matrix with units in headers
  - "Detailed"    : long format, every value with source URL & confidence
  - "Failures"    : every (company, metric) pair that returned null,
                    with reason and list of sources attempted
  - "Attempts"    : raw HTTP attempt log for full debuggability
  - "Charts data" : one block per metric, sorted, ready for native Excel
                    chart insertion

CSV is the long-format equivalent of "Detailed".
"""

from __future__ import annotations
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .models import DataPoint, METRICS


# ── Excel-safe sanitization ─────────────────────────────────────────────────
# openpyxl rejects control characters (anything < 0x20 except \t, \n, \r) in
# string cells.  These can sneak into our Attempts log via response_preview
# when a fetched URL returned a PDF, gzipped HTML, or other binary blob.
# We strip them here at write-time so a single bad byte can't fail the whole
# export.  See: https://openpyxl.readthedocs.io/en/stable/_modules/openpyxl/cell/cell.html
_ILLEGAL_CHARS_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f]"   # XML 1.0 forbids these
)


def _sanitize_for_excel(value: Any) -> Any:
    """Strip control characters from string values so openpyxl can write them.
    Truncates very long strings (>32,767 chars — Excel's per-cell limit).
    Non-string values pass through unchanged."""
    if isinstance(value, str):
        cleaned = _ILLEGAL_CHARS_RE.sub("", value)
        # Excel cell limit is 32,767 chars; longer values are silently
        # truncated by openpyxl but with a warning we'd rather avoid.
        if len(cleaned) > 32700:
            cleaned = cleaned[:32700] + "…[truncated]"
        return cleaned
    return value


def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply _sanitize_for_excel to every potentially-string column.
    Pandas may infer string-only columns as 'string' or 'str' dtype rather
    than 'object', so we check by trying to apply the function rather than
    by dtype — non-string values pass through unchanged anyway."""
    if df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        # Skip purely numeric columns for speed; everything else gets the map.
        if pd.api.types.is_numeric_dtype(out[col]) or pd.api.types.is_bool_dtype(out[col]):
            continue
        out[col] = out[col].map(_sanitize_for_excel)
    return out



# ── CSV ─────────────────────────────────────────────────────────────────────


def to_dataframe(datapoints: Iterable[DataPoint]) -> pd.DataFrame:
    return pd.DataFrame([dp.to_flat_row() for dp in datapoints])


def to_csv(datapoints: Iterable[DataPoint], path: str | Path) -> Path:
    path = Path(path)
    df = to_dataframe(datapoints)
    df.to_csv(path, index=False)
    return path


# ── Excel ───────────────────────────────────────────────────────────────────


def to_excel(
    datapoints: list[DataPoint],
    path: str | Path,
    *,
    summary_text: str | None = None,
    insights: str | None = None,
    audit_summary: str | None = None,
) -> Path:
    """Write a multi-sheet workbook.  Pure data + provenance + summary,
    no fallback / fabricated rows."""

    path = Path(path)
    companies = sorted({dp.company for dp in datapoints})
    metrics_in_run = [m for m in METRICS if any(dp.metric == m for dp in datapoints)]

    ok_dps = [dp for dp in datapoints if dp.ok]
    fail_dps = [dp for dp in datapoints if not dp.ok]
    confidences = [dp.confidence_score for dp in ok_dps if dp.confidence_score is not None]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    # Source-mix tally (only for successful values)
    source_mix: dict[str, int] = defaultdict(int)
    for dp in ok_dps:
        # Group by extractor base name (before the " — " annotation)
        key = (dp.source_name or "unknown").split(" — ")[0]
        source_mix[key] += 1

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # ── Summary ────────────────────────────────────────────────────────
        summary_rows = [
            ["Utility Benchmark Pipeline — Run Summary"],
            [f"Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z"],
            [""],
            ["Companies", ", ".join(companies)],
            ["Metrics",   ", ".join(METRICS[m]["label"] for m in metrics_in_run)],
            [""],
            ["── Coverage ──"],
            ["Total (company × metric) pairs", len(datapoints)],
            ["Successful extractions",         len(ok_dps)],
            ["Failed extractions (null)",      len(fail_dps)],
            ["Coverage %", f"{(len(ok_dps) / max(1, len(datapoints))) * 100:.1f}%"],
            [""],
            ["── Confidence ──"],
            ["Average confidence (successful values)", round(avg_conf, 3)],
        ]
        for threshold, label in [(0.90, "≥ 0.90 (gov API exact)"),
                                  (0.75, "≥ 0.75 (PP 990 / derived)"),
                                  (0.60, "≥ 0.60 (CSR PDF)"),
                                  (0.50, "≥ 0.50 (third-party survey)")]:
            n = sum(1 for c in confidences if c >= threshold)
            summary_rows.append([f"Values {label}", n])
        summary_rows.append([""])
        summary_rows.append(["── Source mix ──"])
        for src, n in sorted(source_mix.items(), key=lambda x: -x[1]):
            summary_rows.append([src, n])

        if audit_summary:
            summary_rows += [[""], ["── AI verification ──"], [audit_summary]]
        if insights:
            summary_rows += [[""], ["── AI-generated insights ──"]]
            for line in (insights or "").split("\n"):
                line = line.strip()
                if line:
                    summary_rows.append([line.replace("**", "")])
        if summary_text:
            summary_rows += [[""], ["── Notes ──"], [summary_text]]

        # Sanitize each cell in the AOA — AI-generated insights and audits
        # can contain embedded control characters that crash openpyxl.
        sanitized_summary = [
            [_sanitize_for_excel(c) for c in row] for row in summary_rows
        ]
        pd.DataFrame(sanitized_summary).to_excel(
            writer, sheet_name="Summary", index=False, header=False
        )

        # ── Benchmark (wide) ───────────────────────────────────────────────
        wide_rows = []
        for c in companies:
            row = {"Company": c}
            for m in metrics_in_run:
                meta = METRICS[m]
                col = f"{meta['label']} ({meta['unit']})"
                dp = next((d for d in datapoints if d.company == c and d.metric == m), None)
                row[col] = dp.value if dp and dp.ok else None
            wide_rows.append(row)
        _sanitize_dataframe(pd.DataFrame(wide_rows)).to_excel(
            writer, sheet_name="Benchmark", index=False)

        # ── Detailed (long) ────────────────────────────────────────────────
        detailed = pd.DataFrame([dp.to_flat_row() for dp in datapoints])
        _sanitize_dataframe(detailed).to_excel(
            writer, sheet_name="Detailed", index=False)

        # ── Failures ───────────────────────────────────────────────────────
        if fail_dps:
            fail_df = pd.DataFrame([{
                "company": dp.company,
                "metric": dp.metric,
                "metric_label": METRICS.get(dp.metric, {}).get("label", dp.metric),
                "reason": (dp.error or {}).get("reason"),
                "attempted_sources": ", ".join((dp.error or {}).get("attempted_sources", [])),
                "attempt_count": len(dp.attempts),
                "notes": dp.notes,
            } for dp in fail_dps])
            _sanitize_dataframe(fail_df).to_excel(
                writer, sheet_name="Failures", index=False)

        # ── Attempts log ───────────────────────────────────────────────────
        # The response_preview column is the most likely to contain illegal
        # control chars (binary PDFs, gzipped pages).  _sanitize_dataframe
        # strips them so the Excel writer doesn't crash.
        attempt_rows = []
        for dp in datapoints:
            for a in dp.attempts:
                attempt_rows.append({
                    "company": dp.company,
                    "metric": dp.metric,
                    "source": a.source,
                    "url": a.url,
                    "method": a.method,
                    "status_code": a.status_code,
                    "content_type": a.content_type,
                    "response_bytes": a.response_bytes,
                    "duration_ms": a.duration_ms,
                    "success": a.success,
                    "error": a.error,
                    "preview": a.response_preview,
                    "timestamp": a.timestamp,
                })
        if attempt_rows:
            _sanitize_dataframe(pd.DataFrame(attempt_rows)).to_excel(
                writer, sheet_name="Attempts", index=False)

        # ── Charts data (per metric, sorted) ───────────────────────────────
        chart_rows = []
        for m in metrics_in_run:
            meta = METRICS[m]
            chart_rows.append({"chart": f"{meta['label']} ({meta['unit']})"})
            chart_rows.append({"chart": "Company", "value": "Value", "rank": "Rank"})
            metric_dps = sorted(
                [dp for dp in datapoints if dp.metric == m and dp.ok],
                key=lambda d: d.value if d.value is not None else 0,
                reverse=not meta["lower_is_better"],
            )
            for i, dp in enumerate(metric_dps, 1):
                chart_rows.append({"chart": dp.company, "value": dp.value, "rank": i})
            chart_rows.append({})
        _sanitize_dataframe(pd.DataFrame(chart_rows)).to_excel(
            writer, sheet_name="Charts data", index=False, header=False
        )

    return path
