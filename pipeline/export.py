"""
Export utilities — CSV and Excel.

The Excel workbook is the artifact a Community Partnerships analyst can
hand off:
  - "Summary"          run metadata, coverage, source mix
  - "Benchmark"        wide companies × metrics matrix, formatted
  - "Detailed"         long format with flags, values, sources, confidence
  - "Failures & Notes" pairs that returned null, with reasons
  - "AI Audit"         AI verification + insights (when AI ran)
  - "Attempts log"     full HTTP audit trail (collapsed/hidden by default)

CSV is the long-format equivalent of "Detailed".
"""

from __future__ import annotations
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from .models import DataPoint, METRICS


# ── Excel-safe sanitization ─────────────────────────────────────────────────
_ILLEGAL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize_for_excel(value: Any) -> Any:
    if isinstance(value, str):
        cleaned = _ILLEGAL_CHARS_RE.sub("", value)
        if len(cleaned) > 32700:
            cleaned = cleaned[:32700] + "…[truncated]"
        return cleaned
    return value


def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in out.columns:
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


def _apply_formatting(ws, *, title: str, freeze_at: str = "A2",
                      col_widths: Optional[dict] = None,
                      number_format_cols: Optional[dict] = None):
    """Apply consistent formatting: header style, frozen panes, column widths,
    number formatting, autofilter."""
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    # Header row styling
    header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    header_align = Alignment(horizontal="left", vertical="center", wrap_text=True)
    thin = Side(border_style="thin", color="BBBBBB")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    if ws.max_row >= 1:
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = header_align
            cell.border = border

    # Body rows: light borders, wrap text
    body_align = Alignment(vertical="top", wrap_text=True)
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row:
            cell.alignment = body_align
            cell.border = border

    # Header row height
    if ws.max_row >= 1:
        ws.row_dimensions[1].height = 32

    # Column widths
    if col_widths:
        for col_name, width in col_widths.items():
            # Find the column letter for this header
            for cell in ws[1]:
                if str(cell.value).strip() == col_name:
                    ws.column_dimensions[get_column_letter(cell.column)].width = width
                    break

    # Number formats
    if number_format_cols:
        for col_name, fmt in number_format_cols.items():
            for cell in ws[1]:
                if str(cell.value).strip() == col_name:
                    col_letter = get_column_letter(cell.column)
                    for r in range(2, ws.max_row + 1):
                        ws[f"{col_letter}{r}"].number_format = fmt
                    break

    # Freeze top row, enable autofilter
    ws.freeze_panes = freeze_at
    if ws.max_row > 1 and ws.max_column > 0:
        from openpyxl.utils import get_column_letter as _gc
        last_col = _gc(ws.max_column)
        ws.auto_filter.ref = f"A1:{last_col}{ws.max_row}"


def to_excel(
    datapoints: list[DataPoint],
    path: str | Path,
    *,
    summary_text: str | None = None,
    insights: str | None = None,
    audit_summary: str | None = None,
    flags: dict | None = None,
) -> Path:
    """Write a multi-sheet workbook formatted for analyst hand-off."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    path = Path(path)
    flags = flags or {}

    companies = sorted({dp.company for dp in datapoints})
    metrics_in_run = []
    seen = set()
    # Preserve user's metric order: standard metrics first (in METRICS order),
    # then any custom metrics in alphabetical order
    for m in METRICS:
        if any(dp.metric == m for dp in datapoints) and m not in seen:
            metrics_in_run.append(m)
            seen.add(m)
    custom_metrics = sorted({dp.metric for dp in datapoints if dp.metric not in seen})
    metrics_in_run.extend(custom_metrics)

    ok_dps = [dp for dp in datapoints if dp.ok]
    fail_dps = [dp for dp in datapoints if not dp.ok]
    confidences = [dp.confidence_score for dp in ok_dps if dp.confidence_score is not None]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    source_mix: dict[str, int] = defaultdict(int)
    for dp in ok_dps:
        key = (dp.source_name or "unknown").split(" — ")[0]
        source_mix[key] += 1

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # ── Sheet 1: Summary ───────────────────────────────────────────────
        summary_rows = [
            ["Utility Benchmark Pipeline — Run Summary", ""],
            [f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", ""],
            ["", ""],
            ["Companies", ", ".join(companies)],
            ["Metrics", ", ".join(METRICS.get(m, {}).get("label", m) for m in metrics_in_run)],
            ["", ""],
            ["── Coverage ──", ""],
            ["Total (company × metric) pairs", len(datapoints)],
            ["Successful extractions", len(ok_dps)],
            ["Failed extractions (null)", len(fail_dps)],
            ["Coverage", f"{(len(ok_dps) / max(1, len(datapoints))) * 100:.1f}%"],
            ["", ""],
            ["── Confidence ──", ""],
            ["Average confidence (successful values)", round(avg_conf, 3)],
        ]
        for threshold, label in [(0.90, "Government API exact (≥ 0.90)"),
                                 (0.75, "ProPublica 990 / derived (≥ 0.75)"),
                                 (0.60, "CSR PDF (≥ 0.60)"),
                                 (0.50, "Third-party survey (≥ 0.50)")]:
            n = sum(1 for c in confidences if c >= threshold)
            summary_rows.append([label, n])
        summary_rows.append(["", ""])
        summary_rows.append(["── Source mix ──", ""])
        for src, n in sorted(source_mix.items(), key=lambda x: -x[1]):
            summary_rows.append([src, n])

        sanitized_summary = [[_sanitize_for_excel(c) for c in r] for r in summary_rows]
        df_summary = pd.DataFrame(sanitized_summary, columns=["Metric", "Value"])
        df_summary.to_excel(writer, sheet_name="Summary", index=False, header=False)

        # Style summary
        ws = writer.sheets["Summary"]
        ws.column_dimensions["A"].width = 45
        ws.column_dimensions["B"].width = 60
        title_font = Font(name="Calibri", size=14, bold=True, color="1F4E79")
        section_font = Font(name="Calibri", size=11, bold=True, color="1F4E79")
        ws["A1"].font = title_font
        for r in range(1, ws.max_row + 1):
            v = ws.cell(row=r, column=1).value
            if v and isinstance(v, str) and v.startswith("──"):
                ws.cell(row=r, column=1).font = section_font

        # ── Sheet 2: Benchmark (wide, formatted) ───────────────────────────
        wide_rows = []
        for c in companies:
            row = {"Company": c}
            for m in metrics_in_run:
                meta = METRICS.get(m, {})
                label = meta.get("label", m)
                unit = meta.get("unit", "")
                col = f"{label} ({unit})" if unit else label
                dp = next((d for d in datapoints if d.company == c and d.metric == m), None)
                if dp and dp.ok:
                    row[col] = dp.value
                else:
                    row[col] = None
            wide_rows.append(row)
        df_wide = pd.DataFrame(wide_rows)
        _sanitize_dataframe(df_wide).to_excel(writer, sheet_name="Benchmark", index=False)

        ws = writer.sheets["Benchmark"]
        # First column wider for company names; metric columns 18-22 chars
        col_widths = {"Company": 28}
        for col in df_wide.columns[1:]:
            col_widths[col] = 22
        # Number format: $B → 0.00, $M → 0.00, % → 0.0, etc.
        number_formats = {}
        for m in metrics_in_run:
            meta = METRICS.get(m, {})
            label = meta.get("label", m)
            unit = meta.get("unit", "")
            col = f"{label} ({unit})" if unit else label
            if unit == "%":
                number_formats[col] = "0.0"
            elif unit in ("$B", "$M"):
                number_formats[col] = "#,##0.00"
            elif unit in ("hrs/yr", "grants", "min/yr"):
                number_formats[col] = "#,##0"
            elif unit == "M MT CO2":
                number_formats[col] = "0.00"
            elif unit == "/100":
                number_formats[col] = "0.0"
        _apply_formatting(ws, title="Benchmark", col_widths=col_widths,
                          number_format_cols=number_formats)

        # ── Sheet 3: Detailed ──────────────────────────────────────────────
        detailed_rows = []
        for dp in datapoints:
            flag_info = flags.get((dp.company, dp.metric), {})
            meta = METRICS.get(dp.metric, {})
            detailed_rows.append({
                "Flag": flag_info.get("flag", ""),
                "Company": dp.company,
                "Metric": meta.get("label", dp.metric),
                "Value": dp.value,
                "Unit": dp.unit,
                "Year": dp.year,
                "Confidence": dp.confidence_score,
                "Source": (dp.source_name or "")[:120],
                "Source URL": dp.source_url,
                "Notes": dp.notes,
                "Flag reason": flag_info.get("reason", ""),
            })
        df_detail = pd.DataFrame(detailed_rows)
        _sanitize_dataframe(df_detail).to_excel(writer, sheet_name="Detailed", index=False)

        ws = writer.sheets["Detailed"]
        _apply_formatting(
            ws, title="Detailed",
            col_widths={
                "Flag": 6, "Company": 24, "Metric": 28, "Value": 12,
                "Unit": 10, "Year": 10, "Confidence": 11, "Source": 40,
                "Source URL": 50, "Notes": 50, "Flag reason": 50,
            },
            number_format_cols={"Value": "#,##0.00", "Confidence": "0.00"},
        )

        # ── Sheet 4: Failures & Notes (only if any failures) ───────────────
        if fail_dps:
            failure_rows = []
            for dp in fail_dps:
                flag_info = flags.get((dp.company, dp.metric), {})
                meta = METRICS.get(dp.metric, {})
                failure_rows.append({
                    "Flag": flag_info.get("flag", "🚩"),
                    "Company": dp.company,
                    "Metric": meta.get("label", dp.metric),
                    "Reason": (dp.error or {}).get("reason", "") if dp.error else "",
                    "What this means": flag_info.get("reason", ""),
                    "Sources tried": ", ".join((dp.error or {}).get("attempted_sources", [])),
                })
            df_fail = pd.DataFrame(failure_rows)
            _sanitize_dataframe(df_fail).to_excel(writer, sheet_name="Failures & Notes", index=False)
            ws = writer.sheets["Failures & Notes"]
            _apply_formatting(
                ws, title="Failures & Notes",
                col_widths={"Flag": 6, "Company": 24, "Metric": 28,
                            "Reason": 50, "What this means": 60, "Sources tried": 35},
            )

        # ── Sheet 5: AI Audit (only if AI ran) ─────────────────────────────
        if audit_summary or insights:
            audit_rows = []
            if audit_summary:
                audit_rows.append(["Data quality audit", ""])
                for line in (audit_summary or "").split("\n"):
                    audit_rows.append(["", _sanitize_for_excel(line)])
                audit_rows.append(["", ""])
            if insights:
                audit_rows.append(["Strategic insights", ""])
                for line in (insights or "").split("\n"):
                    audit_rows.append(["", _sanitize_for_excel(line)])
            df_audit = pd.DataFrame(audit_rows, columns=["Section", "Content"])
            df_audit.to_excel(writer, sheet_name="AI Audit", index=False, header=False)
            ws = writer.sheets["AI Audit"]
            ws.column_dimensions["A"].width = 30
            ws.column_dimensions["B"].width = 100
            for r in range(1, ws.max_row + 1):
                ws.cell(row=r, column=2).alignment = Alignment(wrap_text=True, vertical="top")
                v = ws.cell(row=r, column=1).value
                if v:
                    ws.cell(row=r, column=1).font = Font(bold=True, color="1F4E79")

        # ── Sheet 6: Attempts log (audit trail, hidden by default) ─────────
        attempt_rows = []
        for dp in datapoints:
            for a in dp.attempts:
                attempt_rows.append({
                    "Company": dp.company,
                    "Metric": dp.metric,
                    "Source": a.source,
                    "URL": a.url,
                    "Method": a.method,
                    "Status": a.status_code,
                    "OK": a.success,
                    "Time (ms)": a.duration_ms,
                    "Error": a.error,
                    "Preview": (a.response_preview or "")[:200],
                })
        if attempt_rows:
            df_att = pd.DataFrame(attempt_rows)
            _sanitize_dataframe(df_att).to_excel(writer, sheet_name="Attempts log", index=False)
            ws = writer.sheets["Attempts log"]
            _apply_formatting(
                ws, title="Attempts log",
                col_widths={"Company": 22, "Metric": 24, "Source": 30, "URL": 60,
                            "Method": 12, "Status": 8, "OK": 6, "Time (ms)": 10,
                            "Error": 40, "Preview": 50},
            )
            ws.sheet_state = "hidden"  # hidden by default — user can unhide for debugging

    return path
