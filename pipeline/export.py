"""
Excel + CSV export.

The Excel workbook is the artifact a Community Partnerships analyst hands to
their VP.  It needs to look professional and be navigable without the analyst
having to explain every column.

Sheets:
  - Cover            single-page brief: companies, metrics, coverage, top insights
  - Benchmark        wide table — companies × metrics, formatted values
  - Per metric       one block per metric: clear label, every company, source
  - Failures & gaps  null cells with plain-English explanations
  - AI Audit         AI-generated audit + insights, formatted for reading
  - Attempts log     full HTTP audit trail (hidden by default)
"""

from __future__ import annotations
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from .models import DataPoint, METRICS


# ── Excel-safe sanitization ────────────────────────────────────────────────
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


# ── CSV ────────────────────────────────────────────────────────────────────


def to_dataframe(datapoints: Iterable[DataPoint]) -> pd.DataFrame:
    return pd.DataFrame([dp.to_flat_row() for dp in datapoints])


def to_csv(datapoints: Iterable[DataPoint], path: str | Path) -> Path:
    path = Path(path)
    df = to_dataframe(datapoints)
    df.to_csv(path, index=False)
    return path


# ── Styling palette ─────────────────────────────────────────────────────────
# Used consistently across every sheet for a polished look.

PALETTE = {
    "primary":      "1F4E79",   # Con Edison-ish dark blue (header bar)
    "primary_text": "FFFFFF",
    "accent":       "2E75B6",
    "subtle":       "F2F6FA",   # zebra stripe (very light blue)
    "border":       "BFBFBF",
    "good":         "C6EFCE",
    "good_text":    "006100",
    "warn":         "FFEB9C",
    "warn_text":    "9C5700",
    "bad":          "FFC7CE",
    "bad_text":     "9C0006",
    "info":         "DDEBF7",
    "muted":        "808080",
}


def _styles():
    """Pre-built openpyxl style objects (computed once at import)."""
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    thin = Side(border_style="thin", color=PALETTE["border"])
    return {
        "border":         Border(left=thin, right=thin, top=thin, bottom=thin),
        "header_fill":    PatternFill(start_color=PALETTE["primary"],
                                      end_color=PALETTE["primary"],
                                      fill_type="solid"),
        "header_font":    Font(name="Calibri", size=11, bold=True,
                               color=PALETTE["primary_text"]),
        "header_align":   Alignment(horizontal="left", vertical="center",
                                    wrap_text=True),
        "title_font":     Font(name="Calibri", size=18, bold=True,
                               color=PALETTE["primary"]),
        "subtitle_font":  Font(name="Calibri", size=11, italic=True,
                               color=PALETTE["muted"]),
        "section_font":   Font(name="Calibri", size=13, bold=True,
                               color=PALETTE["primary"]),
        "label_font":     Font(name="Calibri", size=10, bold=True),
        "body_font":      Font(name="Calibri", size=10),
        "body_align":     Alignment(vertical="top", wrap_text=True),
        "zebra_fill":     PatternFill(start_color=PALETTE["subtle"],
                                      end_color=PALETTE["subtle"],
                                      fill_type="solid"),
        "good_fill":      PatternFill(start_color=PALETTE["good"],
                                      end_color=PALETTE["good"],
                                      fill_type="solid"),
        "warn_fill":      PatternFill(start_color=PALETTE["warn"],
                                      end_color=PALETTE["warn"],
                                      fill_type="solid"),
        "bad_fill":       PatternFill(start_color=PALETTE["bad"],
                                      end_color=PALETTE["bad"],
                                      fill_type="solid"),
        "info_fill":      PatternFill(start_color=PALETTE["info"],
                                      end_color=PALETTE["info"],
                                      fill_type="solid"),
    }


# ── Per-metric formatting ──────────────────────────────────────────────────


def _number_format_for(unit: str) -> str:
    """Excel format string for a given unit."""
    return {
        "$B":         '"$"#,##0.00"B"',
        "$M":         '"$"#,##0.00"M"',
        "%":          "0.0%",
        "/100":       "0.0",
        "M MT CO2":   "0.00",
        "min/yr":     "#,##0",
        "hrs/yr":     "#,##0",
        "grants":     "#,##0",
    }.get(unit, "#,##0.00")


def _value_for_excel(value, unit: str):
    """Adjust value to match the unit's expected display.  E.g. % is stored
    as a fraction (0.42 not 42) so '%' format prints '42.0%' correctly."""
    if value is None:
        return None
    if unit == "%":
        # If the value is already 0–1, leave it; if 0–100, divide
        try:
            return float(value) / 100 if float(value) > 1 else float(value)
        except (TypeError, ValueError):
            return value
    return value


# ── Excel writer ───────────────────────────────────────────────────────────


def to_excel(
    datapoints: list[DataPoint],
    path: str | Path,
    *,
    summary_text: str | None = None,
    insights: str | None = None,
    audit_summary: str | None = None,
    flags: dict | None = None,
) -> Path:
    """Write a polished multi-sheet workbook for analyst hand-off."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.dimensions import ColumnDimension

    path = Path(path)
    flags = flags or {}
    s = _styles()

    companies = sorted({dp.company for dp in datapoints})
    metric_keys_seen: list[str] = []
    for dp in datapoints:
        if dp.metric not in metric_keys_seen:
            metric_keys_seen.append(dp.metric)
    # Prefer the canonical METRICS order, then any custom metrics
    ordered_metrics = [m for m in METRICS if m in metric_keys_seen]
    custom = [m for m in metric_keys_seen if m not in METRICS]
    ordered_metrics.extend(sorted(custom))

    ok_dps = [dp for dp in datapoints if dp.ok]
    fail_dps = [dp for dp in datapoints if not dp.ok]

    wb = Workbook()
    # Remove the default sheet — we'll create our own
    wb.remove(wb.active)

    # ── Sheet 1: Cover ─────────────────────────────────────────────────────
    cover = wb.create_sheet("Cover")
    _write_cover_sheet(cover, datapoints, ok_dps, fail_dps, companies,
                       ordered_metrics, audit_summary, insights, s)

    # ── Sheet 2: Where Con Edison Stands ───────────────────────────────────
    standing_sheet = wb.create_sheet("Where Con Ed Stands")
    _write_standing_sheet(standing_sheet, datapoints, ordered_metrics, s)

    # ── Sheet 3: Benchmark (wide, formatted) ───────────────────────────────
    bench = wb.create_sheet("Benchmark")
    _write_benchmark_sheet(bench, datapoints, companies, ordered_metrics, flags, s)

    # ── Sheet 4: Per metric (one labeled block per metric) ─────────────────
    per_metric = wb.create_sheet("Per metric")
    _write_per_metric_sheet(per_metric, datapoints, companies, ordered_metrics, flags, s)

    # ── Sheet 5: Failures & gaps ───────────────────────────────────────────
    if fail_dps:
        fails = wb.create_sheet("Failures & gaps")
        _write_failures_sheet(fails, fail_dps, flags, s)

    # ── Sheet 6: AI Audit ─────────────────────────────────────────────────
    if audit_summary or insights:
        audit = wb.create_sheet("AI Audit")
        _write_audit_sheet(audit, audit_summary, insights, s)

    # ── Sheet 7: Attempts log (hidden) ─────────────────────────────────────
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
                "Preview": (a.response_preview or "")[:240],
            })
    if attempt_rows:
        att = wb.create_sheet("Attempts log")
        _write_attempts_sheet(att, attempt_rows, s)
        att.sheet_state = "hidden"

    wb.save(path)
    return path


# ── Sheet writers ──────────────────────────────────────────────────────────


def _write_cover_sheet(ws, datapoints, ok_dps, fail_dps, companies,
                       ordered_metrics, audit_summary, insights, s):
    """One-page executive brief."""
    from openpyxl.utils import get_column_letter

    confidences = [dp.confidence_score for dp in ok_dps if dp.confidence_score]
    avg_conf = sum(confidences) / max(1, len(confidences))

    # Title
    ws["A1"] = "Utility Benchmark — Run Summary"
    ws["A1"].font = s["title_font"]
    ws.row_dimensions[1].height = 30

    ws["A2"] = f"Compiled {datetime.utcnow().strftime('%B %d, %Y · %H:%M UTC')}"
    ws["A2"].font = s["subtitle_font"]

    ws.row_dimensions[3].height = 10  # spacer

    row = 4
    def _section(text):
        nonlocal row
        ws.cell(row=row, column=1, value=text).font = s["section_font"]
        row += 1

    def _kv(label, value):
        nonlocal row
        c1 = ws.cell(row=row, column=1, value=_sanitize_for_excel(label))
        c1.font = s["label_font"]
        c1.alignment = s["body_align"]
        c2 = ws.cell(row=row, column=2, value=_sanitize_for_excel(str(value)))
        c2.font = s["body_font"]
        c2.alignment = s["body_align"]
        row += 1

    _section("Scope")
    _kv("Companies", f"{len(companies)} — {', '.join(companies)}")
    _kv("Metrics", f"{len(ordered_metrics)} — " +
        ", ".join(METRICS.get(m, {}).get("label", m) for m in ordered_metrics))
    row += 1

    _section("Coverage")
    _kv("Total (company × metric) pairs", len(datapoints))
    _kv("Successful extractions", f"{len(ok_dps)} ({len(ok_dps)/max(1,len(datapoints))*100:.0f}%)")
    _kv("Null (with reason)", len(fail_dps))
    _kv("Average confidence (when value present)", f"{avg_conf:.2f}")
    row += 1

    _section("Confidence breakdown")
    for threshold, label in [(0.90, "Government API exact (≥ 0.90)"),
                              (0.75, "ProPublica 990 / derived (≥ 0.75)"),
                              (0.60, "CSR PDF (≥ 0.60)"),
                              (0.50, "Third-party survey (≥ 0.50)")]:
        n = sum(1 for c in confidences if c >= threshold)
        _kv(label, n)
    row += 1

    _section("Source mix (successful values only)")
    source_count: dict[str, int] = defaultdict(int)
    for dp in ok_dps:
        key = (dp.source_name or "unknown").split(" — ")[0]
        source_count[key] += 1
    for src, n in sorted(source_count.items(), key=lambda x: -x[1]):
        _kv(src, n)
    row += 1

    if audit_summary:
        _section("Data quality audit (excerpt)")
        # Show first ~6 lines
        excerpt = "\n".join((audit_summary or "").split("\n")[:8])
        c = ws.cell(row=row, column=1, value=_sanitize_for_excel(excerpt))
        c.alignment = s["body_align"]
        c.font = s["body_font"]
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)
        ws.row_dimensions[row].height = 100
        row += 1
        ws.cell(row=row, column=1, value="(Full audit on the 'AI Audit' sheet.)").font = s["subtitle_font"]
        row += 2

    # Column widths
    ws.column_dimensions["A"].width = 38
    ws.column_dimensions["B"].width = 80

    # Hide gridlines for a cleaner look
    ws.sheet_view.showGridLines = False


def _write_standing_sheet(ws, datapoints, ordered_metrics, s):
    """Sheet showing Con Edison's standing on each metric: rank, peer median,
    leader, laggard, narrative.  Drives the executive 'how do we look?' read."""
    from openpyxl.styles import Alignment
    from pipeline.ai_layer import compute_standing

    standings = compute_standing(datapoints, focus_company="Con Edison")

    ws["A1"] = "Where Con Edison Stands"
    ws["A1"].font = s["title_font"]
    ws.row_dimensions[1].height = 28
    ws["A2"] = ("For every metric, this sheet shows Con Edison's rank, "
                "distance from the peer median, and who leads / lags the "
                "peer set.")
    ws["A2"].font = s["subtitle_font"]
    ws.row_dimensions[2].height = 20
    ws.merge_cells("A2:H2")

    headers = ["Metric", "Verdict", "Con Ed value", "Rank", "of",
               "Peer median", "vs Median %", "Leader (Company / value)",
               "Laggard (Company / value)", "Narrative"]
    for col_idx, h in enumerate(headers, 1):
        cell = ws.cell(row=4, column=col_idx, value=h)
        cell.fill = s["header_fill"]
        cell.font = s["header_font"]
        cell.alignment = s["header_align"]
        cell.border = s["border"]
    ws.row_dimensions[4].height = 28

    row = 5
    for metric_key in ordered_metrics:
        info = standings.get(metric_key, {})
        meta = METRICS.get(metric_key, {})
        label = meta.get("label", metric_key)
        unit = meta.get("unit", "")
        verdict = info.get("verdict", "—")

        # Background color cue per verdict
        verdict_fill = {
            "Top performer": s["good_fill"],
            "Above median":  s["good_fill"],
            "Median":        s["warn_fill"],
            "Below median":  s["warn_fill"],
            "Lowest":        s["bad_fill"],
            "No data":       s["info_fill"],
        }.get(verdict)

        cells_to_color = []
        # Metric
        c = ws.cell(row=row, column=1, value=label)
        c.font = s["label_font"]
        cells_to_color.append(c)
        # Verdict
        c = ws.cell(row=row, column=2, value=verdict)
        c.font = s["body_font"]
        cells_to_color.append(c)
        # Con Ed value
        if info.get("focus_value") is not None:
            c = ws.cell(row=row, column=3, value=info["focus_value"])
            c.number_format = _number_format_for(unit)
        else:
            c = ws.cell(row=row, column=3, value="—")
        c.alignment = Alignment(horizontal="right", vertical="center")
        cells_to_color.append(c)
        # Rank, of
        ws.cell(row=row, column=4, value=info.get("rank") or "—")
        ws.cell(row=row, column=5, value=info.get("of_total") or "—")
        cells_to_color += [ws.cell(row=row, column=4),
                           ws.cell(row=row, column=5)]
        # Peer median
        if info.get("peer_median") is not None:
            c = ws.cell(row=row, column=6, value=info["peer_median"])
            c.number_format = _number_format_for(unit)
        else:
            c = ws.cell(row=row, column=6, value="—")
        c.alignment = Alignment(horizontal="right", vertical="center")
        cells_to_color.append(c)
        # vs Median %
        if info.get("vs_median_pct") is not None:
            ws.cell(row=row, column=7, value=info["vs_median_pct"] / 100)
            ws.cell(row=row, column=7).number_format = "+0.0%;-0.0%"
        else:
            ws.cell(row=row, column=7, value="—")
        cells_to_color.append(ws.cell(row=row, column=7))
        # Leader
        if info.get("best"):
            ws.cell(row=row, column=8,
                    value=f"{info['best'][0]}: {info['best'][1]:g}")
        else:
            ws.cell(row=row, column=8, value="—")
        cells_to_color.append(ws.cell(row=row, column=8))
        # Laggard
        if info.get("worst"):
            ws.cell(row=row, column=9,
                    value=f"{info['worst'][0]}: {info['worst'][1]:g}")
        else:
            ws.cell(row=row, column=9, value="—")
        cells_to_color.append(ws.cell(row=row, column=9))
        # Narrative
        c = ws.cell(row=row, column=10, value=_sanitize_for_excel(info.get("narrative", "")))
        c.alignment = s["body_align"]
        c.font = s["body_font"]
        cells_to_color.append(c)

        for cc in cells_to_color:
            cc.border = s["border"]
            if verdict_fill is not None:
                cc.fill = verdict_fill
        row += 1

    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 16
    ws.column_dimensions["C"].width = 14
    ws.column_dimensions["D"].width = 8
    ws.column_dimensions["E"].width = 6
    ws.column_dimensions["F"].width = 14
    ws.column_dimensions["G"].width = 12
    ws.column_dimensions["H"].width = 30
    ws.column_dimensions["I"].width = 30
    ws.column_dimensions["J"].width = 60
    ws.freeze_panes = "A5"
    ws.sheet_view.showGridLines = False


def _write_benchmark_sheet(ws, datapoints, companies, ordered_metrics, flags, s):
    """Wide companies × metrics table, cleanly formatted."""
    from openpyxl.utils import get_column_letter
    from openpyxl.styles import Alignment

    # Title
    ws["A1"] = "Benchmark — Companies × Metrics"
    ws["A1"].font = s["title_font"]
    ws.row_dimensions[1].height = 28
    ws.merge_cells(start_row=1, start_column=1,
                   end_row=1, end_column=1 + len(ordered_metrics) + 1)

    ws["A2"] = (
        "Each cell shows the value with a flag emoji.  ✅ verified · "
        "⚠️ caveat · 🚩 data exists but couldn't be retrieved · "
        "ℹ️ not applicable · — null"
    )
    ws["A2"].font = s["subtitle_font"]
    ws.row_dimensions[2].height = 20
    ws.merge_cells(start_row=2, start_column=1,
                   end_row=2, end_column=1 + len(ordered_metrics) + 1)

    # Header row at row 4
    header_row = 4
    headers = ["Company"] + [
        f"{METRICS.get(m, {}).get('label', m)}\n({METRICS.get(m, {}).get('unit', '')})"
        for m in ordered_metrics
    ] + ["Flags summary"]
    for col_idx, h in enumerate(headers, 1):
        cell = ws.cell(row=header_row, column=col_idx, value=_sanitize_for_excel(h))
        cell.fill = s["header_fill"]
        cell.font = s["header_font"]
        cell.alignment = s["header_align"]
        cell.border = s["border"]
    ws.row_dimensions[header_row].height = 36

    # Body rows
    for r_offset, company in enumerate(companies):
        excel_row = header_row + 1 + r_offset
        # Company name cell
        c = ws.cell(row=excel_row, column=1, value=_sanitize_for_excel(company))
        c.font = s["label_font"]
        c.alignment = s["body_align"]
        c.border = s["border"]

        flag_summary: dict[str, int] = defaultdict(int)
        for col_idx, metric in enumerate(ordered_metrics, 2):
            dp = next((d for d in datapoints if d.company == company and d.metric == metric), None)
            flag_info = flags.get((company, metric), {})
            flag = flag_info.get("flag", "")
            cell = ws.cell(row=excel_row, column=col_idx)
            unit = METRICS.get(metric, {}).get("unit", "")

            if dp and dp.ok:
                # Real number: store as number, format via number_format
                cell.value = _value_for_excel(dp.value, unit)
                cell.number_format = _number_format_for(unit)
                cell.alignment = Alignment(horizontal="right", vertical="center")
                # Set flag in a comment so the cell stays a number
                from openpyxl.comments import Comment
                if flag:
                    flag_text = (flag_info.get("reason") or "")[:200]
                    cell.comment = Comment(f"{flag}  {flag_text}", "Pipeline")
            else:
                # Null cell: show flag + em-dash
                cell.value = f"{flag} —"
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.font = s["body_font"]

            cell.border = s["border"]
            if flag:
                flag_summary[flag] += 1
            # Zebra striping
            if r_offset % 2 == 1 and not (dp and dp.ok):
                cell.fill = s["zebra_fill"]
            elif r_offset % 2 == 1:
                cell.fill = s["zebra_fill"]

        # Flags summary column
        summary_str = "  ".join(f"{flag}{n}" for flag, n in sorted(flag_summary.items()))
        last_cell = ws.cell(row=excel_row, column=len(headers), value=summary_str)
        last_cell.alignment = Alignment(horizontal="left", vertical="center")
        last_cell.border = s["border"]
        last_cell.font = s["body_font"]
        if r_offset % 2 == 1:
            last_cell.fill = s["zebra_fill"]

    # Column widths
    ws.column_dimensions["A"].width = 28
    for col_idx in range(2, len(ordered_metrics) + 2):
        ws.column_dimensions[get_column_letter(col_idx)].width = 18
    ws.column_dimensions[get_column_letter(len(headers))].width = 20

    # Freeze panes at top-left of data
    ws.freeze_panes = ws.cell(row=header_row + 1, column=2).coordinate

    # Autofilter
    last_col = get_column_letter(len(headers))
    last_row = header_row + len(companies)
    ws.auto_filter.ref = f"A{header_row}:{last_col}{last_row}"

    ws.sheet_view.showGridLines = False


def _write_per_metric_sheet(ws, datapoints, companies, ordered_metrics, flags, s):
    """One labeled block per metric, with each company's value clearly tied to it."""
    from openpyxl.utils import get_column_letter
    from openpyxl.styles import Alignment

    ws["A1"] = "Per metric — companies and values"
    ws["A1"].font = s["title_font"]
    ws.row_dimensions[1].height = 28

    ws["A2"] = ("One block per metric. Each company is on its own row inside "
                "the block, so you can see exactly which value belongs to which company.")
    ws["A2"].font = s["subtitle_font"]
    ws.row_dimensions[2].height = 18

    row = 4
    for metric in ordered_metrics:
        meta = METRICS.get(metric, {})
        label = meta.get("label", metric)
        unit = meta.get("unit", "")
        description = meta.get("description", "(custom metric)")
        expected = meta.get("expected_range", "n/a")
        lower_better = meta.get("lower_is_better", False)

        # Section header
        section_title = f"{label}" + (f"  ·  {unit}" if unit else "")
        cell = ws.cell(row=row, column=1, value=section_title)
        cell.font = s["section_font"]
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=6)
        ws.row_dimensions[row].height = 22
        row += 1

        # Description line
        c = ws.cell(row=row, column=1,
                    value=f"{description} · expected {expected} · "
                          f"{'lower is better' if lower_better else 'higher is better'}")
        c.font = s["subtitle_font"]
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
        row += 1

        # Column headers for this block
        headers = ["Flag", "Company", "Value", "Unit", "Year",
                   "Confidence", "Source", "Source URL"]
        for col_idx, h in enumerate(headers, 1):
            hc = ws.cell(row=row, column=col_idx, value=h)
            hc.fill = s["header_fill"]
            hc.font = s["header_font"]
            hc.alignment = s["header_align"]
            hc.border = s["border"]
        ws.row_dimensions[row].height = 22
        row += 1

        # Sort companies by value (high→low or low→high based on lower_is_better),
        # nulls at bottom
        rows_data = []
        for company in companies:
            dp = next((d for d in datapoints if d.company == company and d.metric == metric), None)
            flag_info = flags.get((company, metric), {})
            rows_data.append((company, dp, flag_info))

        def sort_key(t):
            _, dp, _ = t
            if dp and dp.ok:
                return (0, -dp.value if not lower_better else dp.value)
            return (1, 0)
        rows_data.sort(key=sort_key)

        for r_offset, (company, dp, flag_info) in enumerate(rows_data):
            flag = flag_info.get("flag", "")
            zebra = (r_offset % 2 == 1)

            cells = []
            # Flag
            c = ws.cell(row=row, column=1, value=flag)
            c.alignment = Alignment(horizontal="center", vertical="center")
            cells.append(c)
            # Company
            c = ws.cell(row=row, column=2, value=_sanitize_for_excel(company))
            c.font = s["label_font"]
            cells.append(c)
            # Value
            if dp and dp.ok:
                vc = ws.cell(row=row, column=3, value=_value_for_excel(dp.value, unit))
                vc.number_format = _number_format_for(unit)
                vc.alignment = Alignment(horizontal="right", vertical="center")
            else:
                # Show "—" with flag-colored fill so failures are scannable
                vc = ws.cell(row=row, column=3, value="—")
                vc.alignment = Alignment(horizontal="right", vertical="center")
                if flag == "🚩":
                    vc.fill = s["bad_fill"]
                elif flag == "ℹ️":
                    vc.fill = s["info_fill"]
            cells.append(vc)
            # Unit
            c = ws.cell(row=row, column=4, value=unit)
            c.alignment = Alignment(horizontal="left", vertical="center")
            cells.append(c)
            # Year
            c = ws.cell(row=row, column=5, value=_sanitize_for_excel(dp.year if dp and dp.ok else "—"))
            cells.append(c)
            # Confidence (always present — 0.00 for missing)
            conf_val = dp.confidence_score if dp else 0.0
            cc = ws.cell(row=row, column=6, value=conf_val if conf_val is not None else 0.0)
            cc.number_format = "0.00"
            cc.alignment = Alignment(horizontal="right", vertical="center")
            cells.append(cc)
            # Source name (or reason for failure)
            if dp and dp.ok:
                source_text = (dp.source_name or "").split(" — ")[0]
            else:
                source_text = (flag_info.get("reason") or "(see Failures sheet)")[:200]
            c = ws.cell(row=row, column=7, value=_sanitize_for_excel(source_text))
            c.alignment = s["body_align"]
            cells.append(c)
            # Source URL (clickable)
            if dp and dp.ok and dp.source_url:
                url_cell = ws.cell(row=row, column=8, value=dp.source_url)
                url_cell.hyperlink = dp.source_url
                url_cell.style = "Hyperlink"
                url_cell.alignment = s["body_align"]
            else:
                url_cell = ws.cell(row=row, column=8, value="—")
                url_cell.alignment = Alignment(horizontal="center")
            cells.append(url_cell)

            # Apply zebra striping + borders + body font
            for cc in cells:
                cc.border = s["border"]
                if cc.font is None or cc.font.bold is False:
                    cc.font = s["body_font"]
                if zebra and cc.fill.start_color.rgb in (None, "00000000"):
                    cc.fill = s["zebra_fill"]
            row += 1

        row += 1  # spacer between metrics

    # Column widths
    ws.column_dimensions["A"].width = 8     # Flag
    ws.column_dimensions["B"].width = 28    # Company
    ws.column_dimensions["C"].width = 14    # Value
    ws.column_dimensions["D"].width = 12    # Unit
    ws.column_dimensions["E"].width = 12    # Year
    ws.column_dimensions["F"].width = 12    # Confidence
    ws.column_dimensions["G"].width = 32    # Source
    ws.column_dimensions["H"].width = 50    # URL

    ws.sheet_view.showGridLines = False


def _write_failures_sheet(ws, fail_dps, flags, s):
    """Dedicated failures sheet with 'what this means' plain-English column."""
    from openpyxl.utils import get_column_letter
    from openpyxl.styles import Alignment

    ws["A1"] = "Failures & gaps"
    ws["A1"].font = s["title_font"]
    ws.row_dimensions[1].height = 28

    ws["A2"] = ("These are honest reports of what couldn't be retrieved.  "
                "Some are legitimate (ℹ️) — the data doesn't exist for that combination.  "
                "Others (🚩) are likely retrievable manually — see 'What this means' for guidance.")
    ws["A2"].font = s["subtitle_font"]
    ws.row_dimensions[2].height = 32
    ws.merge_cells("A2:F2")

    headers = ["Flag", "Company", "Metric", "What this means (plain English)",
               "Technical reason", "Sources tried"]
    for col_idx, h in enumerate(headers, 1):
        cell = ws.cell(row=4, column=col_idx, value=h)
        cell.fill = s["header_fill"]
        cell.font = s["header_font"]
        cell.alignment = s["header_align"]
        cell.border = s["border"]
    ws.row_dimensions[4].height = 28

    for r_offset, dp in enumerate(fail_dps):
        excel_row = 5 + r_offset
        flag_info = flags.get((dp.company, dp.metric), {})
        flag = flag_info.get("flag", "🚩")
        meta = METRICS.get(dp.metric, {})

        cells = [
            ws.cell(row=excel_row, column=1, value=flag),
            ws.cell(row=excel_row, column=2, value=_sanitize_for_excel(dp.company)),
            ws.cell(row=excel_row, column=3, value=_sanitize_for_excel(meta.get("label", dp.metric))),
            ws.cell(row=excel_row, column=4, value=_sanitize_for_excel(flag_info.get("reason", "")[:500])),
            ws.cell(row=excel_row, column=5, value=_sanitize_for_excel((dp.error or {}).get("reason", "")[:500])),
            ws.cell(row=excel_row, column=6, value=_sanitize_for_excel(", ".join((dp.error or {}).get("attempted_sources", []))[:200])),
        ]
        for c in cells:
            c.border = s["border"]
            c.alignment = s["body_align"]
            c.font = s["body_font"]
        # Color-code the row by flag
        if flag == "ℹ️":
            for c in cells:
                c.fill = s["info_fill"]
        elif flag == "🚩":
            for c in cells:
                c.fill = s["bad_fill"]

    # Column widths
    ws.column_dimensions["A"].width = 6
    ws.column_dimensions["B"].width = 24
    ws.column_dimensions["C"].width = 28
    ws.column_dimensions["D"].width = 60
    ws.column_dimensions["E"].width = 50
    ws.column_dimensions["F"].width = 35

    ws.freeze_panes = "A5"
    ws.sheet_view.showGridLines = False


def _write_audit_sheet(ws, audit_summary, insights, s):
    """AI audit + insights, formatted for reading."""
    from openpyxl.utils import get_column_letter
    from openpyxl.styles import Alignment

    ws["A1"] = "AI Audit & Insights"
    ws["A1"].font = s["title_font"]
    ws.row_dimensions[1].height = 28

    ws["A2"] = ("Generated by Claude after the data was compiled.  "
                "Audit reviews each value for plausibility; insights highlight "
                "strategic findings.")
    ws["A2"].font = s["subtitle_font"]
    ws.row_dimensions[2].height = 18

    row = 4
    if audit_summary:
        ws.cell(row=row, column=1, value="Data quality audit").font = s["section_font"]
        row += 1
        for line in (audit_summary or "").split("\n"):
            line = line.strip()
            if not line:
                row += 1
                continue
            c = ws.cell(row=row, column=1, value=_sanitize_for_excel(line))
            c.alignment = s["body_align"]
            c.font = s["body_font"]
            ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=4)
            row += 1
        row += 2

    if insights:
        ws.cell(row=row, column=1, value="Strategic insights").font = s["section_font"]
        row += 1
        for line in (insights or "").split("\n"):
            line = line.strip()
            if not line:
                row += 1
                continue
            c = ws.cell(row=row, column=1, value=_sanitize_for_excel(line))
            c.alignment = s["body_align"]
            c.font = s["body_font"]
            ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=4)
            row += 1

    ws.column_dimensions["A"].width = 90
    ws.sheet_view.showGridLines = False


def _write_attempts_sheet(ws, attempt_rows, s):
    """Full HTTP attempt log for debugging.  Not pretty but complete."""
    from openpyxl.utils import get_column_letter

    df = pd.DataFrame(attempt_rows)
    df = _sanitize_dataframe(df)

    # Header
    for col_idx, h in enumerate(df.columns, 1):
        cell = ws.cell(row=1, column=col_idx, value=h)
        cell.fill = s["header_fill"]
        cell.font = s["header_font"]
        cell.alignment = s["header_align"]
        cell.border = s["border"]

    for r_offset, (_, rowdata) in enumerate(df.iterrows()):
        for col_idx, h in enumerate(df.columns, 1):
            ws.cell(row=2 + r_offset, column=col_idx,
                    value=_sanitize_for_excel(rowdata[h])).border = s["border"]

    widths = {"Company": 24, "Metric": 24, "Source": 28, "URL": 60, "Method": 12,
              "Status": 8, "OK": 6, "Time (ms)": 10, "Error": 40, "Preview": 50}
    for col_idx, h in enumerate(df.columns, 1):
        ws.column_dimensions[get_column_letter(col_idx)].width = widths.get(h, 18)

    ws.freeze_panes = "A2"
    if df.shape[0] > 0:
        last_col = get_column_letter(df.shape[1])
        ws.auto_filter.ref = f"A1:{last_col}{df.shape[0]+1}"
