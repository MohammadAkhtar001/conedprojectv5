"""
Streamlit UI for the utility benchmark pipeline.

Deploys cleanly to Streamlit Cloud.  See STREAMLIT_DEPLOYMENT.md.
"""

from __future__ import annotations
import io
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ── Path bootstrap (Streamlit Cloud sometimes runs from a different CWD) ────
# Make sure the directory containing this file is on sys.path so the
# `pipeline/` and `extractors/` packages can be imported regardless of how
# the app was launched.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import streamlit as st

# Wrap third-party deps in try/except so a missing package produces a
# clean diagnostic page in the browser instead of a redacted crash.
_missing_deps = []
try:
    import pandas as pd
except ModuleNotFoundError:
    _missing_deps.append("pandas")
try:
    import plotly.express as px
except ModuleNotFoundError:
    _missing_deps.append("plotly")

if _missing_deps:
    st.set_page_config(page_title="Utility Benchmark — missing dependency", page_icon="⚠️")
    st.error(f"**Missing Python package(s):** `{', '.join(_missing_deps)}`")
    st.markdown(
        f"""
        Streamlit Cloud installs packages listed in `requirements.txt` at the
        repo root.  If you're seeing this, it usually means one of:

        1. **`requirements.txt` is not at the same level as `app.py`** in your
           GitHub repo.  Both files must sit at the **repo root**.  Go to your
           GitHub repo's main page — you should see `app.py` AND `requirements.txt`
           in the file listing without clicking into any folder.

        2. **Streamlit cached a build from before `requirements.txt` existed.**
           In Streamlit Cloud → **Manage app → ⋯ menu → Reboot app**.  If that
           doesn't work, try **Delete app** and create it fresh from the same
           repo — this forces a clean install.

        3. **`requirements.txt` is missing or empty.**  It should contain at
           minimum:
           ```
           streamlit>=1.30
           pandas>=2.0
           plotly>=5.18
           openpyxl>=3.1
           pdfplumber>=0.11
           beautifulsoup4>=4.12
           lxml>=5.0
           requests>=2.31
           tenacity>=8.2
           python-dotenv>=1.0
           anthropic>=0.39
           ```

        **Diagnostics:**
        - `app.py` ran from: `{_HERE}`
        - Files next to app.py: `{sorted(p.name for p in _HERE.iterdir())}`
        - `requirements.txt` present at this level: `{(_HERE / 'requirements.txt').exists()}`
        """
    )
    st.stop()

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Surface import errors clearly in the Streamlit UI so deploy issues are
# easy to diagnose, instead of "data leaks redacted" cryptic messages.
try:
    from pipeline.orchestrator import (run_pipeline, _fuzzy_match_metric,
                                        infer_metric_unit)
    from pipeline.export import to_excel
    from pipeline.ai_layer import verify, generate_insights, merged_flags
    from pipeline.models import METRICS
    from extractors.company_registry import REGISTRY, resolve as resolve_company
except ModuleNotFoundError as e:
    st.set_page_config(page_title="Utility Benchmark — import error", page_icon="⚠️")
    st.error(f"**Import failed:** `{e.name}` could not be found.")
    st.markdown(
        f"""
        This usually means the project's folder layout is wrong on the deploy host.

        **Expected layout** (relative to `app.py`):
        ```
        app.py
        run.py
        requirements.txt
        pipeline/
            __init__.py
            orchestrator.py
            ...
        extractors/
            __init__.py
            company_registry.py
            ...
        ```

        **Diagnostics from this run:**
        - `app.py` is at: `{_HERE}`
        - Files next to app.py: `{sorted(p.name for p in _HERE.iterdir())}`
        - `sys.path[0]`: `{sys.path[0]}`

        **Likely fixes:**
        1. On GitHub, make sure `pipeline/` and `extractors/` are at the **same level** as `app.py` — not nested inside another folder.
        2. Verify both folders contain a (possibly empty) `__init__.py`.
        3. In Streamlit Cloud → **Manage app → Settings**, set the *Main file path* to point at `app.py` at the project root (e.g. `app.py`, not `utility_pipeline/app.py`).
        """
    )
    st.stop()


st.set_page_config(
    page_title="Utility Benchmark",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Utility Benchmark Pipeline")
st.caption(
    "Government APIs first · per-source extractors · no fallback / mock / seed data."
)


# ── Helper functions ────────────────────────────────────────────────────────


def _draw_metric_chart(df_long, metric_key: str, flags_map: dict, *, height: int = 360):
    """Draw a single metric's bar chart, sorted by value, color = confidence."""
    meta = METRICS.get(metric_key, {})
    sub = df_long[(df_long["metric"] == metric_key) & (df_long["ok"])].copy()
    if sub.empty:
        st.info(f"No successful values for {meta.get('label', metric_key)}.")
        return

    sub = sub.sort_values(
        "value",
        ascending=not meta.get("lower_is_better", False),
    )

    # Build hover text with flag + reason
    sub["flag"] = sub.apply(
        lambda r: flags_map.get((r["company"], r["metric"]), {}).get("flag", ""),
        axis=1,
    )
    sub["company_with_flag"] = sub["flag"] + " " + sub["company"]

    label = meta.get("label", metric_key)
    unit = meta.get("unit", "")

    fig = px.bar(
        sub,
        x="value",
        y="company_with_flag",
        orientation="h",
        text="value",
        color="confidence_score",
        color_continuous_scale="Blues",
        range_color=[0.3, 1.0],
        custom_data=["source_name", "year", "confidence_score"],
    )
    fig.update_traces(
        texttemplate="%{x:,.2f}",
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            f"{label}: %{{x:,.2f}} {unit}<br>"
            "Source: %{customdata[0]}<br>"
            "Year: %{customdata[1]}<br>"
            "Confidence: %{customdata[2]:.2f}"
            "<extra></extra>"
        ),
    )
    fig.update_layout(
        title=dict(text=f"{label} ({unit})" if unit else label, font=dict(size=15)),
        xaxis_title=unit if unit else None,
        yaxis_title=None,
        height=height,
        margin=dict(l=10, r=20, t=50, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(title="Conf.", thickness=10, len=0.6),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    st.plotly_chart(fig, use_container_width=True)


def _build_analysis_prompt(datapoints, flags_map: dict) -> str:
    """Compose a self-contained, per-metric structured prompt for another AI
    to analyze.  The data is organized as one section per metric so values
    are unambiguous — never just numbers in a row."""

    companies = sorted({dp.company for dp in datapoints})
    metric_keys_seen: list[str] = []
    for dp in datapoints:
        if dp.metric not in metric_keys_seen:
            metric_keys_seen.append(dp.metric)

    # ── Header / context ────────────────────────────────────────────────────
    n_total = len(datapoints)
    n_ok = sum(1 for d in datapoints if d.ok)
    n_fail = n_total - n_ok
    avg_conf = (
        sum(d.confidence_score for d in datapoints if d.ok and d.confidence_score)
        / max(1, sum(1 for d in datapoints if d.ok and d.confidence_score))
    )

    header = f"""# Utility Benchmark Data — Compiled for Analysis

**Audience:** Con Edison Community Partnerships team (and their VP)
**Companies benchmarked ({len(companies)}):** {", ".join(companies)}
**Metrics benchmarked ({len(metric_keys_seen)}):** {", ".join(METRICS.get(m, {}).get("label", m) for m in metric_keys_seen)}
**Coverage:** {n_ok}/{n_total} pairs returned a value ({n_ok/max(1,n_total)*100:.0f}%) · average confidence {avg_conf:.2f}

## How to read this brief

Each metric below is its own section containing:
- The unit, what it means, expected range
- ONE labeled value per company, with flag, source, year, and confidence
- A "Methodology notes" line explaining the source the pipeline used

**Flag legend** (next to every value):
- ✅  Verified from a credible source, no concerns
- ⚠️  Caveat applies (partial coverage, e.g. only the foundation slice of total giving)
- 🚩  Data exists but the automated tool could not retrieve it — needs manual lookup
- ℹ️   Legitimately not applicable for this combination (structural reason)
- —   No value (see flag for why)

## Critical structural facts (do NOT treat these as data errors)

1. **Pure transmission/distribution utilities** (Con Edison, National Grid USA, Eversource, PSEG Long Island) do NOT own electricity-generation plants tracked by EPA eGRID. Their Scope 1 from generation is genuinely zero. Real Scope 1 (vehicle fleet, fugitive gas leaks) is small and only disclosed in CSR reports. ℹ️ does NOT mean the value is missing — it means it doesn't exist in the form requested.

2. **Integrated generators** (Duke Energy, Southern Company, PG&E, Dominion) emit ~10× more Scope 1 CO₂ than distributors. That ratio is structural, not an outlier.

3. **"Total Charitable Giving" from ProPublica is the FOUNDATION-PAID slice only** (IRS Form 990-PF, contributions paid line). It does NOT include direct corporate contributions, energy assistance programs, in-kind giving, or community investment from the operating budget. To assess total philanthropy, look at "Total Charitable Giving" + "Energy Assistance Programs" + "Community Investment" together.

4. **National Grid USA's parent is UK-listed** and does not file US 10-Ks. Its US operating subsidiaries (Niagara Mohawk Power, KeySpan, Massachusetts Electric) historically filed but do not appear in SEC's current company-facts API. Revenue 🚩 is therefore expected and should be filled in manually from National Grid plc's UK Annual Report (US segment).

---

# Data, organized by metric"""

    sections: list[str] = []
    for m in metric_keys_seen:
        meta = METRICS.get(m, {})
        label = meta.get("label", m)
        unit = meta.get("unit", "")
        description = meta.get("description", "(custom metric — no standard description)")
        expected = meta.get("expected_range", "n/a")
        lower_better = meta.get("lower_is_better", False)
        direction = "lower is better" if lower_better else "higher is better"

        section_lines = [
            f"\n## {label}",
            "",
            f"- **Unit:** {unit if unit else '(unitless / custom)'}",
            f"- **What it measures:** {description}",
            f"- **Expected range:** {expected}",
            f"- **Direction:** {direction}",
            "",
            "**Values by company:**",
            "",
        ]

        # Collect rows for this metric
        rows_for_metric = []
        for c in companies:
            dp = next((d for d in datapoints if d.company == c and d.metric == m), None)
            flag_info = flags_map.get((c, m), {})
            flag = flag_info.get("flag", "")
            flag_reason = flag_info.get("reason", "")
            if dp is None:
                continue
            if dp.ok:
                conf_pct = f"{int((dp.confidence_score or 0) * 100)}% confidence" if dp.confidence_score else "—"
                year_str = dp.year or "year unknown"
                source_str = (dp.source_name or "unknown source").split(" — ")[0]
                rows_for_metric.append(
                    f"- {flag} **{c}**: **{dp.value:g} {unit}**  "
                    f"_(source: {source_str}, {year_str}, {conf_pct})_"
                )
                if dp.notes and ("foundation" in dp.notes.lower() and "only" in dp.notes.lower()):
                    rows_for_metric.append(
                        f"    - ⚠️ Caveat: foundation 990-PF slice only; "
                        f"excludes direct corporate contributions, energy "
                        f"assistance, and in-kind."
                    )
            else:
                rows_for_metric.append(
                    f"- {flag} **{c}**: **null** — {flag_reason or '(see Failures section)'}"
                )

        # Sort successful values by value (rank order); failures at the bottom
        # We've appended in order so sort with a key
        ok_rows = []
        fail_rows = []
        for c in companies:
            dp = next((d for d in datapoints if d.company == c and d.metric == m), None)
            if dp is None:
                continue
            if dp.ok:
                ok_rows.append((c, dp))
            else:
                fail_rows.append(c)

        # Recompute the sorted output now that we've split
        section_lines = [
            f"\n## {label}",
            "",
            f"- **Unit:** {unit if unit else '(unitless / custom)'}",
            f"- **What it measures:** {description}",
            f"- **Expected range:** {expected}",
            f"- **Direction:** {direction}",
        ]
        if ok_rows:
            ok_rows.sort(key=lambda x: x[1].value, reverse=not lower_better)
            section_lines.append("")
            section_lines.append(
                f"**Values by company** (sorted "
                f"{'low → high' if lower_better else 'high → low'}):"
            )
            section_lines.append("")
            for rank, (c, dp) in enumerate(ok_rows, 1):
                flag = flags_map.get((c, m), {}).get("flag", "")
                conf_pct = (
                    f"{int((dp.confidence_score or 0) * 100)}% confidence"
                    if dp.confidence_score else ""
                )
                year_str = dp.year or "year unknown"
                source_str = (dp.source_name or "unknown source").split(" — ")[0]
                bullet = (
                    f"{rank}. {flag} **{c}**: **{dp.value:g} {unit}**".rstrip()
                    + f"  _({source_str}, {year_str}, {conf_pct})_"
                )
                section_lines.append(bullet)
                # Inline caveat for foundation-only data
                if (dp.notes and "foundation" in dp.notes.lower()
                        and "only" in dp.notes.lower()):
                    section_lines.append(
                        "    - ⚠️ Foundation 990-PF slice only; excludes "
                        "direct corporate contributions and in-kind."
                    )
        if fail_rows:
            section_lines.append("")
            section_lines.append("**No value retrieved for:**")
            for c in fail_rows:
                flag = flags_map.get((c, m), {}).get("flag", "🚩")
                reason = flags_map.get((c, m), {}).get("reason", "(no reason)")
                section_lines.append(f"- {flag} **{c}** — {reason}")

        # Source methodology note
        method_note = _methodology_note_for_metric(m)
        if method_note:
            section_lines.append("")
            section_lines.append(f"**Methodology:** {method_note}")

        sections.append("\n".join(section_lines))

    # ── What we want from the analyst AI ────────────────────────────────────
    closing = """\

---

# What I want from you

Now that you've read every metric, please produce the following five sections.

## 1. Critical review of values
For each metric section above, scan the ✅ values quickly and call out anything that looks wrong: a unit error, a scale mismatch, a value that's implausible given the company's size or business model. Be specific — quote the exact metric and company.

## 2. Plug the gaps for 🚩 and — values
For each null, name the specific manual lookup that would resolve it. Examples: "For National Grid USA Revenue, look at the 'US Performance' section of National Grid plc's most recent UK Annual Report." or "For PSEG Long Island Foundation Assets, search ProPublica for 'PSEG Foundation' (parent foundation; LI is a subsidiary)."

## 3. Strategic insights for the Community Partnerships VP (4–6 bullets)
Each insight starts with a bolded headline, then 1–2 sentences. Focus on:
- How Con Edison stacks up vs peers, normalized for company size where possible
- Specific competitive advantages or gaps in Con Edison's philanthropy posture
- A storyline an executive should know before a board meeting

## 4. Three concrete follow-up data pulls
Specific metrics from specific sources. Examples: "Pull Con Edison's K-12 STEM scholarship total from page 24 of their 2024 Sustainability Report." Be precise enough that an analyst could execute each one in under 15 minutes.

## 5. Caveats the VP should know before quoting numbers
Anything that, if mis-stated to the board, would damage credibility. Examples: "Con Edison's $0.03M foundation giving figure is the foundation-paid line only. Total corporate philanthropy is a multiple of that — quoting $0.03M would be misleading."

Do not invent numbers. If you don't have data, say "I don't have a verifiable source for that." Be ruthlessly specific."""

    return header + "\n".join(sections) + closing


def _methodology_note_for_metric(metric_key: str) -> str:
    """Return a short methodology note explaining what source/method was
    used for this metric."""
    notes = {
        "revenue": (
            "SEC EDGAR XBRL company-facts API. Pulled the most recent 10-K "
            "fiscal-year value of the `us-gaap:Revenues` tag (or close "
            "equivalent). Confidence 0.95."
        ),
        "charitable_giving": (
            "ProPublica Nonprofit Explorer (IRS 990-PF). Pulled grants paid out "
            "from the most recent fiscal year. ⚠️ This is the FOUNDATION-PAID "
            "slice only — does not include direct corporate contributions, "
            "energy assistance, or in-kind giving. Confidence 0.65."
        ),
        "foundation_grants_paid": (
            "ProPublica Nonprofit Explorer (IRS 990-PF). Same source as Total "
            "Charitable Giving but explicitly scoped to foundation grants only. "
            "Confidence 0.85."
        ),
        "foundation_assets": (
            "ProPublica Nonprofit Explorer (IRS 990-PF Part II, totassetsend). "
            "Confidence 0.85."
        ),
        "carbon_emissions": (
            "EPA eGRID2022 plant-level dataset. Annual CO₂ in short tons summed "
            "across all plants matching the company's eGRID operator name(s), "
            "converted to million metric tons (× 0.907185 / 1e6). Confidence "
            "0.95 for integrated generators; ℹ️ for pure distributors who do not "
            "own generation."
        ),
        "renewable_pct": (
            "Primary: company CSR/sustainability report. Secondary: EIA Form 923 "
            "generation mix derivation. Confidence 0.60."
        ),
        "saidi": (
            "EIA Form 861 reliability workbook (when implemented) or state PUC "
            "filings. Confidence 0.85."
        ),
        "customer_satisfaction": (
            "J.D. Power Residential Electric Satisfaction Study press release. "
            "Raw score is /1000, normalized to /100. Note: J.D. Power's free "
            "press release names only regional category WINNERS — most utilities "
            "will return null with no fault of the pipeline. Confidence 0.50."
        ),
        "energy_assistance": (
            "Company CSR/sustainability report — narrative section on energy "
            "assistance, LIHEAP supplements, and customer hardship programs. "
            "Confidence 0.60."
        ),
        "stem_education_giving": (
            "Company CSR/sustainability report — STEM/education narrative or "
            "foundation 990 grant listings. Confidence 0.60."
        ),
        "community_investment": (
            "Company CSR/sustainability report — community investment summary "
            "(typically the broadest line that includes philanthropy + economic "
            "development + customer assistance). Confidence 0.60."
        ),
        "volunteer_hours": (
            "Company CSR/sustainability report. Confidence 0.60."
        ),
        "employee_match": (
            "Company CSR/sustainability report — matching gift program total. "
            "Confidence 0.60."
        ),
        "num_grants": (
            "ProPublica 990-PF Part XV grant counts, or CSR-report disclosure. "
            "Confidence 0.85 (regulator) / 0.60 (CSR)."
        ),
    }
    return notes.get(metric_key, "Custom metric — extracted via fuzzy match to "
                                 "closest standard metric, or AI web search.")


def _build_standalone_copilot_prompt() -> str:
    """A self-contained prompt for Microsoft Copilot (or any AI with web
    search) that lets a Con Edison analyst run a benchmark without needing
    access to this Streamlit app.  Mirrors the tool's logic but no data
    is pre-embedded — the analyst types in their companies and metrics."""

    return """# Utility Benchmark — Standalone Copilot Mode

You are an analyst's research assistant compiling a benchmark of US electric / gas utilities for the Con Edison Community Partnerships team. Behave like a hand-built data pipeline, not a generic LLM. Refuse to invent values. Cite every number.

## How the analyst will use you

1. They'll tell you a list of companies (e.g. "Con Edison, Duke Energy, PG&E, Eversource, National Grid USA").
2. They'll tell you a list of metrics (e.g. "Revenue, Charitable Giving, Foundation Assets, Carbon Emissions Scope 1, Customer Satisfaction").
3. You will compile a benchmark by walking the source priority list below for each (company × metric) pair.
4. You will produce a per-metric breakdown, a peer-comparison standing for Con Edison, and 4-6 strategic insights.

## Source priority order — walk this for EVERY (company, metric) pair

For each pair, search ONLINE (use your web search tool) and stop at the FIRST source that returns a credible value.

### Tier 1: Government / regulator-of-record (preferred — confidence 0.85–0.95)

**Revenue → SEC EDGAR XBRL**
- URL pattern: `https://data.sec.gov/api/xbrl/companyfacts/CIK{10digit}.json`
- Pull most recent 10-K, fiscal-year value of `us-gaap:Revenues` (or close equivalent). Convert to $B.
- If the company doesn't file 10-Ks (UK parents like National Grid plc, public-power utilities), say so explicitly and try the parent's annual report.

**Foundation philanthropy → ProPublica Nonprofit Explorer (IRS Form 990-PF)**
- Search by foundation name: `https://projects.propublica.org/nonprofits/`
- Pull `cttgrntpd` / `grntspaid` (grants paid out) and `totassetsend` (foundation assets at year end).
- ⚠️ This is the FOUNDATION-PAID slice only — note clearly that it does NOT include direct corporate contributions, energy assistance programs, or in-kind giving.

**Carbon Emissions Scope 1 → EPA eGRID**
- URL: `https://www.epa.gov/egrid` (latest annual release)
- Sum plant-level CO₂ where the operator name matches the company. Convert short tons → million metric tons (× 0.907185 / 1e6).
- ℹ️ Pure T&D distributors (Con Edison, Eversource, National Grid USA, PSEG Long Island) do NOT own generation tracked here — return null with "not applicable, pure distributor" rather than hunting.

**Reliability (SAIDI), Renewable %, Generation Mix → EIA Form 861 / 923**
- URL: `https://www.eia.gov/electricity/data/`
- Filter by EIA Operator ID for the company.

### Tier 2: Third-party surveys (confidence 0.50)

**Customer Satisfaction → J.D. Power Residential Electric Study**
- URL: `https://www.jdpower.com/business/press-releases/{year}-us-electric-utility-residential-customer-satisfaction-study`
- Score is /1000 raw; normalize to /100.
- ⚠️ Public press release names ONLY regional category WINNERS (~7-8 utilities). For everyone else, return null with reason — that is correct, not a failure.

### Tier 3: Corporate disclosure (confidence 0.60)

**CSR / Sustainability Report PDF → company website**
- Search "{company name} sustainability report 2024 PDF" or "ESG report".
- Find the most recent PDF on the company's own site.
- Extract metrics for: Energy Assistance, Volunteer Hours, Employee Match, Number of Grants, Community Investment, STEM/Education Giving.
- Anchor your extraction to a distinctive phrase ("$X million in energy assistance", "Y volunteer hours") and validate the captured number.

### Tier 4: News / press release (confidence 0.30)

Last resort. Cite source with URL. Mark as low-confidence.

## Validation rules — REJECT values outside these ranges

| Metric | Unit | Min | Max | Expected |
|---|---|---|---|---|
| Revenue | $B | 1 | 80 | $10–35B large IOU |
| Foundation Grants Paid | $M | 0.01 | 200 | $0.5–20M |
| Foundation Assets | $M | 0.5 | 500 | $5–100M |
| Total Charitable Giving | $M | 0.01 | 500 | $5–100M |
| Energy Assistance | $M | 0.1 | 100 | $1–30M |
| Carbon Emissions Scope 1 | M MT CO2 | 0.5 | 200 | 2–10 distributor / 50–100 generator |
| Customer Satisfaction | /100 | 30 | 80 | 45–55 |
| Renewable % | % | 0 | 100 | 15–50% |
| SAIDI | min/yr | 20 | 500 | 50–200 |
| Volunteer Hours | hrs/yr | 100 | 500K | 5–100K |
| Employee Match | $M | 0.05 | 50 | $0.5–10M |

If a value is outside its plausible range, return null with reason. Do NOT pass it through.

## Output structure

For each metric you benchmark, produce a section in this format:

---

### {Metric Label} ({unit})
**What it measures:** {description}
**Expected range:** {expected_range}
**Direction:** {higher/lower is better}

**Values by company** (sorted best → worst):
1. ✅ **{Company A}**: **{value} {unit}**  _({source}, {year}, {confidence}% confidence)_
2. ✅ **{Company B}**: **{value} {unit}**  _({source}, {year}, {confidence}% confidence)_
3. ⚠️ **{Company C}**: **{value} {unit}**  _({source}, {year}, {confidence}% confidence)_
   - ⚠️ Caveat: foundation-only data; excludes direct corporate giving.

**No value retrieved for:**
- ℹ️ **{Company D}** — Not applicable: pure T&D distributor, no eGRID generation.
- 🚩 **{Company E}** — Foundation not registered with IRS as 501(c)(3); company likely gives from operating budget.

**Source URLs cited:**
- {full URL 1}
- {full URL 2}

**Methodology:** {one-line explanation of where the data came from and any caveats}

---

## Flag system

Every value gets a flag:
- ✅ Verified, high confidence, no concerns
- ⚠️ Caveat applies (partial coverage, foundation-only when total was implied, methodology mismatch)
- 🚩 Data exists but couldn't be retrieved cleanly, OR very low confidence
- ℹ️ Legitimately not applicable for this combination

## After the per-metric breakdown, produce these closing sections

### Where Con Edison stands (peer comparison)

For every metric where Con Edison has a value, note: rank (#X of Y), distance from peer median (e.g. "20% below peer median"), and the leader/laggard. Be direct.

### Strategic insights for the VP (4-6 bullets)

Each insight starts with a **bolded headline** then 1-2 sentences. Focus on:
- How Con Ed stacks up vs peers, normalized for company size
- Specific competitive advantages or gaps
- Anything that would change a board-level talking point

### Recommended manual follow-ups (3 specific data pulls)

For every 🚩 and missing value, name the specific source and section the analyst should check manually. Be precise enough that the lookup takes <15 minutes per pull.

### Caveats the VP should know before quoting numbers

Anything that, if mis-stated to the board, would damage credibility. Foundation-only philanthropy is the classic example.

## Hard rules

1. **Refuse to guess.** If you cannot find a credible source, return null with reason.
2. **Cite every number with a URL.** Every single one. No exceptions.
3. **Validate against the plausibility ranges above.** Reject and report null if outside.
4. **Distinguish "not applicable" from "couldn't find it"** — use ℹ️ vs 🚩.
5. **Foundation-only data must be flagged ⚠️.** Never present it as total philanthropy.

When the analyst gives you a list of companies and metrics, confirm the inputs, then go.
"""


def _build_recreation_prompt() -> str:
    """A detailed meta-prompt that another AI can use to mimic this tool's
    behavior interactively, including all sources, methodology, and rules."""

    return """# Utility Benchmark Pipeline — Interactive Mode

You are a hand-built data pipeline for benchmarking US electric and gas utilities. The user will give you a list of companies and a list of metrics, and you will compile a benchmark — pulling values from real sources, scoring confidence, flagging issues, and refusing to invent numbers.

You are not a generic LLM. You behave like a specific tool with specific rules. Every value you return must be from a real source you can cite. When you can't find a value, you return null with a structured reason — never a guess.

---

## PART 1 — Source priority order (walk this for EVERY metric)

For each (company, metric) pair, walk the source list IN ORDER. Stop at the FIRST source that returns a usable value. If all sources fail, return null with the reasons.

### Tier 1: Government / regulator APIs (preferred, highest confidence)

#### A. SEC EDGAR XBRL company-facts API — Revenue
- **URL pattern:** `https://data.sec.gov/api/xbrl/companyfacts/CIK{10digit}.json`
- **Auth:** None, but SEC requires a contact email in the User-Agent header. Generic UAs get 403.
- **Method:** Fetch the JSON. Look for `facts.us-gaap.Revenues.units.USD`. Filter to entries where `form == "10-K"` and `fp == "FY"`. Sort by `end` descending; take the most recent.
- **Output:** value (in dollars, divide by 1e9 for $B), `fy` field as year, the filing accession number for source URL.
- **Confidence:** 0.95.
- **Common CIKs:** Con Edison 0001047862; Duke 0001326160; Eversource 0000072741; PG&E (parent) 0001004980; Southern Co 0000092122; Dominion 0000715957; Exelon 0001109357; AEP 0000004904.

#### B. ProPublica Nonprofit Explorer — Foundation philanthropy
- **Search URL:** `https://projects.propublica.org/nonprofits/api/v2/search.json?q={name}`
- **Org URL:** `https://projects.propublica.org/nonprofits/api/v2/organizations/{ein_int}.json`
- **Method:** Search by foundation name (e.g. "Consolidated Edison Foundation"). Score candidates: prefer names containing "foundation" + the company's distinctive word. Penalize "welfare benefit", "master trust", "pension", "society", "international". Take the highest-scoring match. Then fetch the full org record. Use the most recent entry in `filings_with_data`.
- **Field mapping (IRS 990-PF):**
  - Charitable giving / grants paid → `cttgrntpd` or `grntspaid` or `cttgfgvprtcamt` (the "contributions, gifts, grants paid" line). Convert dollars → millions.
  - Foundation assets → `totassetsend`. Convert dollars → millions.
- **CRITICAL CAVEAT:** This is the FOUNDATION-PAID slice only. It does NOT include direct corporate contributions, energy assistance, in-kind, or community investment from operating budget. When you return a value here, FLAG it ⚠️ with this note.
- **Confidence:** 0.85 for foundation_grants_paid; 0.65 for "Total Charitable Giving" (downgraded because it's a partial picture).

#### C. EPA eGRID — Scope 1 carbon emissions
- **URL:** `https://www.epa.gov/system/files/documents/2024-01/egrid2022_data.xlsx` (latest as of Q1 2024). For 2025 onward, use eGRID2023: `https://www.epa.gov/system/files/documents/2025-01/egrid2023_data.xlsx`.
- **Method:** Download the Excel workbook. Read sheet `PLNT22` (or `PLNT23`) with `header=1` (row 2 is the column-codes row). Filter rows where `OPRNAME` matches the company's eGRID operator name(s). Sum `PLCO2AN` (annual plant CO₂ in short tons). Convert: `(short_tons * 0.907185) / 1e6` → million metric tons.
- **Operator-name normalization is required:** eGRID writes "Pacific Gas & Electric Company" but the registry might say "Pacific Gas and Electric Company". Collapse `&` ↔ `and`, strip punctuation, normalize whitespace before substring matching.
- **Confidence:** 0.95 for integrated generators.
- **CRITICAL:** Pure T&D distributors (Con Edison, National Grid USA, Eversource, PSEG Long Island) do NOT own generation tracked by eGRID. They will return ZERO matched plants. This is NOT an error — flag ℹ️ "not applicable" with explanation. Do not retry.

#### D. EIA Form 861 — SAIDI, generation mix, renewable %
- **URL:** `https://www.eia.gov/electricity/data/eia861/` (annual ZIP download)
- **Method:** Download zip, read the "Reliability" workbook, filter by EIA Operator ID. Pull SAIDI With MED. For multi-sub utilities, customer-weighted average across subs.
- **Confidence:** 0.85.

### Tier 2: Third-party surveys

#### E. J.D. Power Residential Electric Customer Satisfaction Study
- **URL pattern:** `https://www.jdpower.com/business/press-releases/{year}-us-electric-utility-residential-customer-satisfaction-study`
- **Method:** Fetch HTML, parse with BeautifulSoup, search for the company name and find the nearest 3-digit number in 600–900 range (raw J.D. Power score is /1000). Normalize to /100 by dividing by 10.
- **Confidence:** 0.50.
- **HARD LIMIT — KNOW THIS:** J.D. Power's free press release names ONLY regional category WINNERS (typically 7-8 utilities like PSE&G, Delmarva, MidAmerican, Omaha PPD, Georgia Power, EPB, SRP). Most utilities are NOT named in the public press release. Returning null for non-winners is correct. Detailed scores require a paid subscription.

### Tier 3: Corporate disclosure

#### F. Corporate Sustainability / CSR report PDF
- Used for: renewable_pct, energy_assistance, volunteer_hours, employee_match, num_grants, community_investment, stem_education_giving.
- **Method:** Fetch the most recent CSR report URL (must be hand-curated per company), download PDF, extract text with pdfplumber, run anchored regex patterns for each metric. Patterns are ANCHORED to a distinctive phrase (e.g. "energy assistance" + a $-amount window). Always validate the captured number against the metric's plausible range — if it fails validation, return null with reason rather than passing through.
- **Confidence:** 0.60.

### Tier 4: AI + web search (last resort)

#### G. AI fallback (this is YOU)
- Triggered when: no government source matched, OR custom user-defined metric, OR all higher tiers returned null.
- **Method:** Use web search to find the metric. Prefer regulator filings → CSR reports → reputable news. ALWAYS cite the source URL.
- **Confidence:** 0.85 if from regulator, 0.60 if from CSR, 0.50 if from third-party survey, 0.30 if from press release / news.
- **Refuse to guess.** If you cannot find a credible source, return null with reason.

---

## PART 2 — Validation rules (apply to every value before accepting it)

Reject any value outside the plausibility range. Don't pass it through with a warning — null it out with a clear reason. The whole point is that we'd rather give the user "no data, here's why" than a wrong number.

| Metric | Unit | Min | Max | Expected |
|---|---|---|---|---|
| Revenue | $B | 1 | 80 | $10B–$35B large IOUs |
| Renewable Energy % | % | 0 | 100 | 15%–50% (2024) |
| SAIDI | min/yr | 20 | 500 | 50–200 |
| Customer satisfaction | /100 | 30 | 80 | 45–55 (industry avg ≈ 49.9) |
| Carbon Emissions Scope 1 | M MT CO2 | 0.5 | 200 | 2–10 distributors / 50–100 generators |
| Total Charitable Giving | $M | 0.01 | 500 | $5M–$100M large utility |
| Foundation Grants Paid | $M | 0.01 | 200 | $0.5M–$20M |
| Foundation Assets | $M | 0.5 | 500 | $5M–$100M |
| Energy Assistance | $M | 0.1 | 100 | $1M–$30M |
| STEM/Education Giving | $M | 0.05 | 50 | $0.5M–$10M |
| Community Investment | $M | 0.1 | 500 | $10M–$100M |
| Volunteer Hours | hrs/yr | 100 | 500,000 | 5,000–100,000 |
| Employee Match | $M | 0.05 | 50 | $0.5M–$10M |

---

## PART 3 — The flag system (apply to every value)

Every value you output gets a flag based on these rules:

- **✅** Successful value with confidence ≥ 0.50 from a credible source, NOT a peer outlier, no caveats.
- **⚠️** Value is correct but has a caveat the user should know:
  - Foundation-only data when a "total" was implied
  - Confidence between 0.30 and 0.49
  - Value is 3–10× the peer median for that metric (potential scale mismatch)
- **🚩** Genuine concern OR data exists but couldn't be retrieved:
  - Confidence < 0.30 (press-release / news only)
  - Peer outlier ≥ 10× the median AND low confidence
  - Foundation not found on ProPublica (so we know there's a gap)
  - Custom metric with no extractor and no AI fallback available
  - Anthropic API error (credit, auth)
- **ℹ️** Legitimately not applicable for this combination:
  - Carbon Emissions for a pure T&D distributor (no eGRID generation)
  - SEC Revenue for a non-SEC-registered company (e.g. UK parent's US sub)
  - Customer Satisfaction for a non-J.D.-Power-named company
  - Any case where the data genuinely doesn't exist for this scope

NEVER use 🚩 as a default for "no value". Pick the right flag based on the cause.

---

## PART 3.5 — Confidence scoring rules (explicit)

Every value carries a confidence score 0.00–0.99. The rules:

1. **Source-tier baseline:**
   - SEC EDGAR XBRL exact: 0.95
   - EPA eGRID, EIA datasets exact: 0.95
   - IRS 990 / ProPublica exact: 0.85
   - CSR PDF (anchored phrase + validated number): 0.60
   - Third-party survey (J.D. Power, etc.): 0.50
   - AI fallback with web_search: 0.65 default (up to 0.85 if from a regulator URL)
   - AI fallback without web_search (training data only): cap 0.45
   - Press release / news: 0.30
   - **Missing data: 0.00** (explicit zero, not null — so it sorts correctly)

2. **Cross-source agreement bonus:** When two independent sources return values within 10% of each other for the same (company, metric), raise both confidence scores by +0.05 (capped at 0.99). Note this in the value's notes field.

3. **Validator failure:** If a candidate value falls outside the metric's plausibility range, REJECT it. Do NOT downgrade confidence and pass it through. Reject means null with reason.

4. **Stale data penalty:** If the source year is more than 3 years old and a more recent year is available from the same source family but couldn't be retrieved, downgrade confidence by 0.10.

---

## PART 3.7 — Outlier and unit-error detection (mandatory pass)

Before producing the final output, run an explicit AUDIT pass on every successful value. Specifically check for:

1. **Order-of-magnitude errors:** Revenue $15,000B is wrong; should be $15B. Likely $ vs $B confusion.
2. **Year bleeding into value field:** If revenue=2024 or charitable_giving=2023, that's almost certainly the year being misread as the value.
3. **Unit confusion:** $M vs $B, hours vs days, % stored as 0.42 when the spreadsheet is showing 42.
4. **Scale mismatch with company size:** Con Edison revenue = $0.05B is wrong; it's a $15B company. Check that the value is in the right ballpark for the company's revenue / customer count.
5. **Suspicious uniformity:** A number that's exactly the same across multiple companies (suggests a copy-paste or unit-default error).
6. **Peer outliers:** Value ≥5× peer median for metrics that normally cluster tightly (philanthropy, customer satisfaction).

For each ✅ value, ask yourself: "Is this plausible for this company? Does the unit match the field? Is this the year column or the value column?" If anything is suspicious, downgrade to ⚠️ or 🚩 with explanation.

---

## PART 4 — Hard rules (non-negotiable)

1. **No fallback / mock / seed data.** Failed extraction returns null with a structured reason. Never substitute a placeholder.
2. **Every value carries a confidence score and a source URL.** No values without provenance.
3. **Validate every value against the plausibility range before accepting it.**
4. **Quote at most one short fact per source.** Always paraphrase otherwise.
5. **Distinguish "not applicable" from "we couldn't find it"** — the flag system enforces this.
6. **Never modify or correct a structured value with AI judgment.** AI's job is to ANNOTATE (audit, flag), not to overwrite. If a value looks wrong, FLAG it; don't replace it.

---

## PART 5 — Workflow when the user gives you input

When the user says something like "benchmark Con Edison, Duke, and PG&E on revenue, charitable giving, and carbon emissions":

1. **Confirm the inputs.** Echo back the company list and metric list to make sure you parsed them correctly. If a custom metric isn't in the standard list, fuzzy-match it (e.g. "program_grants_education" → "stem_education_giving") and tell the user which standard metric you're routing it to.

2. **For each (company × metric), walk the source priority list.** Tell the user which source you tried and what happened. Be transparent.

3. **Compile the wide table.** Companies as rows, metrics as columns, flag emoji + value in each cell.

4. **Compile the per-metric breakdown.** For EACH metric, write a section with:
   - Unit and what it measures
   - Expected range
   - Each company's value with flag, source, year, and confidence
   - A one-line methodology note

5. **Run a quality scan.** Re-check every ✅ value against expected range and peer plausibility. Downgrade flags as needed.

6. **Flag failures by cause.** For every null, give the user a tailored explanation, not a generic "no value found."

7. **Write 4–6 strategic insights** for the user's stated audience, with bolded headlines.

8. **Recommend 3 specific follow-up data pulls** that would close gaps in the benchmark.

When uncertain: "I don't have a verifiable source for that — would you like me to flag it for manual review?" Never guess.

---

## PART 6 — Tone

You are speaking to a Community Partnerships analyst. Be direct, specific, and useful. No filler. Every sentence should either inform a decision or warn about a risk. Plain English, not consultant-speak. When the data has a problem, name it clearly. When the data is solid, just say so and move on."""


# ── Sidebar — inputs ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Inputs")

    st.subheader("Companies")
    st.caption(
        "Pick from the registry, or type any utility name in the box below. "
        "Custom companies use the AI fallback (web search) if government APIs "
        "can't address them."
    )

    registry_options = [c.name for c in REGISTRY.values()]

    # ── Preset peer groups — quick way to populate the company list ──────
    # Built from the registry's metadata so we don't hardcode names twice.
    def _preset_companies(preset: str) -> list[str]:
        if preset == "None":
            return registry_options[:4]   # default = first 4
        if preset == "NY/Tristate utilities":
            return [c.name for c in REGISTRY.values()
                    if any(s in (c.state_puc_codes or ())
                           for s in ("NY-PSC", "NY-DPS", "NJ-BPU"))]
        if preset == "New England utilities":
            return [c.name for c in REGISTRY.values()
                    if any(s in (c.state_puc_codes or ())
                           for s in ("CT-PURA", "MA-DPU", "RI-PUC",
                                     "ME-PUC", "NH-PUC", "VT-PSB"))]
        if preset == "Mid-Atlantic / PJM":
            return [c.name for c in REGISTRY.values()
                    if any(s in (c.state_puc_codes or ())
                           for s in ("PA-PUC", "MD-PSC", "DC-PSC", "DE-PSC",
                                     "VA-SCC", "NJ-BPU", "OH-PUCO", "WV-PSC"))]
        if preset == "T&D distributors only":
            return [c.name for c in REGISTRY.values()
                    if not c.is_integrated_generator]
        if preset == "Integrated generators only":
            return [c.name for c in REGISTRY.values()
                    if c.is_integrated_generator]
        if preset == "Top 10 by size (mixed)":
            # Hand-picked national peer set commonly used in benchmarking
            wanted = {"Con Edison", "Duke Energy", "Pacific Gas and Electric",
                      "Eversource Energy", "Exelon Corporation",
                      "Public Service Enterprise Group", "Southern Company",
                      "Dominion Energy", "American Electric Power",
                      "NextEra Energy"}
            return [c.name for c in REGISTRY.values() if c.name in wanted]
        return registry_options[:4]

    preset = st.selectbox(
        "Preset peer group (optional)",
        options=["None", "NY/Tristate utilities", "New England utilities",
                 "Mid-Atlantic / PJM", "T&D distributors only",
                 "Integrated generators only", "Top 10 by size (mixed)"],
        index=0,
        help=("Pick a preset to populate the company list automatically. "
              "You can still add or remove individual companies after. "
              "Selecting a different preset replaces the current selection."),
    )
    # When the preset changes, store the new default in session state so the
    # multiselect picks it up.
    if "_last_preset" not in st.session_state:
        st.session_state["_last_preset"] = "None"
    if preset != st.session_state["_last_preset"]:
        st.session_state["_last_preset"] = preset
        st.session_state["_preset_default"] = _preset_companies(preset)
        st.rerun()

    default_for_multiselect = st.session_state.get(
        "_preset_default", registry_options[:4])
    # Filter the default to only valid registry names (defensive in case the
    # registry changed between session state writes)
    default_for_multiselect = [n for n in default_for_multiselect
                                if n in registry_options]

    selected_registry = st.multiselect(
        "From registry",
        options=registry_options,
        default=default_for_multiselect,
    )

    # ── Custom companies: input + Add button + chip list ────────────────────
    if "custom_companies" not in st.session_state:
        st.session_state["custom_companies"] = []

    st.markdown("**Custom companies**")
    if st.session_state["custom_companies"]:
        st.caption("Click ✕ to remove an entry:")
        # Render in rows of up to 3 chips
        chips = st.session_state["custom_companies"]
        for i, name in enumerate(chips):
            col_name, col_x = st.columns([5, 1])
            with col_name:
                st.markdown(f"• {name}")
            with col_x:
                if st.button("✕", key=f"rm_co_{i}", help=f"Remove {name}"):
                    st.session_state["custom_companies"].pop(i)
                    st.rerun()
    else:
        st.caption("None added yet. Type a name below and click **Add**.")

    new_co_col_input, new_co_col_btn = st.columns([4, 1])
    with new_co_col_input:
        new_co = st.text_input(
            "Add a company",
            key="new_company_input",
            placeholder="Exelon",
            label_visibility="collapsed",
        )
    with new_co_col_btn:
        if st.button("Add", key="add_company_btn", use_container_width=True):
            cleaned = (new_co or "").strip()
            if cleaned and cleaned not in st.session_state["custom_companies"]:
                st.session_state["custom_companies"].append(cleaned)
                st.rerun()

    custom_companies = st.session_state["custom_companies"]
    companies = selected_registry + custom_companies

    st.subheader("Metrics")
    metric_labels = {k: f"{m['label']} ({m['unit']})" for k, m in METRICS.items()}
    selected_metrics = st.multiselect(
        "From standard set",
        options=list(METRICS.keys()),
        default=["revenue", "charitable_giving", "foundation_assets",
                 "carbon_emissions"],
        format_func=lambda k: metric_labels[k],
    )

    # ── Custom metrics: input + Add button + chip list ──────────────────────
    if "custom_metrics" not in st.session_state:
        st.session_state["custom_metrics"] = []

    st.markdown("**Custom metrics**")
    if st.session_state["custom_metrics"]:
        st.caption("Click ✕ to remove:")
        for i, name in enumerate(st.session_state["custom_metrics"]):
            col_name, col_x = st.columns([5, 1])
            with col_name:
                st.markdown(f"• {name}")
            with col_x:
                if st.button("✕", key=f"rm_mt_{i}", help=f"Remove {name}"):
                    st.session_state["custom_metrics"].pop(i)
                    st.rerun()
    else:
        st.caption(
            "None added yet. Type a metric name (custom metrics route to "
            "the closest standard metric, or to the AI fallback)."
        )

    new_mt_col_input, new_mt_col_btn = st.columns([4, 1])
    with new_mt_col_input:
        new_mt = st.text_input(
            "Add a metric",
            key="new_metric_input",
            placeholder="program_grants_education",
            label_visibility="collapsed",
        )
    with new_mt_col_btn:
        if st.button("Add", key="add_metric_btn", use_container_width=True):
            cleaned = (new_mt or "").strip()
            if cleaned and cleaned not in st.session_state["custom_metrics"]:
                st.session_state["custom_metrics"].append(cleaned)
                st.rerun()

    custom_metrics = st.session_state["custom_metrics"]
    metrics = selected_metrics + custom_metrics

    st.divider()
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    has_gemini = bool(os.environ.get("GOOGLE_API_KEY", "").strip())
    has_any_ai = has_anthropic or has_gemini

    # Build a help string explaining what's configured
    ai_status_lines = []
    if has_anthropic:
        ai_status_lines.append("✅ Anthropic Claude (paid)")
    else:
        ai_status_lines.append("⚪ Anthropic Claude — set `ANTHROPIC_API_KEY` to enable")
    if has_gemini:
        ai_status_lines.append("✅ Google Gemini (free tier)")
    else:
        ai_status_lines.append("⚪ Google Gemini — set `GOOGLE_API_KEY` to enable (free)")

    use_ai = st.checkbox(
        "Use AI fallback + insights",
        value=has_any_ai,
        disabled=not has_any_ai,
        help=(
            "AI is used for: (1) finding values for custom companies / "
            "metrics that don't match the registry, (2) the audit and "
            "insights tabs.\n\n"
            "When BOTH keys are set, the app runs both AIs and shows their "
            "analyses in separate sub-tabs so you can compare."
        ),
    )
    st.caption("AI providers status:")
    for line in ai_status_lines:
        st.caption(line)

    # Smarter per-entry feedback: classify each custom entry.
    # - Custom company resolves via registry fuzzy match → no AI needed
    # - Custom metric resolves via fuzzy match to a standard metric → no AI needed
    # - Otherwise: AI fallback is required
    custom_companies_needing_ai = []
    custom_companies_resolved = []
    for name in custom_companies:
        if resolve_company(name):
            custom_companies_resolved.append(name)
        else:
            custom_companies_needing_ai.append(name)

    custom_metrics_needing_ai = []
    custom_metrics_resolved = []  # list of (typed_name, matched_standard_metric)
    for name in custom_metrics:
        matched = _fuzzy_match_metric(name)
        if matched:
            custom_metrics_resolved.append((name, matched))
        else:
            custom_metrics_needing_ai.append(name)

    # Show what resolved (good news first)
    if custom_companies_resolved or custom_metrics_resolved:
        msg_lines = []
        for n in custom_companies_resolved:
            msg_lines.append(f"• Company **{n}** → matched in registry, will use structured extractors")
        for typed, matched in custom_metrics_resolved:
            label = METRICS[matched]["label"]
            unit = METRICS[matched]["unit"]
            msg_lines.append(
                f"• Metric **{typed}** → routed to **{label}** ({unit}), "
                f"will use that metric's extractor"
            )
        st.success("These custom entries will work without AI:\n\n" + "\n\n".join(msg_lines))

    # Show what genuinely needs AI — but with inferred unit/description so
    # the user knows what the tool will look for
    if custom_companies_needing_ai or custom_metrics_needing_ai:
        msg_lines = []
        for n in custom_companies_needing_ai:
            msg_lines.append(f"• Company **{n}** — not in registry; needs AI fallback")
        for n in custom_metrics_needing_ai:
            inferred_unit, inferred_desc = infer_metric_unit(n)
            unit_str = f" — inferred unit: **{inferred_unit}**" if inferred_unit else ""
            msg_lines.append(
                f"• Metric **{n}**{unit_str}\n  _{inferred_desc}_"
            )
        if not use_ai:
            st.warning(
                "These entries need AI to find values, and AI is currently disabled. "
                "They'll return null:\n\n" + "\n\n".join(msg_lines)
            )
        else:
            st.info(
                "These entries will use AI fallback:\n\n" + "\n\n".join(msg_lines)
            )

    run_btn = st.button("Run benchmark", type="primary", use_container_width=True,
                        disabled=not (companies and metrics))


# ── Run ─────────────────────────────────────────────────────────────────────
if run_btn:
    st.session_state.pop("datapoints", None)
    progress = st.empty()
    progress.info(f"Running pipeline for {len(companies)} companies × {len(metrics)} metrics…")

    from pipeline.ai_layer import apply_cross_source_bonus, compute_standing

    with st.spinner("Extracting…"):
        datapoints = run_pipeline(companies, metrics, use_ai_fallback=use_ai)
    # Cross-source agreement boost (rewards values corroborated by 2+ sources)
    apply_cross_source_bonus(datapoints)
    st.session_state["datapoints"] = datapoints

    # Detect which AI providers are configured.  The two are independent —
    # neither, either, or both can be set.  When both, we run both in
    # parallel and show separate tabs for each.
    from pipeline.ai_layer import available_providers
    providers = available_providers() if use_ai else []

    audit_by_provider: dict[str, Optional[str]] = {}
    insights_by_provider: dict[str, Optional[str]] = {}
    flags = None

    if use_ai and providers:
        with st.spinner(f"AI verification ({', '.join(providers)})…"):
            for p in providers:
                audit_by_provider[p] = verify(datapoints, provider=p)
            # Quality-scan flags use Anthropic if available, else Gemini.
            # We don't run both for flags because the merged_flags function
            # combines AI output with rule-based logic and we want one source
            # of truth for the table-cell flag.
            flags = merged_flags(datapoints)
        with st.spinner(f"Generating insights ({', '.join(providers)})…"):
            for p in providers:
                insights_by_provider[p] = generate_insights(
                    datapoints, audit_by_provider.get(p), provider=p
                )
    else:
        from pipeline.ai_layer import rule_based_flags
        flags = rule_based_flags(datapoints)

    # "Where does Con Edison stand" indicator — computed deterministically,
    # always available regardless of AI status.
    standings = compute_standing(datapoints, focus_company="Con Edison")

    # Backwards-compat: keep "audit" / "insights" keys pointing at whichever
    # provider was used (Anthropic preferred), so other parts of the app
    # (Excel export, etc.) still work.
    st.session_state["audit"] = (
        audit_by_provider.get("anthropic") or audit_by_provider.get("gemini")
    )
    st.session_state["insights"] = (
        insights_by_provider.get("anthropic") or insights_by_provider.get("gemini")
    )
    st.session_state["audit_by_provider"] = audit_by_provider
    st.session_state["insights_by_provider"] = insights_by_provider
    st.session_state["providers_used"] = providers
    st.session_state["flags"] = flags
    st.session_state["standings"] = standings

    progress.success("Done.")


# ── Render results ──────────────────────────────────────────────────────────
if "datapoints" in st.session_state:
    dps = st.session_state["datapoints"]
    df_long = pd.DataFrame([dp.to_flat_row() for dp in dps])

    n_ok = int(df_long["ok"].sum()) if not df_long.empty else 0
    n_fail = len(df_long) - n_ok
    confidences = df_long.loc[df_long["ok"], "confidence_score"].dropna()
    avg_conf = float(confidences.mean()) if len(confidences) else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total pairs", len(df_long))
    c2.metric("Successful", n_ok, f"{(n_ok / max(1, len(df_long))) * 100:.0f}%")
    c3.metric("Failed (null)", n_fail)
    c4.metric("Avg confidence", f"{avg_conf:.2f}" if avg_conf else "—")

    (tab_table, tab_standing, tab_charts, tab_failures, tab_audit,
     tab_copilot_prompt, tab_recreate_prompt, tab_export) = st.tabs([
        "Benchmark", "🎯 Where Con Ed Stands", "Charts", "Failures",
        "AI Audit & Insights", "📋 Copilot prompt",
        "🛠️ Recreation prompt", "Export",
    ])

    # ── Where Con Ed Stands tab ────────────────────────────────────────────
    with tab_standing:
        standings = st.session_state.get("standings") or {}
        st.subheader("Con Edison's standing across each metric")
        st.caption(
            "For every metric in this run, this tab shows Con Edison's "
            "ranking, distance from the peer median, and the leader / "
            "laggard. Use this to brief executives without scanning the "
            "full benchmark table."
        )

        if not standings:
            st.info("Run a benchmark to populate this tab.")
        else:
            # Top-level summary metrics
            top_n = sum(1 for s in standings.values() if s.get("verdict") == "Top performer")
            above_n = sum(1 for s in standings.values()
                          if s.get("verdict") in ("Top performer", "Above median"))
            below_n = sum(1 for s in standings.values()
                          if s.get("verdict") in ("Below median", "Lowest"))
            no_data_n = sum(1 for s in standings.values()
                            if s.get("verdict") in ("No data",))

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Top performer (#1)", f"{top_n} metrics")
            c2.metric("Above median", f"{above_n} metrics")
            c3.metric("Below median", f"{below_n} metrics")
            c4.metric("No data", f"{no_data_n} metrics")
            st.divider()

            # Per-metric breakdown
            for metric_key, info in standings.items():
                meta = METRICS.get(metric_key, {})
                label = meta.get("label", metric_key)
                unit = meta.get("unit", "")
                verdict = info.get("verdict", "—")

                # Color-code the verdict
                color_map = {
                    "Top performer": "✅",
                    "Above median": "🟢",
                    "Median": "🟡",
                    "Below median": "🟠",
                    "Lowest": "🔴",
                    "No data": "⚪",
                    "Only data point": "ℹ️",
                }
                emoji = color_map.get(verdict, "•")

                with st.container(border=True):
                    head_col, badge_col = st.columns([3, 1])
                    with head_col:
                        st.markdown(f"### {label}")
                        st.caption(unit if unit else "(custom unit)")
                    with badge_col:
                        st.markdown(
                            f"<div style='text-align:right; font-size:1.6em;'>"
                            f"{emoji} <b>{verdict}</b></div>",
                            unsafe_allow_html=True,
                        )

                    if info.get("focus_value") is None:
                        st.warning(info.get("narrative", "No data."))
                        continue

                    cc1, cc2, cc3, cc4 = st.columns(4)
                    cc1.metric("Con Edison", f"{info['focus_value']:g}")
                    cc2.metric(
                        "Rank",
                        f"#{info['rank']} of {info['of_total']}" if info.get('rank') else "—",
                    )
                    if info.get("peer_median") is not None:
                        cc3.metric("Peer median", f"{info['peer_median']:g}")
                    if info.get("vs_median_pct") is not None:
                        delta = info["vs_median_pct"]
                        cc4.metric("vs Peer median",
                                   f"{abs(delta):.0f}% {'above' if delta >= 0 else 'below'}")

                    if info.get("best") and info.get("worst"):
                        b_co, b_v = info["best"]
                        w_co, w_v = info["worst"]
                        st.caption(
                            f"**Leader:** {b_co} at {b_v:g} {unit} · "
                            f"**Laggard:** {w_co} at {w_v:g} {unit}"
                        )
                    st.markdown(f"_{info.get('narrative', '')}_")

    with tab_table:
        flags_map = st.session_state.get("flags") or {}

        # ── Metric glossary (collapsed by default to save space) ──────────
        metrics_in_run = []
        for dp in dps:
            if dp.metric not in metrics_in_run:
                metrics_in_run.append(dp.metric)

        with st.expander("📖 Metric glossary — what each column means and its unit",
                          expanded=True):
            st.caption(
                "Quick reference for every metric in this run. Standard "
                "metrics show their official definition; custom metrics "
                "show what unit was inferred and how the tool tried to "
                "resolve them."
            )
            for m in metrics_in_run:
                meta = METRICS.get(m)
                if meta:
                    label = meta["label"]
                    unit = meta["unit"]
                    description = meta["description"]
                    expected = meta.get("expected_range", "n/a")
                    direction = ("lower is better"
                                 if meta.get("lower_is_better") else "higher is better")
                    st.markdown(
                        f"**{label}** _({unit})_ · _{direction}_  \n"
                        f"{description}  \n"
                        f"_Expected range: {expected}_"
                    )
                else:
                    inferred_unit, inferred_desc = infer_metric_unit(m)
                    matched = _fuzzy_match_metric(m)
                    if matched:
                        target_meta = METRICS[matched]
                        st.markdown(
                            f"**{m}** _({target_meta['unit']})_ · custom metric  \n"
                            f"Routed to standard metric **{target_meta['label']}** — "
                            f"{target_meta['description']}"
                        )
                    else:
                        unit_disp = f"_({inferred_unit})_ · " if inferred_unit else ""
                        st.markdown(
                            f"**{m}** {unit_disp}custom metric, no fuzzy match  \n"
                            f"_{inferred_desc}_"
                        )

        st.subheader("Wide format (Companies × Metrics)")
        st.caption(
            "Flags: ✅ verified · ⚠️ caveat (partial coverage / known limitation) · "
            "🚩 likely problem (data exists but couldn't be retrieved, or low confidence) · "
            "ℹ️ not applicable (data legitimately doesn't exist for this combo)"
        )
        if df_long.empty:
            st.info("No data.")
        else:
            metric_cols_present = [m for m in METRICS if m in df_long["metric"].unique()]
            for m in df_long["metric"].unique():
                if m not in METRICS and m not in metric_cols_present:
                    metric_cols_present.append(m)

            wide = pd.DataFrame({"Company": sorted(df_long["company"].unique())})
            for m in metric_cols_present:
                meta = METRICS.get(m, {})
                label = meta.get("label", m)
                unit = meta.get("unit", "")
                col_name = f"{label} ({unit})" if unit else label

                def _cell(c, m=m):
                    sub = df_long[(df_long["company"] == c) & (df_long["metric"] == m)]
                    if sub.empty:
                        return ""
                    row = sub.iloc[0]
                    flag_info = flags_map.get((c, m), {})
                    flag = flag_info.get("flag", "")
                    if not row["ok"]:
                        return f"{flag} —"
                    val = row["value"]
                    return f"{flag} {val:g}" if isinstance(val, (int, float)) else f"{flag} {val}"

                wide[col_name] = wide["Company"].map(_cell)
            st.dataframe(wide, use_container_width=True, hide_index=True)

        st.subheader("Long format with flags, sources & confidence")
        long_with_flags = df_long.copy()
        long_with_flags["flag"] = long_with_flags.apply(
            lambda r: flags_map.get((r["company"], r["metric"]), {}).get("flag", ""),
            axis=1,
        )
        long_with_flags["flag_reason"] = long_with_flags.apply(
            lambda r: flags_map.get((r["company"], r["metric"]), {}).get("reason", ""),
            axis=1,
        )
        st.dataframe(
            long_with_flags[
                ["flag", "company", "metric_label", "value", "unit", "year",
                 "confidence_score", "flag_reason", "source_name", "source_url", "ok"]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "source_url": st.column_config.LinkColumn("Source URL", width="medium"),
                "confidence_score": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=1, format="%.2f"
                ),
            },
        )

    # ── Charts tab ──────────────────────────────────────────────────────────
    with tab_charts:
        st.caption("Bars colored by confidence (darker = higher). Hover for source details.")

        # View mode toggle
        view_mode = st.radio(
            "Layout",
            options=["Grouped by metric", "Side-by-side comparison", "Single metric (large)"],
            horizontal=True,
        )

        chartable_metrics = [
            m for m in metric_cols_present
            if not df_long[(df_long["metric"] == m) & (df_long["ok"])].empty
        ]

        if not chartable_metrics:
            st.info("No successful values to chart.")
        elif view_mode == "Grouped by metric":
            # Two charts per row, more compact
            for i in range(0, len(chartable_metrics), 2):
                cols = st.columns(2)
                for j, m in enumerate(chartable_metrics[i:i+2]):
                    with cols[j]:
                        _draw_metric_chart(df_long, m, flags_map, height=320)

        elif view_mode == "Side-by-side comparison":
            # Heatmap-style: companies × metrics with value labels and flag emojis
            heat_rows = []
            for c in sorted(df_long["company"].unique()):
                for m in chartable_metrics:
                    sub = df_long[(df_long["company"] == c) & (df_long["metric"] == m)
                                  & (df_long["ok"])]
                    if sub.empty:
                        continue
                    val = sub.iloc[0]["value"]
                    conf = sub.iloc[0]["confidence_score"] or 0
                    flag = flags_map.get((c, m), {}).get("flag", "")
                    meta = METRICS.get(m, {})
                    heat_rows.append({
                        "Company": c,
                        "Metric": meta.get("label", m),
                        "Value": val,
                        "Confidence": conf,
                        "Display": f"{flag} {val:g}",
                    })
            if heat_rows:
                df_heat = pd.DataFrame(heat_rows)
                # Pivot to a matrix for the heatmap
                pivot_val = df_heat.pivot(index="Company", columns="Metric", values="Confidence")
                pivot_label = df_heat.pivot(index="Company", columns="Metric", values="Display")
                fig = px.imshow(
                    pivot_val,
                    text_auto=False,
                    color_continuous_scale="Blues",
                    aspect="auto",
                    labels=dict(color="Confidence"),
                )
                # Overlay text from pivot_label
                fig.update_traces(text=pivot_label.values, texttemplate="%{text}")
                fig.update_layout(
                    height=max(300, 80 * len(pivot_val)),
                    xaxis={"side": "top"},
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Cell color shows confidence (darker = higher). "
                    "Numbers are values. Different metrics have different units — "
                    "compare within a column, not across."
                )

        else:  # Single metric (large)
            metric_choice = st.selectbox(
                "Metric to display",
                options=chartable_metrics,
                format_func=lambda m: METRICS.get(m, {}).get("label", m),
            )
            _draw_metric_chart(df_long, metric_choice, flags_map, height=520)

    # ── Failures tab ────────────────────────────────────────────────────────
    with tab_failures:
        fails = df_long[~df_long["ok"]]
        if fails.empty:
            st.success("No failures — every (company, metric) pair returned a value.")
        else:
            st.warning(
                f"{len(fails)} pairs returned null. **These are honest reports** — "
                "not all are bugs. ℹ️ flags are legitimate "
                "*not-applicable* cases (e.g. carbon emissions for a pure "
                "distributor); 🚩 flags are cases where data likely exists but "
                "we couldn't retrieve it."
            )
            fail_with_flags = fails.copy()
            fail_with_flags["flag"] = fail_with_flags.apply(
                lambda r: flags_map.get((r["company"], r["metric"]), {}).get("flag", ""),
                axis=1,
            )
            fail_with_flags["what_this_means"] = fail_with_flags.apply(
                lambda r: flags_map.get((r["company"], r["metric"]), {}).get("reason", ""),
                axis=1,
            )
            st.dataframe(
                fail_with_flags[["flag", "company", "metric_label",
                                 "what_this_means", "error_reason", "num_attempts"]],
                use_container_width=True, hide_index=True,
            )

    # ── AI Audit & Insights tab ─────────────────────────────────────────────
    with tab_audit:
        audit_by_provider = st.session_state.get("audit_by_provider") or {}
        insights_by_provider = st.session_state.get("insights_by_provider") or {}
        providers_used = st.session_state.get("providers_used") or []

        if not providers_used:
            st.info(
                "AI verification + insights are off, or no AI API key is set "
                "/ working. To populate this tab:\n\n"
                "- **Anthropic Claude** (paid, higher quality): set "
                "`ANTHROPIC_API_KEY` in Streamlit secrets.\n"
                "- **Google Gemini** (free tier, 1500 req/day): set "
                "`GOOGLE_API_KEY` in Streamlit secrets.\n\n"
                "When BOTH keys are set, the tool runs both AIs and shows "
                "their analyses side-by-side so you can compare."
            )
        else:
            # Build a sub-tab per provider that produced output
            provider_labels = {
                "anthropic": "🤖 Claude (Anthropic)",
                "gemini": "✨ Gemini (Google)",
            }
            sub_tabs = st.tabs([provider_labels.get(p, p) for p in providers_used])
            for sub_tab, p in zip(sub_tabs, providers_used):
                with sub_tab:
                    if p == "anthropic":
                        st.caption(
                            "Claude Sonnet 4.6 · paid · most rigorous on "
                            "structured-data audit and unit-error detection."
                        )
                    elif p == "gemini":
                        st.caption(
                            "Gemini 2.0 Flash · free tier · faster response, "
                            "broadly capable. Use to cross-check Claude's "
                            "audit when both keys are configured."
                        )
                    audit_for = audit_by_provider.get(p)
                    insights_for = insights_by_provider.get(p)
                    if audit_for:
                        st.subheader("Data quality audit")
                        st.markdown(audit_for)
                    if insights_for:
                        st.subheader("Strategic insights")
                        st.markdown(insights_for)
                    if not (audit_for or insights_for):
                        st.warning(
                            f"{provider_labels.get(p, p)} returned no output. "
                            "Check API key validity and rate limits."
                        )

            # When both ran, offer a side-by-side comparison view
            if len(providers_used) > 1:
                st.divider()
                st.subheader("Cross-AI comparison")
                st.caption(
                    "If the two AIs disagree on flagging or insight emphasis, "
                    "that disagreement is itself signal — investigate the "
                    "underlying value."
                )

    # ── Copilot / Claude analysis prompt ────────────────────────────────────
    with tab_copilot_prompt:
        st.subheader("📋 Copy-paste prompt for analysis in Copilot")
        st.caption(
            "Two prompt versions are available below. The **Analysis** prompt "
            "embeds the data this tool just pulled; paste into Copilot for "
            "deeper narrative. The **Standalone Run** prompt mirrors the tool's "
            "logic — paste into Copilot with NO embedded data and Copilot will "
            "run the same analysis on any company set you ask about. This is "
            "what to share with Con Edison staff who don't have access to this app."
        )

        prompt_mode = st.radio(
            "Prompt version",
            options=["Analysis (this run's data embedded)",
                     "Standalone Run (no data — analyst types companies/metrics into Copilot)"],
            horizontal=False,
        )

        if prompt_mode.startswith("Analysis"):
            prompt_text = _build_analysis_prompt(dps, flags_map)
            filename = "copilot_analysis_prompt.txt"
        else:
            prompt_text = _build_standalone_copilot_prompt()
            filename = "copilot_standalone_prompt.txt"

        st.text_area(
            "Prompt (click into the box, ⌘/Ctrl+A to select all, ⌘/Ctrl+C to copy)",
            value=prompt_text,
            height=400,
        )
        st.download_button(
            "Download as .txt",
            data=prompt_text.encode("utf-8"),
            file_name=filename,
            mime="text/plain",
        )

    # ── Tool-recreation prompt ──────────────────────────────────────────────
    with tab_recreate_prompt:
        st.subheader("🛠️ Detailed prompt to recreate this tool with another AI")
        st.caption(
            "If this Streamlit app ever breaks or is unavailable, paste this "
            "prompt into Claude, ChatGPT, or another capable assistant to "
            "get an interactive walkthrough that mimics what the tool does. "
            "It includes the architecture, data sources, methodology, and "
            "rules — without the code."
        )
        recreate_text = _build_recreation_prompt()
        st.text_area(
            "Prompt",
            value=recreate_text,
            height=500,
        )
        st.download_button(
            "Download as .txt",
            data=recreate_text.encode("utf-8"),
            file_name="utility_benchmark_recreation_prompt.txt",
            mime="text/plain",
        )

    # ── Export tab ──────────────────────────────────────────────────────────
    with tab_export:
        st.subheader("Download")
        st.caption(
            "The Excel workbook contains 5 sheets: Summary, Benchmark (wide "
            "table), Detailed (with flags & sources), Failures & Notes, and "
            "AI Audit. The full Attempts log is included as a hidden sheet "
            "(unhide in Excel for debugging)."
        )

        buf = io.BytesIO()
        tmp_path = Path(".cache/_export.xlsx")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        to_excel(
            dps, tmp_path,
            audit_summary=st.session_state.get("audit"),
            insights=st.session_state.get("insights"),
            flags=flags_map,
        )
        buf.write(tmp_path.read_bytes())

        st.download_button(
            "📊 Download Excel workbook (.xlsx)",
            data=buf.getvalue(),
            file_name="utility_benchmark.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.download_button(
            "📄 Download long-format CSV",
            data=df_long.to_csv(index=False).encode("utf-8"),
            file_name="utility_benchmark.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "📦 Download JSON (full DataPoints + attempts log)",
            data=json.dumps(
                [dp.to_dict() for dp in dps],
                indent=2,
                default=str,
            ).encode("utf-8"),
            file_name="utility_benchmark.json",
            mime="application/json",
            use_container_width=True,
        )

else:
    st.info("Configure inputs in the sidebar and click **Run benchmark**.")
