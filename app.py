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
    from pipeline.orchestrator import run_pipeline
    from pipeline.export import to_excel
    from pipeline.ai_layer import verify, generate_insights, merged_flags
    from pipeline.models import METRICS
    from extractors.company_registry import REGISTRY
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
    """Compose a self-contained prompt that another AI can use to do deeper
    analysis on the just-pulled benchmark data.  Embeds the data as a
    structured table and asks for a specific kind of analysis."""

    companies = sorted({dp.company for dp in datapoints})
    metric_keys_seen = []
    for dp in datapoints:
        if dp.metric not in metric_keys_seen:
            metric_keys_seen.append(dp.metric)

    # Build a markdown table of values
    header = "| Company | " + " | ".join(
        f"{METRICS.get(m, {}).get('label', m)} ({METRICS.get(m, {}).get('unit', '')})".strip(" ()")
        for m in metric_keys_seen
    ) + " | Notes |"
    sep = "|" + "|".join("---" for _ in range(len(metric_keys_seen) + 2)) + "|"
    body_lines = [header, sep]
    for c in companies:
        cells = [c]
        notes_for_row = []
        for m in metric_keys_seen:
            dp = next((d for d in datapoints if d.company == c and d.metric == m), None)
            if dp and dp.ok:
                flag = flags_map.get((c, m), {}).get("flag", "")
                cells.append(f"{flag} {dp.value:g}".strip())
            else:
                flag = flags_map.get((c, m), {}).get("flag", "🚩")
                reason = (flags_map.get((c, m), {}).get("reason") or "")[:80]
                cells.append(f"{flag} —")
                if reason:
                    notes_for_row.append(f"{METRICS.get(m, {}).get('label', m)}: {reason}")
        cells.append("; ".join(notes_for_row[:2]) if notes_for_row else "")
        body_lines.append("| " + " | ".join(cells) + " |")

    table = "\n".join(body_lines)

    prompt = f"""I have just compiled the following utility-industry benchmark for the Con Edison Community Partnerships team. The data was pulled from regulator-of-record sources where available (SEC EDGAR XBRL for revenue, EPA eGRID for carbon emissions, IRS 990-PF via ProPublica for foundation philanthropy, J.D. Power for customer satisfaction, and CSR/sustainability reports for narrative-disclosed metrics).

Each value carries a flag:
- ✅ verified from a credible source
- ⚠️ caveat applies (e.g. partial coverage, or only the foundation slice of total giving)
- 🚩 data exists somewhere but couldn't be retrieved by the automated tool
- ℹ️ not applicable for this combination (e.g. carbon emissions for a pure transmission/distribution utility that owns no generation)
- — indicates a null

Important structural facts:
1. Pure T&D distributors (Con Edison, National Grid USA, Eversource, PSEG Long Island) do NOT own generation plants tracked in EPA eGRID, so their Scope 1 emissions reported there will be null. Their actual Scope 1 (vehicle fleet, fugitive gas leaks) is small and disclosed only in CSR reports.
2. Integrated generators (Duke Energy, Southern Company, PG&E) emit ~10× more Scope 1 CO₂ than distributors — that is structural, not an outlier.
3. "Charitable Giving" pulled from ProPublica is the FOUNDATION-PAID slice only (IRS Form 990-PF, line for grants paid). It does NOT include direct corporate contributions, energy assistance programs (LIHEAP supplements, hardship funds), or in-kind giving. To get the full picture, look at "Total Charitable Giving" together with "Energy Assistance Programs" and "Community Investment".

## Benchmark data

{table}

## What I want from you

1. **Read the data critically.** For each value, quickly decide if it looks plausible given company size and structure. Call out anything that looks like a unit error, scale mismatch, or peer outlier — especially among the ✅ rows (those are the ones I trust most, so an error there is the most dangerous).

2. **Spot the data gaps.** For each 🚩 or — value, suggest where the analyst should look manually (e.g. specific CSR report sections, specific 10-K line items, state PUC dockets).

3. **Write 4–6 strategic insights for the Community Partnerships VP**, focused on:
   - How Con Edison's philanthropy stacks up vs peers, *normalized for company size* (revenue or customer count)
   - Where Con Edison appears to have a competitive advantage or gap
   - Any storyline an executive should know before a board meeting

4. **Recommend 3 follow-up data pulls** that would sharpen this benchmark — specific metrics from specific sources.

Keep your response in plain English. No fluff. Do not invent numbers; if you don't have data, say so."""

    return prompt


def _build_recreation_prompt() -> str:
    """A detailed meta-prompt that another AI can use to mimic this tool's
    behavior interactively."""

    return """# Utility Benchmark Pipeline — interactive recreation

You are an analyst's research assistant. The user is going to ask you to benchmark US electric and gas utilities on operational, ESG, and philanthropy metrics. Behave like a hand-built, per-source data pipeline — not a generic LLM that hallucinates numbers. The exact rules below are non-negotiable.

## Your data-source priority order

For each (company, metric) pair, work down this list and STOP at the first source that returns a value:

1. **SEC EDGAR XBRL company-facts API** — for Revenue and any other GAAP-tagged financials. URL pattern: `https://data.sec.gov/api/xbrl/companyfacts/CIK{10digit}.json`. Pull from the most recent 10-K, fiscal-year (`fp=FY`) entry. Tag is `us-gaap:Revenues` or its equivalents. Confidence: 0.95.

2. **ProPublica Nonprofit Explorer** — for foundation philanthropy. URL pattern: `https://projects.propublica.org/nonprofits/api/v2/organizations/{ein_numeric}.json`. Search by foundation name first (`/search.json?q=...`) since hard-coded EINs are fragile. The relevant 990-PF fields are `cttgrntpd` / `grntspaid` (grants paid out) and `totassetsend` (total assets at year end). Confidence: 0.85. If you only return foundation-paid giving, FLAG that it's a partial picture — it excludes direct corporate contributions and energy assistance.

3. **EPA eGRID** — for Scope 1 carbon emissions. Latest release URL: `https://www.epa.gov/system/files/documents/2024-01/egrid2022_data.xlsx`. Read the PLNT22 sheet (header row 2). Sum `PLCO2AN` (annual plant CO₂ in short tons) by `OPRNAME` (operator name). Convert short tons to million metric tons: × 0.907185 / 1e6. Confidence: 0.95. Pure T&D distributors will return zero plants and that is correct — their Scope 1 from generation is genuinely zero; flag as "not applicable" not "missing data".

4. **EIA Form 861** — for SAIDI and renewable energy %. Annual XLSX from `https://www.eia.gov/electricity/data/eia861/`. Filter by EIA Operator ID. Confidence: 0.85.

5. **J.D. Power** — for residential customer satisfaction. Press-release page: `https://www.jdpower.com/business/press-releases/2024-us-electric-utility-residential-customer-satisfaction-study`. ONLY regional category winners are publicly named — for everyone else this is a null and should not be considered a bug. Confidence: 0.50.

6. **Corporate sustainability / CSR report PDF** — for narrative-disclosed metrics: energy assistance programs, volunteer hours, employee match, community investment, STEM education giving. Locate the most recent PDF, extract text, search for distinctive phrases ("$X million in energy assistance", "Y volunteer hours", etc.). Confidence: 0.60.

7. **AI / web-search fallback** (this is YOU, when other sources fail) — search the web for the metric, prefer regulator filings and CSR reports, fall back to news with reduced confidence. Confidence: 0.30–0.85 depending on source. **Refuse to guess.** If no credible source exists, return null with reason.

## Hard rules

- **No fallback / mock / seed data, ever.** A failed extraction returns null with a structured reason — never a substituted value.
- **Every value carries a confidence score and a source URL.** No values without provenance.
- **Validate every value against a plausibility range** before accepting it. Revenue $5,000B is wrong (probably USD vs billions confusion). Renewable % above 100 is wrong. Reject and report null with reason.
- **Quote at most one short fact per source.** Always paraphrase otherwise.
- **Distinguish "not applicable" from "we couldn't find it"** — see flag philosophy below.

## Plausible ranges per metric (reject values outside these)

| Metric | Unit | Min | Max | Expected |
|---|---|---|---|---|
| Revenue | $B | 1 | 80 | $10B–$35B |
| Renewable Energy % | % | 0 | 100 | 15%–50% |
| SAIDI | min/yr | 20 | 500 | 50–200 |
| Customer satisfaction | /100 | 30 | 80 | 45–55 |
| Carbon Emissions Scope 1 | M MT CO2 | 0.5 | 200 | 2–10 distributor / 50–100 generator |
| Total Charitable Giving | $M | 0.01 | 500 | $5M–$100M large utility |
| Foundation Grants Paid | $M | 0.01 | 200 | $0.5M–$20M |
| Foundation Assets | $M | 0.5 | 500 | $5M–$100M |
| Energy Assistance | $M | 0.1 | 100 | $1M–$30M |
| STEM/Education Giving | $M | 0.05 | 50 | $0.5M–$10M |
| Community Investment | $M | 0.1 | 500 | $10M–$100M |
| Volunteer Hours | hrs/yr | 100 | 500,000 | 5,000–100,000 |
| Employee Match | $M | 0.05 | 50 | $0.5M–$10M |

## Flag system on every value

- ✅ Verified from a credible source, no concerns
- ⚠️ Caveat applies (partial coverage, foundation slice only, mid-tier confidence)
- 🚩 Data likely exists but couldn't be retrieved, OR very low confidence (< 0.30)
- ℹ️ Not applicable for this combo (genuine structural reason — pure distributor for carbon, no SEC filings for foreign-parent companies, etc.)

## Workflow when the user asks

1. Ask the user for the list of companies and metrics they want to benchmark.
2. For each (company, metric) pair, walk the source priority list, ATTEMPT EACH SOURCE, and report what you find — including failures and the reasons.
3. Output a wide table (companies × metrics) with flag emoji + value in each cell.
4. List any null cells in a "Failures & Notes" section explaining why each one came back null.
5. Run a quality-scan pass: for each ✅ value, ask yourself whether it's plausible vs the company's size and the peer median. Downgrade to ⚠️ or 🚩 if not.
6. Write 3–5 strategic insights for the user's stated audience.

When uncertain, say "I don't have a verifiable source for that — would you like me to flag it for manual review?" Never guess."""


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
    selected_registry = st.multiselect(
        "From registry",
        options=registry_options,
        default=registry_options[:4],
    )
    custom_companies_raw = st.text_area(
        "Custom companies (one per line)",
        value="",
        height=70,
        placeholder="Exelon\nDominion Energy",
        help="Any company name. The pipeline will try AI fallback (web search) "
             "if no government-API ID can be resolved.",
    )
    custom_companies = [
        line.strip() for line in custom_companies_raw.splitlines() if line.strip()
    ]
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
    custom_metrics_raw = st.text_area(
        "Custom metrics (one per line)",
        value="",
        height=70,
        placeholder="women_in_leadership_pct\nprogram_grants_education",
        help="Any metric. Custom metrics route directly to the AI fallback "
             "(web search) since there's no purpose-built extractor.",
    )
    custom_metrics = [
        line.strip() for line in custom_metrics_raw.splitlines() if line.strip()
    ]
    metrics = selected_metrics + custom_metrics

    st.divider()
    has_known_key = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    use_ai = st.checkbox(
        "Use AI fallback + insights",
        value=has_known_key,
        disabled=not has_known_key,
        help=("Requires ANTHROPIC_API_KEY in Streamlit secrets. "
              "When enabled: custom companies / metrics become possible, and "
              "structured-extractor failures get one more shot via Claude+web_search.")
        if has_known_key else
        "Set ANTHROPIC_API_KEY in Streamlit secrets to enable this.",
    )
    if not has_known_key:
        st.caption("ℹ️ AI features are disabled — `ANTHROPIC_API_KEY` is not set.")
    if custom_companies or custom_metrics:
        if not use_ai:
            st.warning(
                "Custom companies/metrics need the AI fallback to find values. "
                "Without it, they'll return null."
            )

    run_btn = st.button("Run benchmark", type="primary", use_container_width=True,
                        disabled=not (companies and metrics))


# ── Run ─────────────────────────────────────────────────────────────────────
if run_btn:
    st.session_state.pop("datapoints", None)
    progress = st.empty()
    progress.info(f"Running pipeline for {len(companies)} companies × {len(metrics)} metrics…")

    with st.spinner("Extracting…"):
        datapoints = run_pipeline(companies, metrics, use_ai_fallback=use_ai)
    st.session_state["datapoints"] = datapoints

    audit_text = None
    insights_text = None
    flags = None
    if use_ai:
        with st.spinner("AI verification + quality scan…"):
            audit_text = verify(datapoints)
            flags = merged_flags(datapoints)
        with st.spinner("Generating insights…"):
            insights_text = generate_insights(datapoints, audit_text)
    else:
        # Without AI we still apply the deterministic rule-based flags
        # (low-confidence values and obvious peer-group outliers).
        from pipeline.ai_layer import rule_based_flags
        flags = rule_based_flags(datapoints)

    st.session_state["audit"] = audit_text
    st.session_state["insights"] = insights_text
    st.session_state["flags"] = flags

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

    (tab_table, tab_charts, tab_failures, tab_audit,
     tab_copilot_prompt, tab_recreate_prompt, tab_export) = st.tabs([
        "Benchmark", "Charts", "Failures", "AI Audit & Insights",
        "📋 Copilot prompt", "🛠️ Recreation prompt", "Export",
    ])

    with tab_table:
        flags_map = st.session_state.get("flags") or {}

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
        a = st.session_state.get("audit")
        i = st.session_state.get("insights")
        if not (a or i):
            st.info(
                "AI verification + insights are off, or `ANTHROPIC_API_KEY` is "
                "not set / out of credit. Run with the AI checkbox enabled "
                "and a valid API key to populate this tab."
            )
        if a:
            st.subheader("Data quality audit")
            st.markdown(a)
        if i:
            st.subheader("Strategic insights")
            st.markdown(i)

    # ── Copilot / Claude analysis prompt ────────────────────────────────────
    with tab_copilot_prompt:
        st.subheader("📋 Copy-paste prompt for analysis in Copilot or Claude")
        st.caption(
            "This prompt embeds the data we just pulled. Paste into Microsoft "
            "Copilot, ChatGPT, Claude, or any other AI assistant to get a deeper "
            "narrative analysis than what fits in the AI Audit tab."
        )
        prompt_text = _build_analysis_prompt(dps, flags_map)
        st.text_area(
            "Prompt (click into the box, ⌘/Ctrl+A to select all, ⌘/Ctrl+C to copy)",
            value=prompt_text,
            height=400,
        )
        st.download_button(
            "Download as .txt",
            data=prompt_text.encode("utf-8"),
            file_name="copilot_analysis_prompt.txt",
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
