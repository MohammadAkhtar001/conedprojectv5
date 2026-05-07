"""
Optional AI-driven verification + insights layer.

Supports two providers:
  - Anthropic Claude (paid, with web_search support for AI fallback)
  - Google Gemini (free tier 1500 req/day, no web_search)

When BOTH keys are set, verify/insights/quality_scan can run twice — once
per provider — so the user gets two parallel views and can compare.

If neither key is set, the AI layer is fully disabled and the pipeline
still produces complete data and exports.
"""

from __future__ import annotations
import json
import logging
import os
from typing import Optional, Literal

from .models import DataPoint, METRICS

log = logging.getLogger("ai_layer")

Provider = Literal["anthropic", "gemini"]


# ── Provider clients ───────────────────────────────────────────────────────


def _anthropic_client():
    """Return an Anthropic client, or None if key missing or SDK unavailable."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key or key.startswith("sk-ant-...") or len(key) < 20:
        return None
    try:
        from anthropic import Anthropic
        return Anthropic(api_key=key)
    except ImportError:
        log.warning("anthropic package not installed; Anthropic backend disabled")
        return None


def _gemini_client():
    """Return a Gemini client, or None if key missing or SDK unavailable."""
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key or len(key) < 20:
        return None
    try:
        from google import genai
        return genai.Client(api_key=key)
    except ImportError:
        log.warning("google-genai not installed; Gemini backend disabled")
        return None


# Backwards compat: existing call sites use _client() — keep it pointing to
# Anthropic so nothing breaks if I missed a spot.
def _client():
    return _anthropic_client()


def available_providers() -> list[Provider]:
    """Return list of providers with a working configured key."""
    out: list[Provider] = []
    if _anthropic_client():
        out.append("anthropic")
    if _gemini_client():
        out.append("gemini")
    return out


def _call_llm(provider: Provider, prompt: str, *, max_tokens: int = 2000) -> str:
    """Make a single text-completion call to the given provider.  Returns
    the response text, or raises on failure (caller handles)."""
    if provider == "anthropic":
        client = _anthropic_client()
        if client is None:
            raise RuntimeError("Anthropic backend not configured")
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            b.text for b in msg.content if getattr(b, "type", None) == "text"
        ).strip()

    elif provider == "gemini":
        client = _gemini_client()
        if client is None:
            raise RuntimeError("Gemini backend not configured")
        # Use the most capable free-tier model.  gemini-2.0-flash is the
        # current default; gemini-2.5-flash is also free with similar limits.
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return (resp.text or "").strip()

    raise ValueError(f"Unknown provider: {provider}")


def verify(
    datapoints: list[DataPoint],
    *,
    provider: Provider = "anthropic",
) -> Optional[str]:
    """Returns an audit summary string from the given provider, or None
    if that provider isn't configured."""
    if provider == "anthropic" and not _anthropic_client():
        return None
    if provider == "gemini" and not _gemini_client():
        return None

    rows = []
    for dp in datapoints:
        if not dp.ok:
            continue
        meta = METRICS.get(dp.metric, {})
        rows.append(
            f"{dp.company} | {meta.get('label', dp.metric)} = {dp.value} {dp.unit} "
            f"(source: {dp.source_name}, confidence {dp.confidence_score})"
        )
    if not rows:
        return None

    range_block = "\n".join(
        f"- {meta['label']} ({key}): expected {meta['expected_range']}, "
        f"plausible {meta['plausible_min']}–{meta['plausible_max']} {meta['unit']}"
        for key, meta in METRICS.items()
        if any(dp.metric == key and dp.ok for dp in datapoints)
    )

    prompt = f"""You are auditing a freshly compiled benchmark of US electric/gas utilities. Catch values that look wrong: wrong order of magnitude, wrong unit, peer-group outliers, scale mismatches.

Expected ranges:
{range_block}

Compiled values:
{chr(10).join(rows)}

Structural context that is NOT an error:
- Integrated generators (Duke, Southern, Dominion) legitimately emit ~10× more Scope 1 CO₂ than pure distributors (Con Edison, Eversource, National Grid).
- Charitable Giving and Foundation Assets correlate with company size; small absolute giving from a small utility is not an outlier.
- Charitable Giving from ProPublica is the FOUNDATION 990-PF slice only — does not include direct corporate giving or energy assistance, so small numbers (e.g. $0.03M) are real for that scope.

Write 2–4 sentences summarizing data quality, then list any specific values you'd flag for review with a one-line reason each. Do not invent or suggest replacement values."""

    try:
        return _call_llm(provider, prompt, max_tokens=2000)
    except Exception as e:
        log.warning("AI verify (%s) failed: %s", provider, e)
        return f"(AI verification failed [{provider}]: {e})"


def generate_insights(
    datapoints: list[DataPoint],
    audit: Optional[str] = None,
    *,
    provider: Provider = "anthropic",
) -> Optional[str]:
    if provider == "anthropic" and not _anthropic_client():
        return None
    if provider == "gemini" and not _gemini_client():
        return None

    summary_rows = []
    for dp in datapoints:
        if not dp.ok:
            continue
        meta = METRICS.get(dp.metric, {})
        summary_rows.append(
            f"{dp.company} | {meta.get('label', dp.metric)} = {dp.value} {dp.unit}"
        )
    if not summary_rows:
        return None

    prompt = f"""You are a utility-industry strategy analyst writing for a Community Partnerships executive briefing. Below is a benchmark of US electric/gas utilities. Write 4–6 concise insights, each starting with a bolded headline, followed by 1–2 sentences.

Focus on:
- Competitive positioning on operations AND philanthropy
- Structural differences (integrated generator vs distributor)
- Where the company sits on size-normalized philanthropy metrics
- Clear strategic implications a Community Partnerships VP could act on

Plain English. No preamble.

Benchmark:
{chr(10).join(summary_rows)}

{f'Auditor notes: {audit}' if audit else ''}

Return a markdown bullet list. Use **bold** for headlines."""

    try:
        return _call_llm(provider, prompt, max_tokens=2000)
    except Exception as e:
        log.warning("AI insights (%s) failed: %s", provider, e)
        return f"(AI insights failed [{provider}]: {e})"


# ── Quality scan: per-value flags ────────────────────────────────────────────


def apply_cross_source_bonus(datapoints: list[DataPoint]) -> None:
    """Cross-source agreement boost: when an extracted value is corroborated
    by attempts from two or more independent source families AND the
    values agree within 10%, raise the confidence by +0.05 (capped at 0.99).

    This rewards data points where multiple sources independently confirmed
    the number — a signal of correctness much stronger than any single
    source's tier.

    Mutates datapoints in place.  Adds a sentence to the .notes field
    describing the corroboration when applied.
    """
    for dp in datapoints:
        if not dp.ok or dp.value is None or dp.confidence_score is None:
            continue
        # Look at attempts that succeeded with a numeric response and a
        # different source family than the one that produced dp
        primary_source = (dp.source_name or "").split(" ")[0].lower()
        corroborators: list[tuple[str, float]] = []
        for a in dp.attempts:
            if not a.success:
                continue
            other_source = (a.source or "").split(" ")[0].lower()
            if other_source == primary_source or not other_source:
                continue
            # Try to parse a number from the response_preview
            import re
            for m in re.finditer(r"[-+]?\d+(?:\.\d+)?", a.response_preview or ""):
                try:
                    n = float(m.group(0))
                    # Only consider numbers in the same order of magnitude
                    if dp.value > 0 and 0.1 <= (n / dp.value) <= 10:
                        corroborators.append((other_source, n))
                        break
                except ValueError:
                    pass

        # Are any corroborators within 10% of dp.value?
        agreed = [
            (src, n) for src, n in corroborators
            if abs(n - dp.value) / max(abs(dp.value), 1e-9) <= 0.10
        ]
        if agreed:
            old_conf = dp.confidence_score
            dp.confidence_score = min(0.99, old_conf + 0.05)
            sources_agreeing = ", ".join(sorted(set(s for s, _ in agreed)))
            cross_note = (f"Cross-source agreement bonus: corroborated by "
                          f"{sources_agreeing} (Δ ≤ 10%); confidence "
                          f"raised {old_conf:.2f} → {dp.confidence_score:.2f}.")
            dp.notes = ((dp.notes or "") + " | " + cross_note).strip(" |")


def compute_standing(
    datapoints: list[DataPoint], focus_company: str = "Con Edison"
) -> dict[str, dict]:
    """For each metric, compute where the focus company stands vs peers.

    Returns: {metric_key: {
        "focus_value":     <focus company's value or None>,
        "rank":            <1-based rank, 1 = best>,
        "of_total":        <total companies with values>,
        "peer_median":     <median across peers>,
        "peer_average":    <mean across peers>,
        "best":            (company_name, value),
        "worst":           (company_name, value),
        "vs_median_pct":   <percentage above/below peer median>,
        "verdict":         "Top performer" | "Above median" | "Median" | "Below median" | "Lowest" | "No data",
        "narrative":       "<one-line plain-English summary>",
    }}

    "Best" honors lower_is_better — for SAIDI and carbon emissions, lower is better.
    """
    from .models import METRICS as _M

    standings: dict[str, dict] = {}
    metrics_seen: set[str] = set()
    for dp in datapoints:
        metrics_seen.add(dp.metric)

    focus_lower = focus_company.lower()

    for metric in metrics_seen:
        meta = _M.get(metric, {})
        lower_better = meta.get("lower_is_better", False)
        label = meta.get("label", metric)

        # All values for this metric
        values = []
        focus_dp = None
        for dp in datapoints:
            if dp.metric != metric:
                continue
            if dp.company.lower() == focus_lower:
                focus_dp = dp
            if dp.ok and dp.value is not None:
                values.append((dp.company, dp.value))

        if not values:
            standings[metric] = {
                "focus_value": None, "rank": None, "of_total": 0,
                "peer_median": None, "peer_average": None,
                "best": None, "worst": None, "vs_median_pct": None,
                "verdict": "No data",
                "narrative": f"No {label} values were retrieved for any company in this run.",
            }
            continue

        # Sort by value: best first (smallest if lower_better, else largest)
        values_sorted = sorted(values, key=lambda x: x[1], reverse=not lower_better)
        n = len(values_sorted)

        focus_value = focus_dp.value if (focus_dp and focus_dp.ok) else None
        rank = None
        if focus_value is not None:
            for i, (c, v) in enumerate(values_sorted, 1):
                if c.lower() == focus_lower:
                    rank = i
                    break

        # Peer-only stats (exclude focus company)
        peer_vals = [v for c, v in values_sorted if c.lower() != focus_lower]
        peer_median = (sorted(peer_vals)[len(peer_vals) // 2]
                       if peer_vals else None)
        peer_average = (sum(peer_vals) / len(peer_vals)) if peer_vals else None

        vs_median_pct = None
        if focus_value is not None and peer_median:
            vs_median_pct = ((focus_value - peer_median) / peer_median) * 100

        # Verdict
        verdict = "No data"
        narrative = ""
        if focus_value is None:
            verdict = "No data"
            narrative = f"{focus_company} has no value for {label}."
        elif n == 1:
            verdict = "Only data point"
            narrative = f"{focus_company} is the only company with a value for {label}."
        elif rank == 1:
            verdict = "Top performer"
            narrative = (
                f"{focus_company} ranks #1 of {n} on {label} "
                f"({focus_value:g} {meta.get('unit','')}), "
                f"{'lowest' if lower_better else 'highest'} in the peer set."
            )
        elif rank == n:
            verdict = "Lowest"
            narrative = (
                f"{focus_company} ranks last (#{n} of {n}) on {label} "
                f"({focus_value:g} {meta.get('unit','')}). "
                f"Peer leader: {values_sorted[0][0]} at {values_sorted[0][1]:g}."
            )
        else:
            top_half = rank <= n // 2
            verdict = "Above median" if top_half else "Below median"
            ratio_text = ""
            if vs_median_pct is not None:
                if vs_median_pct >= 0:
                    ratio_text = f"{abs(vs_median_pct):.0f}% above peer median"
                else:
                    ratio_text = f"{abs(vs_median_pct):.0f}% below peer median"
            narrative = (
                f"{focus_company} ranks #{rank} of {n} on {label} "
                f"({focus_value:g} {meta.get('unit','')}) — {ratio_text}."
            )

        standings[metric] = {
            "focus_value": focus_value,
            "rank": rank,
            "of_total": n,
            "peer_median": peer_median,
            "peer_average": peer_average,
            "best": values_sorted[0] if values_sorted else None,
            "worst": values_sorted[-1] if values_sorted else None,
            "vs_median_pct": vs_median_pct,
            "verdict": verdict,
            "narrative": narrative,
        }

    return standings


def rule_based_flags(datapoints: list[DataPoint]) -> dict[tuple[str, str], dict]:
    """Quick, deterministic flagging that runs without an API key.

    Returns: {(company, metric): {"flag": "✅"|"⚠️"|"🚩"|"ℹ️", "reason": "..."}}

    Calibration philosophy: data from real government APIs and ProPublica is
    almost always correct.  We avoid false-positive flags — a value at 0.50
    confidence from J.D. Power or a CSR PDF is NOT a problem, it's a known
    third-party source.  Only flag 🚩 when something is *wrong*, not just
    "lower-tier".

    Rules:
      🚩  Genuine problems:
          - peer outlier ≥10× the median (likely scale/unit error)
          - confidence < 0.30 (press-release tier — we explicitly distrust)
      ⚠️  Caveats worth knowing:
          - foundation-only when "total" was requested
          - confidence 0.30–0.49 AND value is also peer-outlier 3-10×
      ℹ️  Failures by cause:
          - no value, AI fallback was disabled
          - no value, structural (e.g. company has no eGRID generation)
          - no value, AI tried but failed (credit / API error)
      ✅  Default for any successful value with confidence ≥ 0.50.
    """
    flags: dict[tuple[str, str], dict] = {}

    # Peer medians for outlier detection
    peer_values: dict[str, list[float]] = {}
    for dp in datapoints:
        if dp.ok and dp.value is not None and dp.value > 0:
            peer_values.setdefault(dp.metric, []).append(dp.value)

    for dp in datapoints:
        key = (dp.company, dp.metric)

        # ── Failure cases — distinguish by cause ─────────────────────────────
        # Flag philosophy:
        #   ℹ️ = legitimate "not applicable" — the data doesn't exist for this
        #        company×metric combo (pure distributor for carbon, no public
        #        filings for a company that doesn't file).
        #   🚩 = data exists somewhere but we failed to retrieve it (foundation
        #        on ProPublica wasn't matched, custom metric needs CSR/AI to
        #        find it).
        #   AI suggestions are only added when AI would actually help.
        if not dp.ok:
            reason_text = ((dp.error or {}).get("reason") or "").lower()
            notes_text = (dp.notes or "").lower()

            if "company is not an integrated generator" in reason_text:
                # Pure distributor + carbon emissions: this is a structural
                # mismatch, not an extraction failure.  No AI can fix this.
                flags[key] = {
                    "flag": "ℹ️",
                    "reason": ("Not applicable — this company is a pure "
                               "transmission/distribution utility and does NOT own "
                               "generation plants tracked in EPA eGRID. Their Scope 1 "
                               "emissions are negligible and reported only in their "
                               "CSR report (vehicle fleet, gas leaks)."),
                }
            elif "no SEC CIK" in reason_text or "may not be SEC-registered" in reason_text:
                # Could be UK parent (NG) or non-investor-owned (PSEG LI is a
                # LIPA contractor, not a separate SEC filer).  AI MIGHT find
                # revenue from annual reports or news.
                flags[key] = {
                    "flag": "🚩",
                    "reason": ("No SEC 10-K filings — company may have a foreign "
                               "parent, be a privately-held subsidiary, or be a "
                               "non-IOU public entity. Revenue likely available "
                               "from annual report or parent's filing; enable "
                               "AI fallback to try those sources."),
                }
            elif "could not resolve foundation" in reason_text:
                # Foundation might exist under a different name, OR the
                # company might not have a registered 501(c)(3) foundation
                # at all (giving via direct corporate budget instead).
                flags[key] = {
                    "flag": "🚩",
                    "reason": ("Foundation not found on ProPublica. Either the "
                               "company has no 501(c)(3) corporate foundation "
                               "(gives directly from operating budget — common for "
                               "smaller utilities), or the foundation uses a "
                               "different legal name. AI fallback can search CSR "
                               "reports and press releases for this."),
                }
            elif ("ai fallback is disabled" in reason_text or
                  "could not be auto-matched" in reason_text or
                  "matched to" in reason_text):
                # Custom metric situation
                flags[key] = {
                    "flag": "🚩",
                    "reason": ("Custom metric — value likely exists in a CSR/"
                               "sustainability report or company news but no "
                               "automated extractor could find it. AI fallback "
                               "(Anthropic API key) would search the web for it."),
                }
            elif "credit balance" in reason_text or "credit balance" in notes_text:
                flags[key] = {
                    "flag": "🚩",
                    "reason": ("AI fallback failed — Anthropic API credit is $0. "
                               "Add billing at console.anthropic.com to enable, "
                               "or remove ANTHROPIC_API_KEY from secrets."),
                }
            elif "csr_url" in notes_text or "no csr url on file" in notes_text:
                # CSR URL not in the registry — would need manual config OR AI
                flags[key] = {
                    "flag": "🚩",
                    "reason": ("No CSR report URL configured for this company. "
                               "Add one to extractors/csr_report.py CSR_URL_HINTS, "
                               "or enable AI fallback to find values via web search."),
                }
            elif ("ai fallback failed across all providers" in reason_text
                  or "ai fallback failed across all providers" in notes_text):
                # Both Anthropic and Gemini were tried and both failed.
                # Show the combined reason so the user sees which provider
                # said what.
                combined = reason_text if "all providers" in reason_text else notes_text
                # Strip the lead-in
                idx = combined.lower().find("ai fallback failed across all providers")
                detail = combined[idx:].split("—", 1)[-1].strip() if idx >= 0 else combined
                flags[key] = {
                    "flag": "🚩",
                    "reason": (f"Both AI providers failed — {detail[:250]}. "
                               "If only Anthropic was tried, set GOOGLE_API_KEY "
                               "in Streamlit secrets to add a free Gemini fallback."),
                }
            elif "ai fallback api call failed" in notes_text or "ai fallback api call failed" in reason_text:
                # Pull the actual API error out of the notes string.
                # Notes look like: "AiFallback: AI fallback API call failed: <real error>"
                actual_error = ""
                for source_str in (notes_text, reason_text):
                    if "ai fallback api call failed:" in source_str:
                        idx = source_str.index("ai fallback api call failed:")
                        actual_error = source_str[idx + len("ai fallback api call failed:"):].strip()
                        # Truncate long error messages
                        actual_error = actual_error.split(" | ")[0][:240]
                        break
                # Detect which provider this was (look for [Claude] or [Gemini]
                # in the source_name or attempts)
                provider_label = "Anthropic"
                if "[gemini]" in (notes_text + reason_text):
                    provider_label = "Gemini"
                if "model:" in actual_error and "not_found" in actual_error:
                    msg = (f"{provider_label} failed — model name not recognized. "
                           f"Raw error: {actual_error}")
                elif "credit balance" in actual_error or "billing" in actual_error:
                    msg = (f"{provider_label} failed — API credit is $0. "
                           "Add billing or set GOOGLE_API_KEY for free Gemini fallback.")
                elif "invalid x-api-key" in actual_error or "401" in actual_error:
                    msg = (f"{provider_label} failed — invalid API key. Generate "
                           "a fresh one and update Streamlit secrets.")
                elif "rate" in actual_error and "limit" in actual_error:
                    msg = (f"{provider_label} failed — hit rate limit. "
                           "Wait a minute and re-run, OR set GOOGLE_API_KEY in "
                           "Streamlit secrets so the tool can fall through to "
                           "Gemini's free tier (1500 req/day) when Anthropic "
                           "rate-limits.")
                elif "quota" in actual_error.lower():
                    msg = (f"{provider_label} failed — daily quota exhausted. "
                           "Add the other provider's key for redundancy.")
                else:
                    msg = f"{provider_label} failed: {actual_error or '(unknown error)'}"
                flags[key] = {"flag": "🚩", "reason": msg}
            else:
                # Generic failure — the structured extractor was tried and
                # came back empty. Probably retrievable with a different
                # source or AI fallback.
                flags[key] = {
                    "flag": "🚩",
                    "reason": ((dp.error or {}).get("reason") or "no value extracted")[:240],
                }
            continue

        # ── Successful values: default to ✅, downgrade only when warranted ─
        c = dp.confidence_score or 0
        notes_lower = (dp.notes or "").lower()

        # Press-release tier — we distrust this enough to flag
        if c < 0.30:
            flags[key] = {
                "flag": "🚩",
                "reason": f"very low confidence ({c:.2f}) — press release / news only",
            }
            continue

        # Peer-outlier check.  Skip metrics where structural variance is huge
        # (carbon_emissions: integrated generator vs distributor; revenue:
        # parent vs sub varies hugely).
        outlier_skip = ("carbon_emissions", "revenue", "renewable_pct")
        is_outlier = False
        outlier_ratio = 0.0
        outlier_median = 0.0
        if dp.metric not in outlier_skip:
            peers = [v for v in peer_values.get(dp.metric, []) if v != dp.value]
            if len(peers) >= 3:   # need at least 3 peers for stable median
                peers.sort()
                median = peers[len(peers) // 2]
                if median > 0:
                    ratio = dp.value / median
                    # Only flag truly extreme outliers (10×) — real philanthropy
                    # at large utilities legitimately spans an order of magnitude
                    if ratio >= 10.0 or (0 < ratio <= 0.1):
                        is_outlier = True
                        outlier_ratio = ratio
                        outlier_median = median

        if is_outlier and c < 0.50:
            flags[key] = {
                "flag": "🚩",
                "reason": (f"extreme outlier: {dp.value} is {outlier_ratio:.1f}× "
                           f"peer median ({outlier_median:g}) AND low confidence"),
            }
            continue

        if is_outlier:
            flags[key] = {
                "flag": "⚠️",
                "reason": (f"value is {outlier_ratio:.1f}× peer median "
                           f"({outlier_median:g}) — verify it's the same scope/unit"),
            }
            continue

        # "Foundation only" caveat for charitable_giving
        if dp.metric == "charitable_giving" and "foundation" in notes_lower and "only" in notes_lower:
            flags[key] = {
                "flag": "⚠️",
                "reason": ("only the foundation 990-PF slice — does not include "
                           "direct corporate giving, energy assistance, or in-kind"),
            }
            continue

        # Default: looks fine
        flags[key] = {"flag": "✅", "reason": f"confidence {c:.2f}, no concerns"}

    return flags


def ai_quality_scan(datapoints: list[DataPoint]) -> Optional[dict[tuple[str, str], dict]]:
    """AI-powered quality scan.  Augments rule_based_flags with Claude's
    judgment about plausibility, scale errors, and source-credibility issues.

    Returns the same shape as rule_based_flags or None if AI unavailable.
    """
    client = _client()
    if not client:
        return None

    rows = []
    for dp in datapoints:
        if not dp.ok:
            continue
        meta = METRICS.get(dp.metric, {})
        rows.append({
            "company": dp.company,
            "metric": dp.metric,
            "label": meta.get("label", dp.metric),
            "value": dp.value,
            "unit": dp.unit,
            "year": dp.year,
            "source": dp.source_name,
            "confidence": dp.confidence_score,
            "expected_range": meta.get("expected_range", ""),
        })
    if not rows:
        return None

    prompt = f"""You are an experienced utility-industry data analyst auditing a freshly compiled benchmark.  Be skeptical.  Your job is to catch errors before they reach an executive deck.

For EACH row below, decide if the value looks correct.

Output a single JSON object: a map keyed by "company||metric" (use double-pipe), value is {{"flag": "✅"|"⚠️"|"🚩", "reason": "<one short sentence>"}}.

Use:
  ✅  Value looks correct and from a credible source
  ⚠️  Value is plausible but has a caveat (partial coverage, dated, methodology mismatch, foundation-only when total was implied)
  🚩  Value looks WRONG.  Specifically check for:
        - Order-of-magnitude error (e.g. revenue showing as $15,000B instead of $15B — likely $ vs $B confusion)
        - Year number bleeding into a value field (e.g. revenue=2024, charitable_giving=2023 — check if the value matches the year field)
        - Unit confusion ($M vs $B, hours vs days, % stored as 0.42 when displayed as 42)
        - Implausibly tiny value for a company that size (e.g. Con Edison Revenue = $0.05B is wrong; it's a $15B company)
        - Implausibly large value for what's being measured
        - A number that's exactly the same across multiple companies (suggesting a copy-paste or unit-default error)
        - Outliers ≥5× peer median when the metric is normally tightly clustered (philanthropy, customer satisfaction)

Structural facts that are NOT errors (do not flag):
- Integrated generators (Duke, Southern, Dominion, PG&E) emit ~10× more Scope 1 CO₂ than pure distributors. That is structural, not an outlier.
- Foundation grants paid varies year-to-year; small amounts in off-years (down to a few thousand dollars) are real.
- Customer-satisfaction scores cluster tightly around 50/100; a 1-2 point gap IS meaningful but is NOT a 🚩.

Reference ranges (use these to spot-check):
- Revenue: $5–60B for large IOUs
- Foundation grants paid: $0.1–20M
- Foundation assets: $5–150M
- Carbon Emissions Scope 1 (generator): 30–120 M MT CO2; (distributor): 0–5 M MT CO2 (usually 0)
- Customer satisfaction: 45–80 / 100
- Renewable %: 5–60%
- SAIDI: 50–300 min/yr (more for storm-prone utilities)

Rows to audit:
{json.dumps(rows, indent=2, default=str)}

Return ONLY the JSON object.  No prose.  No markdown fences."""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
    except Exception as e:
        log.warning("AI quality_scan failed: %s", e)
        return None

    # Parse the JSON map
    import re
    cleaned = re.sub(r"^```json\s*|```$", "", text, flags=re.MULTILINE).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        raw = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as e:
        log.warning("AI quality_scan returned invalid JSON: %s", e)
        return None

    # Convert "company||metric" → (company, metric) keys
    out: dict[tuple[str, str], dict] = {}
    for k, v in raw.items():
        if "||" not in k or not isinstance(v, dict):
            continue
        company, metric = k.split("||", 1)
        flag = v.get("flag", "")
        if flag not in ("✅", "⚠️", "🚩"):
            continue
        out[(company, metric)] = {
            "flag": flag,
            "reason": str(v.get("reason", ""))[:300],
        }
    return out


def merged_flags(datapoints: list[DataPoint]) -> dict[tuple[str, str], dict]:
    """Combine rule-based flags with AI scan when available.  AI flags take
    precedence when they're more severe (🚩 > ⚠️ > ✅) — the rule-based
    confidence floor is preserved as a baseline."""
    rules = rule_based_flags(datapoints)
    ai = ai_quality_scan(datapoints) or {}

    severity = {"🚩": 3, "⚠️": 2, "✅": 1}
    merged: dict[tuple[str, str], dict] = {}
    for key in set(rules) | set(ai):
        r = rules.get(key)
        a = ai.get(key)
        if r and a:
            if severity[a["flag"]] >= severity[r["flag"]]:
                merged[key] = {
                    "flag": a["flag"],
                    "reason": f"AI: {a['reason']}",
                }
            else:
                merged[key] = {
                    "flag": r["flag"],
                    "reason": r["reason"],
                }
        else:
            merged[key] = a or r
    return merged
