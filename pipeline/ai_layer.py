"""
Optional AI-driven verification + insights layer.

If ANTHROPIC_API_KEY is set, after the pipeline produces DataPoints we
ask Claude to:
  1. Audit the table for outliers, scale errors, peer inconsistencies
     (with structural context — integrated generators legitimately emit
     ~10× more Scope 1 CO₂ than pure distributors).
  2. Generate strategic insights for a Community Partnerships team
     reading the benchmark.

This layer NEVER modifies values.  If verification flags an outlier, it
is reported in the audit text — the underlying DataPoint is unchanged.
The strict "no fallback" rule means we don't let the AI invent or
substitute values; it only annotates.

If ANTHROPIC_API_KEY is not set, both functions return None and the
pipeline still produces complete data and exports.
"""

from __future__ import annotations
import json
import logging
import os
from typing import Optional

from .models import DataPoint, METRICS

log = logging.getLogger("ai_layer")


def _client():
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return None
    try:
        from anthropic import Anthropic
        return Anthropic(api_key=key)
    except ImportError:
        log.warning("anthropic package not installed; AI layer disabled")
        return None


def verify(datapoints: list[DataPoint]) -> Optional[str]:
    """Returns an audit summary string, or None if AI layer disabled."""
    client = _client()
    if not client:
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

Write 2–4 sentences summarizing data quality, then list any specific values you'd flag for review with a one-line reason each. Do not invent or suggest replacement values."""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()
    except Exception as e:
        log.warning("AI verify failed: %s", e)
        return f"(AI verification failed: {e})"


def generate_insights(datapoints: list[DataPoint], audit: Optional[str] = None) -> Optional[str]:
    client = _client()
    if not client:
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
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()
    except Exception as e:
        log.warning("AI insights failed: %s", e)
        return f"(AI insights failed: {e})"


# ── Quality scan: per-value flags ────────────────────────────────────────────


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
            elif "ai fallback api call failed" in notes_text:
                flags[key] = {
                    "flag": "🚩",
                    "reason": "AI fallback errored — see Attempts log for the API error",
                }
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

    prompt = f"""You are an experienced utility-industry data analyst auditing a freshly compiled benchmark. For EACH row below, decide if the value looks correct.

Output a single JSON object: a map keyed by "company||metric" (use double-pipe), value is {{"flag": "✅"|"⚠️"|"🚩", "reason": "<one short sentence>"}}.

Use:
  ✅  value looks correct and from a credible source
  ⚠️  value is plausible but has a caveat (partial coverage, dated, methodology mismatch)
  🚩  value looks wrong (off by an order of magnitude, wrong unit, implausibly small/large for company size)

Structural facts that are NOT errors (so do not flag for these):
- Integrated generators (Duke, Southern, Dominion, PG&E) emit ~10× more Scope 1 CO₂ than pure distributors (Con Edison, National Grid USA, Eversource).
- Foundation grants paid varies year-to-year; small amounts in off-years can be real.
- Pure T&D distributors legitimately have null/zero eGRID Scope 1 emissions (no owned generation).

Rows:
{json.dumps(rows, indent=2, default=str)}

Return ONLY the JSON object — no prose, no markdown fences."""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
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
