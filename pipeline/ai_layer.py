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

    Returns: {(company, metric): {"flag": "✅"|"⚠️"|"🚩", "reason": "..."}}

    Rules:
      🚩  no value (failed extraction)
      🚩  confidence < 0.50  (third-party survey or worse)
      🚩  value is a peer-group outlier (≥5× peer median, except for
          structurally-explained metrics like carbon_emissions)
      ⚠️  confidence 0.50–0.69 (CSR PDF, scraped sources)
      ⚠️  value flagged as foundation-only when "total" was implied
      ✅  confidence ≥ 0.70  AND  no other concerns
    """
    flags: dict[tuple[str, str], dict] = {}

    # First pass: per-metric peer medians for outlier detection
    peer_values: dict[str, list[float]] = {}
    for dp in datapoints:
        if dp.ok and dp.value is not None:
            peer_values.setdefault(dp.metric, []).append(dp.value)

    for dp in datapoints:
        key = (dp.company, dp.metric)

        if not dp.ok:
            flags[key] = {"flag": "🚩", "reason": "no value extracted"}
            continue

        c = dp.confidence_score or 0
        notes_lower = (dp.notes or "").lower()

        if c < 0.50:
            flags[key] = {
                "flag": "🚩",
                "reason": f"low confidence ({c:.2f}) — third-party / scraped source",
            }
            continue

        # Peer-outlier check (skip metrics where structural variance is normal)
        if dp.metric not in ("carbon_emissions",):
            peers = [v for v in peer_values.get(dp.metric, []) if v != dp.value]
            if len(peers) >= 2:
                peers.sort()
                median = peers[len(peers) // 2]
                if median > 0:
                    ratio = dp.value / median
                    if ratio >= 5.0 or (0 < ratio <= 0.2):
                        flags[key] = {
                            "flag": "🚩",
                            "reason": (f"peer outlier: {dp.value} is "
                                       f"{ratio:.1f}× peer median ({median})"),
                        }
                        continue

        # "Foundation only" caveat for charitable_giving
        if dp.metric == "charitable_giving" and "foundation" in notes_lower:
            flags[key] = {
                "flag": "⚠️",
                "reason": "foundation slice only — does not include direct corporate giving",
            }
            continue

        if 0.50 <= c < 0.70:
            flags[key] = {
                "flag": "⚠️",
                "reason": f"medium confidence ({c:.2f}) — CSR / non-regulator source",
            }
            continue

        flags[key] = {"flag": "✅", "reason": f"confidence {c:.2f}, no concerns flagged"}

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
