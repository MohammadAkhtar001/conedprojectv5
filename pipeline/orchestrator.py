"""
Orchestrator.

Walks the (company, metric) matrix.  For each pair:
  1. If the metric isn't in METRICS, fuzzy-match it to a known metric
     (so 'program_grants_education' routes to 'stem_education_giving').
  2. Try registered extractors in priority order.
  3. If all primary extractors fail AND ANTHROPIC_API_KEY is set, try the
     AI fallback (Claude + web_search).
  4. If still nothing, emit a structured failure DataPoint with a reason
     that distinguishes "AI off" from genuine data limits.

For unknown companies, synthesize a minimal Company on the fly.

There is no fallback to mock/seed data anywhere.  Failure stays failure.
"""

from __future__ import annotations
import logging
import os
import re
from typing import Iterable, Optional

from .models import DataPoint, ExtractionAttempt, make_failure, METRICS
from extractors import EXTRACTOR_PRIORITY
from extractors.company_registry import Company, resolve as registry_resolve

log = logging.getLogger("orchestrator")


# ── Fuzzy metric matching ──────────────────────────────────────────────────
# Maps free-text metric names typed by users to standard METRICS keys.
# Each tuple: (keywords-that-must-all-be-present, target_metric_key).
# Checked in order — first match wins, so list more specific groups first.
_METRIC_ALIASES: list[tuple[tuple[str, ...], str]] = [
    # Foundation-specific
    (("foundation", "asset"),       "foundation_assets"),
    (("foundation", "grant"),       "foundation_grants_paid"),
    (("foundation", "giving"),      "foundation_grants_paid"),
    # STEM / education
    (("stem",),                     "stem_education_giving"),
    (("education", "giving"),       "stem_education_giving"),
    (("education", "grant"),        "stem_education_giving"),
    (("education", "program"),      "stem_education_giving"),
    (("program", "education"),      "stem_education_giving"),
    (("scholarship",),              "stem_education_giving"),
    (("workforce", "develop"),      "stem_education_giving"),
    # Energy assistance
    (("energy", "assistance"),      "energy_assistance"),
    (("liheap",),                   "energy_assistance"),
    (("hardship",),                 "energy_assistance"),
    (("bill", "assistance"),        "energy_assistance"),
    (("low", "income"),             "energy_assistance"),
    # Community investment
    (("community", "invest"),       "community_investment"),
    (("community", "giving"),       "community_investment"),
    (("community", "develop"),      "community_investment"),
    (("economic", "develop"),       "community_investment"),
    (("infrastructure", "invest"),  "community_investment"),
    # Volunteering / matching
    (("volunteer",),                "volunteer_hours"),
    (("employee", "match"),         "employee_match"),
    (("matching", "gift"),          "employee_match"),
    # Grant counts
    (("number", "grant"),           "num_grants"),
    (("count", "grant"),            "num_grants"),
    # Charitable giving generic
    (("total", "giving"),           "charitable_giving"),
    (("charitable",),               "charitable_giving"),
    (("philanthrop",),              "charitable_giving"),
    # Renewable / clean energy
    (("renewable",),                "renewable_pct"),
    (("clean", "energy"),           "renewable_pct"),
    (("solar",),                    "renewable_pct"),
    (("wind",),                     "renewable_pct"),
    # Carbon
    (("scope", "1"),                "carbon_emissions"),
    (("co2",),                      "carbon_emissions"),
    (("carbon", "emission"),        "carbon_emissions"),
    (("greenhouse",),               "carbon_emissions"),
    (("ghg",),                      "carbon_emissions"),
    # Reliability
    (("saidi",),                    "saidi"),
    (("outage",),                   "saidi"),
    (("reliability",),              "saidi"),
    # Customer satisfaction
    (("customer", "satisf"),        "customer_satisfaction"),
    (("jdpower",),                  "customer_satisfaction"),
    (("j.d. power",),               "customer_satisfaction"),
    (("nps",),                      "customer_satisfaction"),
    # Revenue
    (("revenue",),                  "revenue"),
    (("sales",),                    "revenue"),
]


def _fuzzy_match_metric(user_metric: str) -> Optional[str]:
    """Return the closest known METRICS key for a free-text metric name,
    or None if nothing reasonable matches."""
    if user_metric in METRICS:
        return user_metric
    n = re.sub(r"[_\-/.,]", " ", user_metric.lower())
    n = re.sub(r"\s+", " ", n).strip()
    if not n:
        return None
    for keywords, target in _METRIC_ALIASES:
        if all(kw in n for kw in keywords):
            return target
    return None


def run_pipeline(
    company_names: Iterable[str],
    metrics: Iterable[str],
    *,
    use_ai_fallback: Optional[bool] = None,
) -> list[DataPoint]:
    """Main entry point.  Returns one DataPoint per (company, metric).

    use_ai_fallback: if None, auto-detect from ANTHROPIC_API_KEY or
                     GOOGLE_API_KEY env vars.
    """
    if use_ai_fallback is None:
        anth_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        gem_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        use_ai_fallback = (
            (bool(anth_key) and not anth_key.startswith("sk-ant-..."))
            or (bool(gem_key) and len(gem_key) >= 20)
        )

    results: list[DataPoint] = []

    for raw_name in company_names:
        company = registry_resolve(raw_name)
        if not company:
            company = Company(
                name=raw_name.strip(),
                aliases=(),
                is_integrated_generator=False,
                cik=None,
                foundation_ein=None,
                foundation_name=None,
                eia_op_ids=(),
                egrid_operator_names=(),
                state_puc_codes=(),
                notes=("Unknown company — not in registry. "
                       "Government-API extractors require CIK/EIN/operator IDs "
                       "and will return null. AI fallback may find values via "
                       "web search."),
            )

        for metric in metrics:
            results.append(_extract_one(company, metric, use_ai_fallback))

    return results


def _extract_one(company: Company, metric: str, use_ai_fallback: bool) -> DataPoint:
    """Try to extract one metric for one company.  Output's `metric` field
    always matches the user-typed metric so tables stay aligned."""

    is_known_metric = metric in METRICS

    # Fuzzy-match unknown metrics to a registered metric
    routed_metric = metric
    routing_note = None
    if not is_known_metric:
        matched = _fuzzy_match_metric(metric)
        if matched:
            routed_metric = matched
            routing_note = (
                f"Custom metric '{metric}' was matched to standard metric "
                f"'{matched}' ({METRICS[matched]['label']}). Value comes from "
                f"the same source as that standard metric."
            )
            log.info("→ %s :: %s :: fuzzy-matched to %s", company.name, metric, matched)

    extractor_classes = list(EXTRACTOR_PRIORITY.get(routed_metric, []))

    all_attempts: list[ExtractionAttempt] = []
    failure_reasons: list[str] = []

    # ── Phase 1: registered extractors ──────────────────────────────────────
    for cls in extractor_classes:
        log.info("→ %s :: %s :: trying %s", company.name, routed_metric, cls.__name__)
        try:
            dps = cls().extract(company, [routed_metric])
        except Exception as e:
            log.exception("extractor %s blew up: %s", cls.__name__, e)
            failure_reasons.append(f"{cls.__name__}: unhandled exception: {e}")
            continue

        for dp in dps:
            if dp.metric != routed_metric:
                continue
            all_attempts.extend(dp.attempts)
            if dp.ok:
                # Restore the user-typed metric name on the output so column
                # headers in the UI match what the user typed
                dp.metric = metric
                dp.attempts = all_attempts
                if routing_note:
                    dp.notes = ((dp.notes or "") + " | " + routing_note).strip(" |")
                return dp
            reason = (dp.error or {}).get("reason") if dp.error else "unknown"
            failure_reasons.append(f"{cls.__name__}: {reason}")

    # ── Phase 2: AI fallback (when enabled) ─────────────────────────────────
    if use_ai_fallback:
        log.info("→ %s :: %s :: trying AI fallback", company.name, metric)
        try:
            from extractors.ai_fallback import AiFallbackExtractor
            dps = AiFallbackExtractor().extract(company, [metric])
            for dp in dps:
                if dp.metric != metric:
                    continue
                all_attempts.extend(dp.attempts)
                if dp.ok:
                    dp.attempts = all_attempts
                    if failure_reasons:
                        prefix = (
                            f"Recovered via AI fallback after "
                            f"{len(failure_reasons)} structured extractor(s) failed. "
                        )
                        dp.notes = (prefix + (dp.notes or "")).strip()
                    return dp
                reason = (dp.error or {}).get("reason") if dp.error else "unknown"
                failure_reasons.append(f"AiFallback: {reason}")
        except Exception as e:
            log.exception("AI fallback blew up: %s", e)
            failure_reasons.append(f"AiFallback: unhandled exception: {e}")

    # ── Phase 3: emit final failure with a tailored reason ──────────────────
    if not is_known_metric and not extractor_classes and not use_ai_fallback:
        reason = (
            f"custom metric '{metric}' could not be auto-matched to a standard "
            "metric, and AI fallback is disabled. Either rename to match a "
            "standard metric (try 'stem_education_giving', 'energy_assistance', "
            "'community_investment', 'foundation_grants_paid'), or set "
            "ANTHROPIC_API_KEY in Streamlit secrets to enable AI search."
        )
    elif not is_known_metric and routing_note and not use_ai_fallback:
        reason = (
            f"custom metric '{metric}' was matched to '{routed_metric}' but "
            f"that source returned no value. Set ANTHROPIC_API_KEY for AI fallback."
        )
    elif not extractor_classes and is_known_metric:
        reason = f"no extractor registered for metric '{metric}'"
    elif not use_ai_fallback:
        reason = (
            "structured extractors failed and AI fallback is disabled. "
            "Set ANTHROPIC_API_KEY in Streamlit secrets to enable AI search "
            "for stragglers like this one."
        )
    else:
        reason = "all registered extractors and AI fallback failed; see attempts log"

    return make_failure(
        company=company.name, metric=metric,
        reason=reason,
        attempts=all_attempts,
        notes=" | ".join(failure_reasons) if failure_reasons else routing_note,
    )
