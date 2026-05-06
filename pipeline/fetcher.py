"""
Orchestrator.

Walks the (company, metric) matrix.  For each pair:
  1. Try registered extractors in priority order (SEC, ProPublica, eGRID, etc.).
  2. If all primary extractors fail AND ANTHROPIC_API_KEY is set, try the
     AI fallback extractor (Claude + web_search).
  3. If that also fails, emit a structured failure DataPoint.

For unknown companies (not in extractors/company_registry.py), the
orchestrator constructs a synthetic Company on the fly with the inputs
the user provided.  Most government-API extractors will return null for
unknown companies (no CIK, no foundation EIN, no eGRID operator), but
the AI fallback can still try to find values for them.

For unknown metrics (not in pipeline/models.py METRICS), the orchestrator
ALSO accepts them and routes directly to the AI fallback — there's no
structured extractor for arbitrary user-defined metrics, so AI is the
only option.  The metric key the user typed becomes the metric name in
the prompt; the pipeline does no validation since plausible-range rules
can't exist for an unknown unit.

There is no fallback to mock/seed data anywhere.  Failure stays failure.
"""

from __future__ import annotations
import logging
import os
from typing import Iterable, Optional

from .models import DataPoint, ExtractionAttempt, make_failure, METRICS
from extractors import EXTRACTOR_PRIORITY
from extractors.company_registry import Company, resolve as registry_resolve

log = logging.getLogger("orchestrator")


def run_pipeline(
    company_names: Iterable[str],
    metrics: Iterable[str],
    *,
    use_ai_fallback: Optional[bool] = None,
) -> list[DataPoint]:
    """Main entry point.  Returns one DataPoint per (company, metric).

    use_ai_fallback: if None, auto-detect from ANTHROPIC_API_KEY env var.
                     If False, skip the AI fallback even when a key is set.
    """
    if use_ai_fallback is None:
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        use_ai_fallback = bool(key) and not key.startswith("sk-ant-...")

    results: list[DataPoint] = []

    for raw_name in company_names:
        company = registry_resolve(raw_name)
        if not company:
            # Unknown company: synthesize a minimal Company record so the AI
            # fallback (if enabled) can still try.  Government-API extractors
            # will return null because cik/ein/operator IDs are missing —
            # that's correct behavior, not a regression.
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
    is_known_metric = metric in METRICS
    extractor_classes = list(EXTRACTOR_PRIORITY.get(metric, []))

    all_attempts: list[ExtractionAttempt] = []
    failure_reasons: list[str] = []

    # ── Phase 1: registered extractors (only for known metrics) ─────────────
    for cls in extractor_classes:
        log.info("→ %s :: %s :: trying %s", company.name, metric, cls.__name__)
        try:
            dps = cls().extract(company, [metric])
        except Exception as e:
            log.exception("extractor %s blew up: %s", cls.__name__, e)
            failure_reasons.append(f"{cls.__name__}: unhandled exception: {e}")
            continue

        for dp in dps:
            if dp.metric != metric:
                continue
            all_attempts.extend(dp.attempts)
            if dp.ok:
                dp.attempts = all_attempts
                return dp
            reason = (dp.error or {}).get("reason") if dp.error else "unknown"
            failure_reasons.append(f"{cls.__name__}: {reason}")

    # ── Phase 2: AI fallback (when enabled) ─────────────────────────────────
    # Runs both when no structured extractor exists (custom metric / unknown
    # company) and when structured extractors all failed.
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

    # ── Phase 3: emit final failure ─────────────────────────────────────────
    if not is_known_metric and not extractor_classes and not use_ai_fallback:
        reason = (
            f"unknown metric '{metric}' and AI fallback is disabled "
            "(set ANTHROPIC_API_KEY to enable web-search-based extraction "
            "for custom metrics)"
        )
    elif not extractor_classes and is_known_metric:
        reason = f"no extractor registered for metric '{metric}'"
    else:
        reason = "all registered extractors failed; see attempts and notes for details"

    return make_failure(
        company=company.name, metric=metric,
        reason=reason,
        attempts=all_attempts,
        notes=" | ".join(failure_reasons) if failure_reasons else None,
    )
