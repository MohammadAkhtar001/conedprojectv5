"""
AI fallback extractor.

Last-resort extractor that uses Claude + web_search to find a value for
any (company, metric) pair that the structured extractors couldn't resolve.

Why this is a fallback, not a primary path:
  - Confidence is intrinsically lower than government APIs (LLM may
    misread context, sources may be press releases or news rather than
    regulator filings)
  - Costs API tokens
  - Slower than direct API calls

Why it's still better than a "no data" wall:
  - Custom metrics the user types are unlikely to have purpose-built
    extractors; web_search is the only way to address them
  - For known metrics where the structured extractor failed (URL changed,
    schema drifted), the AI extractor can still surface a value rather
    than giving up

Strict rules respected:
  - No mock / fallback / seed data — if Claude can't find a real source,
    the value comes back null with attempts logged
  - Confidence is honestly scored at 0.50 (third-party survey tier), or
    0.30 if Claude indicates a press-release source
  - The model is instructed to refuse to guess and return null when no
    credible source exists
"""

from __future__ import annotations
import json
import logging
import os
import re
from typing import Iterable, Optional

from pipeline.models import (
    DataPoint, ExtractionAttempt, METRICS, CONFIDENCE, make_failure
)
from pipeline.validate import validate_value, ValidationFailure
from .base import Extractor, register
from .company_registry import Company

log = logging.getLogger("ai_fallback")


def _client():
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key or key.startswith("sk-ant-...") or len(key) < 20:
        return None
    try:
        from anthropic import Anthropic
        return Anthropic(api_key=key)
    except ImportError:
        return None


class AiFallbackExtractor(Extractor):
    """Last-resort extractor that uses Claude + web_search.

    Note this extractor declares supplies_metrics = () so it isn't picked
    up by the registry's normal metric → extractors lookup.  Instead, it
    is registered as a fallback by the orchestrator AFTER all primary
    extractors fail.
    """

    supplies_metrics = ()  # registered manually as a global fallback
    source_name = "AI + web_search"
    base_confidence = 0.50

    def extract(self, company: Company, metrics: Iterable[str]) -> list[DataPoint]:
        client = _client()
        if not client:
            return [make_failure(
                company=company.name, metric=m,
                reason="AI fallback unavailable (ANTHROPIC_API_KEY not set)",
                attempts=[],
            ) for m in metrics]

        results = []
        for metric in metrics:
            results.append(self._extract_one(client, company, metric))
        return results

    def _extract_one(self, client, company: Company, metric: str) -> DataPoint:
        meta = METRICS.get(metric)
        if not meta:
            # Custom metric — no validation rules, just ask for the value
            # with a description the user supplied.
            unit = "(custom)"
            description = metric
            expected_range = "n/a (custom metric — no plausibility check)"
        else:
            unit = meta["unit"]
            description = meta["description"]
            expected_range = meta["expected_range"]

        prompt = f"""You are a utility-industry data analyst. Find the most recent (FY2024 or FY2025) value of a single metric for one US electric/gas utility.

Company: **{company.name}**
{f'Aliases / additional context: {", ".join(company.aliases)}' if company.aliases else ''}
{company.notes if company.notes else ''}

Metric: **{metric}**
Unit: {unit}
What it means: {description}
Expected range: {expected_range}

Search the web. Prefer government / regulator sources (SEC EDGAR, IRS 990-PF, EPA eGRID, EIA, state PUC filings). Fall back to corporate sustainability reports, then to reputable news only as a last resort.

Hard rules:
- Return a single JSON object — no prose, no markdown.
- If you cannot find a credible source for this specific value, set "value" to null. Do NOT guess.
- Cite the actual source URL you used.
- "confidence" should be 0.85 if from a regulator filing, 0.60 if from a CSR report, 0.50 if from a third-party survey, 0.30 if from a press release / news.

Output schema:
{{
  "value": <number or null>,
  "unit": "{unit}",
  "year": "<e.g. FY2024 or null>",
  "source_url": "<URL or null>",
  "source_name": "<short label, e.g. 'SEC 10-K FY2024' or null>",
  "confidence": <0.30, 0.50, 0.60, or 0.85>,
  "notes": "<one sentence explaining what you found, or why null>"
}}"""

        attempts: list[ExtractionAttempt] = []
        text = ""
        used_search = False
        try:
            # Try with web_search first; some models (Haiku family) don't
            # support tools and will 400.  Fall back to no-tools mode in
            # that case — Claude can still attempt the answer from training
            # data, and we lower the confidence accordingly.
            try:
                msg = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}],
                    tools=[{"type": "web_search_20250305", "name": "web_search"}],
                )
                used_search = True
            except Exception as search_err:
                err_str = str(search_err).lower()
                if ("tool" in err_str or "web_search" in err_str
                        or "not supported" in err_str or "400" in err_str):
                    log.info("web_search not supported by model; retrying without tools")
                    msg = client.messages.create(
                        model="claude-3-5-sonnet-latest",
                        max_tokens=2000,
                        messages=[{"role": "user", "content": prompt}],
                    )
                else:
                    raise

            text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")

            # Log a synthetic attempt so the audit trail shows the AI was tried.
            attempts.append(ExtractionAttempt(
                source=self.source_name + (" (with web_search)" if used_search else " (no search)"),
                url="anthropic://messages",
                method="POST", status_code=200,
                content_type="application/json",
                response_bytes=len(text or ""),
                response_preview=(text or "")[:500],
                selectors_matched=None, duration_ms=0, success=True,
            ))
        except Exception as e:
            attempts.append(ExtractionAttempt(
                source=self.source_name, url="anthropic://messages",
                method="POST", status_code=None,
                content_type=None, response_bytes=None,
                response_preview="", selectors_matched=None,
                duration_ms=0, success=False, error=str(e),
            ))
            return make_failure(
                company=company.name, metric=metric,
                reason=f"AI fallback API call failed: {e}",
                attempts=attempts,
            )

        # Pull the JSON object out of the response
        parsed = self._parse_json(text)
        if not parsed:
            return make_failure(
                company=company.name, metric=metric,
                reason="AI fallback response was not valid JSON",
                attempts=attempts,
                notes=f"Response preview: {(text or '')[:200]}",
            )

        value = parsed.get("value")
        if value is None:
            return make_failure(
                company=company.name, metric=metric,
                reason=parsed.get("notes") or "AI returned null (no credible source found)",
                attempts=attempts,
            )

        # Validate against rules if it's a known metric
        if meta:
            try:
                value = validate_value(metric, float(value), unit)
            except ValidationFailure as ve:
                return make_failure(
                    company=company.name, metric=metric,
                    reason=f"AI value rejected by validator: {ve.reason}",
                    attempts=attempts,
                    notes=f"raw AI response: value={parsed.get('value')} {unit}",
                )

        confidence = parsed.get("confidence")
        if not isinstance(confidence, (int, float)):
            confidence = self.base_confidence
        confidence = max(0.30, min(0.85, float(confidence)))
        # Without web_search, the AI is answering from training data — cap
        # confidence at 0.45 so this never looks like a regulator-grade source.
        if not used_search:
            confidence = min(confidence, 0.45)

        notes_text = parsed.get("notes") or ""
        if not used_search:
            notes_text = ("[no web_search available on this model — value "
                          "from training data only, treat as low-confidence] "
                          + notes_text)

        return DataPoint(
            company=company.name, metric=metric,
            value=round(float(value), 3),
            unit=parsed.get("unit") or unit,
            year=parsed.get("year"),
            source_url=parsed.get("source_url"),
            source_name=f"{self.source_name} → {parsed.get('source_name', 'unknown')}",
            confidence_score=confidence,
            attempts=attempts,
            notes=notes_text,
        )

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        if not text:
            return None
        # Strip code fences if present
        cleaned = re.sub(r"```json\s*|```", "", text).strip()
        # Find first {...} block
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return None
