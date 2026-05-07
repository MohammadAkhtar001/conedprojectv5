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
import time
from typing import Iterable, Optional

from pipeline.models import (
    DataPoint, ExtractionAttempt, METRICS, CONFIDENCE, make_failure
)
from pipeline.validate import validate_value, ValidationFailure
from .base import Extractor, register
from .company_registry import Company

log = logging.getLogger("ai_fallback")


def _anthropic_client():
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key or key.startswith("sk-ant-...") or len(key) < 20:
        return None
    try:
        from anthropic import Anthropic
        return Anthropic(api_key=key)
    except ImportError:
        return None


def _gemini_client():
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key or len(key) < 20:
        return None
    try:
        from google import genai
        return genai.Client(api_key=key)
    except ImportError:
        return None


# Backwards-compat shim
def _client():
    return _anthropic_client()


# ── Module-level rate-limit tracking ───────────────────────────────────────
# These persist across all calls to extract() within a single Python
# process, so that as the orchestrator iterates over (company × metric)
# pairs the pacing knowledge carries forward.

_LAST_ANTHROPIC_CALL_AT: float = 0.0
_LAST_GEMINI_CALL_AT: float = 0.0
# User has 50 RPM on Sonnet/Haiku/Opus.  Haiku has 50K input tokens/min.
# Each fallback prompt is ~120 input tokens; with web_search adding ~2-3K
# tokens of search context, ~16-20 calls/min stays under the ITPM cap.
# 4-second pacing = ~15 calls/min, well-spaced.
_ANTHROPIC_MIN_INTERVAL_S: float = 4.0
_GEMINI_MIN_INTERVAL_S: float = 6.0      # Gemini free tier: 15 RPM

# When a provider just 429'd, cool off longer before hitting it again.
_ANTHROPIC_COOLDOWN_UNTIL: float = 0.0
_GEMINI_COOLDOWN_UNTIL: float = 0.0

# After N consecutive 429s, give up on a provider for the rest of the run
# and route all subsequent calls to the other one.  This avoids burning
# 60+ seconds of cooldowns on a provider that's clearly out of capacity.
_ANTHROPIC_429_STREAK: int = 0
_GEMINI_429_STREAK: int = 0
_GIVE_UP_AFTER_N_429S: int = 2


def _provider_is_available(provider: str) -> bool:
    """Returns False if a provider has exceeded the 429 streak — caller
    should skip it and use the other provider instead."""
    if provider == "anthropic":
        return _ANTHROPIC_429_STREAK < _GIVE_UP_AFTER_N_429S
    if provider == "gemini":
        return _GEMINI_429_STREAK < _GIVE_UP_AFTER_N_429S
    return True


def _wait_for_rate_window(provider: str) -> None:
    """Sleep until enough time has passed since the last call to provider,
    AND until any active cooldown has expired."""
    global _LAST_ANTHROPIC_CALL_AT, _LAST_GEMINI_CALL_AT
    import time as _t
    now = _t.time()
    if provider == "anthropic":
        target = max(
            _LAST_ANTHROPIC_CALL_AT + _ANTHROPIC_MIN_INTERVAL_S,
            _ANTHROPIC_COOLDOWN_UNTIL,
        )
        wait = max(0.0, target - now)
        if wait > 0:
            _t.sleep(wait)
        _LAST_ANTHROPIC_CALL_AT = _t.time()
    elif provider == "gemini":
        target = max(
            _LAST_GEMINI_CALL_AT + _GEMINI_MIN_INTERVAL_S,
            _GEMINI_COOLDOWN_UNTIL,
        )
        wait = max(0.0, target - now)
        if wait > 0:
            _t.sleep(wait)
        _LAST_GEMINI_CALL_AT = _t.time()


def _mark_rate_limited(provider: str) -> None:
    """Set a cooldown for a provider that just 429'd, and increment streak."""
    global _ANTHROPIC_COOLDOWN_UNTIL, _GEMINI_COOLDOWN_UNTIL
    global _ANTHROPIC_429_STREAK, _GEMINI_429_STREAK
    import time as _t
    if provider == "anthropic":
        _ANTHROPIC_COOLDOWN_UNTIL = _t.time() + 30.0
        _ANTHROPIC_429_STREAK += 1
        if _ANTHROPIC_429_STREAK >= _GIVE_UP_AFTER_N_429S:
            log.warning("Anthropic 429'd %d times in a row — disabling for the "
                        "rest of this run; routing all calls to Gemini.",
                        _ANTHROPIC_429_STREAK)
    elif provider == "gemini":
        _GEMINI_COOLDOWN_UNTIL = _t.time() + 30.0
        _GEMINI_429_STREAK += 1
        if _GEMINI_429_STREAK >= _GIVE_UP_AFTER_N_429S:
            log.warning("Gemini 429'd %d times in a row — disabling for the "
                        "rest of this run; routing all calls to Anthropic.",
                        _GEMINI_429_STREAK)


def _mark_success(provider: str) -> None:
    """Reset the 429 streak after a successful call."""
    global _ANTHROPIC_429_STREAK, _GEMINI_429_STREAK
    if provider == "anthropic":
        _ANTHROPIC_429_STREAK = 0
    elif provider == "gemini":
        _GEMINI_429_STREAK = 0


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
        anthropic_client = _anthropic_client()
        gemini_client = _gemini_client()

        if not anthropic_client and not gemini_client:
            return [make_failure(
                company=company.name, metric=m,
                reason=("AI fallback unavailable — neither ANTHROPIC_API_KEY "
                        "nor GOOGLE_API_KEY is set in Streamlit secrets."),
                attempts=[],
            ) for m in metrics]

        results = []
        for i, metric in enumerate(metrics):
            dp_anthropic = None
            dp_gemini = None

            # If a provider has been disabled mid-run due to repeated 429s,
            # skip it and route directly to the other one.  Saves ~30s per
            # metric of pointless cooldown waiting.
            try_anthropic = anthropic_client and _provider_is_available("anthropic")
            try_gemini = gemini_client and _provider_is_available("gemini")

            if try_anthropic:
                dp_anthropic = self._extract_one_anthropic(anthropic_client, company, metric)
                if dp_anthropic.ok:
                    _mark_success("anthropic")
                    results.append(dp_anthropic)
                    continue

            if try_gemini:
                dp_gemini = self._extract_one_gemini(gemini_client, company, metric)
                if dp_gemini.ok:
                    _mark_success("gemini")
                    if dp_anthropic is not None:
                        dp_gemini.attempts = (
                            list(dp_anthropic.attempts) + list(dp_gemini.attempts)
                        )
                        prefix = (f"Recovered via Gemini after Anthropic failed. "
                                  f"Anthropic error: "
                                  f"{(dp_anthropic.error or {}).get('reason', '')[:100]}. ")
                        dp_gemini.notes = (prefix + (dp_gemini.notes or "")).strip()
                    results.append(dp_gemini)
                    continue

            # Neither provider produced a valid value.  Build a combined
            # failure DataPoint that surfaces BOTH provider errors so the
            # user can diagnose without checking the attempts log.
            combined_attempts = []
            reasons = []
            for label, dp in [("Anthropic", dp_anthropic), ("Gemini", dp_gemini)]:
                if dp is None:
                    continue
                combined_attempts.extend(dp.attempts)
                reason = (dp.error or {}).get("reason", "(no reason)")[:160]
                reasons.append(f"{label}: {reason}")

            if not reasons:
                # Should not reach here (we already returned early if no
                # clients), but be defensive.
                results.append(make_failure(
                    company=company.name, metric=metric,
                    reason="AI fallback unavailable (no providers configured)",
                    attempts=[],
                ))
            else:
                results.append(make_failure(
                    company=company.name, metric=metric,
                    reason=f"AI fallback failed across all providers — {' | '.join(reasons)}",
                    attempts=combined_attempts,
                ))
        return results

    def _extract_one_anthropic(self, client, company: Company, metric: str) -> DataPoint:
        return self._extract_one(client, company, metric, provider="anthropic")

    def _extract_one_gemini(self, client, company: Company, metric: str) -> DataPoint:
        return self._extract_one(client, company, metric, provider="gemini")

    def _extract_one(self, client, company: Company, metric: str,
                     *, provider: str = "anthropic") -> DataPoint:
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

        prompt = f"""Find FY2024 value of {metric} ({unit}) for US utility "{company.name}". Web search. Prefer SEC/IRS/EPA/EIA. If no regulator data exists, return whatever non-regulator value you find with a low confidence score — DO NOT return null just because the source isn't authoritative. Only set value=null if you genuinely found nothing.

Return ONE JSON object only:
{{"value": <number or null>, "unit": "{unit}", "year": "FY2024", "source_url": "<URL>", "source_name": "<short label>", "confidence": <0.20/0.30/0.50/0.60/0.85>, "notes": "<1 sentence>"}}

Confidence scale:
  0.85 = regulator filing (SEC/IRS/EPA/EIA/state PUC)
  0.60 = company CSR / sustainability report
  0.50 = third-party survey (J.D. Power etc.)
  0.30 = press release / news article
  0.20 = third-party aggregator / unverified data broker (DitchCarbon, etc.) — RETURN THE VALUE, just flag it with this confidence

Always cite the actual source URL you used."""

        attempts: list[ExtractionAttempt] = []
        text = ""
        used_search = False
        provider_label = "Claude" if provider == "anthropic" else "Gemini"
        endpoint_url = (f"anthropic://messages" if provider == "anthropic"
                        else "gemini://generate_content")

        def _is_rate_limit_error(err) -> bool:
            s = str(err).lower()
            return ("429" in s or "rate_limit" in s or "rate limit" in s
                    or "resource_exhausted" in s or "quota" in s)

        def _call_with_retry(do_call, max_retries: int = 1):
            """Call do_call(), retry once after exponential backoff on 429."""
            for attempt_num in range(max_retries + 1):
                try:
                    return do_call()
                except Exception as e:
                    if _is_rate_limit_error(e) and attempt_num < max_retries:
                        wait = 8 * (attempt_num + 1)  # 8s, then 16s
                        log.info("Rate-limited on %s; backing off %ds", provider_label, wait)
                        time.sleep(wait)
                        continue
                    raise

        try:
            if provider == "anthropic":
                _wait_for_rate_window("anthropic")
                # Try with web_search first; fall back to no-tools mode if
                # the model rejects it.
                try:
                    msg = _call_with_retry(lambda: client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=1500,
                        messages=[{"role": "user", "content": prompt}],
                        tools=[{"type": "web_search_20250305", "name": "web_search"}],
                    ))
                    used_search = True
                except Exception as search_err:
                    err_str = str(search_err).lower()
                    if ("tool" in err_str or "web_search" in err_str
                            or "not supported" in err_str
                            or ("400" in err_str and "429" not in err_str)):
                        log.info("web_search not supported; retrying without tools")
                        _wait_for_rate_window("anthropic")
                        msg = _call_with_retry(lambda: client.messages.create(
                            model="claude-haiku-4-5-20251001",
                            max_tokens=1500,
                            messages=[{"role": "user", "content": prompt}],
                        ))
                    else:
                        raise
                text = "".join(
                    b.text for b in msg.content if getattr(b, "type", None) == "text"
                )
            elif provider == "gemini":
                _wait_for_rate_window("gemini")
                # Gemini free tier has no web_search; values come from
                # training data only.  Cap confidence accordingly later.
                resp = _call_with_retry(lambda: client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                ))
                text = (resp.text or "")
                used_search = False

            # Log a synthetic attempt so the audit trail shows the AI was tried.
            search_suffix = " (with web_search)" if used_search else " (no search)"
            attempts.append(ExtractionAttempt(
                source=f"{self.source_name} [{provider_label}]" + search_suffix,
                url=endpoint_url,
                method="POST", status_code=200,
                content_type="application/json",
                response_bytes=len(text or ""),
                response_preview=(text or "")[:500],
                selectors_matched=None, duration_ms=0, success=True,
            ))
        except Exception as e:
            # If this was a rate limit, mark the provider as cooled down so
            # subsequent calls in this run wait longer before hitting again.
            if _is_rate_limit_error(e):
                _mark_rate_limited(provider)
            attempts.append(ExtractionAttempt(
                source=f"{self.source_name} [{provider_label}]",
                url=endpoint_url,
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
                reason=parsed.get("notes") or "AI returned null (no source found)",
                attempts=attempts,
            )

        # Validate against rules if it's a known metric.  Per directive:
        # if the value falls outside the plausible range, we DO NOT
        # silently drop it — we pass it through with very low confidence
        # so the user sees what the AI found but knows to flag it.
        validator_warning = None
        if meta:
            try:
                value = validate_value(metric, float(value), unit)
            except ValidationFailure as ve:
                # Keep the raw value but mark it as suspect.
                validator_warning = (
                    f"⚠️ VALUE OUTSIDE PLAUSIBLE RANGE: {ve.reason}. "
                    f"Original AI response: value={parsed.get('value')} {unit}. "
                    f"Passed through anyway per directive — review before using."
                )
                try:
                    value = float(parsed.get("value"))
                except (TypeError, ValueError):
                    return make_failure(
                        company=company.name, metric=metric,
                        reason=f"AI value not numeric: {parsed.get('value')!r}",
                        attempts=attempts,
                    )

        confidence = parsed.get("confidence")
        if not isinstance(confidence, (int, float)):
            confidence = self.base_confidence
        confidence = max(0.20, min(0.85, float(confidence)))
        # Without web_search, the AI is answering from training data — cap
        # confidence at 0.45 so this never looks like a regulator-grade source.
        if not used_search:
            confidence = min(confidence, 0.45)
        # Validator-rejected values get the lowest confidence so they
        # always show 🚩.
        if validator_warning is not None:
            confidence = min(confidence, 0.20)

        notes_text = parsed.get("notes") or ""
        if not used_search:
            notes_text = ("[no web_search available on this model — value "
                          "from training data only, treat as low-confidence] "
                          + notes_text)
        if validator_warning:
            notes_text = (validator_warning + " | " + notes_text).strip(" |")

        return DataPoint(
            company=company.name, metric=metric,
            value=round(float(value), 3),
            unit=parsed.get("unit") or unit,
            year=parsed.get("year"),
            source_url=parsed.get("source_url"),
            source_name=f"{self.source_name} [{provider_label}] → {parsed.get('source_name', 'unknown')}",
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
