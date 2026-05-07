"""
FERC Form 1 connector — annual financial report from electric utilities.

Source:    https://elibrary.ferc.gov  +  https://www.ferc.gov/form-1
Key forms: Form 1 (electric utilities) and Form 1-F (small utilities)
Coverage:  All US investor-owned electric utilities with > $1M annual revenue
Cadence:   Annual (filed by April 30 for the prior calendar year)

Why a FERC connector at all?  FERC Form 1 contains data NOT in 10-Ks:
  • Schedule 422: Transmission line statistics (miles, voltage, ownership)
  • Schedule 200: Net plant in service (rate base proxy)
  • Schedule 320: Detailed O&M expenses by function
  • Schedule 408: Taxes other than income taxes by jurisdiction

This connector uses FERC's Public XBRL viewer API to pull the latest filed
form for a given respondent ID.  The XBRL API exposes structured tagged data
for all forms filed in 2021 and later (FERC migrated from VFP to XBRL in 2021).

Confidence:  0.90 (gov_dataset_exact tier — slightly below SEC's 0.95
because FERC's XBRL extractor APIs are newer and occasionally return stale
or empty values for the most recent fiscal year if the filing is still
being processed).

Limitations:
  • Pre-2021 filings used Visual FoxPro (.dbf) format, not supported here.
  • Some respondents (small utilities, REA cooperatives) file Form 1-F or
    are exempt entirely; this connector returns null for those.
  • A holding company's consolidated revenue is in 10-K, not Form 1 — Form 1
    is filed by the OPERATING SUBSIDIARY, so a parent like Duke Energy maps
    to multiple respondent IDs (one per subsidiary).
"""

from __future__ import annotations
import json
import logging
import re
from typing import Iterable, Optional

from pipeline.models import (
    DataPoint, ExtractionAttempt, METRICS, CONFIDENCE, make_failure
)
from pipeline.fetcher import fetch, FetchError
from pipeline.validate import validate_value, ValidationFailure
from .base import Extractor, register
from .company_registry import Company

log = logging.getLogger("ferc_form1")


# FERC's open data XBRL API.  Returns JSON-formatted line items.
# Documented at https://ecollection.ferc.gov/api/help
_FERC_XBRL_BASE = "https://ecollection.ferc.gov/api"


# Map our metric keys to FERC XBRL concept tags.  These are stable across
# years and standardized in the FERC taxonomy (FERCElectric).
_FERC_CONCEPT_MAP = {
    "transmission_line_miles": (
        # Schedule 422, line "Total" — sum of miles across all
        # transmission line entries
        "ferc:TransmissionLineLengthInMiles",
    ),
    "rate_base": (
        # Schedule 200 — Total electric utility plant in service
        # (net of accumulated depreciation)
        "ferc:NetUtilityPlantInService",
        "ferc:UtilityPlantInService",
    ),
    "operating_expenses_ferc": (
        # Schedule 320 — Total operation and maintenance expenses
        "ferc:OperationAndMaintenanceExpenseElectric",
        "ferc:TotalElectricOperationAndMaintenanceExpenses",
    ),
}


class FercForm1Extractor(Extractor):
    supplies_metrics = (
        "transmission_line_miles",
        "rate_base",
        "operating_expenses_ferc",
    )
    source_name = "FERC Form 1 (XBRL)"
    base_confidence = 0.90

    def extract(self, company: Company, metrics: Iterable[str]) -> list[DataPoint]:
        wanted = [m for m in metrics if m in self.supplies_metrics]
        if not wanted:
            return []

        attempts: list[ExtractionAttempt] = []
        results: list[DataPoint] = []

        if not company.ferc_respondent_ids:
            for m in wanted:
                results.append(make_failure(
                    company=company.name, metric=m,
                    reason=("no FERC respondent ID on file for this company. "
                            "Add ferc_respondent_ids to the registry entry. "
                            "Cooperatives, public-power utilities, and many "
                            "holding companies do not file Form 1 directly — "
                            "their operating subsidiaries do."),
                    attempts=[],
                ))
            return results

        # For each respondent ID this company has, fetch the latest filing.
        # If a holding company has multiple subsidiaries, we sum the values
        # across them (transmission miles, opex are additive).
        per_metric_total: dict[str, float] = {m: 0.0 for m in wanted}
        per_metric_year: dict[str, Optional[str]] = {m: None for m in wanted}
        per_metric_units: dict[str, str] = {m: METRICS[m]["unit"] for m in wanted}
        per_metric_found: dict[str, bool] = {m: False for m in wanted}
        per_metric_attempts: dict[str, list[ExtractionAttempt]] = {
            m: [] for m in wanted
        }

        for respondent_id in company.ferc_respondent_ids:
            for metric in wanted:
                concepts = _FERC_CONCEPT_MAP.get(metric, ())
                if not concepts:
                    continue
                for concept in concepts:
                    url = (f"{_FERC_XBRL_BASE}/data?respondent={respondent_id}"
                           f"&concept={concept}&form=1&latest=true")
                    try:
                        resp = fetch(url, timeout=20)
                    except FetchError as e:
                        per_metric_attempts[metric].append(ExtractionAttempt(
                            source=self.source_name, url=url,
                            method="GET", status_code=None,
                            content_type=None, response_bytes=None,
                            response_preview="", selectors_matched=None,
                            duration_ms=0, success=False, error=str(e),
                        ))
                        continue

                    per_metric_attempts[metric].append(ExtractionAttempt(
                        source=self.source_name, url=url,
                        method="GET", status_code=resp.status_code,
                        content_type=resp.headers.get("content-type"),
                        response_bytes=len(resp.content),
                        response_preview=(resp.text or "")[:300],
                        selectors_matched=None,
                        duration_ms=int(resp.elapsed_ms or 0), success=True,
                    ))

                    # Parse the JSON response
                    try:
                        data = resp.json()
                    except (ValueError, json.JSONDecodeError):
                        continue

                    # FERC API returns a list of facts
                    facts = data if isinstance(data, list) else data.get("facts", [])
                    if not facts:
                        continue

                    # Take the most recent fiscal year for this respondent
                    facts_sorted = sorted(
                        facts,
                        key=lambda f: f.get("period_end") or f.get("fy") or "",
                        reverse=True,
                    )
                    fact = facts_sorted[0]
                    raw_val = fact.get("value")
                    if raw_val is None:
                        continue
                    try:
                        n = float(raw_val)
                    except (TypeError, ValueError):
                        continue

                    # Normalize units.  FERC reports rate base / opex in
                    # dollars; we want $B.  Transmission miles are already
                    # in miles.
                    if metric in ("rate_base", "operating_expenses_ferc"):
                        n = n / 1e9   # → $B

                    per_metric_total[metric] += n
                    per_metric_found[metric] = True
                    fy = fact.get("fy") or fact.get("period_end", "")[:4]
                    per_metric_year[metric] = f"FY{fy}" if fy else None
                    break  # got a value for this concept; don't try others

        # Build result DataPoints
        for metric in wanted:
            attempts_for_m = per_metric_attempts[metric]
            if not per_metric_found[metric]:
                results.append(make_failure(
                    company=company.name, metric=metric,
                    reason=("FERC Form 1 returned no value for this metric "
                            "across the company's registered respondent IDs. "
                            "Possible causes: filing not yet posted for the "
                            "most recent fiscal year, respondent files Form "
                            "1-F instead of Form 1, or respondent ID is wrong."),
                    attempts=attempts_for_m,
                ))
                continue

            value = per_metric_total[metric]
            unit = per_metric_units[metric]
            try:
                value = validate_value(metric, value, unit)
            except ValidationFailure as ve:
                results.append(make_failure(
                    company=company.name, metric=metric,
                    reason=f"FERC value rejected by validator: {ve.reason}",
                    attempts=attempts_for_m,
                    notes=f"raw FERC sum: {per_metric_total[metric]} {unit}",
                ))
                continue

            results.append(DataPoint(
                company=company.name, metric=metric,
                value=round(float(value), 3),
                unit=unit,
                year=per_metric_year[metric],
                source_url=(f"https://ecollection.ferc.gov/Filings/?form=1"
                            f"&respondent={company.ferc_respondent_ids[0]}"),
                source_name=self.source_name,
                confidence_score=self.base_confidence,
                attempts=attempts_for_m,
                notes=(f"Summed across {len(company.ferc_respondent_ids)} "
                       f"FERC respondent ID(s)."
                       if len(company.ferc_respondent_ids) > 1 else None),
            ))

        return results


for m in FercForm1Extractor.supplies_metrics:
    register(m, FercForm1Extractor)
