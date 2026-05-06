"""
Data models for the pipeline.

The contract: every extractor returns a DataPoint.  A DataPoint with
`value=None` and `error` populated represents a failed extraction.  There
is no fallback / mock / seed data path — failure stays failure.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional


# ── Metric registry ──────────────────────────────────────────────────────────
# Single source of truth for metric metadata: units, valid ranges, whether
# lower is better.  Validation reads from here; the UI reads from here.
METRICS: dict[str, dict[str, Any]] = {
    "revenue": {
        "label": "Revenue",
        "unit": "$B",
        "category": "operations",
        "lower_is_better": False,
        "plausible_min": 1.0,
        "plausible_max": 80.0,
        "expected_range": "$10B – $35B for large US utilities",
        "description": "Total annual revenue (FY most recent).",
    },
    "renewable_pct": {
        "label": "Renewable Energy %",
        "unit": "%",
        "category": "operations",
        "lower_is_better": False,
        "plausible_min": 0.0,
        "plausible_max": 100.0,
        "expected_range": "15% – 50% for US utilities (2024)",
        "description": "Share of energy supply from renewable sources.",
    },
    "saidi": {
        "label": "Outage Frequency (SAIDI)",
        "unit": "min/yr",
        "category": "operations",
        "lower_is_better": True,
        "plausible_min": 20.0,
        "plausible_max": 500.0,
        "expected_range": "50 – 200 min/yr",
        "description": "System Average Interruption Duration Index.",
    },
    "customer_satisfaction": {
        "label": "Customer Satisfaction Score",
        "unit": "/100",
        "category": "operations",
        "lower_is_better": False,
        "plausible_min": 30.0,
        "plausible_max": 80.0,
        "expected_range": "45 – 55 / 100 (industry avg ≈ 49.9)",
        "description": "J.D. Power Residential Electric Satisfaction Study (raw /1000 ÷ 10).",
    },
    "carbon_emissions": {
        "label": "Carbon Emissions (Scope 1)",
        "unit": "M MT CO2",
        "category": "operations",
        "lower_is_better": True,
        "plausible_min": 0.5,
        "plausible_max": 200.0,
        "expected_range": "Distributors 2–10 M MT  |  Integrated generators 50–100 M MT",
        "description": "Scope 1 CO₂e from owned generation. EPA eGRID is the regulator-of-record.",
    },
    # ── Philanthropy: split into specific sub-metrics ───────────────────────
    # The previous single "charitable_giving" was misleading because it only
    # captured foundation-paid grants from the corporate foundation's 990-PF.
    # Real corporate philanthropy comes from multiple channels:
    #   1. The corporate foundation (separate 501(c)(3)) — IRS 990-PF data
    #   2. Direct corporate contributions (operating-budget giving) — CSR
    #   3. Energy assistance programs (LIHEAP supplements, hardship funds)
    #   4. Economic / community development grants
    #   5. In-kind contributions (volunteer time, tree-planting, etc.)
    # We expose them separately so the user knows what they're comparing.
    "foundation_grants_paid": {
        "label": "Foundation Grants Paid",
        "unit": "$M",
        "category": "philanthropy",
        "lower_is_better": False,
        "plausible_min": 0.01,
        "plausible_max": 200.0,
        "expected_range": "$0.5M – $20M for utility corporate foundations",
        "description": ("Grants paid out by the company's 501(c)(3) corporate "
                        "foundation (IRS 990-PF Part I, contributions paid). "
                        "Excludes operating-budget giving."),
    },
    "charitable_giving": {
        "label": "Total Charitable Giving",
        "unit": "$M",
        "category": "philanthropy",
        "lower_is_better": False,
        "plausible_min": 0.01,
        "plausible_max": 500.0,
        "expected_range": "$5M – $100M for large US utilities",
        "description": ("Total corporate giving — foundation grants paid PLUS "
                        "direct corporate contributions PLUS in-kind. Combines "
                        "990-PF data with CSR disclosure when available."),
    },
    "community_investment": {
        "label": "Community Investment",
        "unit": "$M",
        "category": "philanthropy",
        "lower_is_better": False,
        "plausible_min": 0.1,
        "plausible_max": 500.0,
        "expected_range": "$10M – $100M for large US utilities (broader than philanthropy alone)",
        "description": ("Total community-facing investment — philanthropy + "
                        "economic development grants + customer assistance + "
                        "in-kind. Typically CSR-disclosed."),
    },
    "foundation_assets": {
        "label": "Foundation Assets",
        "unit": "$M",
        "category": "philanthropy",
        "lower_is_better": False,
        "plausible_min": 0.5,
        "plausible_max": 500.0,
        "expected_range": "$5M – $100M for utility corporate foundations",
        "description": "Foundation total assets at fiscal year-end (IRS 990-PF Part II).",
    },
    "energy_assistance": {
        "label": "Energy Assistance Programs",
        "unit": "$M",
        "category": "philanthropy",
        "lower_is_better": False,
        "plausible_min": 0.1,
        "plausible_max": 100.0,
        "expected_range": "$1M – $30M depending on territory size",
        "description": ("Utility-funded customer hardship + LIHEAP supplements + "
                        "bill-pay assistance + weatherization. From CSR disclosure."),
    },
    "stem_education_giving": {
        "label": "STEM / Education Giving",
        "unit": "$M",
        "category": "philanthropy",
        "lower_is_better": False,
        "plausible_min": 0.05,
        "plausible_max": 50.0,
        "expected_range": "$0.5M – $10M for utility STEM/education programs",
        "description": ("Targeted giving to STEM, K-12 education, scholarships, "
                        "workforce development. From CSR or foundation 990 grant lists."),
    },
    "num_grants": {
        "label": "Number of Grants",
        "unit": "grants",
        "category": "philanthropy",
        "lower_is_better": False,
        "plausible_min": 5,
        "plausible_max": 5000,
        "expected_range": "50 – 1500 grants/yr for utility foundations",
        "description": "Total grants paid (IRS 990-PF Part XV).",
    },
    "volunteer_hours": {
        "label": "Volunteer Hours",
        "unit": "hrs/yr",
        "category": "philanthropy",
        "lower_is_better": False,
        "plausible_min": 100,
        "plausible_max": 500000,
        "expected_range": "5,000 – 100,000 hrs/yr",
        "description": "Employee volunteer hours (CSR-disclosed).",
    },
    "employee_match": {
        "label": "Employee Match",
        "unit": "$M",
        "category": "philanthropy",
        "lower_is_better": False,
        "plausible_min": 0.05,
        "plausible_max": 50.0,
        "expected_range": "$0.5M – $10M",
        "description": "Matching gift program total (CSR-disclosed).",
    },
}


# ── Confidence scores by source kind ─────────────────────────────────────────
CONFIDENCE = {
    "sec_xbrl_exact": 0.95,
    "gov_dataset_exact": 0.95,
    "propublica_990": 0.85,
    "gov_dataset_derived": 0.85,
    "csr_report": 0.60,
    "third_party_survey": 0.50,
    "press_news": 0.30,
}


@dataclass
class ExtractionAttempt:
    """One attempt by one extractor at one source.  Always logged regardless of
    outcome — the directive requires that failures be transparent."""

    source: str               # e.g. "SEC EDGAR XBRL"
    url: str                  # the actual URL that was hit
    method: str               # "GET", "POST", "playwright", etc.
    status_code: Optional[int]
    content_type: Optional[str]
    response_bytes: Optional[int]
    response_preview: str     # first 500 chars of the body, truncated
    selectors_matched: Optional[bool]  # for HTML scraping; None for JSON APIs
    duration_ms: int
    success: bool
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self):
        return asdict(self)


@dataclass
class DataPoint:
    """The unit of pipeline output.

    On success: value, unit, source_url, source_name, confidence_score are
    populated and error is None.
    On failure: value is None and error is populated with reason +
    attempted_sources.  No silent substitution.
    """

    company: str
    metric: str               # key from METRICS
    value: Optional[float]
    unit: str
    year: Optional[str]
    source_url: Optional[str]
    source_name: Optional[str]    # human-friendly source label
    confidence_score: Optional[float]
    attempts: list[ExtractionAttempt] = field(default_factory=list)
    error: Optional[dict] = None  # { reason, attempted_sources, ... }
    notes: Optional[str] = None
    extracted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    @property
    def ok(self) -> bool:
        return self.value is not None and self.error is None

    def to_dict(self):
        d = asdict(self)
        # ExtractionAttempt is a dataclass already converted by asdict
        return d

    def to_flat_row(self) -> dict:
        """Flat dict suitable for a DataFrame / CSV row."""
        meta = METRICS.get(self.metric, {})
        return {
            "company": self.company,
            "metric": self.metric,
            "metric_label": meta.get("label", self.metric),
            "value": self.value,
            "unit": self.unit,
            "year": self.year,
            "confidence_score": self.confidence_score,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "ok": self.ok,
            "error_reason": (self.error or {}).get("reason") if self.error else None,
            "num_attempts": len(self.attempts),
            "notes": self.notes,
            "extracted_at": self.extracted_at,
        }


def make_failure(
    company: str,
    metric: str,
    reason: str,
    attempts: list[ExtractionAttempt],
    notes: Optional[str] = None,
) -> DataPoint:
    """Construct the canonical failure DataPoint.  This is the ONLY way a
    failed extraction is represented — there is no fallback path."""
    meta = METRICS.get(metric, {})
    return DataPoint(
        company=company,
        metric=metric,
        value=None,
        unit=meta.get("unit", ""),
        year=None,
        source_url=None,
        source_name=None,
        confidence_score=None,
        attempts=attempts,
        error={
            "reason": reason,
            "attempted_sources": [a.source for a in attempts],
            "attempt_count": len(attempts),
        },
        notes=notes,
    )
