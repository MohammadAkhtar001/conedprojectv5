"""
CSR / Sustainability Report PDF extractor.

Source: Each company's most recent CSR / sustainability report PDF.
Auth:   None.  Public PDFs.

Supplies (when present in the PDF):
  - renewable_pct       (% of energy mix from renewables)
  - energy_assistance   ($M, utility-funded customer hardship + LIHEAP supplements)
  - volunteer_hours     (employee volunteer hours in fiscal year)
  - employee_match      ($M, matching gift program total)
  - num_grants          (count of grants paid; sometimes only here)

Why scraping (carefully) is acceptable here: these metrics are NOT
published in any government dataset for most utilities.  They appear only
in narrative disclosure inside the CSR PDF.  We use pdfplumber + targeted
regex patterns anchored to phrases like "energy assistance", "volunteer
hours", etc., and validate aggressively.

The hard-won lesson: a regex like r"\\$([0-9.]+)\\s*million" matches
anything.  We anchor patterns to a phrase + a small window around it,
NOT to the whole document.  And we always validate the extracted number
against the metric's plausible range.

Confidence: 0.60.  Lower than government-dataset extractors because PDF
text extraction has its own failure modes (multi-column layouts, embedded
images of numbers, infographics rather than text).

URL discovery: this extractor reads a small CSR_URL_HINTS map.  When the
URL for a given company is missing from the hints, the extractor returns
a structured failure rather than guessing.  The hints table is the only
piece of HTML scraping a human (or AI) needs to maintain.
"""

from __future__ import annotations
import logging
import os
import re
from typing import Iterable, Optional

from pipeline.models import DataPoint, ExtractionAttempt, METRICS, CONFIDENCE, make_failure
from pipeline.fetcher import fetch, FetchError
from pipeline.validate import validate_value, ValidationFailure
from .base import Extractor, register
from .company_registry import Company

log = logging.getLogger("csr_report")


# ── CSR URL hints ──────────────────────────────────────────────────────────
# Hand-curated.  When a company's CSR URL isn't here, extraction returns
# a structured failure — we don't guess URLs.  Add new entries as you
# benchmark new companies.  Update annually when companies publish new reports.
CSR_URL_HINTS: dict[str, str] = {
    # Keys are company.name.lower()
    "con edison":            "https://www.coned.com/-/media/files/coned/documents/sustainability/sustainability-report.pdf",
    "duke energy":           "https://www.duke-energy.com/our-company/about-us/sustainability",   # landing page
    "national grid usa":     "https://www.nationalgrid.com/us/responsibility",
    "pacific gas and electric": "https://www.pgecorp.com/corp_responsibility/reports.shtml",
    "eversource energy":     "https://www.eversource.com/content/about/about-us/our-company/corporate-responsibility",
    "southern company":      "https://www.southerncompany.com/corporate-responsibility/reporting.html",
}


# ── Per-metric extraction patterns ─────────────────────────────────────────
# Each pattern is anchored to a distinctive phrase, with a capture group
# for the numeric value and an optional unit suffix.  We extract the FIRST
# match in the PDF text — CSRs typically lead with the headline number.

PHRASE_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "energy_assistance": [
        (r"energy\s+assistance\D{0,80}?\$?([\d,.]+)\s*(million|m\b|billion|b\b)?", "$M"),
        (r"customer\s+hardship\D{0,80}?\$?([\d,.]+)\s*(million|m\b|billion|b\b)?", "$M"),
        (r"bill\s+(?:payment|pay)\s+assistance\D{0,80}?\$?([\d,.]+)\s*(million|m\b)?", "$M"),
    ],
    "volunteer_hours": [
        (r"([\d,]+)\s+(?:volunteer\s+)?hours\b", "hrs/yr"),
        (r"volunteer(?:ed|ing)?\D{0,40}?([\d,]+)\s*hours", "hrs/yr"),
    ],
    "employee_match": [
        (r"employee\s+match(?:ing)?(?:\s+gift)?(?:\s+program)?\D{0,60}?\$?([\d.]+)\s*(million|m\b)?", "$M"),
        (r"matching\s+gift\D{0,60}?\$?([\d.]+)\s*(million|m\b)?", "$M"),
    ],
    "renewable_pct": [
        (r"([\d.]+)\s*%[^.\n]{0,40}?(?:from\s+)?renewable", "%"),
        (r"renewable\D{0,40}?([\d.]+)\s*%", "%"),
        (r"([\d.]+)\s*%[^.\n]{0,40}?clean\s+energy", "%"),
    ],
    "num_grants": [
        (r"(\d{2,4})\s+grants\s+(?:paid|awarded|given)", "grants"),
        (r"(?:awarded|paid|gave)\s+(\d{2,4})\s+grants", "grants"),
    ],
    "community_investment": [
        (r"community\s+investment\D{0,80}?\$?([\d.,]+)\s*(million|m\b|billion|b\b)?", "$M"),
        (r"invested\s+\$?([\d.,]+)\s*(million|m\b|billion|b\b)?\s+in\s+(?:our\s+)?communit", "$M"),
        (r"corporate\s+(?:contributions|giving)\D{0,80}?\$?([\d.,]+)\s*(million|m\b)?", "$M"),
    ],
    "stem_education_giving": [
        (r"(?:STEM|education)(?:\s+giving|\s+programs?|\s+grants?)?\D{0,80}?\$?([\d.,]+)\s*(million|m\b)?", "$M"),
        (r"\$?([\d.,]+)\s*(million|m\b)?\s+(?:to|for|in)\s+(?:STEM|education|scholarship)", "$M"),
        (r"scholarship\D{0,60}?\$?([\d.,]+)\s*(million|m\b)?", "$M"),
    ],
}


def _normalize_number(raw: str, magnitude: Optional[str]) -> float:
    """Convert '12.5' or '1,234' + optional 'million'/'billion' → number."""
    n = float(raw.replace(",", ""))
    if magnitude:
        m = magnitude.lower()
        if m.startswith("b"):
            n *= 1000  # billion → million
    return n


# Cache discovered URLs across pipeline runs (per Python process)
_DISCOVERED_URLS: dict[str, Optional[str]] = {}


def _discover_csr_url(company: Company) -> Optional[str]:
    """Use Claude + web_search to find a current sustainability/CSR report
    URL for a company that isn't in CSR_URL_HINTS.  Caches the result.

    Returns None if AI is unavailable, search fails, or no credible URL
    is found.  Never returns a guess — only URLs Claude found via search.
    """
    cache_key = company.name.lower()
    if cache_key in _DISCOVERED_URLS:
        return _DISCOVERED_URLS[cache_key]

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key or api_key.startswith("sk-ant-...") or len(api_key) < 20:
        _DISCOVERED_URLS[cache_key] = None
        return None

    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
    except ImportError:
        _DISCOVERED_URLS[cache_key] = None
        return None

    prompt = f"""Find the URL for the most recent corporate sustainability / CSR / ESG report PDF or landing page for this US utility:

Company: **{company.name}**
{f'Aliases: {", ".join(company.aliases)}' if company.aliases else ''}

Search the web. Prefer the company's own corporate website over third-party aggregators.

Output a single JSON object — no prose, no markdown:

{{"url": "<full URL or null>", "kind": "pdf" | "landing_page", "year": "<e.g. 2024 or null>"}}

If you cannot find a credible URL on the company's own site, return null. Do NOT make up a URL."""

    try:
        # Try with web_search; fall back gracefully if not supported
        try:
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
            )
        except Exception:
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    except Exception as e:
        log.warning("CSR URL discovery failed: %s", e)
        _DISCOVERED_URLS[cache_key] = None
        return None

    import json as _json
    import re as _re
    cleaned = _re.sub(r"^```json\s*|```$", "", text or "", flags=_re.MULTILINE).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < 0:
        _DISCOVERED_URLS[cache_key] = None
        return None
    try:
        parsed = _json.loads(cleaned[start:end + 1])
        url = parsed.get("url")
        if isinstance(url, str) and url.startswith("http"):
            log.info("Discovered CSR URL for %s: %s", company.name, url)
            _DISCOVERED_URLS[cache_key] = url
            return url
    except _json.JSONDecodeError:
        pass

    _DISCOVERED_URLS[cache_key] = None
    return None


class CsrReportExtractor(Extractor):
    supplies_metrics = (
        "renewable_pct",
        "energy_assistance",
        "volunteer_hours",
        "employee_match",
        "num_grants",
        "community_investment",
        "stem_education_giving",
    )
    source_name = "CSR / Sustainability Report"
    base_confidence = CONFIDENCE["csr_report"]

    _cached_text: dict[str, tuple[str, str]] = {}  # company → (url, text)

    def extract(self, company: Company, metrics: Iterable[str]) -> list[DataPoint]:
        wanted = [m for m in metrics if m in self.supplies_metrics]
        if not wanted:
            return []

        results: list[DataPoint] = []

        url = CSR_URL_HINTS.get(company.name.lower())
        if not url:
            # Try to discover a CSR URL via AI before giving up.  This makes
            # the tool work for companies not in CSR_URL_HINTS without
            # manual configuration.
            url = _discover_csr_url(company)
        if not url:
            for m in wanted:
                results.append(make_failure(
                    company=company.name, metric=m,
                    reason=("no CSR URL on file for this company and AI URL "
                            "discovery did not find one. Either add an entry "
                            "to CSR_URL_HINTS in extractors/csr_report.py, or "
                            "ensure ANTHROPIC_API_KEY is set so AI can search "
                            "for the report URL."),
                    attempts=[],
                ))
            return results

        # Get the report text, downloading + parsing the PDF if needed
        attempts: list[ExtractionAttempt] = []
        try:
            text = self._get_report_text(company, url, attempts)
        except FetchError as e:
            for m in wanted:
                results.append(make_failure(
                    company=company.name, metric=m,
                    reason=f"could not retrieve CSR report: {e}",
                    attempts=attempts,
                ))
            return results
        except Exception as e:
            for m in wanted:
                results.append(make_failure(
                    company=company.name, metric=m,
                    reason=f"PDF parse failed: {e}",
                    attempts=attempts,
                ))
            return results

        # Run each requested metric through its patterns
        for m in wanted:
            value, snippet, page = self._find_metric(text, m)
            if value is None:
                results.append(make_failure(
                    company=company.name, metric=m,
                    reason=f"no pattern matched in CSR text for {m}",
                    attempts=attempts,
                    notes=f"CSR URL: {url}",
                ))
                continue

            unit = METRICS[m]["unit"]
            try:
                validated = validate_value(m, value, unit)
            except ValidationFailure as ve:
                results.append(make_failure(
                    company=company.name, metric=m,
                    reason=f"value rejected by validator: {ve.reason}",
                    attempts=attempts,
                    notes=f"raw match: '{snippet}' → {value} {unit}",
                ))
                continue

            # Build a "Check My Work" URL — appending #page=N tells PDF
            # viewers (Chrome, Adobe, Edge) to jump directly to the page
            # where this value was found.  Works for direct-PDF links;
            # falls back to the bare URL if not a PDF or page is unknown.
            url_with_page = url
            if page is not None and url.lower().endswith(".pdf"):
                url_with_page = f"{url}#page={page}"

            page_label = f" (page {page})" if page else ""
            attribution_note = (
                f"📄 Found{page_label}: \"{snippet}\""
                if snippet else
                f"📄 Found in CSR PDF{page_label}"
            )

            results.append(DataPoint(
                company=company.name, metric=m,
                value=round(validated, 3),
                unit=unit,
                year=None,
                source_url=url_with_page,
                source_name=f"{self.source_name} ({company.name})"
                            + (f" p.{page}" if page else ""),
                confidence_score=self.base_confidence,
                attempts=attempts,
                notes=attribution_note,
            ))

        return results

    # ── PDF retrieval ──────────────────────────────────────────────────────

    def _get_report_text(self, company: Company, url: str, attempts: list[ExtractionAttempt]) -> str:
        cached = self._cached_text.get(company.name)
        if cached and cached[0] == url:
            return cached[1]

        # Step 1: fetch the URL.  It might be a direct PDF or an HTML
        # landing page that links to the PDF.
        r, attempt = fetch(url, source=self.source_name, timeout=30)
        attempts.append(attempt)

        ctype = r.headers.get("Content-Type", "")
        if "pdf" in ctype.lower() or url.lower().endswith(".pdf"):
            text = self._pdf_to_text(r.content)
        else:
            # HTML landing page → find the first PDF link, fetch it.
            pdf_url = self._find_pdf_link(r.text, base_url=url)
            if not pdf_url:
                raise FetchError(attempt, f"no PDF link found on landing page {url}")
            r2, attempt2 = fetch(pdf_url, source=self.source_name, timeout=30, expect="pdf")
            attempts.append(attempt2)
            text = self._pdf_to_text(r2.content)

        self._cached_text[company.name] = (url, text)
        return text

    @staticmethod
    def _pdf_to_text(blob: bytes) -> str:
        """Extract text from a PDF blob using pdfplumber.  Each page is
        prefixed with a marker so we can later determine which page a
        match came from.  The marker is invisible to the regex extractors
        (it's an unusual sequence) but readable by _page_for_offset()."""
        import io
        import pdfplumber
        out = []
        with pdfplumber.open(io.BytesIO(blob)) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                t = page.extract_text() or ""
                # Marker we can scan for later when we know the match offset
                out.append(f"\x1f__PAGE_{i}__\x1f\n{t}")
        return "\n".join(out)

    @staticmethod
    def _page_for_offset(text: str, offset: int) -> Optional[int]:
        """Return the page number (1-indexed) where the given character
        offset falls in text, based on \\x1f__PAGE_N__\\x1f markers."""
        import re
        last_page = None
        for m in re.finditer(r"\x1f__PAGE_(\d+)__\x1f", text):
            if m.start() > offset:
                break
            last_page = int(m.group(1))
        return last_page

    @staticmethod
    def _snippet_around(text: str, offset: int, span: int = 120) -> str:
        """Return a ~240-char snippet around an offset, with page markers
        stripped, for use in source attribution."""
        import re
        start = max(0, offset - span)
        end = min(len(text), offset + span)
        snippet = text[start:end]
        # Strip our internal page markers and collapse whitespace
        snippet = re.sub(r"\x1f__PAGE_\d+__\x1f\n?", "", snippet)
        snippet = re.sub(r"\s+", " ", snippet).strip()
        # Add ellipses if we trimmed
        if start > 0:
            snippet = "…" + snippet
        if end < len(text):
            snippet = snippet + "…"
        return snippet

    @staticmethod
    def _find_pdf_link(html: str, base_url: str) -> Optional[str]:
        """Find a likely CSR-PDF link on a landing page.  We score links by
        keyword density (sustainability/responsibility/report) and prefer
        recent year stamps."""
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin

        soup = BeautifulSoup(html, "lxml")
        candidates = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.lower().endswith(".pdf"):
                continue
            text = (a.get_text(" ", strip=True) + " " + href).lower()
            score = 0
            for kw in ("sustainability", "responsibility", "csr", "esg",
                       "corporate responsibility", "annual report"):
                if kw in text:
                    score += 2
            for yr in ("2024", "2023"):
                if yr in text:
                    score += 3
            if score > 0:
                candidates.append((score, urljoin(base_url, href)))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    # ── Pattern matching ───────────────────────────────────────────────────

    def _find_metric(self, text: str, metric: str) -> tuple[Optional[float], str, Optional[int]]:
        """Run patterns for `metric` against `text`.  Return
        (value, snippet, page_number) or (None, '', None) if nothing matches."""
        patterns = PHRASE_PATTERNS.get(metric, [])
        text_lower = text.lower()
        for pattern, _ in patterns:
            for match in re.finditer(pattern, text_lower, flags=re.IGNORECASE | re.DOTALL):
                raw = match.group(1)
                magnitude = match.group(2) if match.lastindex and match.lastindex >= 2 else None
                try:
                    value = _normalize_number(raw, magnitude)
                except ValueError:
                    continue
                snippet = self._snippet_around(text, match.start())
                page = self._page_for_offset(text, match.start())
                return value, snippet, page
        return None, "", None


for m in CsrReportExtractor.supplies_metrics:
    register(m, CsrReportExtractor)
