"""
EPA eGRID extractor.

Source: EPA eGRID (Emissions & Generation Resource Integrated Database)
        https://www.epa.gov/egrid/download-data
        Annual XLSX dataset; we use the plant-level sheet (PLNT{year}) and
        sum CO₂ tonnage for plants whose operator name matches the
        company's eGRID operator names.

Auth:   None.
Rate:   Single ~10MB download, cached locally for 7 days.

Supplies: carbon_emissions (Scope 1, M MT CO₂)

Why this works: eGRID is the EPA's authoritative dataset for US power-plant
emissions.  Each plant has a unique ORISPL code, an operator (utility),
and a reported CO₂ value in short tons — we aggregate by operator and
convert to metric tons.

For companies that own NO generation (Con Edison, Eversource, National
Grid USA), this extractor returns the value 0 with a note.  That's
correct — those utilities don't have direct Scope 1 emissions from
generation.  Their Scope 1 is small (vehicle fleet, gas leaks) and not
reported via eGRID; for those we return null and let CSR pick it up.

Confidence:
  0.95 — exact match, owns generation
  N/A  — pure distributor (no Scope 1 from generation; null returned)
"""

from __future__ import annotations
import logging
import os
import time
from pathlib import Path
from typing import Iterable, Optional

from pipeline.models import DataPoint, ExtractionAttempt, METRICS, CONFIDENCE, make_failure
from pipeline.fetcher import fetch, FetchError
from pipeline.validate import validate_value, ValidationFailure
from .base import Extractor, register
from .company_registry import Company

log = logging.getLogger("epa_egrid")

# Most-recent eGRID release.  EPA publishes annually with a ~2-year lag —
# at the time of writing the latest is eGRID2022, released in early 2024.
# Update this URL when EPA publishes a newer release; the file name
# convention has been stable for several years.
EGRID_URL = "https://www.epa.gov/system/files/documents/2024-01/egrid2022_data.xlsx"
EGRID_YEAR = 2022

CACHE_DIR = Path(os.environ.get("EGRID_CACHE_DIR", ".cache"))
CACHE_FILE = CACHE_DIR / f"egrid{EGRID_YEAR}_data.xlsx"
CACHE_TTL_SECONDS = 7 * 24 * 3600

# Short ton → metric ton conversion
SHORT_TON_TO_MT = 0.907185


class EpaEgridExtractor(Extractor):
    supplies_metrics = ("carbon_emissions",)
    source_name = "EPA eGRID"
    base_confidence = CONFIDENCE["gov_dataset_exact"]

    _cached_df = None  # class-level cache: read XLSX once per process

    def extract(self, company: Company, metrics: Iterable[str]) -> list[DataPoint]:
        if "carbon_emissions" not in metrics:
            return []

        # Pure distributors: no plants, no Scope 1 from generation.  This
        # is a real "no data" — eGRID legitimately doesn't have a value
        # for them — so we return null with a clear reason.
        if not company.egrid_operator_names:
            return [make_failure(
                company=company.name, metric="carbon_emissions",
                reason=("company is not an integrated generator; "
                        "Scope 1 emissions from generation are not applicable. "
                        "Fugitive Scope 1 (gas leaks, fleet) requires CSR report."),
                attempts=[],
            )]

        # Load (and cache) the eGRID dataset
        attempts: list[ExtractionAttempt] = []
        try:
            df = self._load_egrid(attempts)
        except FetchError as e:
            return [make_failure(
                company=company.name, metric="carbon_emissions",
                reason=f"could not load eGRID dataset: {e}",
                attempts=attempts,
            )]
        except Exception as e:
            return [make_failure(
                company=company.name, metric="carbon_emissions",
                reason=f"eGRID parse failed: {e}",
                attempts=attempts,
            )]

        # Find plants operated by any of this company's eGRID operator names
        # eGRID's plant-level sheet has columns including OPRNAME (operator
        # name) and PLCO2AN (plant annual CO₂ emissions in short tons).
        # Column codes are stable across years; if EPA changes them we'll
        # see it loudly here.
        operator_col = self._find_col(df, ("OPRNAME", "OPERATOR_NAME", "OPNAME"))
        co2_col = self._find_col(df, ("PLCO2AN", "PLCO2EQA", "PLCO2RTA"))
        if not operator_col or not co2_col:
            return [make_failure(
                company=company.name, metric="carbon_emissions",
                reason=f"eGRID dataset missing expected columns; got {[str(c) for c in df.columns][:30]}",
                attempts=attempts,
            )]

        # Build operator-name target list defensively: only non-empty strings.
        targets = [
            str(n).lower().strip()
            for n in company.egrid_operator_names
            if isinstance(n, str) and n.strip()
        ]
        if not targets:
            return [make_failure(
                company=company.name, metric="carbon_emissions",
                reason=("no usable eGRID operator names in registry for this company "
                        "(empty or non-string entries)"),
                attempts=attempts,
            )]

        # Match plant rows whose OPRNAME contains (or is contained by) any target.
        op_lower = df[operator_col].astype(str).str.lower().fillna("")

        def matches(s: str) -> bool:
            if not isinstance(s, str) or not s:
                return False
            for t in targets:
                if not isinstance(t, str) or not t:
                    continue
                if t in s or s in t:
                    return True
            return False

        try:
            mask = op_lower.apply(matches)
            matched = df[mask]
        except Exception as e:
            return [make_failure(
                company=company.name, metric="carbon_emissions",
                reason=f"eGRID row-matching failed: {e}",
                attempts=attempts,
                notes=f"operator targets: {targets}",
            )]

        if matched.empty:
            sample = df[operator_col].dropna().astype(str).unique()[:8].tolist()
            return [make_failure(
                company=company.name, metric="carbon_emissions",
                reason=(f"no eGRID plants matched operator names {targets}. "
                        f"Sample operators in dataset: {sample}"),
                attempts=attempts,
            )]

        # Sum and convert short tons → million metric tons.
        # Coerce CO2 column to numeric (eGRID sometimes has blank cells).
        import pandas as pd
        co2_numeric = pd.to_numeric(matched[co2_col], errors="coerce").fillna(0)
        total_short_tons = float(co2_numeric.sum())
        total_mmt = (total_short_tons * SHORT_TON_TO_MT) / 1e6

        unit = METRICS["carbon_emissions"]["unit"]
        try:
            validated = validate_value("carbon_emissions", total_mmt, unit)
        except ValidationFailure as ve:
            return [make_failure(
                company=company.name, metric="carbon_emissions",
                reason=f"value rejected by validator: {ve.reason}",
                attempts=attempts,
                notes=f"raw aggregate {total_mmt:.2f} M MT from {len(matched)} plants",
            )]

        return [DataPoint(
            company=company.name, metric="carbon_emissions",
            value=round(validated, 3),
            unit=unit,
            year=f"FY{EGRID_YEAR}",
            source_url="https://www.epa.gov/egrid/download-data",
            source_name=f"{self.source_name} {EGRID_YEAR} — plant-level CO₂ summed by operator",
            confidence_score=self.base_confidence,
            attempts=attempts,
            notes=(f"Aggregated {len(matched)} plants matching operator names "
                   f"{list(company.egrid_operator_names)}. Short tons → metric tons via "
                   f"factor {SHORT_TON_TO_MT}."),
        )]

    # ── Dataset loading ────────────────────────────────────────────────────

    def _load_egrid(self, attempts: list[ExtractionAttempt]):
        """Download and parse the eGRID workbook.  Cached for CACHE_TTL_SECONDS."""
        # Lazy import: pandas + openpyxl are heavy
        import pandas as pd

        if EpaEgridExtractor._cached_df is not None:
            return EpaEgridExtractor._cached_df

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        need_download = True
        if CACHE_FILE.exists():
            age = time.time() - CACHE_FILE.stat().st_mtime
            if age < CACHE_TTL_SECONDS:
                need_download = False
                log.info("eGRID cache hit (age %ds)", int(age))

        if need_download:
            r, attempt = fetch(EGRID_URL, source=self.source_name, timeout=60)
            attempts.append(attempt)
            CACHE_FILE.write_bytes(r.content)
            log.info("eGRID downloaded → %s (%d bytes)", CACHE_FILE, CACHE_FILE.stat().st_size)
        else:
            attempts.append(ExtractionAttempt(
                source=self.source_name, url=str(CACHE_FILE), method="CACHE",
                status_code=200, content_type="application/vnd.openxmlformats",
                response_bytes=CACHE_FILE.stat().st_size, response_preview="[cached file]",
                selectors_matched=None, duration_ms=0, success=True,
            ))

        # Find the plant-level sheet.  eGRID workbook naming convention varies
        # slightly across releases ("PLNT22", "PLNT23", ...) and sheets like
        # "ST22", "BA22" exist for state and balancing-authority levels — we
        # specifically want plant-level data.  We auto-discover the right
        # sheet by looking at sheet names rather than hard-coding.
        all_sheets = pd.ExcelFile(CACHE_FILE).sheet_names
        log.info("eGRID sheets available: %s", all_sheets)

        plant_sheet = None
        for s in all_sheets:
            su = str(s).upper()
            # Plant-level sheets always start with PLNT followed by year digits
            if su.startswith("PLNT") and any(c.isdigit() for c in su):
                plant_sheet = s
                break
        if not plant_sheet:
            raise RuntimeError(
                f"Could not find plant-level sheet in eGRID workbook. "
                f"Available sheets: {all_sheets}"
            )

        # Plant data starts on row 2 (index 1) — row 0 has long descriptions,
        # row 1 has the short element codes (OPRNAME, PLCO2AN…).
        df = pd.read_excel(CACHE_FILE, sheet_name=plant_sheet, header=1)
        log.info("eGRID plant sheet '%s': %d rows, columns: %s",
                 plant_sheet, len(df), list(df.columns)[:20])
        EpaEgridExtractor._cached_df = df
        return df

    @staticmethod
    def _find_col(df, candidates):
        cols_upper = {c.upper(): c for c in df.columns if isinstance(c, str)}
        for cand in candidates:
            if cand.upper() in cols_upper:
                return cols_upper[cand.upper()]
        return None


register("carbon_emissions", EpaEgridExtractor)
