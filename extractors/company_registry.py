"""
Company registry.

Maps utility company names to the stable identifiers each government data
source uses.  Hand-curated rather than scraped — these IDs are stable and
the only correct way to address a company in each system.

CIK   = SEC Central Index Key (10 digit, used by data.sec.gov)
EIN   = IRS Employer Identification Number (used by ProPublica /nonprofits)
        Each utility's CORPORATE FOUNDATION has its own EIN distinct from
        the operating company.  The foundation EIN is what we want for 990
        philanthropy data.
EIA_OP_ID = EIA Operator ID from EIA Form 861/923 — used to attribute
            generation and reliability data.
EGRID_ORIS = EPA eGRID parent operator name — used to sum plant-level CO₂.

Adding a new utility: find the four IDs from public sources, add an entry.
This file is the single place company-specific knowledge lives.

Sources for IDs:
- CIK:        https://www.sec.gov/cgi-bin/browse-edgar (search by company name)
- EIN:        https://projects.propublica.org/nonprofits (search by foundation name)
- EIA_OP_ID:  https://www.eia.gov/electricity/data/eia861/ (download zip,
              find operator name in the workbook)
- EGRID:      https://www.epa.gov/egrid (operator names in the latest
              published year's Excel file)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class Company:
    name: str
    aliases: tuple[str, ...]      # alternate names users might type
    is_integrated_generator: bool # True for Duke, Southern, Dominion etc.
    cik: Optional[str]            # SEC, 10-digit zero-padded, e.g. "0001047862"
    foundation_ein: Optional[str] # IRS EIN of the corporate foundation
    foundation_name: Optional[str]
    eia_op_ids: tuple[int, ...] = ()    # may be multiple operating subs
    egrid_operator_names: tuple[str, ...] = ()
    state_puc_codes: tuple[str, ...] = ()  # informational
    notes: str = ""


REGISTRY: dict[str, Company] = {
    "con_edison": Company(
        name="Con Edison",
        aliases=("Consolidated Edison", "ConEd", "Consolidated Edison Inc",
                 "Consolidated Edison Company of New York", "CECONY"),
        is_integrated_generator=False,
        cik="0001047862",                       # Consolidated Edison, Inc.
        foundation_ein=None,                    # resolved by name lookup at runtime
        foundation_name="Consolidated Edison Foundation",
        eia_op_ids=(13511, 49328),              # CECONY + O&R
        egrid_operator_names=(),                # no owned generation
        state_puc_codes=("NY-PSC",),
        notes="Pure T&D distributor (NY); divested all generation by 2008.",
    ),
    "duke_energy": Company(
        name="Duke Energy",
        aliases=("Duke Energy Corporation", "DUK"),
        is_integrated_generator=True,
        cik="0001326160",                       # Duke Energy Corporation
        foundation_ein=None,                    # resolved by name lookup at runtime
        foundation_name="Duke Energy Foundation",
        eia_op_ids=(3046, 7140, 5416),          # multiple subs (DEC, DEP, DEF, DEI, DEK, DEO)
        egrid_operator_names=(
            "Duke Energy Carolinas, LLC",
            "Duke Energy Progress, LLC",
            "Duke Energy Florida, LLC",
            "Duke Energy Indiana, LLC",
            "Duke Energy Kentucky, Inc",
            "Duke Energy Ohio, Inc",
        ),
        state_puc_codes=("NC-NCUC", "SC-PSC", "FL-PSC", "IN-IURC", "OH-PUCO", "KY-PSC"),
        notes="Integrated generator across 6 states.",
    ),
    "national_grid": Company(
        name="National Grid USA",
        aliases=("National Grid", "NGG", "Niagara Mohawk", "KeySpan",
                 "Massachusetts Electric", "Narragansett Electric"),
        is_integrated_generator=False,
        # National Grid USA itself doesn't file 10-Ks (parent is UK-listed),
        # but its NY operating subsidiary Niagara Mohawk Power does.  This
        # gives us US-jurisdiction revenue for the largest piece of the
        # business; smaller subs (Mass Electric, KeySpan, Narragansett) file
        # less consistently with SEC.
        # National Grid USA itself doesn't file 10-Ks (parent is UK-listed),
        # and historically-filing US subs (Niagara Mohawk, KeySpan, Mass
        # Electric) have ceased posting current XBRL data to SEC's
        # companyfacts API.  We leave cik=None and let the AI fallback try
        # to find revenue from National Grid's annual report or
        # press release if ANTHROPIC_API_KEY is set.
        cik=None,
        foundation_ein=None,                    # resolved by name lookup at runtime
        foundation_name="National Grid Foundation",
        eia_op_ids=(13501, 40209),              # Niagara Mohawk + Mass Electric
        egrid_operator_names=(),
        state_puc_codes=("NY-PSC", "MA-DPU", "RI-PUC"),
        notes=("US subsidiary of UK-listed parent.  Revenue from Niagara Mohawk "
               "10-K (NY op sub) — partial coverage; full US revenue requires "
               "summing all US subs.  Foundation name varies — also try "
               "'National Grid Foundation', 'National Grid USA Service'."),
    ),
    "pge": Company(
        name="Pacific Gas and Electric",
        aliases=("PG&E", "PG&E Corporation", "Pacific Gas & Electric", "PCG"),
        is_integrated_generator=True,
        cik="0001004980",                       # PG&E Corp
        foundation_ein=None,                    # resolved by name lookup at runtime
        foundation_name="PG&E Corporation Foundation",
        eia_op_ids=(14328,),
        egrid_operator_names=("Pacific Gas and Electric Company",),
        state_puc_codes=("CA-CPUC",),
        notes="Owns Diablo Canyon nuclear + hydro fleet (renewable-heavy generation).",
    ),
    "eversource": Company(
        name="Eversource Energy",
        aliases=("Eversource", "ES", "NSTAR", "Northeast Utilities"),
        is_integrated_generator=False,
        cik="0000072741",                       # Eversource Energy
        foundation_ein=None,                    # resolved by name lookup at runtime
        foundation_name="Eversource Energy Foundation",
        eia_op_ids=(13998, 13771),              # Connecticut Light & Power + NSTAR
        egrid_operator_names=(),
        state_puc_codes=("CT-PURA", "MA-DPU", "NH-PUC"),
        notes="Pure T&D in New England.",
    ),
    "southern": Company(
        name="Southern Company",
        aliases=("Southern Co", "SO", "Georgia Power", "Alabama Power",
                 "Mississippi Power", "Southern Power"),
        is_integrated_generator=True,
        cik="0000092122",                       # The Southern Company
        foundation_ein=None,                    # resolved by name lookup at runtime
        foundation_name="Southern Company Charitable Foundation",
        eia_op_ids=(7140, 195, 12686),          # Ga Power, Ala Power, Miss Power
        egrid_operator_names=(
            "Georgia Power Company",
            "Alabama Power Company",
            "Mississippi Power Company",
        ),
        state_puc_codes=("GA-PSC", "AL-PSC", "MS-PSC"),
        notes="Integrated generator across the Southeast.",
    ),
    "pseg_long_island": Company(
        name="PSEG Long Island",
        aliases=("PSEG LI", "PSEG-LI", "Long Island Power Authority operations"),
        is_integrated_generator=False,
        # PSEG LI is a wholly-owned subsidiary of Public Service Enterprise
        # Group Inc (parent CIK 0000788784) operating LIPA's T&D system
        # under contract.  PSEG LI itself doesn't file separate 10-Ks, so
        # we route to the parent for revenue.  Note: parent revenue covers
        # NJ utility + power gen + LI ops combined; LI-only revenue isn't
        # broken out as a separate line.
        cik="0000788784",                       # Public Service Enterprise Group Inc
        foundation_ein=None,                    # resolved by name lookup at runtime
        foundation_name="PSEG Foundation",
        eia_op_ids=(),                          # LIPA is the registered utility, not PSEG LI
        egrid_operator_names=(),                # no owned generation; LIPA contracts
        state_puc_codes=("NY-DPS",),
        notes=("Operates Long Island Power Authority (LIPA) T&D system under "
               "contract.  Pure distributor with no owned generation.  "
               "Revenue is from parent PSEG Inc consolidated 10-K — covers "
               "NJ utility (PSE&G) + LI ops + other; LI segment not "
               "separately reported.  Foundation is the parent-level "
               "PSEG Foundation, not LI-specific."),
    ),
    "pseg": Company(
        name="Public Service Enterprise Group",
        aliases=("PSEG", "PSE&G", "Public Service Electric and Gas",
                 "Public Service Enterprise Group Inc", "PEG"),
        is_integrated_generator=True,
        cik="0000788784",                       # Public Service Enterprise Group Inc
        foundation_ein=None,
        foundation_name="PSEG Foundation",
        eia_op_ids=(15472,),                    # Public Service Elec & Gas Co
        egrid_operator_names=(
            "PSEG Fossil LLC",
            "PSEG Nuclear LLC",
            "Public Service Electric & Gas Co",
        ),
        state_puc_codes=("NJ-BPU",),
        notes=("NJ-based holding company.  Includes PSE&G (regulated NJ "
               "T&D), PSEG Power (merchant generation), and PSEG LI (LIPA "
               "operator).  Revenue figure is consolidated."),
    ),

    # ── Northeast / Mid-Atlantic Tristate ─────────────────────────────────

    "exelon": Company(
        name="Exelon Corporation",
        aliases=("Exelon", "EXC", "ComEd", "PECO", "BGE", "Pepco",
                 "Atlantic City Electric", "Delmarva Power"),
        is_integrated_generator=False,
        cik="0001109357",
        foundation_ein=None,
        foundation_name="Exelon Foundation",
        eia_op_ids=(4110, 14940, 1311, 14127, 13407, 4922),
        egrid_operator_names=(),
        state_puc_codes=("IL-ICC", "PA-PUC", "MD-PSC", "DC-PSC", "NJ-BPU", "DE-PSC"),
        notes=("Largest US T&D-only holding co after 2022 Constellation "
               "spinoff.  Owns ComEd (IL), PECO (PA), BGE (MD), Pepco/ACE/"
               "Delmarva (DC/NJ/DE/MD).  No owned generation."),
    ),

    "fenoc": Company(
        name="FirstEnergy",
        aliases=("FirstEnergy Corp", "FE", "JCP&L", "Jersey Central Power & Light",
                 "Met-Ed", "Penelec", "Penn Power", "West Penn Power",
                 "Ohio Edison", "Toledo Edison", "Cleveland Electric",
                 "Mon Power", "Potomac Edison"),
        is_integrated_generator=False,
        cik="0001031296",
        foundation_ein=None,
        foundation_name="FirstEnergy Foundation",
        eia_op_ids=(13998, 18642, 14015, 14020),
        egrid_operator_names=(),
        state_puc_codes=("OH-PUCO", "PA-PUC", "NJ-BPU", "WV-PSC", "MD-PSC", "NY-PSC"),
        notes=("OH-headquartered T&D holding co.  10 operating companies "
               "across OH, PA, NJ, WV, MD, NY (including JCP&L in NJ "
               "tristate area).  Sold all merchant generation in 2020."),
    ),

    "central_hudson": Company(
        name="Central Hudson Gas & Electric",
        aliases=("Central Hudson", "CH Energy Group", "CenHud", "CHG&E"),
        is_integrated_generator=False,
        # Central Hudson's parent is Fortis Inc (Canadian). The US sub
        # historically filed 10-Ks but stopped after Fortis acquisition in
        # 2013.  AI fallback will hit Fortis annual report or NY DPS filings.
        cik=None,
        foundation_ein=None,
        foundation_name="Central Hudson Gas & Electric Corporation Foundation",
        eia_op_ids=(3266,),
        egrid_operator_names=(),
        state_puc_codes=("NY-PSC",),
        notes=("Hudson Valley utility serving ~300K customers in 8 NY "
               "counties.  Subsidiary of Fortis Inc (Canada)."),
    ),

    "nysed": Company(
        name="New York State Electric & Gas",
        aliases=("NYSEG", "NY State Electric and Gas", "Avangrid Networks NY"),
        is_integrated_generator=False,
        # NYSEG parent is Avangrid (subsidiary of Iberdrola).
        cik="0001601072",   # Avangrid Inc
        foundation_ein=None,
        foundation_name="Avangrid Foundation",
        eia_op_ids=(13573,),
        egrid_operator_names=(),
        state_puc_codes=("NY-PSC",),
        notes=("Upstate NY utility.  Subsidiary of Avangrid Inc, which is "
               "majority-owned by Spain's Iberdrola.  Revenue routed via "
               "Avangrid 10-K (consolidated NY + ME + CT)."),
    ),

    "rochester_gas_electric": Company(
        name="Rochester Gas and Electric",
        aliases=("RG&E", "Rochester Gas & Electric", "Avangrid RG&E"),
        is_integrated_generator=False,
        cik="0001601072",   # Avangrid Inc parent
        foundation_ein=None,
        foundation_name="Avangrid Foundation",
        eia_op_ids=(16387,),
        egrid_operator_names=(),
        state_puc_codes=("NY-PSC",),
        notes=("Western NY utility serving Rochester area.  Subsidiary "
               "of Avangrid Inc (Iberdrola).  Revenue from Avangrid "
               "consolidated 10-K."),
    ),

    "orange_rockland": Company(
        name="Orange and Rockland Utilities",
        aliases=("O&R", "Orange & Rockland", "ORU"),
        is_integrated_generator=False,
        # O&R is a subsidiary of Con Edison, Inc.  Files with SEC as part
        # of the Con Edison consolidated 10-K.
        cik="0001047862",   # Con Edison parent
        foundation_ein=None,
        foundation_name="Consolidated Edison Foundation",   # parent foundation
        eia_op_ids=(49328,),
        egrid_operator_names=(),
        state_puc_codes=("NY-PSC", "NJ-BPU"),
        notes=("Subsidiary of Con Edison serving NY counties of Orange/"
               "Rockland and parts of NJ/PA.  Revenue is consolidated "
               "into Con Edison 10-K, not separately reported."),
    ),

    # New England

    "national_grid_ne": Company(
        name="National Grid New England",
        aliases=("National Grid Massachusetts", "Mass Electric",
                 "Narragansett Electric", "National Grid RI"),
        is_integrated_generator=False,
        cik=None,   # subsidiaries don't file current XBRL
        foundation_ein=None,
        foundation_name="National Grid Foundation",
        eia_op_ids=(13501, 40209, 13260),
        egrid_operator_names=(),
        state_puc_codes=("MA-DPU", "RI-PUC"),
        notes=("New England operations of National Grid USA: Mass "
               "Electric (MA) + Narragansett Electric (RI). Subsidiary of "
               "UK-listed National Grid plc."),
    ),

    "united_illuminating": Company(
        name="United Illuminating",
        aliases=("UI", "United Illuminating Company", "Avangrid UI"),
        is_integrated_generator=False,
        cik="0001601072",   # Avangrid Inc parent
        foundation_ein=None,
        foundation_name="Avangrid Foundation",
        eia_op_ids=(19436,),
        egrid_operator_names=(),
        state_puc_codes=("CT-PURA",),
        notes=("Connecticut utility serving New Haven area, ~340K "
               "customers.  Subsidiary of Avangrid Inc."),
    ),

    "central_maine_power": Company(
        name="Central Maine Power",
        aliases=("CMP", "Avangrid CMP"),
        is_integrated_generator=False,
        cik="0001601072",   # Avangrid Inc parent
        foundation_ein=None,
        foundation_name="Avangrid Foundation",
        eia_op_ids=(3266,),
        egrid_operator_names=(),
        state_puc_codes=("ME-PUC",),
        notes=("Largest electric utility in Maine, ~640K customers. "
               "Subsidiary of Avangrid Inc (Iberdrola)."),
    ),

    "unitil": Company(
        name="Unitil Corporation",
        aliases=("Unitil", "UTL", "Fitchburg Gas and Electric"),
        is_integrated_generator=False,
        cik="0000755001",
        foundation_ein=None,
        foundation_name="Unitil Charitable Foundation",
        eia_op_ids=(7251, 13573),
        egrid_operator_names=(),
        state_puc_codes=("NH-PUC", "MA-DPU", "ME-PUC"),
        notes=("Small T&D utility serving NH, MA, and ME (~108K "
               "customers).  Foundation is small; CSR data limited."),
    ),

    "versant": Company(
        name="Versant Power",
        aliases=("Versant", "Bangor Hydro", "Maine Public Service"),
        is_integrated_generator=False,
        # Versant's parent is ENMAX (Calgary, Canada). No US 10-K.
        cik=None,
        foundation_ein=None,
        foundation_name=None,
        eia_op_ids=(1167, 11522),
        egrid_operator_names=(),
        state_puc_codes=("ME-PUC",),
        notes=("Northern Maine utility (Bangor Hydro + Maine Public "
               "Service consolidated 2020). Owned by Calgary's ENMAX. "
               "Limited US disclosure."),
    ),

    # NY public power / authorities

    "lipa": Company(
        name="Long Island Power Authority",
        aliases=("LIPA", "Long Island Power"),
        is_integrated_generator=False,
        # LIPA is a NY State public-benefit corporation; files with NY
        # Comptroller, not SEC. No 10-K available.
        cik=None,
        foundation_ein=None,
        foundation_name=None,   # public authority, no foundation
        eia_op_ids=(11243,),
        egrid_operator_names=(),
        state_puc_codes=("NY-PSC",),
        notes=("NY State public-benefit corporation that owns Long "
               "Island's electric T&D system.  Operates via PSEG LI "
               "service contract.  No 10-K filings.  Annual report "
               "available from LIPA website only."),
    ),

    "nypa": Company(
        name="New York Power Authority",
        aliases=("NYPA", "Power Authority of the State of New York", "PASNY"),
        is_integrated_generator=True,
        # NYPA is a NY State public-benefit corporation; no SEC filings.
        cik=None,
        foundation_ein=None,
        foundation_name=None,
        eia_op_ids=(13407,),
        egrid_operator_names=("Power Authority of the State of New York",
                              "New York Power Authority"),
        state_puc_codes=("NY-PSC",),
        notes=("Largest US state-owned electric utility.  Operates "
               "Niagara, St. Lawrence, and other hydro generation, plus "
               "transmission.  Public-benefit corp — no SEC filings.  "
               "Revenue from NY State Comptroller filings."),
    ),

    # PJM-region (PA / NJ adjacent)

    "ppl": Company(
        name="PPL Corporation",
        aliases=("PPL", "PPL Electric Utilities", "Talen Energy",
                 "Louisville Gas & Electric", "Kentucky Utilities"),
        is_integrated_generator=False,
        cik="0000922224",
        foundation_ein=None,
        foundation_name="PPL Foundation",
        eia_op_ids=(15296, 11249, 11241),
        egrid_operator_names=(),
        state_puc_codes=("PA-PUC", "KY-PSC", "RI-PUC"),
        notes=("Allentown PA-based holding co.  Owns PPL Electric (PA), "
               "Rhode Island Energy (acquired 2022), LG&E and KU (KY).  "
               "Sold UK Western Power Distribution in 2021."),
    ),

    "rhode_island_energy": Company(
        name="Rhode Island Energy",
        aliases=("RI Energy", "Narragansett Electric (post-2022)"),
        is_integrated_generator=False,
        cik="0000922224",   # PPL Corp parent
        foundation_ein=None,
        foundation_name="PPL Foundation",
        eia_op_ids=(13260,),
        egrid_operator_names=(),
        state_puc_codes=("RI-PUC",),
        notes=("Acquired by PPL from National Grid in May 2022. Serves "
               "~770K electric and ~280K gas customers in RI.  Revenue "
               "consolidated into PPL 10-K."),
    ),

    # Mid-Atlantic

    "dominion": Company(
        name="Dominion Energy",
        aliases=("Dominion", "D", "Dominion Resources",
                 "Virginia Electric and Power"),
        is_integrated_generator=True,
        cik="0000715957",
        foundation_ein=None,
        foundation_name="Dominion Energy Charitable Foundation",
        eia_op_ids=(19876, 14328),
        egrid_operator_names=(
            "Virginia Electric & Power Company",
            "Dominion Energy South Carolina, Inc.",
        ),
        state_puc_codes=("VA-SCC", "NC-NCUC", "SC-PSC"),
        notes=("VA-based integrated utility.  Owns Virginia Electric "
               "(VA + NC) and Dominion Energy South Carolina."),
    ),

    "aep": Company(
        name="American Electric Power",
        aliases=("AEP", "American Electric Power Co", "AEP Ohio",
                 "AEP Texas", "Appalachian Power", "Indiana Michigan Power",
                 "Kentucky Power", "Public Service Co of Oklahoma"),
        is_integrated_generator=True,
        cik="0000004904",
        foundation_ein=None,
        foundation_name="American Electric Power Foundation",
        eia_op_ids=(715, 16572, 814, 9267, 11241, 15470),
        egrid_operator_names=(
            "Appalachian Power Co",
            "Indiana Michigan Power Co",
            "Kentucky Power Co",
            "Public Service Co of Oklahoma",
            "Southwestern Electric Power Co",
        ),
        state_puc_codes=("OH-PUCO", "TX-PUC", "VA-SCC", "WV-PSC", "IN-IURC",
                         "MI-PSC", "KY-PSC", "OK-OCC", "AR-PSC", "LA-PSC", "TN-TRA"),
        notes=("OH-based integrated generator across 11 states. One of "
               "the largest US power generators by capacity."),
    ),

    "edison_intl": Company(
        name="Edison International",
        aliases=("Edison Intl", "EIX", "Southern California Edison", "SCE"),
        is_integrated_generator=False,
        cik="0000827052",
        foundation_ein=None,
        foundation_name="Edison International Foundation",
        eia_op_ids=(17609,),
        egrid_operator_names=(),
        state_puc_codes=("CA-CPUC",),
        notes=("CA-based.  Owns Southern California Edison (largest CA "
               "utility by customers).  Largely T&D after divestiture."),
    ),

    "sempra": Company(
        name="Sempra Energy",
        aliases=("Sempra", "SRE", "San Diego Gas & Electric", "SDG&E",
                 "Southern California Gas", "SoCalGas"),
        is_integrated_generator=True,
        cik="0001032208",
        foundation_ein=None,
        foundation_name="Sempra Energy Foundation",
        eia_op_ids=(16451, 17609),
        egrid_operator_names=("San Diego Gas & Electric Co",),
        state_puc_codes=("CA-CPUC",),
        notes=("San Diego-based.  Owns SDG&E and SoCalGas (largest US "
               "gas utility by customers)."),
    ),

    "xcel": Company(
        name="Xcel Energy",
        aliases=("Xcel", "XEL", "Northern States Power", "Public Service Co of Colorado",
                 "Southwestern Public Service"),
        is_integrated_generator=True,
        cik="0000072903",
        foundation_ein=None,
        foundation_name="Xcel Energy Foundation",
        eia_op_ids=(13781, 15466, 17716),
        egrid_operator_names=(
            "Northern States Power Co - Minnesota",
            "Public Service Co of Colorado",
            "Southwestern Public Service Co",
        ),
        state_puc_codes=("MN-PUC", "CO-PUC", "TX-PUC", "WI-PSC", "ND-PSC",
                         "SD-PUC", "NM-PRC", "MI-PSC"),
        notes=("Minneapolis-based integrated utility across 8 states."),
    ),

    "nextera": Company(
        name="NextEra Energy",
        aliases=("NextEra", "NEE", "Florida Power & Light", "FPL", "Gulf Power"),
        is_integrated_generator=True,
        cik="0000753308",
        foundation_ein=None,
        foundation_name="NextEra Energy Foundation",
        eia_op_ids=(6452, 5416),
        egrid_operator_names=(
            "Florida Power & Light Co",
            "Gulf Power Co",
        ),
        state_puc_codes=("FL-PSC",),
        notes=("FL-based, owns FPL (largest US utility by customers and "
               "renewable capacity).  Massive renewable generation fleet "
               "via NextEra Energy Resources."),
    ),

    "ameren": Company(
        name="Ameren Corporation",
        aliases=("Ameren", "AEE", "Union Electric", "Ameren Missouri", "Ameren Illinois"),
        is_integrated_generator=True,
        cik="0001002910",
        foundation_ein=None,
        foundation_name="Ameren Corporation Charitable Trust",
        eia_op_ids=(19436, 813),
        egrid_operator_names=(
            "Union Electric Co",
            "Ameren Illinois Co",
        ),
        state_puc_codes=("MO-PSC", "IL-ICC"),
        notes=("St. Louis-based.  Owns Ameren Missouri (integrated) and "
               "Ameren Illinois (T&D-only post-2022)."),
    ),

    "wec": Company(
        name="WEC Energy Group",
        aliases=("WEC", "Wisconsin Energy", "Wisconsin Electric", "We Energies",
                 "Peoples Gas", "North Shore Gas", "Wisconsin Public Service"),
        is_integrated_generator=True,
        cik="0000783325",
        foundation_ein=None,
        foundation_name="WEC Energy Group Foundation",
        eia_op_ids=(20860, 4254, 4271),
        egrid_operator_names=(
            "Wisconsin Electric Power Co",
            "Wisconsin Public Service Corp",
        ),
        state_puc_codes=("WI-PSC", "IL-ICC", "MI-PSC", "MN-PUC"),
        notes=("WI-based.  Owns We Energies (WI) and Peoples Gas (Chicago)."),
    ),

    "dte": Company(
        name="DTE Energy",
        aliases=("DTE", "DTE Energy Co", "Detroit Edison"),
        is_integrated_generator=True,
        cik="0000936340",
        foundation_ein=None,
        foundation_name="DTE Energy Foundation",
        eia_op_ids=(5109,),
        egrid_operator_names=("DTE Electric Company",),
        state_puc_codes=("MI-PSC",),
        notes=("MI-based integrated utility (DTE Electric + DTE Gas). "
               "Detroit-area service territory."),
    ),

    "cms": Company(
        name="CMS Energy",
        aliases=("CMS", "Consumers Energy"),
        is_integrated_generator=True,
        cik="0000811156",
        foundation_ein=None,
        foundation_name="Consumers Energy Foundation",
        eia_op_ids=(4254,),
        egrid_operator_names=("Consumers Energy Co",),
        state_puc_codes=("MI-PSC",),
        notes=("MI-based.  Owns Consumers Energy (Lower Michigan)."),
    ),
}


def resolve(name_or_alias: str) -> Optional[Company]:
    """Resolve a free-text company name to a Company record.  Case-insensitive,
    matches both canonical names and aliases.  Loose matching is conservative:
    we require either an exact match or a very high-coverage substring (the
    needle covers ≥60% of the candidate name OR vice versa) so that "Some
    Unknown Utility" doesn't match "Southern Company" via "South"."""
    needle = name_or_alias.strip().lower()
    if not needle:
        return None
    # Exact name or alias match
    for c in REGISTRY.values():
        if c.name.lower() == needle:
            return c
        if any(a.lower() == needle for a in c.aliases):
            return c
    # Conservative loose match: require dominant overlap
    for c in REGISTRY.values():
        all_names = (c.name,) + c.aliases
        for n in all_names:
            n_low = n.lower()
            if needle in n_low and len(needle) >= max(6, 0.6 * len(n_low)):
                return c
            if n_low in needle and len(n_low) >= max(6, 0.6 * len(needle)):
                return c
    return None
