"""Branche -> relevante Bewertungs-Multiples fuer die CRV-Berechnung.

Ersetzt die frueheren, auf 23 Symbole zugeschnittenen COMPANY_SECTORS /
BRANCH_MULTIPLES_MAP in agent/AgentAction.py: statt eigener, hardcodierter
Branchen-Strings nutzt dieses Modul yfinances Standard-Taxonomie (11 Sektoren,
~130 Industrien - stabil, unabhaengig davon, welche Ticker gerade abgefragt
werden), damit JEDES NYSE/NASDAQ-Symbol eine Zuordnung bekommt, nicht nur
eine kuratierte Liste.

Freigabe-Status: Vom Betreiber fachlich geprueft und freigegeben; seither
aktive Quelle in AgentAction.calculate_crv_by_sector_multiples (die
frueheren COMPANY_SECTORS/BRANCH_MULTIPLES_MAP wurden entfernt).

Herleitungslogik je Multiple (fuer die Review-Tabelle):
- EV_Sales: kapitalintensive/wachstumsstarke Geschaeftsmodelle ohne
  (verlaessliche) Gewinne - Umsatz ist die stabilste Bezugsgroesse.
- EV_EBIT / EV_EBITDA: operatives Ergebnis vor Kapitalstruktur-Effekten -
  Standard fuer Industrien mit unterschiedlicher Verschuldung/Abschreibung.
- Price_Book / Price_TangibleBookValue: bilanzlastige Geschaeftsmodelle
  (Finanzwerte, Versicherungen, vorumsatzstarke Biotechs) - Buchwert als
  Bewertungsanker statt Ertrag.
- Price_Sales: wie EV_Sales, aber wenn Nettoschulden/-cash das Bild verzerren
  wuerden (z. B. sehr unterschiedliche Kapitalausstattung je nach Finanzierungsrunde).
- Price_EBIT: aehnlich EV_EBIT, bevorzugt wenn Multiple auf Aktionaersebene
  (statt Unternehmenswert) ueblich ist (z. B. Konsumgueter/Einzelhandel).
- Price_NetCurrentAssets: Substanzwert-/Net-Net-Ansatz - Bodenbewertung ueber
  Nettoumlaufvermoegen, klassisch bei vorumsatzstarken/Cash-brennenden Biotechs.
- Price_OperatingCashflow: wenn GuV durch nicht-zahlungswirksame Posten
  (Abschreibungen, Rueckstellungen) verzerrt ist (Gesundheitswesen, Versorger).
- Price_FreeCashflow: reife, cash-generative Geschaeftsmodelle (Software,
  Konsumgueter, Telekommunikation, Medien).
"""

# Sicherer Rueckfall, falls weder Industrie- noch Sektor-Override greift
# (z. B. Sektor fehlt bei yfinance) - liquiditaetsnahe Standard-Kombination.
GLOBAL_FALLBACK_MULTIPLES = ["EV_EBITDA", "Price_FreeCashflow"]

# yfinances 11 Standard-Sektoren -> Default-Multiples, falls keine
# spezifischere Industrie-Zuordnung existiert.
SECTOR_DEFAULT_MULTIPLES: dict[str, list[str]] = {
    "Technology": ["Price_FreeCashflow", "EV_EBITDA"],
    "Healthcare": ["EV_EBIT", "Price_OperatingCashflow"],
    "Financial Services": ["Price_Book", "Price_TangibleBookValue"],
    "Consumer Cyclical": ["Price_EBIT", "Price_FreeCashflow"],
    "Consumer Defensive": ["Price_FreeCashflow", "EV_EBITDA"],
    "Industrials": ["EV_EBIT", "EV_EBITDA"],
    "Energy": ["EV_Sales", "EV_EBITDA"],
    "Basic Materials": ["EV_EBITDA", "Price_TangibleBookValue"],
    "Communication Services": ["EV_EBITDA", "Price_FreeCashflow"],
    "Utilities": ["EV_EBITDA", "Price_OperatingCashflow"],
    "Real Estate": ["Price_Book", "Price_OperatingCashflow"],
}

# Industrie-Overrides (yfinance "industry"-Feld) - greifen VOR dem
# Sektor-Default, wenn vorhanden. Deckt yfinances oeffentliche Standard-
# Taxonomie ab (Stand 2026); neue/seltene Industrie-Strings fallen einfach
# auf den Sektor-Default zurueck, kein Fehler.
INDUSTRY_MULTIPLES_OVERRIDES: dict[str, list[str]] = {
    # --- Technology ---
    "Information Technology Services": ["Price_FreeCashflow", "EV_EBITDA"],
    "Software - Application": ["Price_FreeCashflow", "EV_Sales"],
    "Software - Infrastructure": ["Price_FreeCashflow", "EV_Sales"],
    "Communication Equipment": ["EV_EBITDA", "EV_EBIT"],
    "Computer Hardware": ["EV_EBITDA", "Price_FreeCashflow"],
    "Consumer Electronics": ["EV_EBITDA", "Price_FreeCashflow"],
    "Electronic Components": ["EV_EBITDA", "EV_EBIT"],
    "Electronics & Computer Distribution": ["EV_EBITDA", "Price_TangibleBookValue"],
    "Scientific & Technical Instruments": ["EV_EBIT", "Price_FreeCashflow"],
    "Semiconductor Equipment & Materials": ["EV_EBITDA", "EV_EBIT"],
    "Semiconductors": ["EV_EBITDA", "EV_EBIT"],
    "Solar": ["EV_Sales", "EV_EBITDA"],

    # --- Healthcare ---
    "Biotechnology": ["Price_TangibleBookValue", "Price_NetCurrentAssets"],
    "Drug Manufacturers - General": ["EV_EBIT", "Price_FreeCashflow"],
    "Drug Manufacturers - Specialty & Generic": ["Price_TangibleBookValue", "Price_NetCurrentAssets"],
    "Healthcare Plans": ["Price_OperatingCashflow", "Price_Book"],
    "Medical Care Facilities": ["EV_EBITDA", "Price_OperatingCashflow"],
    "Medical Devices": ["EV_EBIT", "Price_FreeCashflow"],
    "Medical Distribution": ["EV_EBITDA", "Price_OperatingCashflow"],
    "Medical Instruments & Supplies": ["EV_EBIT", "Price_FreeCashflow"],
    "Diagnostics & Research": ["EV_EBIT", "Price_OperatingCashflow"],
    "Health Information Services": ["Price_FreeCashflow", "EV_Sales"],
    "Pharmaceutical Retailers": ["EV_EBITDA", "Price_OperatingCashflow"],

    # --- Financial Services ---
    "Asset Management": ["Price_Book", "Price_FreeCashflow"],
    "Banks - Diversified": ["Price_Book", "Price_TangibleBookValue"],
    "Banks - Regional": ["Price_Book", "Price_TangibleBookValue"],
    "Capital Markets": ["Price_Book", "Price_OperatingCashflow"],
    # yfinance sortiert Zahlungsabwickler (PayPal, Visa, Mastercard) hier ein —
    # cash-generative Geschäftsmodelle mit hohem Goodwill, Buchwert-Multiples
    # verzerren (gleiche Begründung wie is_financial_sector in DataLoader.py).
    "Credit Services": ["Price_FreeCashflow", "EV_EBITDA"],
    "Financial Conglomerates": ["Price_Book", "Price_TangibleBookValue"],
    "Financial Data & Stock Exchanges": ["Price_FreeCashflow", "EV_EBITDA"],
    "Insurance - Diversified": ["Price_Book", "Price_TangibleBookValue"],
    "Insurance - Life": ["Price_Book", "Price_TangibleBookValue"],
    "Insurance - Property & Casualty": ["Price_Book", "Price_TangibleBookValue"],
    "Insurance - Reinsurance": ["Price_Book", "Price_TangibleBookValue"],
    "Insurance - Specialty": ["Price_Book", "Price_TangibleBookValue"],
    "Insurance Brokers": ["Price_FreeCashflow", "EV_EBITDA"],
    "Mortgage Finance": ["Price_Book", "Price_TangibleBookValue"],
    "Shell Companies": ["Price_TangibleBookValue", "Price_NetCurrentAssets"],

    # --- Consumer Cyclical ---
    "Apparel Manufacturing": ["Price_EBIT", "Price_FreeCashflow"],
    "Apparel Retail": ["Price_EBIT", "Price_FreeCashflow"],
    "Auto Manufacturers": ["EV_Sales", "EV_EBITDA"],
    "Auto Parts": ["EV_EBIT", "EV_EBITDA"],
    "Auto & Truck Dealerships": ["EV_EBITDA", "Price_TangibleBookValue"],
    "Department Stores": ["Price_EBIT", "Price_TangibleBookValue"],
    "Footwear & Accessories": ["Price_EBIT", "Price_FreeCashflow"],
    "Furnishings, Fixtures & Appliances": ["Price_EBIT", "EV_EBITDA"],
    "Gambling": ["EV_EBITDA", "Price_FreeCashflow"],
    "Home Improvement Retail": ["Price_EBIT", "Price_FreeCashflow"],
    "Internet Retail": ["EV_Sales", "Price_FreeCashflow"],
    "Leisure": ["EV_EBITDA", "Price_FreeCashflow"],
    "Lodging": ["EV_EBITDA", "Price_OperatingCashflow"],
    "Luxury Goods": ["Price_EBIT", "Price_FreeCashflow"],
    "Packaging & Containers": ["EV_EBITDA", "EV_EBIT"],
    "Personal Services": ["Price_EBIT", "Price_FreeCashflow"],
    "Residential Construction": ["Price_Book", "Price_EBIT"],
    "Resorts & Casinos": ["EV_EBITDA", "Price_OperatingCashflow"],
    "Restaurants": ["EV_EBITDA", "Price_FreeCashflow"],
    "Specialty Retail": ["Price_EBIT", "Price_FreeCashflow"],
    "Textile Manufacturing": ["Price_EBIT", "EV_EBITDA"],
    "Travel Services": ["EV_Sales", "Price_FreeCashflow"],

    # --- Consumer Defensive ---
    "Beverages - Brewers": ["EV_EBITDA", "Price_FreeCashflow"],
    "Beverages - Non-Alcoholic": ["EV_EBITDA", "Price_FreeCashflow"],
    "Beverages - Wineries & Distilleries": ["EV_EBITDA", "Price_FreeCashflow"],
    "Confectioners": ["EV_EBITDA", "Price_FreeCashflow"],
    "Discount Stores": ["Price_EBIT", "Price_FreeCashflow"],
    "Education & Training Services": ["Price_FreeCashflow", "EV_EBITDA"],
    "Farm Products": ["EV_EBITDA", "Price_TangibleBookValue"],
    "Food Distribution": ["EV_EBITDA", "Price_OperatingCashflow"],
    "Grocery Stores": ["EV_EBITDA", "Price_FreeCashflow"],
    "Household & Personal Products": ["EV_EBITDA", "Price_FreeCashflow"],
    "Packaged Foods": ["EV_EBITDA", "Price_FreeCashflow"],
    "Tobacco": ["Price_FreeCashflow", "EV_EBITDA", "Price_OperatingCashflow"],

    # --- Industrials ---
    "Aerospace & Defense": ["EV_EBIT", "Price_FreeCashflow"],
    "Airlines": ["EV_EBITDA", "Price_OperatingCashflow"],
    "Airports & Air Services": ["EV_EBITDA", "Price_OperatingCashflow"],
    "Building Products & Equipment": ["EV_EBIT", "EV_EBITDA"],
    "Business Equipment & Supplies": ["EV_EBIT", "Price_FreeCashflow"],
    "Conglomerates": ["EV_EBITDA", "Price_Book"],
    "Consulting Services": ["Price_FreeCashflow", "EV_EBITDA"],
    "Electrical Equipment & Parts": ["EV_EBIT", "EV_EBITDA"],
    "Engineering & Construction": ["EV_EBIT", "Price_OperatingCashflow"],
    "Farm & Heavy Construction Machinery": ["EV_EBIT", "EV_EBITDA"],
    "Industrial Distribution": ["EV_EBITDA", "Price_TangibleBookValue"],
    "Infrastructure Operations": ["EV_EBITDA", "Price_OperatingCashflow"],
    "Integrated Freight & Logistics": ["EV_EBITDA", "Price_TangibleBookValue"],
    "Marine Shipping": ["EV_EBITDA", "Price_TangibleBookValue"],
    "Metal Fabrication": ["EV_EBIT", "EV_EBITDA"],
    "Pollution & Treatment Controls": ["EV_EBIT", "EV_EBITDA"],
    "Railroads": ["EV_EBITDA", "Price_FreeCashflow"],
    "Rental & Leasing Services": ["EV_EBITDA", "Price_Book"],
    "Security & Protection Services": ["EV_EBIT", "Price_FreeCashflow"],
    "Specialty Business Services": ["Price_FreeCashflow", "EV_EBITDA"],
    "Specialty Industrial Machinery": ["EV_EBIT", "EV_EBITDA"],
    "Staffing & Employment Services": ["EV_EBIT", "Price_FreeCashflow"],
    "Tools & Accessories": ["EV_EBIT", "EV_EBITDA"],
    "Trucking": ["EV_EBITDA", "Price_TangibleBookValue"],
    "Waste Management": ["EV_EBITDA", "Price_OperatingCashflow"],

    # --- Energy ---
    "Oil & Gas Drilling": ["EV_EBITDA", "Price_TangibleBookValue"],
    "Oil & Gas E&P": ["EV_EBITDA", "Price_TangibleBookValue"],
    "Oil & Gas Equipment & Services": ["EV_EBITDA", "EV_EBIT"],
    "Oil & Gas Integrated": ["EV_EBITDA", "Price_FreeCashflow"],
    "Oil & Gas Midstream": ["EV_EBITDA", "Price_FreeCashflow"],
    "Oil & Gas Refining & Marketing": ["EV_EBITDA", "Price_OperatingCashflow"],
    "Thermal Coal": ["EV_EBITDA", "Price_TangibleBookValue"],
    "Uranium": ["EV_Sales", "Price_TangibleBookValue"],

    # --- Basic Materials ---
    "Agricultural Inputs": ["EV_EBITDA", "EV_EBIT"],
    "Aluminum": ["EV_EBITDA", "Price_TangibleBookValue"],
    "Building Materials": ["EV_EBIT", "EV_EBITDA"],
    "Chemicals": ["EV_EBITDA", "EV_EBIT"],
    "Coking Coal": ["EV_EBITDA", "Price_TangibleBookValue"],
    "Copper": ["EV_EBITDA", "Price_TangibleBookValue"],
    "Gold": ["EV_EBITDA", "Price_TangibleBookValue"],
    "Lumber & Wood Production": ["EV_EBITDA", "Price_TangibleBookValue"],
    "Other Industrial Metals & Mining": ["EV_EBITDA", "Price_TangibleBookValue"],
    "Other Precious Metals & Mining": ["EV_EBITDA", "Price_TangibleBookValue"],
    "Paper & Paper Products": ["EV_EBITDA", "Price_TangibleBookValue"],
    "Silver": ["EV_EBITDA", "Price_TangibleBookValue"],
    "Specialty Chemicals": ["EV_EBITDA", "EV_EBIT"],
    "Steel": ["EV_EBITDA", "Price_TangibleBookValue"],

    # --- Communication Services ---
    "Advertising Agencies": ["Price_FreeCashflow", "EV_EBITDA"],
    "Broadcasting": ["EV_EBITDA", "Price_FreeCashflow"],
    "Electronic Gaming & Multimedia": ["Price_FreeCashflow", "EV_EBIT"],
    "Entertainment": ["EV_EBITDA", "Price_FreeCashflow"],
    "Internet Content & Information": ["EV_Sales", "Price_Sales"],
    "Publishing": ["EV_EBITDA", "Price_FreeCashflow"],
    "Telecom Services": ["EV_EBITDA", "Price_FreeCashflow"],

    # --- Utilities ---
    "Utilities - Diversified": ["EV_EBITDA", "Price_OperatingCashflow"],
    "Utilities - Independent Power Producers": ["EV_EBITDA", "Price_OperatingCashflow"],
    "Utilities - Regulated Electric": ["EV_EBITDA", "Price_Book"],
    "Utilities - Regulated Gas": ["EV_EBITDA", "Price_Book"],
    "Utilities - Regulated Water": ["EV_EBITDA", "Price_Book"],
    "Utilities - Renewable": ["EV_EBITDA", "Price_OperatingCashflow"],

    # --- Real Estate ---
    "Real Estate - Development": ["Price_Book", "Price_TangibleBookValue"],
    "Real Estate - Diversified": ["Price_Book", "Price_OperatingCashflow"],
    "Real Estate Services": ["Price_FreeCashflow", "EV_EBITDA"],
    "REIT - Diversified": ["Price_Book", "Price_OperatingCashflow"],
    "REIT - Healthcare Facilities": ["Price_Book", "Price_OperatingCashflow"],
    "REIT - Hotel & Motel": ["Price_Book", "Price_OperatingCashflow"],
    "REIT - Industrial": ["Price_Book", "Price_OperatingCashflow"],
    "REIT - Mortgage": ["Price_Book", "Price_TangibleBookValue"],
    "REIT - Office": ["Price_Book", "Price_OperatingCashflow"],
    "REIT - Residential": ["Price_Book", "Price_OperatingCashflow"],
    "REIT - Retail": ["Price_Book", "Price_OperatingCashflow"],
    "REIT - Specialty": ["Price_Book", "Price_OperatingCashflow"],
}


def resolve_multiples(sector: str | None, industry: str | None) -> tuple[list[str], str]:
    """Loest die 2 (Ausnahme Tabak: 3) relevanten Multiples fuer ein Symbol
    auf. Reihenfolge: Industrie-Override > Sektor-Default > globaler
    Fallback. Gibt (multiples, source) zurueck, source in
    {"industry", "sector", "fallback"} - nur fuers Debugging/Anzeige."""
    if industry and industry in INDUSTRY_MULTIPLES_OVERRIDES:
        return INDUSTRY_MULTIPLES_OVERRIDES[industry], "industry"
    if sector and sector in SECTOR_DEFAULT_MULTIPLES:
        return SECTOR_DEFAULT_MULTIPLES[sector], "sector"
    return GLOBAL_FALLBACK_MULTIPLES, "fallback"
