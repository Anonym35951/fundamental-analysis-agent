export type MetricUnit = "%" | "currency" | "ratio";

export type MetricConfig = {
  label: string;
  unit: MetricUnit;
  decimals?: number;
  /** Most "%"-unit metrics already return a pre-scaled 0-100 value from
   * Model.py (e.g. cashflow_margin does `* 100` itself). A few methods
   * (calculate_roe, calculate_cash_to_market_cap) intentionally return the
   * raw 0-1 fraction instead — that scale is relied on elsewhere (e.g.
   * AgentAction.py's `roe >= 0.15` / `cash_ratio >= 0.20` threshold checks),
   * so it can't be changed in the backend. Set scale: 100 here for those
   * metrics so the frontend display still shows the correct percentage. */
  scale?: number;
  /** Short, user-facing explanation of what the metric measures. Drives the
   * InfoTooltip icon — metrics without this render no icon at all rather
   * than an empty popover. */
  description?: string;
  /** The calculation formula as implemented in agent/Model.py /
   * agent/DataLoader.py / agent/AgentAction.py, written for end users
   * rather than copied verbatim from the docstrings. */
  formula?: string;
};

export const METRICS_CONFIG: Record<string, MetricConfig> = {
  // ---------------------------------------------------------------------
  // Vollanalyse (hardcoded AgentAction criteria) — these are the literal,
  // unprefixed dict keys AgentAction.py returns, e.g. result["dividend_yield"].
  // ---------------------------------------------------------------------
  profit_growth: {
    label: "Profit Growth",
    unit: "%",
    decimals: 2,
    description: "Veränderung des Nettogewinns gegenüber der Vorperiode.",
    formula: "Profit Growth = (Aktueller Nettogewinn / Vorperioden-Nettogewinn − 1) × 100",
  },

  dividend_yield: {
    label: "Dividend Yield",
    unit: "%",
    decimals: 2,
    description: "Setzt die gezahlte Dividende ins Verhältnis zum aktuellen Aktienkurs.",
    formula: "Dividendenrendite = Dividende (TTM) / Aktueller Kurs × 100",
  },

  roe: {
    label: "ROE",
    unit: "%",
    decimals: 2,
    scale: 100,
    description: "Eigenkapitalrendite — zeigt, wie profitabel ein Unternehmen mit dem Kapital seiner Aktionäre wirtschaftet.",
    formula: "ROE = Nettogewinn / Eigenkapital",
  },

  cashflow_margin: {
    label: "Cashflow Margin",
    unit: "%",
    decimals: 2,
    description: "Anteil des Umsatzes, der als operativer Cashflow erwirtschaftet wird.",
    formula: "Cashflow-Marge = Operativer Cashflow / Umsatz × 100",
  },

  cash_to_market_cap: {
    label: "Cash / Market Cap",
    unit: "%",
    decimals: 2,
    scale: 100,
    description: "Anteil der Marktkapitalisierung, der durch liquide Mittel gedeckt ist.",
    formula: "Cash / Market Cap = Cash & Cash Equivalents / Marktkapitalisierung",
  },

  payout_ratio: {
    label: "Payout Ratio",
    unit: "%",
    decimals: 2,
    description: "Anteil des Gewinns pro Aktie, der als Dividende ausgeschüttet wird.",
    formula: "Payout Ratio = Dividende je Aktie (TTM) / Gewinn je Aktie × 100",
  },

  roic: {
    label: "ROIC",
    unit: "%",
    decimals: 2,
    description: "Rendite auf das insgesamt investierte Kapital (Eigen- und Fremdkapital).",
    formula: "ROIC = Nettogewinn / Investiertes Kapital × 100, Investiertes Kapital = Eigenkapital + Gesamtverschuldung",
  },

  // -- weitere Vollanalyse-Kennzahlen (analyze_dividend_companies,
  // analyze_average_grower, analyze_wachstumswerte, analyze_typical_cyclers,
  // analyze_cycler_turnarounds) --
  debt_to_equity: {
    label: "Debt-to-Equity",
    unit: "ratio",
    decimals: 2,
    description: "Verschuldungsgrad — Verhältnis von Fremdkapital zu Eigenkapital.",
    formula: "Debt-to-Equity = Gesamtverbindlichkeiten / Eigenkapital",
  },

  kgv: {
    label: "KGV",
    unit: "ratio",
    decimals: 2,
    description: "Kurs-Gewinn-Verhältnis — wie viele Jahresgewinne der aktuelle Kurs kostet.",
    formula: "KGV = Aktueller Kurs / Gewinn je Aktie (EPS)",
  },

  peg_ratio: {
    label: "PEG Ratio",
    unit: "ratio",
    decimals: 2,
    description: "Setzt das KGV ins Verhältnis zum Gewinnwachstum, um Wachstum in die Bewertung einzupreisen.",
    formula: "PEG = KGV / Gewinnwachstum (%), Gewinnwachstum = (Aktueller Nettogewinn − Vorjahres-Nettogewinn) / |Vorjahres-Nettogewinn| × 100",
  },

  ev_to_ebit: {
    label: "EV / EBIT",
    unit: "ratio",
    decimals: 2,
    description: "Unternehmenswert im Verhältnis zum operativen Ergebnis vor Zinsen und Steuern.",
    formula: "EV / EBIT = Enterprise Value / EBIT",
  },

  ev_to_ebitda: {
    label: "EV / EBITDA",
    unit: "ratio",
    decimals: 2,
    description: "Unternehmenswert im Verhältnis zum operativen Ergebnis vor Zinsen, Steuern und Abschreibungen.",
    formula: "EV / EBITDA = (Marktkapitalisierung + Nettoverschuldung) / EBITDA",
  },

  ev_to_sales: {
    label: "EV / Sales",
    unit: "ratio",
    decimals: 2,
    description: "Unternehmenswert im Verhältnis zum Umsatz.",
    formula: "EV / Sales = Enterprise Value / Umsatz",
  },

  free_cashflow: {
    label: "Free Cashflow",
    unit: "currency",
    decimals: 2,
    description: "Frei verfügbarer Cashflow nach Investitionen ins Anlagevermögen.",
    formula: "Free Cashflow = Operativer Cashflow + Capital Expenditures (CapEx negativ verbucht) bzw. direkt aus der Kapitalflussrechnung",
  },

  gross_margin: {
    label: "Gross Margin",
    unit: "%",
    decimals: 2,
    description: "Anteil des Umsatzes, der nach Abzug der Herstellungskosten als Bruttogewinn verbleibt.",
    formula: "Gross Margin = Bruttogewinn / Umsatz × 100",
  },

  interest_coverage_ratio: {
    label: "Interest Coverage Ratio",
    unit: "ratio",
    decimals: 2,
    description: "Zinsdeckungsgrad — wie oft das operative Ergebnis die Zinsaufwendungen deckt.",
    formula: "Interest Coverage = EBIT / Zinsaufwand",
  },

  inventory_to_revenue: {
    label: "Inventory / Revenue",
    unit: "%",
    decimals: 2,
    description: "Anteil des Umsatzes, der in Lagerbeständen gebunden ist.",
    formula: "Inventory / Revenue = Vorräte / Umsatz × 100",
  },

  net_current_assets: {
    label: "Net Current Assets",
    unit: "currency",
    decimals: 0,
    description: "Substanzpuffer aus dem Umlaufvermögen abzüglich kurzfristiger Verbindlichkeiten (Net-Net-Ansatz).",
    formula: "Net Current Assets = Umlaufvermögen − Kurzfristige Verbindlichkeiten",
  },

  net_debt_to_ebitda: {
    label: "Net Debt / EBITDA",
    unit: "ratio",
    decimals: 2,
    description: "Zeigt, wie viele Jahres-EBITDA nötig wären, um die Nettoverschuldung zu tilgen.",
    formula: "Net Debt / EBITDA = Nettoverschuldung / EBITDA",
  },

  price_ebit: {
    label: "P/EBIT (aktuell)",
    unit: "ratio",
    decimals: 2,
    description: "Aktuelles Kurs/EBIT-Multiple im Vergleich zum historischen 10-Jahres-Durchschnitt (Kaufkriterium: aktuell unter Durchschnitt).",
    formula: "P/EBIT = Marktkapitalisierung / EBIT",
  },

  price_tbv: {
    label: "P/TBV (aktuell)",
    unit: "ratio",
    decimals: 2,
    description: "Aktuelles Kurs/TBV-Multiple im Vergleich zum historischen Median (Kaufzone: unter Median und ≤ 1,5).",
    formula: "P/TBV = Aktueller Kurs / Materieller Buchwert je Aktie",
  },

  price_to_fcf: {
    label: "Price / Free Cashflow",
    unit: "ratio",
    decimals: 2,
    description: "Bewertung des Unternehmens im Verhältnis zum frei verfügbaren Cashflow.",
    formula: "Price / FCF = Marktkapitalisierung / Free Cashflow",
  },

  price_to_tangible_book: {
    label: "Price / Tangible Book Value",
    unit: "ratio",
    decimals: 2,
    description: "Aktueller Kurs im Verhältnis zum materiellen Buchwert je Aktie.",
    formula: "P/TBV = Aktueller Kurs / (Eigenkapital − Goodwill) je Aktie",
  },

  reinvested_profit: {
    label: "Reinvested Profit",
    unit: "currency",
    decimals: 0,
    description: "Teil des Nettogewinns, der nicht als Dividende ausgeschüttet, sondern im Unternehmen reinvestiert wird.",
    formula: "Reinvested Profit = Nettogewinn − Gezahlte Dividenden",
  },

  reinvestment_rate: {
    label: "Reinvestitionsrate",
    unit: "%",
    decimals: 2,
    description: "Anteil des Nettogewinns, der reinvestiert statt ausgeschüttet wird.",
    formula: "Reinvestitionsrate = Reinvested Profit / Nettogewinn × 100",
  },

  asset_value_quality: {
    label: "Asset Value Quality",
    unit: "ratio",
    decimals: 0,
    description: "Qualitative Prüfung stiller Reserven (Immobilien, Rohstoffe, Beteiligungen, Patente) anhand der Geschäftsberichte — wird nicht automatisch berechnet.",
  },

  dividend_history: {
    label: "Dividendenhistorie",
    unit: "ratio",
    decimals: 0,
    description: "Anzahl der Jahre mit Dividendenzahlung bzw. -erhöhung sowie die langfristige Wachstumsrate der Dividende.",
    formula: "Erfasst u.a. Jahre mit Dividende, Jahre mit Erhöhung und die Dividenden-CAGR (bis zu 30 Jahre)",
  },

  earnings_growth_vs_inflation: {
    label: "Gewinnwachstum vs. Inflation",
    unit: "%",
    decimals: 2,
    description: "Vergleicht das jährliche/quartalsweise Gewinnwachstum mit der Inflationsrate, um reales Wachstum zu prüfen.",
    formula: "Vergleich von AAGR/AQGR (Gewinnwachstum) mit der CPI-basierten Inflationsrate (FRED)",
  },

  // ---------------------------------------------------------------------
  // Custom-Analysis-Katalog (api/services/metric_catalog.py) — these use the
  // catalog's own "calculate_"/"analyze_"/"compare_"-prefixed keys, which are
  // a different naming convention than the Vollanalyse keys above even where
  // the underlying metric is conceptually the same (e.g. ROE).
  // ---------------------------------------------------------------------
  calculate_annual_inflation_rate: {
    label: "Annual Inflation Rate",
    unit: "%",
    decimals: 2,
    description: "Jährliche Veränderungsrate der Verbraucherpreise (CPI, FRED-Daten).",
    formula: "Inflationsrate = (Aktueller CPI − CPI Vorjahr) / CPI Vorjahr × 100",
  },

  calculate_total_inflation_for_period: {
    label: "Total Inflation",
    unit: "%",
    decimals: 2,
    description: "Kumulierte Inflation über einen frei wählbaren Zeitraum.",
    formula: "Gesamtinflation = (End-CPI / Start-CPI − 1) × 100",
  },

  calculate_avg_quarterly_profit_growth: {
    label: "Avg. Quarterly Profit Growth",
    unit: "%",
    decimals: 2,
    description: "Durchschnittliches Gewinnwachstum von Quartal zu Quartal.",
    formula: "AQGR = (Nettogewinn aktuelles Quartal / Nettogewinn Vorquartal − 1) × 100",
  },

  calculate_avg_annual_profit_growth: {
    label: "Avg. Annual Profit Growth",
    unit: "%",
    decimals: 2,
    description: "Durchschnittliches Gewinnwachstum von Jahr zu Jahr.",
    formula: "AAGR = (Nettogewinn aktuelles Jahr / Nettogewinn Vorjahr − 1) × 100",
  },

  compare_avg_quarterly_growth_to_inflation: {
    label: "Quarterly Profit Growth vs. Inflation",
    unit: "%",
    decimals: 2,
    description: "Vergleicht das durchschnittliche Quartalswachstum des Gewinns mit der Inflationsrate im selben Zeitraum.",
    formula: "Vergleich AQGR vs. kumulierte Inflation (CPI, FRED) im gleichen Zeitraum",
  },

  compare_avg_annual_growth_to_inflation: {
    label: "Annual Profit Growth vs. Inflation",
    unit: "%",
    decimals: 2,
    description: "Vergleicht das durchschnittliche Jahreswachstum des Gewinns mit der Inflationsrate im selben Zeitraum.",
    formula: "Vergleich AAGR vs. kumulierte Inflation (CPI, FRED) im gleichen Zeitraum",
  },

  calculate_historical_dividend_yield_average: {
    label: "Dividendenrendite (Ø 10 Jahre)",
    unit: "%",
    decimals: 2,
    description: "Durchschnittliche Dividendenrendite der letzten N Jahre (Standard: 10).",
    formula: "Ø Dividendenrendite = Mittelwert aus (Jahresdividende / Jahresendkurs) über N Jahre",
  },

  calculate_roe: {
    label: "Eigenkapitalrendite (ROE)",
    unit: "%",
    decimals: 2,
    scale: 100,
    description: "Eigenkapitalrendite — zeigt, wie profitabel ein Unternehmen mit dem Kapital seiner Aktionäre wirtschaftet. Nutzt TTM-Nettogewinn (annual) bzw. den neuesten Quartalsgewinn (quarterly).",
    formula: "ROE = Nettogewinn / Eigenkapital",
  },

  calculate_cashflow_margin: {
    label: "Cashflow-Marge",
    unit: "%",
    decimals: 2,
    description: "Anteil des Umsatzes, der als operativer Cashflow erwirtschaftet wird.",
    formula: "Cashflow-Marge = Operativer Cashflow / Umsatz × 100",
  },

  calculate_cash_to_market_cap: {
    label: "Cash / Market Cap",
    unit: "%",
    decimals: 2,
    scale: 100,
    description: "Anteil der Marktkapitalisierung, der durch liquide Mittel gedeckt ist (nicht aussagekräftig für Finanzinstitute).",
    formula: "Cash / Market Cap = Cash & Cash Equivalents / Marktkapitalisierung",
  },

  analyze_payout_ratio: {
    label: "Ausschüttungsquote",
    unit: "%",
    decimals: 2,
    description: "Anteil des Gewinns pro Aktie, der als Dividende ausgeschüttet wird.",
    formula: "Payout Ratio = Dividende je Aktie (TTM) / Gewinn je Aktie × 100",
  },

  calculate_roic: {
    label: "ROIC",
    unit: "%",
    decimals: 2,
    description: "Rendite auf das insgesamt investierte Kapital (Eigen- und Fremdkapital).",
    formula: "ROIC = Nettogewinn / Investiertes Kapital × 100, Investiertes Kapital = Eigenkapital + Gesamtverschuldung",
  },

  calculate_inventory_to_revenue_ratio: {
    label: "Inventory / Revenue",
    unit: "%",
    decimals: 2,
    description: "Anteil des Umsatzes, der in Lagerbeständen gebunden ist (nicht aussagekräftig für Dienstleister ohne Lager).",
    formula: "Inventory / Revenue = Vorräte / Umsatz × 100",
  },

  // -- Bewertungsmultiplikatoren --
  calculate_kgv: {
    label: "KGV (P/E)",
    unit: "ratio",
    decimals: 2,
    description: "Kurs-Gewinn-Verhältnis — wie viele Jahresgewinne der aktuelle Kurs kostet. Negative Werte werden bewusst zugelassen (relevant für Zykliker-Analysen). Hinweis: nutzt immer den letzten Jahresbericht — der Quartal/Jahr-Umschalter hat für das KGV keine Wirkung.",
    formula: "KGV = Aktueller Kurs / Gewinn je Aktie (EPS, aus dem letzten Jahresbericht)",
  },

  calculate_peg_ratio: {
    label: "PEG Ratio",
    unit: "ratio",
    decimals: 2,
    description: "Setzt das KGV ins Verhältnis zum Gewinnwachstum, um Wachstum in die Bewertung einzupreisen.",
    formula: "PEG = KGV / Gewinnwachstum (% p.a.), Gewinnwachstum = annualisierte CAGR aus dem ältesten und dem neuesten verfügbaren Jahresgewinn",
  },

  calculate_kuv: {
    label: "KUV (P/S)",
    unit: "ratio",
    decimals: 2,
    description: "Kurs-Umsatz-Verhältnis — Bewertung im Verhältnis zum Umsatz, unabhängig von der Profitabilität.",
    formula: "KUV = Marktkapitalisierung / Umsatz",
  },

  calculate_ev_to_sales: {
    label: "EV / Sales",
    unit: "ratio",
    decimals: 2,
    description: "Unternehmenswert im Verhältnis zum Umsatz.",
    formula: "EV / Sales = Enterprise Value / Umsatz",
  },

  calculate_ev_to_ebit: {
    label: "EV / EBIT",
    unit: "ratio",
    decimals: 2,
    description: "Unternehmenswert im Verhältnis zum operativen Ergebnis vor Zinsen und Steuern.",
    formula: "EV / EBIT = Enterprise Value / EBIT",
  },

  calculate_ev_to_ebitda: {
    label: "EV / EBITDA",
    unit: "ratio",
    decimals: 2,
    description: "Unternehmenswert im Verhältnis zum operativen Ergebnis vor Zinsen, Steuern und Abschreibungen.",
    formula: "EV / EBITDA = (Marktkapitalisierung + Nettoverschuldung) / EBITDA",
  },

  calculate_price_to_ebit: {
    label: "Price / EBIT",
    unit: "ratio",
    decimals: 2,
    description: "Bewertung des Unternehmens im Verhältnis zum operativen Ergebnis.",
    formula: "Price / EBIT = Marktkapitalisierung / EBIT",
  },

  calculate_price_to_freecashflow: {
    label: "Price / Free Cashflow",
    unit: "ratio",
    decimals: 2,
    description: "Bewertung des Unternehmens im Verhältnis zum frei verfügbaren Cashflow.",
    formula: "Price / FCF = Marktkapitalisierung / Free Cashflow",
  },

  get_current_tbv_and_price: {
    label: "TBV & Kurs",
    unit: "ratio",
    decimals: 2,
    description: "Kombinierte Kennzahl aus materiellem Buchwert je Aktie, aktuellem Kurs und dem daraus abgeleiteten P/TBV-Multiple.",
    formula: "TBV je Aktie = (Eigenkapital − Goodwill) / Aktienanzahl; P/TBV = Aktueller Kurs / TBV je Aktie",
  },

  calculate_book_value_per_share: {
    label: "Buchwert je Aktie",
    unit: "currency",
    decimals: 2,
    description: "Materieller Buchwert je Aktie nach dem pragmatischen Lynch-Ansatz.",
    formula: "Buchwert je Aktie = (Eigenkapital − Goodwill) / Aktienanzahl",
  },

  // -- Verschuldung / Liquidität --
  calculate_debt_to_equity: {
    label: "Debt-to-Equity",
    unit: "ratio",
    decimals: 2,
    description: "Verschuldungsgrad — Verhältnis von Fremdkapital zu Eigenkapital.",
    formula: "Debt-to-Equity = Gesamtverbindlichkeiten / Eigenkapital",
  },

  calculate_net_debt_to_ebitda: {
    label: "Net Debt / EBITDA",
    unit: "ratio",
    decimals: 2,
    description: "Zeigt, wie viele Jahres-EBITDA nötig wären, um die Nettoverschuldung zu tilgen.",
    formula: "Net Debt / EBITDA = Nettoverschuldung / EBITDA",
  },

  calculate_interest_coverage_ratio: {
    label: "Interest Coverage Ratio",
    unit: "ratio",
    decimals: 2,
    description: "Zinsdeckungsgrad — wie oft das operative Ergebnis die Zinsaufwendungen deckt.",
    formula: "Interest Coverage = EBIT / Zinsaufwand",
  },

  calculate_current_netcurrentassets: {
    label: "Net Current Assets",
    unit: "currency",
    decimals: 0,
    description: "Substanzpuffer aus dem Umlaufvermögen abzüglich kurzfristiger Verbindlichkeiten (Net-Net-Ansatz).",
    formula: "Net Current Assets = Umlaufvermögen − Kurzfristige Verbindlichkeiten",
  },

  // -- Dividende --
  calculate_current_dividend_yield: {
    label: "Dividendenrendite (aktuell)",
    unit: "%",
    decimals: 2,
    description: "Aktuelle Dividendenrendite auf Basis der zuletzt bekannten Dividende und des aktuellen Kurses.",
    formula: "Dividendenrendite = Dividende (jährlich, yfinance) / Aktueller Kurs × 100",
  },

  analyze_dividend_history: {
    label: "Dividendenhistorie",
    unit: "ratio",
    decimals: 0,
    description: "Anzahl der Jahre mit Dividendenzahlung bzw. -erhöhung sowie die langfristige Wachstumsrate der Dividende.",
    formula: "Erfasst u.a. Jahre mit Dividende, Jahre mit Erhöhung und die Dividenden-CAGR (bis zu 30 Jahre)",
  },

  // -- Historische Zeitreihen (result_shape: timeseries) --
  calculate_historical_market_cap: {
    label: "Historische Marktkapitalisierung",
    unit: "currency",
    decimals: 0,
    description: "Verlauf der Marktkapitalisierung über die Zeit.",
    formula: "Market Cap = Aktienkurs × Ausstehende Aktien, je Zeitpunkt",
  },

  calculate_historical_ev: {
    label: "Historischer Enterprise Value",
    unit: "currency",
    decimals: 0,
    description: "Verlauf des Unternehmenswerts (Marktkapitalisierung + Nettoverschuldung) über die Zeit.",
    formula: "EV = Marktkapitalisierung + Nettoverschuldung, je Zeitpunkt",
  },

  calculate_historical_sales: {
    label: "Historischer Umsatz",
    unit: "currency",
    decimals: 0,
    description: "Verlauf des Umsatzes über mehrere Perioden.",
    formula: "Direkt aus der historischen Gewinn- und Verlustrechnung (SEC-Daten)",
  },

  calculate_historical_ev_sales: {
    label: "Historisch EV/Sales",
    unit: "ratio",
    decimals: 2,
    description: "Verlauf des EV/Sales-Multiples über die Zeit.",
    formula: "EV / Sales = Enterprise Value / Umsatz, je Zeitpunkt",
  },

  calculate_historical_ebit: {
    label: "Historisches EBIT",
    unit: "currency",
    decimals: 0,
    description: "Verlauf des operativen Ergebnisses (EBIT) über mehrere Perioden.",
    formula: 'Direkt aus der Gewinn- und Verlustrechnung (Position "EBIT"/"Operating Income")',
  },

  calculate_historical_ev_to_ebit: {
    label: "Historisch EV/EBIT",
    unit: "ratio",
    decimals: 2,
    description: "Verlauf des EV/EBIT-Multiples über die Zeit.",
    formula: "EV / EBIT = Enterprise Value / EBIT, je Zeitpunkt",
  },

  calculate_historical_ebitda: {
    label: "Historisches EBITDA",
    unit: "currency",
    decimals: 0,
    description: "Verlauf des EBITDA über mehrere Perioden.",
    formula: "EBITDA = direkt aus der Bilanz oder EBIT + Abschreibungen (D&A)",
  },

  calculate_historical_ev_to_ebitda: {
    label: "Historisch EV/EBITDA",
    unit: "ratio",
    decimals: 2,
    description: "Verlauf des EV/EBITDA-Multiples über die Zeit.",
    formula: "EV / EBITDA = Enterprise Value / EBITDA, je Zeitpunkt",
  },

  calculate_historical_price_to_book: {
    label: "Historisch Price/Book",
    unit: "ratio",
    decimals: 2,
    description: "Verlauf des Kurs-Buchwert-Verhältnisses über die Zeit.",
    formula: "P/B = Marktkapitalisierung / Eigenkapital, je Zeitpunkt",
  },

  calculate_historical_price_to_sales: {
    label: "Historisch Price/Sales",
    unit: "ratio",
    decimals: 2,
    description: "Verlauf des Kurs-Umsatz-Verhältnisses über die Zeit.",
    formula: "P/S = Marktkapitalisierung / Umsatz, je Zeitpunkt",
  },

  calculate_historical_price_to_ebit: {
    label: "Historisch Price/EBIT",
    unit: "ratio",
    decimals: 2,
    description: "Verlauf des Kurs/EBIT-Multiples über die Zeit — Basis für die EBIT-Bandbreiten-Bewertung.",
    formula: "P/EBIT = Marktkapitalisierung / EBIT, je Zeitpunkt",
  },

  calculate_historical_netcurrentassets: {
    label: "Historische Net Current Assets",
    unit: "currency",
    decimals: 0,
    description: "Verlauf des Substanzpuffers aus Umlaufvermögen minus kurzfristigen Verbindlichkeiten.",
    formula: "Net Current Assets = Umlaufvermögen − Kurzfristige Verbindlichkeiten, je Zeitpunkt",
  },

  calculate_historical_price_netcurrentassets: {
    label: "Historisch Price/NCA",
    unit: "ratio",
    decimals: 2,
    description: "Verlauf des Kurses im Verhältnis zu den Net Current Assets je Aktie.",
    formula: "Price / NCA = Aktueller Kurs / (Net Current Assets je Aktie), je Zeitpunkt",
  },

  calculate_historical_operatingcashflow: {
    label: "Historischer Operating Cashflow",
    unit: "currency",
    decimals: 0,
    description: "Verlauf des operativen Cashflows über mehrere Perioden.",
    formula: 'Direkt aus der Kapitalflussrechnung (Position "Operating Cash Flow")',
  },

  calculate_historical_price_operatingcashflow: {
    label: "Historisch Price/OCF",
    unit: "ratio",
    decimals: 2,
    description: "Verlauf des Kurses im Verhältnis zum operativen Cashflow je Aktie.",
    formula: "Price / OCF = Aktueller Kurs / (Operativer Cashflow je Aktie), je Zeitpunkt",
  },

  calculate_historical_freecashflow: {
    label: "Historischer Free Cashflow",
    unit: "currency",
    decimals: 0,
    description: "Verlauf des frei verfügbaren Cashflows über mehrere Perioden.",
    formula: "Free Cashflow = Operativer Cashflow + Capital Expenditures (CapEx negativ verbucht)",
  },

  calculate_historical_price_freecashflow: {
    label: "Historisch Price/FCF",
    unit: "ratio",
    decimals: 2,
    description: "Verlauf des Kurses im Verhältnis zum Free Cashflow je Aktie.",
    formula: "Price / FCF = Marktkapitalisierung / Free Cashflow, je Zeitpunkt",
  },

  calculate_historical_tangiblebookvalue: {
    label: "Historischer TBV",
    unit: "currency",
    decimals: 0,
    description: "Verlauf des materiellen Buchwerts über mehrere Perioden.",
    formula: "TBV = Eigenkapital − Goodwill, je Zeitpunkt",
  },

  calculate_historical_price_to_tangiblebookvalue: {
    label: "Historisch Price/TBV",
    unit: "ratio",
    decimals: 2,
    description: "Verlauf des Kurs/TBV-Multiples über die Zeit — Basis für die TBV-Bandbreiten-Bewertung.",
    formula: "P/TBV = Aktueller Kurs / Materieller Buchwert je Aktie, je Zeitpunkt",
  },

  // -- Bandbreiten / CRV / Kursziele (result_shape: complex) --
  evaluate_tbv_bandwidth: {
    label: "TBV-Bandbreite",
    unit: "ratio",
    decimals: 2,
    description: "Bewertet die Aktie anhand des historischen P/TBV-Bereichs nach dem Regression-to-the-Mean-Ansatz und leitet Kauf-/Verkaufszonen ab.",
    formula: "Kaufzone: P/TBV ∈ [1.0, 1.5]; Überbewertung: P/TBV ∈ [3.0, 4.0]; Kursziele: WC = 0.9×TBV/Aktie, BUY = 1.15×TBV/Aktie, SELL = 3×TBV/Aktie",
  },

  evaluate_ebit_bandwidth: {
    label: "EBIT-Bandbreite",
    unit: "ratio",
    decimals: 2,
    description: "Bewertet die Aktie anhand des historischen P/EBIT-Bereichs nach dem Regression-to-the-Mean-Ansatz und leitet Kauf-/Verkaufszonen ab.",
    formula: "Kaufzone: P/EBIT ∈ [6.0, 10.0]; Überbewertung: P/EBIT ∈ [20.0, 25.0]; Kursziele: WC = 7.5×EBIT/Aktie, BUY = 8.5×EBIT/Aktie, SELL = 22×EBIT/Aktie",
  },

  calculate_crv: {
    label: "CRV (Chance-Risiko-Verhältnis)",
    unit: "ratio",
    decimals: 2,
    description: "Setzt das Kurspotenzial nach oben (Upside) ins Verhältnis zum Risiko nach unten (Downside), basierend auf den berechneten Kurszielen.",
    formula: "CRV = Upside / Downside; Upside_konservativ = FV-Kurs − Aktueller Kurs, Upside_aggressiv = SELL-Kurs − Aktueller Kurs, Downside = Aktueller Kurs − WC-Kurs",
  },

  calculate_course_target_pricemultiples: {
    label: "Kursziel (Price-Multiples)",
    unit: "currency",
    decimals: 2,
    description: "Leitet Kursziele (Worst Case / Buy / Fair Value / Sell) aus der historischen Bandbreite eines Price-Multiples ab (z.B. Price/FCF, Price/Book).",
    formula: "Kursziele = historische Multiple-Bandbreite × zugehörige Kennzahl je Aktie",
  },

  calculate_course_target_evmultiples: {
    label: "Kursziel (EV-Multiples)",
    unit: "currency",
    decimals: 2,
    description: "Leitet Kursziele (Worst Case / Buy / Fair Value / Sell) aus der historischen Bandbreite eines EV-Multiples ab (z.B. EV/EBITDA, EV/EBIT).",
    formula: "Kursziele = (historische Multiple-Bandbreite × zugehörige Kennzahl − Nettoverschuldung) / Aktienanzahl",
  },

  // -- Einzelfelder, die in CrvTargetPanel als eigene Zeilen gerendert werden --
  crv_conservative: {
    label: "CRV konservativ",
    unit: "ratio",
    decimals: 2,
    description: "CRV auf Basis des konservativen (Fair-Value-)Kursziels.",
    formula: "CRV konservativ = (Fair-Value-Kurs − Aktueller Kurs) / Downside",
  },

  crv_aggressive: {
    label: "CRV aggressiv",
    unit: "ratio",
    decimals: 2,
    description: "CRV auf Basis des aggressiven (Sell-)Kursziels.",
    formula: "CRV aggressiv = (Sell-Kurs − Aktueller Kurs) / Downside",
  },

  downside: {
    label: "Downside",
    unit: "currency",
    decimals: 2,
    description: "Abstand vom aktuellen Kurs zum Worst-Case-Kursziel — das Risiko nach unten.",
    formula: "Downside = Aktueller Kurs − Worst-Case-Kurs",
  },

  upside_conservative: {
    label: "Upside konservativ",
    unit: "currency",
    decimals: 2,
    description: "Abstand vom aktuellen Kurs zum konservativen (Fair-Value-)Kursziel.",
    formula: "Upside konservativ = Fair-Value-Kurs − Aktueller Kurs",
  },

  upside_aggressive: {
    label: "Upside aggressiv",
    unit: "currency",
    decimals: 2,
    description: "Abstand vom aktuellen Kurs zum aggressiven (Sell-)Kursziel.",
    formula: "Upside aggressiv = Sell-Kurs − Aktueller Kurs",
  },

  price: {
    label: "Aktueller Kurs",
    unit: "currency",
    decimals: 2,
    description: "Aktueller Aktienkurs, wie er der Bandbreiten-Bewertung zugrunde liegt.",
  },

  pb_ratio: {
    label: "P/TBV",
    unit: "ratio",
    decimals: 2,
    description: "Aktuelles Kurs/TBV-Multiple innerhalb der TBV-Bandbreiten-Bewertung.",
    formula: "P/TBV = Aktueller Kurs / Materieller Buchwert je Aktie",
  },

  shares_outstanding: {
    label: "Ausstehende Aktien",
    unit: "ratio",
    decimals: 0,
    description: "Anzahl der ausstehenden Aktien, wie sie der Bandbreiten-Bewertung zugrunde liegt.",
  },

  ebit_per_share: {
    label: "EBIT je Aktie",
    unit: "currency",
    decimals: 2,
    description: "Operatives Ergebnis je Aktie innerhalb der EBIT-Bandbreiten-Bewertung.",
    formula: "EBIT je Aktie = EBIT / Ausstehende Aktien",
  },

  ebit_ratio: {
    label: "P/EBIT",
    unit: "ratio",
    decimals: 2,
    description: "Aktuelles Kurs/EBIT-Multiple innerhalb der EBIT-Bandbreiten-Bewertung.",
    formula: "P/EBIT = Aktueller Kurs / EBIT je Aktie",
  },

  // ---------------------------------------------------------------------
  // Sub-fields of get_current_tbv_and_price's dict result — looked up
  // individually when ComparePivotTable / formatMetricValue render the
  // tooltip breakdown of that metric's three fields.
  // ---------------------------------------------------------------------
  tbv_per_share: {
    label: "TBV je Aktie",
    unit: "currency",
    decimals: 2,
    description: "Materieller Buchwert je Aktie nach dem pragmatischen Lynch-Ansatz.",
    formula: "TBV je Aktie = (Eigenkapital − Goodwill) / Aktienanzahl",
  },

  current_price: {
    label: "Aktueller Kurs",
    unit: "currency",
    decimals: 2,
    description: "Aktueller Aktienkurs zum Zeitpunkt der Analyse.",
  },

  price_to_tbv: {
    label: "Kurs / TBV",
    unit: "ratio",
    decimals: 2,
    description: "Aktueller Kurs im Verhältnis zum materiellen Buchwert je Aktie.",
    formula: "Kurs / TBV = Aktueller Kurs / TBV je Aktie",
  },
};

export function getMetricConfig(key: string): MetricConfig | undefined {
  return METRICS_CONFIG[normalizeMetricKey(key)];
}

export function normalizeMetricKey(key: string): string {
  return key.trim().toLowerCase().replaceAll("-", "_").replaceAll(" ", "_");
}