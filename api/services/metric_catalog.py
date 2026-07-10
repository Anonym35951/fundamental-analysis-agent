# api/services/metric_catalog.py
"""Metadata-driven registry describing every Model.py method that Custom
Analysis exposes to users, plus the parameters each one accepts.

Replaces the old hand-written 11-entry CUSTOM_METRICS_REGISTRY dict in
api/routes/custom_analysis.py. Most metrics share an identical calling
convention (symbol + a handful of optional kwargs matching Model.py's own
parameter names), so they're called through a single generic adapter
(`_generic_call`) instead of one bespoke lambda per metric. Only the CRV /
course-target methods need a bespoke wrapper since they require a
pre-fetched `historical_data` DataFrame.

See the plan's IN/OUT list for why each of the 62 agent/Model.py methods is
either exposed here or deliberately excluded.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from agent.AgentAction import AgentAction

ParamType = Literal["string", "number", "date", "enum"]
ResultShape = Literal["scalar", "dict", "timeseries", "complex"]


@dataclass(frozen=True)
class MetricParam:
    name: str
    type: ParamType
    required: bool = False
    default: Any = None
    enum_values: Optional[list[str]] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "default": self.default,
            "enum_values": self.enum_values,
        }


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    category: str
    requires_symbol: bool = True
    result_shape: ResultShape = "scalar"
    params: list[MetricParam] = field(default_factory=list)
    # Optional override; defaults to _generic_call(action, symbol, params, key, requires_symbol).
    call: Optional[Callable[[AgentAction, Optional[str], dict], Any]] = None

    def to_catalog_entry(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "category": self.category,
            "requires_symbol": self.requires_symbol,
            "result_shape": self.result_shape,
            "params": [p.to_dict() for p in self.params],
        }


FREQUENCY_PARAM = MetricParam(
    name="frequency", type="enum", default="annual", enum_values=["annual", "quarterly"]
)
START_DATE_PARAM = MetricParam(name="start_date", type="date")
END_DATE_PARAM = MetricParam(name="end_date", type="date")
MIN_YEARS_PARAM = MetricParam(name="min_years", type="number", default=10.0)
THRESHOLD_PARAM = MetricParam(name="threshold", type="number", default=75.0)


def _coerce_param_value(value: Any, annotation: Any) -> Any:
    """A "number"-typed param (threshold, min_years, years) arriving as a
    JSON string reaches a Model.py method typed float/int and crashes a
    downstream numeric comparison - not always with a plain Python
    TypeError, but sometimes with a confusing numpy ufunc error whenever
    the other operand happens to be a numpy scalar (e.g. "ufunc 'greater'
    did not contain a loop with signature matching types (Float64DType,
    StrDType) -> None", live bug 2026-07-10, analyze_payout_ratio). The
    frontend now sends real JSON numbers for these params, but a custom
    analysis definition saved before that fix still has the old string
    value persisted in the database - coerce here, at the one place every
    generically-dispatched metric call passes through, so it's fixed for
    already-saved definitions too, not just newly-edited ones."""
    if not isinstance(value, str) or annotation is inspect.Parameter.empty:
        return value
    if annotation is float:
        try:
            return float(value)
        except ValueError:
            return value
    if annotation is int:
        try:
            return int(value)
        except ValueError:
            return value
    return value


def _generic_call(action: AgentAction, symbol: Optional[str], params: dict, key: str, requires_symbol: bool) -> Any:
    """Maps the stored `params` dict onto the real Model.py method's keyword
    arguments via introspection, so most MetricSpec entries don't need a
    bespoke lambda. Mirrors the frequency/use_cache pass-through pattern
    already used in api/routes/analyze.py's start_single_analysis_job."""
    fn = getattr(action.model, key)
    sig = inspect.signature(fn)
    kwargs = {
        k: _coerce_param_value(v, sig.parameters[k].annotation)
        for k, v in params.items() if k in sig.parameters and v is not None
    }
    if requires_symbol:
        return fn(symbol, **kwargs)
    return fn(**kwargs)


# Historical (DataFrame-returning) methods that feed calculate_crv /
# calculate_course_target_* — keyed by the "multiple" a user can pick for
# those bespoke metrics. Mirrors AgentAction.MULTIPLE_METHOD_MAP / the
# multiples agent/industry_multiples.py can resolve, minus the sector-auto-
# selection (the user picks the multiple explicitly here instead).
_CRV_MULTIPLE_FETCHERS: dict[str, str] = {
    "EV_Sales": "calculate_historical_ev_sales",
    "EV_EBIT": "calculate_historical_ev_to_ebit",
    "EV_EBITDA": "calculate_historical_ev_to_ebitda",
    "Price_Book": "calculate_historical_price_to_book",
    "Price_Sales": "calculate_historical_price_to_sales",
    "Price_EBIT": "calculate_historical_price_to_ebit",
    "Price_NetCurrentAssets": "calculate_historical_price_netCurrentAssets",
    "Price_OperatingCashflow": "calculate_historical_price_OperatingCashflow",
    "Price_FreeCashflow": "calculate_historical_Price_FreeCashflow",
    "Price_TangibleBookValue": "calculate_historical_price_to_TangibleBookValue",
}

_CRV_MULTIPLE_PARAM = MetricParam(
    name="multiple",
    type="enum",
    required=True,
    default="EV_EBITDA",
    enum_values=list(_CRV_MULTIPLE_FETCHERS.keys()),
)

# calculate_course_target_PriceMultiples / -EVMultiples each only operate on
# their own multiple family, so they get their own enum lists derived from
# _CRV_MULTIPLE_FETCHERS instead of sharing the combined 10-value list above.
_PRICE_MULTIPLE_PARAM = MetricParam(
    name="multiple",
    type="enum",
    required=True,
    default="Price_FreeCashflow",
    enum_values=[key for key in _CRV_MULTIPLE_FETCHERS if key.startswith("Price_")],
)
_EV_MULTIPLE_PARAM = MetricParam(
    name="multiple",
    type="enum",
    required=True,
    default="EV_EBITDA",
    enum_values=[key for key in _CRV_MULTIPLE_FETCHERS if key.startswith("EV_")],
)


def _call_tbv_and_price(action: AgentAction, symbol: Optional[str], params: dict) -> Any:
    """Model.get_current_tbv_and_price returns a bare (tbv, price, pb) tuple,
    not a dict — _generic_call would let that serialize as an unlabeled JSON
    array. Wrap it into the same {tbv_per_share, current_price, price_to_tbv}
    shape api/routes/metric_routes.py's /tbv-and-price endpoint already uses,
    so the catalog's "dict" result_shape promise actually holds."""
    tbv, price, pb = action.model.get_current_tbv_and_price(symbol)
    if isinstance(tbv, dict):
        return tbv
    return {"symbol": symbol, "tbv_per_share": tbv, "current_price": price, "price_to_tbv": pb}


def _fetch_historical_for_multiple(action: AgentAction, symbol: str, multiple: str):
    fetcher_name = _CRV_MULTIPLE_FETCHERS.get(multiple, "calculate_historical_ev_to_ebitda")
    fetcher = getattr(action.model, fetcher_name)
    return fetcher(symbol=symbol, start_date=None, end_date=None, use_cache=True)


def _call_crv(action: AgentAction, symbol: Optional[str], params: dict) -> Any:
    multiple = params.get("multiple") or "EV_EBITDA"
    historical_df = _fetch_historical_for_multiple(action, symbol, multiple)
    if historical_df is None or getattr(historical_df, "empty", True):
        return {"error": f"Keine historischen Daten für {multiple} verfügbar."}
    return action.model.calculate_crv(symbol, historical_df)


def _call_course_target_price_multiples(action: AgentAction, symbol: Optional[str], params: dict) -> Any:
    multiple = params.get("multiple") or "Price_FreeCashflow"
    historical_df = _fetch_historical_for_multiple(action, symbol, multiple)
    if historical_df is None or getattr(historical_df, "empty", True):
        return {"error": f"Keine historischen Daten für {multiple} verfügbar."}
    return action.model.calculate_course_target_PriceMultiples(historical_df, symbol)


def _call_course_target_ev_multiples(action: AgentAction, symbol: Optional[str], params: dict) -> Any:
    multiple = params.get("multiple") or "EV_EBITDA"
    historical_df = _fetch_historical_for_multiple(action, symbol, multiple)
    if historical_df is None or getattr(historical_df, "empty", True):
        return {"error": f"Keine historischen Daten für {multiple} verfügbar."}
    return action.model.calculate_course_target_EVMultiples(symbol, historical_df)


METRIC_CATALOG: list[MetricSpec] = [
    # ---- Bewertung ----
    MetricSpec(key="calculate_KGV", label="KGV (P/E)", category="Bewertung"),
    MetricSpec(key="calculate_peg_ratio", label="PEG Ratio", category="Bewertung",
               params=[START_DATE_PARAM, END_DATE_PARAM]),
    MetricSpec(key="calculate_kuv", label="KUV (P/S)", category="Bewertung", params=[FREQUENCY_PARAM]),
    MetricSpec(key="calculate_ev_to_sales", label="EV / Sales", category="Bewertung", params=[FREQUENCY_PARAM]),
    MetricSpec(key="calculate_ev_to_ebit", label="EV / EBIT", category="Bewertung", params=[FREQUENCY_PARAM]),
    MetricSpec(key="calculate_ev_to_ebitda", label="EV / EBITDA", category="Bewertung", params=[FREQUENCY_PARAM]),
    MetricSpec(key="calculate_price_to_ebit", label="Price / EBIT", category="Bewertung", params=[FREQUENCY_PARAM]),
    MetricSpec(key="calculate_price_to_freeCashflow", label="Price / Free Cashflow", category="Bewertung",
               params=[FREQUENCY_PARAM]),
    MetricSpec(key="get_current_tbv_and_price", label="TBV & Kurs", category="Bewertung", result_shape="dict",
               call=_call_tbv_and_price),
    MetricSpec(key="calculate_book_value_per_share", label="Buchwert je Aktie", category="Bewertung"),

    # ---- Profitabilität ----
    MetricSpec(key="calculate_roe", label="Eigenkapitalrendite (ROE)", category="Profitabilität",
               params=[FREQUENCY_PARAM]),
    MetricSpec(key="calculate_ROIC", label="ROIC", category="Profitabilität", params=[FREQUENCY_PARAM]),
    MetricSpec(key="calculate_cashflow_margin", label="Cashflow-Marge", category="Profitabilität",
               params=[FREQUENCY_PARAM]),
    MetricSpec(key="calculate_avg_quarterly_profit_growth", label="Ø Quartalswachstum (Gewinn)",
               category="Profitabilität", result_shape="dict", params=[START_DATE_PARAM, END_DATE_PARAM]),
    MetricSpec(key="calculate_avg_annual_profit_growth", label="Ø Jahreswachstum (Gewinn)",
               category="Profitabilität", result_shape="dict", params=[START_DATE_PARAM, END_DATE_PARAM]),

    # ---- Verschuldung ----
    MetricSpec(key="calculate_debt_to_equity", label="Debt-to-Equity", category="Verschuldung",
               params=[FREQUENCY_PARAM]),
    MetricSpec(key="calculate_net_debt_to_ebitda", label="Net Debt / EBITDA", category="Verschuldung",
               params=[FREQUENCY_PARAM]),
    MetricSpec(key="calculate_interest_coverage_ratio", label="Interest Coverage Ratio", category="Verschuldung",
               params=[FREQUENCY_PARAM]),
    MetricSpec(key="calculate_inventory_to_revenue_ratio", label="Inventory / Revenue", category="Verschuldung",
               params=[FREQUENCY_PARAM]),
    MetricSpec(key="calculate_current_netCurrentAssets", label="Net Current Assets", category="Verschuldung",
               params=[FREQUENCY_PARAM]),

    # ---- Liquidität ----
    MetricSpec(key="calculate_cash_to_market_cap", label="Cash / Market Cap", category="Liquidität",
               params=[FREQUENCY_PARAM]),

    # ---- Dividende ----
    MetricSpec(key="calculate_current_dividend_yield", label="Dividendenrendite (aktuell)",
               category="Dividende", result_shape="dict"),
    MetricSpec(key="calculate_historical_dividend_yield_average", label="Dividendenrendite (Ø 10 Jahre)",
               category="Dividende", params=[MetricParam(name="years", type="number", default=10)]),
    MetricSpec(key="analyze_payout_ratio", label="Ausschüttungsquote", category="Dividende",
               result_shape="dict", params=[THRESHOLD_PARAM]),
    MetricSpec(key="analyze_dividend_history", label="Dividendenhistorie", category="Dividende",
               result_shape="dict"),
    # "Kauf-/Verkaufspunkte" (determine_buy_sell_points) ist bewusst nicht im
    # nutzerseitigen Katalog — spricht eine indirekte Kauf-/Verkaufsempfehlung
    # aus, die hier nicht angezeigt werden soll/darf. Die Model-Methode bleibt
    # für interne Nutzung unverändert bestehen.

    # ---- Inflation / Makro (kein Symbol nötig) ----
    MetricSpec(key="calculate_annual_inflation_rate", label="Jährliche Inflationsrate", category="Makro",
               requires_symbol=False,
               params=[MetricParam(name="current_date_str", type="date"), MetricParam(name="target_date_str", type="date")]),
    MetricSpec(key="calculate_total_inflation_for_period", label="Gesamtinflation im Zeitraum", category="Makro",
               requires_symbol=False,
               params=[MetricParam(name="start_date", type="date", required=True),
                       MetricParam(name="end_date", type="date", required=True)]),

    # ---- Wachstum vs. Inflation ----
    MetricSpec(key="compare_avg_quarterly_growth_to_inflation", label="Quartalswachstum vs. Inflation",
               category="Makro", result_shape="dict", params=[START_DATE_PARAM, END_DATE_PARAM]),
    MetricSpec(key="compare_avg_annual_growth_to_inflation", label="Jahreswachstum vs. Inflation",
               category="Makro", result_shape="dict", params=[START_DATE_PARAM, END_DATE_PARAM]),

    # ---- Historische Zeitreihen ----
    MetricSpec(key="calculate_historical_market_cap", label="Historische Marktkapitalisierung",
               category="Historisch", result_shape="timeseries"),
    MetricSpec(key="calculate_historical_ev", label="Historischer Enterprise Value",
               category="Historisch", result_shape="timeseries"),
    MetricSpec(key="calculate_historical_sales", label="Historischer Umsatz",
               category="Historisch", result_shape="timeseries"),
    MetricSpec(key="calculate_historical_ev_sales", label="Historisch EV/Sales",
               category="Historisch", result_shape="timeseries"),
    MetricSpec(key="calculate_historical_ebit", label="Historisches EBIT",
               category="Historisch", result_shape="timeseries"),
    MetricSpec(key="calculate_historical_ev_to_ebit", label="Historisch EV/EBIT",
               category="Historisch", result_shape="timeseries"),
    MetricSpec(key="calculate_historical_ebitda", label="Historisches EBITDA",
               category="Historisch", result_shape="timeseries"),
    MetricSpec(key="calculate_historical_ev_to_ebitda", label="Historisch EV/EBITDA",
               category="Historisch", result_shape="timeseries"),
    MetricSpec(key="calculate_historical_price_to_book", label="Historisch Price/Book",
               category="Historisch", result_shape="timeseries"),
    MetricSpec(key="calculate_historical_price_to_sales", label="Historisch Price/Sales",
               category="Historisch", result_shape="timeseries"),
    MetricSpec(key="calculate_historical_price_to_ebit", label="Historisch Price/EBIT",
               category="Historisch", result_shape="timeseries"),
    MetricSpec(key="calculate_historical_netCurrentAssets", label="Historische Net Current Assets",
               category="Historisch", result_shape="timeseries"),
    MetricSpec(key="calculate_historical_price_netCurrentAssets", label="Historisch Price/NCA",
               category="Historisch", result_shape="timeseries"),
    MetricSpec(key="calculate_historical_OperatingCashflow", label="Historischer Operating Cashflow",
               category="Historisch", result_shape="timeseries"),
    MetricSpec(key="calculate_historical_price_OperatingCashflow", label="Historisch Price/OCF",
               category="Historisch", result_shape="timeseries"),
    MetricSpec(key="calculate_historical_FreeCashflow", label="Historischer Free Cashflow",
               category="Historisch", result_shape="timeseries"),
    MetricSpec(key="calculate_historical_Price_FreeCashflow", label="Historisch Price/FCF",
               category="Historisch", result_shape="timeseries"),
    MetricSpec(key="calculate_historical_TangibleBookValue", label="Historischer TBV",
               category="Historisch", result_shape="timeseries"),
    MetricSpec(key="calculate_historical_price_to_TangibleBookValue", label="Historisch Price/TBV",
               category="Historisch", result_shape="timeseries"),

    # ---- Bandwidth / komplexe Bewertung ----
    MetricSpec(key="evaluate_tbv_bandwidth", label="TBV-Bandbreite", category="Bewertung (komplex)",
               result_shape="complex", params=[MIN_YEARS_PARAM]),
    MetricSpec(key="evaluate_ebit_bandwidth", label="EBIT-Bandbreite", category="Bewertung (komplex)",
               result_shape="complex", params=[MIN_YEARS_PARAM]),

    # ---- CRV / Kursziele (bespoke wrappers, brauchen vorab geladene historical_data) ----
    MetricSpec(key="calculate_crv", label="CRV (Chance-Risiko-Verhältnis)", category="Bewertung (komplex)",
               result_shape="complex", params=[_CRV_MULTIPLE_PARAM], call=_call_crv),
    MetricSpec(key="calculate_course_target_PriceMultiples", label="Kursziel (Price-Multiples)",
               category="Bewertung (komplex)", result_shape="complex", params=[_PRICE_MULTIPLE_PARAM],
               call=_call_course_target_price_multiples),
    MetricSpec(key="calculate_course_target_EVMultiples", label="Kursziel (EV-Multiples)",
               category="Bewertung (komplex)", result_shape="complex", params=[_EV_MULTIPLE_PARAM],
               call=_call_course_target_ev_multiples),
]

METRIC_CATALOG_BY_KEY: dict[str, MetricSpec] = {spec.key: spec for spec in METRIC_CATALOG}


def get_catalog_entries() -> list[dict]:
    return [spec.to_catalog_entry() for spec in METRIC_CATALOG]


def call_metric(action: AgentAction, key: str, symbol: Optional[str], params: dict) -> Any:
    spec = METRIC_CATALOG_BY_KEY.get(key)
    if spec is None:
        raise KeyError(f"Unbekannte Metrik: {key}")

    if spec.call is not None:
        return spec.call(action, symbol, params)

    return _generic_call(action, symbol, params, spec.key, spec.requires_symbol)
