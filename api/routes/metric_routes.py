import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends
from api.utils.json_sanitize import make_json_safe
from api.routes.analyze import get_action
from api.core.dependencies import get_current_user, require_analysis_access
from api.models.user import User

# Auth auf Router-Ebene: JEDER Metrik-Endpoint erfordert ein gültiges Token.
# Vorher waren fast alle Endpoints hier öffentlich erreichbar (Kosten-/Abuse-Risiko).
router = APIRouter(
    prefix="/metrics",
    tags=["metrics"],
    dependencies=[Depends(get_current_user)],
)

logger = logging.getLogger(__name__)

# Ehrliche, aber generische Client-Meldung; Details landen nur im Server-Log,
# damit Exception-Strings (interne Pfade, URLs) nicht zum Client leaken.
_GENERIC_METRIC_ERROR = (
    "Kennzahl konnte nicht berechnet werden — Datenquelle vorübergehend "
    "nicht verfügbar oder Daten unvollständig."
)


def _norm_symbol(symbol: str) -> str:
    return symbol.strip().upper()


# In-process cache so multiple users/components polling the same symbol's
# live price within a short window don't each trigger their own yfinance
# call — this endpoint is polled every ~20s per visible symbol, well below
# how often the underlying price actually changes.
_PRICE_CACHE: dict[str, dict] = {}
_PRICE_CACHE_TTL_SECONDS = 12

# Quelle/Stand ändern sich pro Symbol höchstens einmal pro Filing (Monate) —
# ein Request-lebenslanger Cache reicht, um wiederholte SEC-Abrufe pro
# Analyse-Session (viele Kennzahl-Karten desselben Symbols) zu vermeiden.
_DATA_SOURCE_CACHE: dict[str, dict] = {}
_DATA_SOURCE_CACHE_TTL_SECONDS = 3600


@router.get("/current-price/{symbol}")
def current_price(
    symbol: str,
    current_user: User = Depends(get_current_user),  # 🔐 PROTECTION
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    cached = _PRICE_CACHE.get(symbol)
    now = time.time()
    if cached and now - cached["fetched_at"] < _PRICE_CACHE_TTL_SECONDS:
        return cached["payload"]

    try:
        price = action.model.dataloader.get_current_price_per_share(symbol)

        if isinstance(price, dict) and "error" in price:
            return {"symbol": symbol, "error": price["error"]}

        payload = {
            "symbol": symbol,
            "price": float(price),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        _PRICE_CACHE[symbol] = {"fetched_at": now, "payload": payload}
        return payload
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}


@router.get("/data-source/{symbol}")
def data_source_summary(
    symbol: str,
    frequency: str = "annual",
    current_user: User = Depends(get_current_user),  # 🔐 PROTECTION
):
    """Herkunft + Stand der Fundamentaldaten eines Symbols — Grundlage für das
    Quellen-Badge im Frontend (Transparenz-Versprechen)."""
    action = get_action()
    symbol = _norm_symbol(symbol)

    cache_key = f"{symbol}:{frequency}"
    cached = _DATA_SOURCE_CACHE.get(cache_key)
    now = time.time()
    if cached and now - cached["fetched_at"] < _DATA_SOURCE_CACHE_TTL_SECONDS:
        return cached["payload"]

    try:
        result = action.model.dataloader.get_data_source_summary(symbol, frequency=frequency)
        _DATA_SOURCE_CACHE[cache_key] = {"fetched_at": now, "payload": result}
        return result
    except Exception:
        logger.exception("Data source summary failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}


# =============================
# EINZELNE KENNZAHLEN
# =============================

@router.get("/profit-growth")
def profit_growth(
    symbol: str,
    frequency: str = "annual",
    current_user: User = Depends(require_analysis_access),  # 🔐 PROTECTION + Quota
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        if frequency == "annual":
            result = action.model.calculate_avg_annual_profit_growth(
                symbol,
                start_date=None,
                end_date=None,
                use_cache=True
            )
        else:
            result = action.model.calculate_avg_quarterly_profit_growth(
                symbol,
                start_date=None,
                end_date=None,
                use_cache=True
            )

        return make_json_safe(result)

    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/kgv")
def kgv(symbol: str,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.calculate_KGV(symbol)
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}


@router.get("/net-debt-to-ebitda")
def net_debt_to_ebitda(symbol: str, frequency: str = "annual",
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.calculate_net_debt_to_ebitda(symbol, frequency=frequency)
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/dividend-yield-average")
def dividend_yield_average(symbol: str, years: int = 10,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.calculate_historical_dividend_yield_average(
            symbol,
            years=years
        )
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/tbv-and-price")
def tbv_and_price(symbol: str,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        tbv, price, pb = action.model.get_current_tbv_and_price(symbol)

        if isinstance(tbv, dict):
            return tbv

        return make_json_safe({
            "symbol": symbol,
            "tbv_per_share": tbv,
            "current_price": price,
            "price_to_tbv": pb,
        })
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}


@router.get("/kuv")
def kuv(symbol: str, frequency: str = "annual",
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.calculate_kuv(symbol, frequency=frequency)
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/roe")
def roe(symbol: str, frequency: str = "annual",
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.calculate_roe(symbol, frequency=frequency)
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/debt-to-equity")
def debt_to_equity(symbol: str, frequency: str = "annual",
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.calculate_debt_to_equity(symbol, frequency=frequency)
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/peg-ratio")
def peg_ratio(symbol: str,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.calculate_peg_ratio(
            symbol,
            start_date=None,
            end_date=None,
            use_cache=True
        )
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/interest-coverage-ratio")
def interest_coverage_ratio(symbol: str, frequency: str = "annual",
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.calculate_interest_coverage_ratio(symbol, frequency=frequency)
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/cashflow-margin")
def cashflow_margin(symbol: str, frequency: str = "annual",
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.calculate_cashflow_margin(symbol, frequency=frequency)
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/inventory-to-revenue")
def inventory_to_revenue(symbol: str, frequency: str = "annual",
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.calculate_inventory_to_revenue_ratio(
            symbol,
            frequency=frequency
        )
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/cash-to-market-cap")
def cash_to_market_cap(symbol: str, frequency: str = "annual",
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.calculate_cash_to_market_cap(
            symbol,
            frequency=frequency,
            use_cache=True
        )
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/payout-ratio")
def payout_ratio(symbol: str, threshold: float = 75.0,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.analyze_payout_ratio(
            symbol=symbol,
            threshold=threshold
        )
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/ev-to-sales")
def ev_to_sales(symbol: str, frequency: str = "annual",
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.calculate_ev_to_sales(
            symbol,
            frequency=frequency,
            use_cache=True
        )
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/price-to-free-cashflow")
def price_to_free_cashflow(symbol: str, frequency: str = "annual",
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.calculate_price_to_freeCashflow(
            symbol,
            use_cache=True,
            frequency=frequency
        )
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/annual-inflation-rate")
def annual_inflation_rate(
    current_date_str: str | None = None,
    target_date_str: str | None = None,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()

    try:
        result = action.model.calculate_annual_inflation_rate(
            current_date_str=current_date_str,
            target_date_str=target_date_str
        )
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed")
        return {"error": _GENERIC_METRIC_ERROR}

@router.get("/total-inflation")
def total_inflation_for_period(start_date: str, end_date: str,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()

    try:
        result = action.model.calculate_total_inflation_for_period(
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed")
        return {"error": _GENERIC_METRIC_ERROR}

@router.get("/avg-quarterly-profit-growth")
def avg_quarterly_profit_growth(symbol: str,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.calculate_avg_quarterly_profit_growth(
            symbol,
            start_date=None,
            end_date=None,
            use_cache=True
        )
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/avg-annual-profit-growth")
def avg_annual_profit_growth(symbol: str,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.calculate_avg_annual_profit_growth(
            symbol,
            start_date=None,
            end_date=None,
            use_cache=True
        )
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/compare-quarterly-profit-growth-to-inflation")
def compare_quarterly_profit_growth_to_inflation(symbol: str,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.compare_avg_quarterly_growth_to_inflation(
            symbol,
            start_date=None,
            end_date=None,
            use_cache=True
        )
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/compare-annual-profit-growth-to-inflation")
def compare_annual_profit_growth_to_inflation(symbol: str,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.compare_avg_annual_growth_to_inflation(
            symbol,
            start_date=None,
            end_date=None,
            use_cache=True
        )
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/ev-to-ebit")
def ev_to_ebit(symbol: str, frequency: str = "annual",
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.calculate_ev_to_ebit(
            symbol,
            use_cache=True,
            frequency=frequency
        )
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/ev-to-ebitda")
def ev_to_ebitda(symbol: str, frequency: str = "annual",
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.calculate_ev_to_ebitda(
            symbol,
            use_cache=True,
            frequency=frequency
        )
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/price-to-ebit")
def price_to_ebit(symbol: str, frequency: str = "annual",
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.calculate_price_to_ebit(
            symbol,
            use_cache=True,
            frequency=frequency
        )
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/roic")
def roic(symbol: str, frequency: str = "annual",
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.calculate_ROIC(
            symbol,
            frequency=frequency,
            use_cache=True
        )
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/net-current-assets")
def net_current_assets(symbol: str, frequency: str = "annual",
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.calculate_current_netCurrentAssets(
            symbol,
            frequency=frequency,
            use_cache=True
        )
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/historical-market-cap")
def historical_market_cap(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        df = action.model.calculate_historical_market_cap(
            symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )

        if df is None:
            return {"symbol": symbol, "error": "Keine historischen Marktkapitalisierungsdaten verfügbar."}

        result = {
            "symbol": symbol,
            "rows": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "market_cap": row["MarketCap"],
                    "shares_outstanding": row["commonStockSharesOutstanding"],
                    "close": row["Close"],
                }
                for idx, row in df.iterrows()
            ]
        }

        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/historical-enterprise-value")
def historical_enterprise_value(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        df = action.model.calculate_historical_ev(
            symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )

        if df is None:
            return {"symbol": symbol, "error": "Keine historischen EV-Daten verfügbar."}

        result = {
            "symbol": symbol,
            "rows": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "enterprise_value": row["EV"],
                }
                for idx, row in df.iterrows()
            ]
        }

        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/historical-sales")
def historical_sales(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        df = action.model.calculate_historical_sales(
            symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )

        if df is None:
            return {"symbol": symbol, "error": "Keine historischen Umsatzdaten verfügbar."}

        result = {
            "symbol": symbol,
            "rows": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "sales": row["Sales"],
                }
                for idx, row in df.iterrows()
            ]
        }

        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/historical-ebit")
def historical_ebit(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        df = action.model.calculate_historical_ebit(
            symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )

        if df is None:
            return {"symbol": symbol, "error": "Keine historischen EBIT-Daten verfügbar."}

        result = {
            "symbol": symbol,
            "rows": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "ebit": row["EBIT"],
                }
                for idx, row in df.iterrows()
            ]
        }

        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/historical-ebitda")
def historical_ebitda(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        df = action.model.calculate_historical_ebitda(
            symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )

        if df is None:
            return {"symbol": symbol, "error": "Keine historischen EBITDA-Daten verfügbar."}

        result = {
            "symbol": symbol,
            "rows": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "ebit": row["EBIT"],
                    "depreciation_and_amortization": row["depreciationAndAmortization"],
                    "ebitda": row["EBITDA"],
                }
                for idx, row in df.iterrows()
            ]
        }

        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/historical-net-current-assets")
def historical_net_current_assets(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        df = action.model.calculate_historical_netCurrentAssets(
            symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )

        if df is None:
            return {"symbol": symbol, "error": "Keine historischen Net-Current-Assets-Daten verfügbar."}

        result = {
            "symbol": symbol,
            "rows": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "net_current_assets": row["NetCurrentAssets"],
                }
                for idx, row in df.iterrows()
            ]
        }

        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/historical-operating-cashflow")
def historical_operating_cashflow(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        df = action.model.calculate_historical_OperatingCashflow(
            symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )

        if df is None:
            return {"symbol": symbol, "error": "Keine historischen Operating-Cashflow-Daten verfügbar."}

        result = {
            "symbol": symbol,
            "rows": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "operating_cashflow": row["OperatingCashflow"],
                }
                for idx, row in df.iterrows()
            ]
        }

        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/historical-free-cashflow")
def historical_free_cashflow(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        df = action.model.calculate_historical_FreeCashflow(
            symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )

        if df is None:
            return {"symbol": symbol, "error": "Keine historischen Free-Cashflow-Daten verfügbar."}

        result = {
            "symbol": symbol,
            "rows": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "free_cashflow": row["FreeCashflow"],
                    "operating_cashflow": row["OperatingCashflow"],
                    "capital_expenditures": row["CapitalExpenditures"],
                }
                for idx, row in df.iterrows()
            ]
        }

        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/historical-tangible-book-value")
def historical_tangible_book_value(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        df = action.model.calculate_historical_TangibleBookValue(
            symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )

        if df is None:
            return {"symbol": symbol, "error": "Keine historischen Tangible-Book-Value-Daten verfügbar."}

        result = {
            "symbol": symbol,
            "rows": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "tangible_book_value": row["TangibleBookValue"],
                    "total_assets": row["totalAssets"],
                    "intangible_assets": row["intangibleAssets"],
                    "goodwill": row["goodwill"],
                    "total_liabilities": row["totalLiabilities"],
                }
                for idx, row in df.iterrows()
            ]
        }

        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}



@router.get("/historical-ev-sales")
def historical_ev_sales(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        df = action.model.calculate_historical_ev_sales(
            symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )

        if df is None:
            return {"symbol": symbol, "error": "Keine historischen EV/Sales-Daten verfügbar."}

        result = {
            "symbol": symbol,
            "rows": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "ev_sales": row["EV_Sales"],
                }
                for idx, row in df.iterrows()
            ]
        }

        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/historical-ev-ebit")
def historical_ev_ebit(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        df = action.model.calculate_historical_ev_to_ebit(
            symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )

        if df is None:
            return {"symbol": symbol, "error": "Keine historischen EV/EBIT-Daten verfügbar."}

        result = {
            "symbol": symbol,
            "rows": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "ev_ebit": row["EV_EBIT"],
                }
                for idx, row in df.iterrows()
            ]
        }

        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/historical-ev-ebitda")
def historical_ev_ebitda(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        df = action.model.calculate_historical_ev_to_ebitda(
            symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )

        if df is None:
            return {"symbol": symbol, "error": "Keine historischen EV/EBITDA-Daten verfügbar."}

        result = {
            "symbol": symbol,
            "rows": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "ev_ebitda": row["EV_EBITDA"],
                }
                for idx, row in df.iterrows()
            ]
        }

        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/historical-price-to-book")
def historical_price_to_book(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        df = action.model.calculate_historical_price_to_book(
            symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )

        if df is None:
            return {"symbol": symbol, "error": "Keine historischen Price-to-Book-Daten verfügbar."}

        result = {
            "symbol": symbol,
            "rows": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "price_to_book": row["Price_Book"],
                    "price": row["Price"],
                    "total_assets": row["totalAssets"],
                    "total_liabilities": row["totalLiabilities"],
                    "shares_outstanding": row["commonStockSharesOutstanding"],
                }
                for idx, row in df.iterrows()
            ]
        }

        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/historical-price-to-sales")
def historical_price_to_sales(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        df = action.model.calculate_historical_price_to_sales(
            symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )

        if df is None:
            return {"symbol": symbol, "error": "Keine historischen Price-to-Sales-Daten verfügbar."}

        result = {
            "symbol": symbol,
            "rows": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "price_to_sales": row["Price_Sales"],
                    "price": row["Price"],
                    "sales": row["Sales"],
                    "shares_outstanding": row["commonStockSharesOutstanding"],
                }
                for idx, row in df.iterrows()
            ]
        }

        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/historical-price-to-ebit")
def historical_price_to_ebit(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        df = action.model.calculate_historical_price_to_ebit(
            symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )

        if df is None:
            return {"symbol": symbol, "error": "Keine historischen Price-to-EBIT-Daten verfügbar."}

        result = {
            "symbol": symbol,
            "rows": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "price_to_ebit": row["Price_EBIT"],
                    "price": row["Price"],
                    "ebit": row["EBIT"],
                    "shares_outstanding": row["commonStockSharesOutstanding"],
                }
                for idx, row in df.iterrows()
            ]
        }

        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}


@router.get("/historical-price-to-net-current-assets")
def historical_price_to_net_current_assets(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        df = action.model.calculate_historical_price_netCurrentAssets(
            symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )

        if df is None:
            return {"symbol": symbol, "error": "Keine historischen Price-to-Net-Current-Assets-Daten verfügbar."}

        result = {
            "symbol": symbol,
            "rows": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "price_to_net_current_assets": row["Price_NetCurrentAssets"],
                    "price": row["Price"],
                    "net_current_assets": row["NetCurrentAssets"],
                    "shares_outstanding": row["commonStockSharesOutstanding"],
                }
                for idx, row in df.iterrows()
            ]
        }

        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/historical-price-to-operating-cashflow")
def historical_price_to_operating_cashflow(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        df = action.model.calculate_historical_price_OperatingCashflow(
            symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )

        if df is None:
            return {"symbol": symbol, "error": "Keine historischen Price-to-Operating-Cashflow-Daten verfügbar."}

        result = {
            "symbol": symbol,
            "rows": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "price_to_operating_cashflow": row["Price_OperatingCashflow"],
                    "price": row["Price"],
                    "operating_cashflow": row["OperatingCashflow"],
                    "shares_outstanding": row["commonStockSharesOutstanding"],
                }
                for idx, row in df.iterrows()
            ]
        }

        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/historical-price-to-free-cashflow")
def historical_price_to_free_cashflow(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        df = action.model.calculate_historical_Price_FreeCashflow(
            symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )

        if df is None:
            return {"symbol": symbol, "error": "Keine historischen Price-to-Free-Cashflow-Daten verfügbar."}

        result = {
            "symbol": symbol,
            "rows": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "price_to_free_cashflow": row["Price_FreeCashflow"],
                    "price": row["Price"],
                    "free_cashflow": row["FreeCashflow"],
                    "operating_cashflow": row["OperatingCashflow"],
                    "capital_expenditures": row["CapitalExpenditures"],
                    "shares_outstanding": row["commonStockSharesOutstanding"],
                }
                for idx, row in df.iterrows()
            ]
        }

        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/historical-price-to-tangible-book-value")
def historical_price_to_tangible_book_value(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        df = action.model.calculate_historical_price_to_TangibleBookValue(
            symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )

        if df is None:
            return {"symbol": symbol, "error": "Keine historischen Price-to-Tangible-Book-Value-Daten verfügbar."}

        result = {
            "symbol": symbol,
            "rows": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "price_to_tangible_book_value": row["Price_TangibleBookValue"],
                    "price": row["Price"],
                    "tangible_book_value": row["TangibleBookValue"],
                    "shares_outstanding": row["commonStockSharesOutstanding"],
                }
                for idx, row in df.iterrows()
            ]
        }

        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/tbv-bandwidth")
def tbv_bandwidth(symbol: str, min_years: float = 10.0,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.evaluate_tbv_bandwidth(
            symbol,
            min_years=min_years,
            use_cache=True
        )

        if isinstance(result, dict) and "error" in result:
            return result

        pb_df = result.get("pb")
        if pb_df is not None and not pb_df.empty:
            result["pb"] = [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "price_to_tangible_book_value": row["Price_TangibleBookValue"],
                    "price": row["Price"],
                    "tangible_book_value": row["TangibleBookValue"],
                    "shares_outstanding": row["commonStockSharesOutstanding"],
                }
                for idx, row in pb_df.iterrows()
            ]

        return make_json_safe(result)

    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/ebit-bandwidth")
def ebit_bandwidth(symbol: str, min_years: float = 10.0,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.model.evaluate_ebit_bandwidth(
            symbol,
            min_years=min_years,
            use_cache=True
        )

        if isinstance(result, dict) and "error" in result:
            return result

        ebit_df = result.get("ebit")
        if ebit_df is not None and not ebit_df.empty:
            result["ebit"] = [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "price_to_ebit": row["Price_EBIT"],
                    "price": row["Price"],
                    "ebit": row["EBIT"],
                }
                for idx, row in ebit_df.iterrows()
            ]

        return make_json_safe(result)

    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/crv")
def crv(symbol: str,
    current_user: User = Depends(require_analysis_access),
):
    action = get_action()
    symbol = _norm_symbol(symbol)

    try:
        result = action.calculate_crv_by_sector_multiples(symbol)
        return make_json_safe(result)
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}
