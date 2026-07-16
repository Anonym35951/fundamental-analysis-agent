import logging
import time
from datetime import datetime, timezone

from dateutil.relativedelta import relativedelta
from fastapi import APIRouter, HTTPException, Depends, Request
from api.utils.json_sanitize import make_json_safe
from api.utils.reporting_currency import resolve_reporting_currency
from api.core.rate_limit import limiter
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
@limiter.limit("60/minute")
def current_price(
    request: Request,
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
            # EVOLVING.md EV-021: das Symbol-Universum ist auf NYSE+NASDAQ
            # beschränkt (siehe scripts/import_symbols.py) - Kurse dort
            # notieren immer in USD, unabhängig von der Berichtswährung der
            # Fundamentaldaten (reporting_currency), die z. B. bei Foreign-
            # Private-Issuer-Filern abweichen kann.
            "currency": "USD",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        _PRICE_CACHE[symbol] = {"fetched_at": now, "payload": payload}
        return payload
    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}


@router.get("/data-source/{symbol}")
@limiter.limit("60/minute")
def data_source_summary(
    request: Request,
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


# EVOLVING.md EV-060: eigener, gröberer Cache als `_PRICE_CACHE` (12s TTL,
# einzelner Live-Kurs) - eine ganze Kursreihe pro Symbol+Range ist teurer zu
# berechnen (Downsampling) und ändert sich innerhalb weniger Minuten nicht
# sichtbar, daher 15 Minuten TTL.
_PRICE_HISTORY_CACHE: dict[str, dict] = {}
_PRICE_HISTORY_CACHE_TTL_SECONDS = 15 * 60

_PRICE_HISTORY_RANGE_MONTHS = {"1m": 1, "2m": 2, "3m": 3, "6m": 6, "1y": 12, "2y": 24, "5y": 60}
_PRICE_HISTORY_VALID_RANGES = frozenset(_PRICE_HISTORY_RANGE_MONTHS) | {"max"}
# Ab diesen Ranges ist die Rohserie (taeglich) zu groß fuer eine schlanke
# Chart-Payload - wöchentliche Downsampling-Stufe (EVOLVING.md EV-060/8.5).
_PRICE_HISTORY_WEEKLY_DOWNSAMPLE_RANGES = frozenset({"2y", "5y"})
# "max" wird nur downgesampelt, wenn die Rohserie tatsächlich groß ist (junge
# Symbole mit kurzer Historie sollen taeglich aufgeloest bleiben).
_PRICE_HISTORY_MAX_POINTS_BEFORE_MONTHLY_DOWNSAMPLE = 1500


def _fetch_price_history_payload(action, symbol: str, range: str) -> dict:
    """Gemeinsame Kern-Logik für `price_history()` und `price_history_batch()`
    (EVOLVING.md EV-060/070): Cache-Check, Laden, Range-Zuschnitt,
    Downsampling. Wirft `HTTPException` (404/502) bei Fehlern -
    `price_history()` lässt das direkt durchreichen, `price_history_batch()`
    fängt es je Symbol ab und baut daraus einen `{symbol, error}`-Eintrag,
    statt den gesamten Batch an einem einzelnen kaputten Symbol scheitern zu
    lassen."""
    cache_key = f"{symbol}:{range}"
    cached = _PRICE_HISTORY_CACHE.get(cache_key)
    now = time.time()
    if cached and now - cached["fetched_at"] < _PRICE_HISTORY_CACHE_TTL_SECONDS:
        return cached["payload"]

    try:
        # interval="1d": get_max_historical_stock_data() defaultet auf "1mo"
        # (fuer die monatsweise Marktkapitalisierungs-Berechnung) - Kurscharts
        # brauchen die taegliche Aufloesung als Rohbasis fuer die Range-/
        # Downsampling-Logik unten.
        df = action.model.dataloader.get_max_historical_stock_data(symbol, use_cache=True, interval="1d")
    except Exception:
        logger.exception("Price-history fetch failed for symbol=%s", symbol)
        raise HTTPException(status_code=502, detail=_GENERIC_METRIC_ERROR)

    if df is None or df.empty or "Close" not in df.columns:
        raise HTTPException(status_code=404, detail=f"Keine Kursdaten für Symbol {symbol} verfügbar.")

    closes = df["Close"].dropna()
    if closes.empty:
        raise HTTPException(status_code=404, detail=f"Keine Kursdaten für Symbol {symbol} verfügbar.")

    if range != "max":
        # Anker = neuester verfuegbarer Handelstag (nicht "heute" - vermeidet
        # eine leere/verkürzte Serie bei Wochenenden/Feiertagen oder
        # Datenverzug), analog zur Frontend-Utility filterSeriesByRange (EV-040).
        anchor = closes.index.max()
        cutoff = anchor - relativedelta(months=_PRICE_HISTORY_RANGE_MONTHS[range])
        closes = closes[closes.index >= cutoff]

    if closes.empty:
        raise HTTPException(status_code=404, detail=f"Keine Kursdaten für Symbol {symbol} im gewählten Zeitraum.")

    if range in _PRICE_HISTORY_WEEKLY_DOWNSAMPLE_RANGES:
        closes = closes.resample("W").last().dropna()
    elif range == "max" and len(closes) > _PRICE_HISTORY_MAX_POINTS_BEFORE_MONTHLY_DOWNSAMPLE:
        closes = closes.resample("ME").last().dropna()

    payload = make_json_safe({
        "symbol": symbol,
        # Kursbasiert, immer USD (NYSE/NASDAQ-Universum) - unabhängig von der
        # Berichtswährung der Fundamentaldaten (EVOLVING.md EV-022).
        "currency": "USD",
        "range": range,
        "rows": [{"date": idx.strftime("%Y-%m-%d"), "close": float(value)} for idx, value in closes.items()],
    })

    _PRICE_HISTORY_CACHE[cache_key] = {"fetched_at": now, "payload": payload}
    return payload


@router.get("/price-history/{symbol}")
@limiter.limit("30/minute")
def price_history(
    request: Request,
    symbol: str,
    range: str = "1y",
    current_user: User = Depends(get_current_user),  # 🔐 PROTECTION, kein Quota-Verbrauch (analog current-price)
):
    """Tägliche (bzw. bei langen Ranges herunterge­samplete) Adjusted-Close-
    Kursreihe für Kurscharts (EVOLVING.md EV-060/061/062). Adjusted Close
    kommt automatisch aus `get_max_historical_stock_data` -> `yfinance`s
    `auto_adjust=True`-Standard (D5) - keine separate Bereinigung nötig."""
    symbol = _norm_symbol(symbol)

    if range not in _PRICE_HISTORY_VALID_RANGES:
        raise HTTPException(
            status_code=422,
            detail=f"Ungültiger range-Parameter: {range}. Erlaubt: {', '.join(sorted(_PRICE_HISTORY_VALID_RANGES))}.",
        )

    action = get_action()
    return _fetch_price_history_payload(action, symbol, range)


# EVOLVING.md EV-070: Dashboard-Sparklines brauchen nur kurze Ranges (Payload-
# Schutz - ein Batch aus bis zu 20 Symbolen × mehrjähriger Historie wäre
# unnötig groß für eine reine Mini-Trend-Anzeige).
_PRICE_HISTORY_BATCH_VALID_RANGES = frozenset({"1m", "3m"})
_PRICE_HISTORY_BATCH_MAX_SYMBOLS = 20
# Delay NUR zwischen tatsächlichen (nicht gecachten) yfinance-Abrufen -
# schont die Datenquelle bei einem Batch mit vielen Cache-Misses, ohne
# bereits gecachte Symbole künstlich zu verlangsamen.
_PRICE_HISTORY_BATCH_THROTTLE_SECONDS = 0.2


@router.get("/price-history-batch")
@limiter.limit("10/minute")
def price_history_batch(
    request: Request,
    symbols: str,
    range: str = "1m",
    current_user: User = Depends(get_current_user),  # 🔐 PROTECTION, kein Quota-Verbrauch (analog current-price)
):
    """Preis-Historien für mehrere Symbole in einem Request (EVOLVING.md
    EV-070) - für die Dashboard-Favoriten-Sparklines (EV-071), damit N
    Favoriten nicht N Einzel-Requests auslösen. Teilfehler pro Symbol landen
    als `{symbol, error}`-Eintrag im Ergebnis statt den gesamten Batch mit
    einem 4xx/5xx scheitern zu lassen."""
    if range not in _PRICE_HISTORY_BATCH_VALID_RANGES:
        raise HTTPException(
            status_code=422,
            detail=f"Ungültiger range-Parameter: {range}. Erlaubt: {', '.join(sorted(_PRICE_HISTORY_BATCH_VALID_RANGES))}.",
        )

    symbol_list = [_norm_symbol(s) for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        raise HTTPException(status_code=422, detail="Mindestens ein Symbol erforderlich.")
    if len(symbol_list) > _PRICE_HISTORY_BATCH_MAX_SYMBOLS:
        raise HTTPException(
            status_code=422,
            detail=f"Maximal {_PRICE_HISTORY_BATCH_MAX_SYMBOLS} Symbole pro Anfrage (erhalten: {len(symbol_list)}).",
        )

    action = get_action()
    results = []
    did_uncached_fetch = False

    for symbol in symbol_list:
        cache_key = f"{symbol}:{range}"
        cached = _PRICE_HISTORY_CACHE.get(cache_key)
        is_cached = bool(cached) and time.time() - cached["fetched_at"] < _PRICE_HISTORY_CACHE_TTL_SECONDS

        if not is_cached and did_uncached_fetch:
            time.sleep(_PRICE_HISTORY_BATCH_THROTTLE_SECONDS)

        try:
            results.append(_fetch_price_history_payload(action, symbol, range))
        except HTTPException as exc:
            results.append({"symbol": symbol, "error": exc.detail})
        except Exception:
            logger.exception("Batch price-history fetch failed for symbol=%s", symbol)
            results.append({"symbol": symbol, "error": _GENERIC_METRIC_ERROR})
        finally:
            if not is_cached:
                did_uncached_fetch = True

    return {"results": results}


# =============================
# EINZELNE KENNZAHLEN
# =============================

@router.get("/profit-growth")
@limiter.limit("60/minute")
def profit_growth(
    request: Request,
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
        elif frequency == "quarterly":
            result = action.model.calculate_avg_quarterly_profit_growth(
                symbol,
                start_date=None,
                end_date=None,
                use_cache=True
            )
        else:
            # EV-133: Ø Jahres-/Quartalswachstum (AAGR/AQGR) unterstützt kein
            # "ttm" (EVOLVING.md §9/§11 E5 - TTM-vs-TTM-Vorjahr wäre neue
            # Numerik, kein reiner Alias auf einen bestehenden Pfad). Vorher
            # fiel jeder Nicht-"annual"-Wert stillschweigend in den
            # Quarterly-Zweig - mit "ttm" als neu erreichbarem Wert wäre das
            # eine fachlich falsche Berechnung statt eines klaren Fehlers.
            return {
                "symbol": symbol,
                "error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.",
            }

        return make_json_safe(result)

    except Exception:
        logger.exception("Metric endpoint failed for symbol=%s", symbol)
        return {"symbol": symbol, "error": _GENERIC_METRIC_ERROR}

@router.get("/kgv")
@limiter.limit("60/minute")
def kgv(request: Request, symbol: str,
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
@limiter.limit("60/minute")
def net_debt_to_ebitda(request: Request, symbol: str, frequency: str = "annual",
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
@limiter.limit("60/minute")
def dividend_yield_average(request: Request, symbol: str, years: int = 10,
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
@limiter.limit("60/minute")
def tbv_and_price(request: Request, symbol: str,
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
@limiter.limit("60/minute")
def kuv(request: Request, symbol: str, frequency: str = "annual",
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
@limiter.limit("60/minute")
def roe(request: Request, symbol: str, frequency: str = "annual",
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
@limiter.limit("60/minute")
def debt_to_equity(request: Request, symbol: str, frequency: str = "annual",
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
@limiter.limit("60/minute")
def peg_ratio(request: Request, symbol: str,
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
@limiter.limit("60/minute")
def interest_coverage_ratio(request: Request, symbol: str, frequency: str = "annual",
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
@limiter.limit("60/minute")
def cashflow_margin(request: Request, symbol: str, frequency: str = "annual",
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
@limiter.limit("60/minute")
def inventory_to_revenue(request: Request, symbol: str, frequency: str = "annual",
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
@limiter.limit("60/minute")
def cash_to_market_cap(request: Request, symbol: str, frequency: str = "annual",
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
@limiter.limit("60/minute")
def payout_ratio(request: Request, symbol: str, threshold: float = 75.0,
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
@limiter.limit("60/minute")
def ev_to_sales(request: Request, symbol: str, frequency: str = "annual",
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
@limiter.limit("60/minute")
def price_to_free_cashflow(request: Request, symbol: str, frequency: str = "annual",
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
@limiter.limit("60/minute")
def annual_inflation_rate(
    request: Request,
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
@limiter.limit("60/minute")
def total_inflation_for_period(request: Request, start_date: str, end_date: str,
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
@limiter.limit("60/minute")
def avg_quarterly_profit_growth(request: Request, symbol: str,
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
@limiter.limit("60/minute")
def avg_annual_profit_growth(request: Request, symbol: str,
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
@limiter.limit("60/minute")
def compare_quarterly_profit_growth_to_inflation(request: Request, symbol: str,
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
@limiter.limit("60/minute")
def compare_annual_profit_growth_to_inflation(request: Request, symbol: str,
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
@limiter.limit("60/minute")
def ev_to_ebit(request: Request, symbol: str, frequency: str = "annual",
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
@limiter.limit("60/minute")
def ev_to_ebitda(request: Request, symbol: str, frequency: str = "annual",
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
@limiter.limit("60/minute")
def price_to_ebit(request: Request, symbol: str, frequency: str = "annual",
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
@limiter.limit("60/minute")
def roic(request: Request, symbol: str, frequency: str = "annual",
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
@limiter.limit("60/minute")
def net_current_assets(request: Request, symbol: str, frequency: str = "annual",
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
@limiter.limit("60/minute")
def historical_market_cap(
    request: Request,
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
            # Marktkapitalisierung = Kurs (immer USD, NYSE/NASDAQ-Universum)
            # × Aktienanzahl - kein SEC-Reporting-Currency-Lookup nötig.
            "currency": "USD",
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
@limiter.limit("60/minute")
def historical_enterprise_value(
    request: Request,
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
            "reporting_currency": resolve_reporting_currency(action, symbol),
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
@limiter.limit("60/minute")
def historical_sales(
    request: Request,
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
            "reporting_currency": resolve_reporting_currency(action, symbol),
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
@limiter.limit("60/minute")
def historical_ebit(
    request: Request,
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
            "reporting_currency": resolve_reporting_currency(action, symbol),
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
@limiter.limit("60/minute")
def historical_ebitda(
    request: Request,
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
            "reporting_currency": resolve_reporting_currency(action, symbol),
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
@limiter.limit("60/minute")
def historical_net_current_assets(
    request: Request,
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
            "reporting_currency": resolve_reporting_currency(action, symbol),
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
@limiter.limit("60/minute")
def historical_operating_cashflow(
    request: Request,
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
            "reporting_currency": resolve_reporting_currency(action, symbol),
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
@limiter.limit("60/minute")
def historical_free_cashflow(
    request: Request,
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
            "reporting_currency": resolve_reporting_currency(action, symbol),
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
@limiter.limit("60/minute")
def historical_tangible_book_value(
    request: Request,
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
            "reporting_currency": resolve_reporting_currency(action, symbol),
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
@limiter.limit("60/minute")
def historical_ev_sales(
    request: Request,
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
@limiter.limit("60/minute")
def historical_ev_ebit(
    request: Request,
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
@limiter.limit("60/minute")
def historical_ev_ebitda(
    request: Request,
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
@limiter.limit("60/minute")
def historical_price_to_book(
    request: Request,
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
@limiter.limit("60/minute")
def historical_price_to_sales(
    request: Request,
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
@limiter.limit("60/minute")
def historical_price_to_ebit(
    request: Request,
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
@limiter.limit("60/minute")
def historical_price_to_net_current_assets(
    request: Request,
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
@limiter.limit("60/minute")
def historical_price_to_operating_cashflow(
    request: Request,
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
@limiter.limit("60/minute")
def historical_price_to_free_cashflow(
    request: Request,
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
@limiter.limit("60/minute")
def historical_price_to_tangible_book_value(
    request: Request,
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
@limiter.limit("60/minute")
def tbv_bandwidth(request: Request, symbol: str, min_years: float = 10.0,
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
@limiter.limit("60/minute")
def ebit_bandwidth(request: Request, symbol: str, min_years: float = 10.0,
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
@limiter.limit("60/minute")
def crv(request: Request, symbol: str,
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
