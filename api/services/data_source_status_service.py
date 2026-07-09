import logging
import time

import requests
import yfinance as yf

from api.core.config import settings

logger = logging.getLogger(__name__)

# Kurzer, gemeinsamer Cache für alle Nutzer, die die Statusseite gleichzeitig
# aufrufen — vermeidet, dass jeder Seitenaufruf einen eigenen Live-Check
# gegen SEC/Yahoo auslöst. 5 Minuten sind aktuell genug für einen "ist der
# Dienst gerade grundsätzlich erreichbar"-Indikator.
_CACHE_TTL_SECONDS = 300
_CHECK_TIMEOUT_SECONDS = 5

_cache: dict | None = None
_cache_checked_at: float = 0.0


def _check_sec() -> dict:
    try:
        response = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": settings.EMAIL_FROM},
            timeout=_CHECK_TIMEOUT_SECONDS,
        )
        status = "ok" if response.ok else "down"
    except Exception:
        status = "down"
    return {"name": "SEC EDGAR", "status": status}


def _check_yahoo() -> dict:
    try:
        # fast_info statt info: löst nur einen leichten Endpoint aus, nicht
        # das volle (deutlich teurere) Profil-/Kennzahlen-Bundle.
        price = yf.Ticker("AAPL").fast_info.get("lastPrice")
        status = "ok" if price else "down"
    except Exception:
        status = "down"
    return {"name": "Yahoo Finance", "status": status}


def get_data_source_status() -> list[dict]:
    """Liefert den aktuellen Erreichbarkeits-Status der externen
    Datenquellen (SEC, Yahoo Finance) — passt zur Transparenz-Positionierung
    ("SEC: ✓ · Yahoo: ✓"). Ergebnis wird kurz gecacht, siehe oben."""
    global _cache, _cache_checked_at

    now = time.time()
    if _cache is not None and now - _cache_checked_at < _CACHE_TTL_SECONDS:
        return _cache

    result = [_check_sec(), _check_yahoo()]
    _cache = result
    _cache_checked_at = now
    return result
