"""Hintergrund-Synchronisation der `symbols`-Tabelle (EVOLVING.md EV-010/
EV-011): scripts/import_symbols.py lief bisher nur manuell. Die Migration
seedet lediglich 23 Zeilen, sodass eine Produktions-DB, auf der der Import
nie von Hand ausgeführt wurde, faktisch nur diese 23 Symbole durchsuchbar
macht statt des vollen NYSE+NASDAQ-Universums (~5900 aktive Symbole lokal).

Dieses Modul kapselt denselben Import (fetch_nasdaq_listed/
fetch_other_listed/upsert_symbols aus scripts/import_symbols.py) für zwei
Aufrufer: einen einmaligen Startup-Check (api.main, EV-010) und einen
wöchentlichen Hintergrund-Worker (api.main, EV-011). Beide teilen sich
_import_lock, damit nie zwei Importe parallel laufen.
"""
import asyncio
import logging

from sqlalchemy import func
from sqlalchemy.orm import Session

from api.models.symbol import Symbol
from scripts.import_symbols import fetch_nasdaq_listed, fetch_other_listed, upsert_symbols

logger = logging.getLogger(__name__)

# Unterhalb dieser Anzahl aktiver Symbole gilt die Tabelle als "faktisch
# leer" (z. B. nur der 23-Zeilen-Migrations-Seed) - das tatsächliche
# NYSE+NASDAQ-Universum hat ~5900 aktive Zeilen (siehe EVOLVING.md EV-010,
# lokal verifizierter Stand).
SYMBOL_IMPORT_MIN_COUNT = 1000

_import_lock = asyncio.Lock()


def count_active_symbols(db: Session) -> int:
    return db.query(func.count(Symbol.id)).filter(Symbol.is_active.is_(True)).scalar() or 0


def run_symbol_import(db: Session) -> dict:
    """Synchron/blockierend (Netzwerk-Download + DB-Upsert) - für den
    Aufruf über asyncio.to_thread gedacht, damit der Event-Loop des Servers
    währenddessen nicht blockiert wird."""
    rows = fetch_nasdaq_listed() + fetch_other_listed()
    upsert_symbols(db, rows)
    return {"fetched": len(rows)}


def _run_import_with_own_session(session_factory) -> dict:
    db = session_factory()
    try:
        return run_symbol_import(db)
    finally:
        db.close()


async def _run_import_locked(session_factory) -> dict | None:
    """Führt den Import aus, außer es läuft bereits einer - dann wird
    sofort (ohne zu warten) übersprungen statt sich anzustellen, damit sich
    z. B. ein Startup-Check und ein zeitgleich fälliger Wochen-Lauf nicht
    gegenseitig blockieren."""
    if _import_lock.locked():
        logger.info("Symbol-Import läuft bereits, dieser Aufruf wird übersprungen.")
        return None

    async with _import_lock:
        try:
            result = await asyncio.to_thread(_run_import_with_own_session, session_factory)
            logger.info("Symbol-Import abgeschlossen: %s Zeilen verarbeitet.", result["fetched"])
            return result
        except Exception:
            logger.exception("Symbol-Import fehlgeschlagen - Serverbetrieb läuft unbeeinflusst weiter.")
            return None


async def sync_symbols_on_startup(session_factory) -> None:
    """Startet den vollen Import nur, wenn die `symbols`-Tabelle unterhalb
    von SYMBOL_IMPORT_MIN_COUNT aktiver Zeilen liegt (Render-Erststart bzw.
    Fälle, in denen scripts/import_symbols.py nie manuell lief). Bereits
    befüllte Tabellen lösen KEINEN Netzwerk-Request aus, damit ein
    Server-Neustart nicht bei jedem Deploy unnötig die NASDAQ-Trader-Dateien
    lädt."""
    db = session_factory()
    try:
        current = count_active_symbols(db)
    finally:
        db.close()

    if current >= SYMBOL_IMPORT_MIN_COUNT:
        logger.info(
            "Symbol-Universum bereits befüllt (%s aktive Symbole, Schwellwert %s) - kein Startup-Import.",
            current, SYMBOL_IMPORT_MIN_COUNT,
        )
        return

    logger.info(
        "Symbol-Universum unterhalb Schwellwert (%s < %s aktive Symbole) - starte Startup-Import.",
        current, SYMBOL_IMPORT_MIN_COUNT,
    )
    await _run_import_locked(session_factory)


async def symbol_sync_worker(session_factory, interval_seconds: float = 7 * 24 * 60 * 60) -> None:
    """Wöchentlicher Hintergrund-Worker (EV-011): ruft den Import
    UNBEDINGT auf (kein Schwellwert-Gate wie beim Startup-Check), damit
    neue Listings erscheinen und delistete Symbole zuverlässig auf
    is_active=False gesetzt werden - genau das würde ein reiner
    Schwellwert-Check nach dem ersten erfolgreichen Import nie wieder
    auslösen, da die Tabelle danach dauerhaft oberhalb von
    SYMBOL_IMPORT_MIN_COUNT liegt."""
    while True:
        await asyncio.sleep(interval_seconds)
        logger.info("Wöchentlicher Symbol-Sync gestartet.")
        await _run_import_locked(session_factory)
