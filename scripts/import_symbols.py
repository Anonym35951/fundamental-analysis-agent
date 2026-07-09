"""Befuellt die `symbols`-Tabelle mit allen NYSE+NASDAQ-Stammaktien.

Nutzung (aus dem Repo-Root, mit aktiver venv):
    python -m scripts.import_symbols                  # nur Listing-Import (schnell)
    python -m scripts.import_symbols --enrich          # zusaetzlich Branchen anreichern (langsam, Stunden)
    python -m scripts.import_symbols --enrich --limit 200 --sleep 1.0
    python -m scripts.import_symbols --enrich --retry-failed

Zwei unabhaengige Schritte:
1. Listing-Import (immer): laedt die NASDAQ-Trader-Symbolverzeichnisse (frei,
   kein API-Key, taeglich aktualisiert), filtert auf NYSE/NASDAQ-Stammaktien
   und gleicht sie mit der `symbols`-Tabelle ab (neu/aktualisiert/delisted).
   Dauert Sekunden.
2. Branchen-Anreicherung (--enrich, optional): holt sector/industry/Name
   je Symbol per yfinance nach, ein Request pro Symbol - bei ~7000 Symbolen
   dauert das mit sinnvollem --sleep mehrere Stunden. Ist bewusst NICHT
   Voraussetzung fuer die Suche (api/routes/analyze.py: search_symbols) -
   die Branchen-Klassifikation fuer die CRV-Berechnung passiert ohnehin lazy
   zur Analysezeit (agent/DataLoader.get_company_profile, siehe
   agent/AgentAction.calculate_crv_by_sector_multiples). Diese Anreicherung
   dient nur der Dropdown-Anzeige und der Betreiber-Uebersicht.

Resumierbar per Konstruktion: enrich_symbols() waehlt via
`WHERE enriched_at IS NULL` und committet nach JEDER Zeile - ein Abbruch
mittendrin verliert keinen Fortschritt, ein Neustart macht dort weiter, wo
es aufgehoert hat. Symbole, deren yfinance-Lookup fehlschlaegt, bekommen
enriched_at trotzdem gesetzt (sector/industry bleiben NULL), damit sie nicht
bei jedem Lauf erneut versucht werden - `--retry-failed` setzt genau diese
Zeilen (enriched_at gesetzt, sector NULL) fuer einen neuen Versuch zurueck.
"""
import argparse
import re
import time
from datetime import datetime

import requests
import yfinance as yf

from api.core.database import SessionLocal
from api.models.symbol import Symbol

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ComAnalysisSymbolImport/1.0)"}
REQUEST_TIMEOUT_SECONDS = 30

# Namen, die auf Nicht-Stammaktien hindeuten (Warrants, Rechte, Vorzugsaktien,
# Notes, Einheiten aus SPAC-Strukturen etc.) - case-insensitive.
# WICHTIG: "Depositary Shares" bewusst NICHT hier - "American Depositary
# Shares" (ADRs) sind die Standard-Notierungsform auslaendischer Stammaktien
# in den USA (Alibaba, Baidu, JD.com, XPeng, SAP, Novo Nordisk, ...) und
# waeren damit faelschlich ausgeschlossen worden. Echte Vorzugs-Depositary-
# Shares enthalten so gut wie immer zusaetzlich "Preferred" im Namen und
# werden darueber bereits erfasst. Bekannter Kompromiss: Limited-Partnership-
# "Units" (z. B. AllianceBernstein Holding L.P., MLPs) werden dadurch
# ebenfalls ausgeschlossen, da sich ihre Bezeichnung nicht von SPAC-Units
# unterscheiden laesst - akzeptiert, da es sich ohnehin um eine andere
# Wertpapierklasse (Partnerschaftsanteile statt Aktien) handelt.
NON_COMMON_STOCK_PATTERN = re.compile(
    r"\b(warrants?|rights?|units?|preferred|notes?)\b", re.IGNORECASE
)


class RawSymbol:
    __slots__ = ("symbol", "name", "exchange")

    def __init__(self, symbol: str, name: str, exchange: str):
        self.symbol = symbol
        self.name = name
        self.exchange = exchange


def _fetch_lines(url: str) -> list[str]:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    lines = response.text.splitlines()
    # Kopfzeile und die abschliessende "File Creation Time: ..."-Zeile raus.
    return [line for line in lines[1:] if line and not line.startswith("File Creation Time")]


def _is_common_stock(security_name: str, symbol: str) -> bool:
    if "$" in symbol:
        return False
    if NON_COMMON_STOCK_PATTERN.search(security_name):
        return False
    return True


def _normalize_for_yfinance(symbol: str) -> str:
    """Aktienklassen-Symbole (z. B. "BRK.B") verwenden am Markt oft einen
    Punkt, yfinance/SEC-EDGAR und der Rest dieser App erwarten den
    Bindestrich-Stil ("BRK-B") - hier direkt normalisiert gespeichert statt
    ein separates Anzeige-Feld zu fuehren, das nirgends sonst existiert."""
    return symbol.replace(".", "-")


def fetch_nasdaq_listed() -> list[RawSymbol]:
    rows: list[RawSymbol] = []
    for line in _fetch_lines(NASDAQ_LISTED_URL):
        fields = line.split("|")
        if len(fields) < 7:
            continue
        symbol, security_name, _market_category, test_issue, _financial_status, _round_lot, etf = fields[:7]
        if test_issue != "N" or etf != "N":
            continue
        if not _is_common_stock(security_name, symbol):
            continue
        rows.append(RawSymbol(_normalize_for_yfinance(symbol), security_name.strip(), "NASDAQ"))
    return rows


def fetch_other_listed() -> list[RawSymbol]:
    rows: list[RawSymbol] = []
    for line in _fetch_lines(OTHER_LISTED_URL):
        fields = line.split("|")
        if len(fields) < 7:
            continue
        act_symbol, security_name, exchange, _cqs_symbol, etf, _round_lot, test_issue = fields[:7]
        if exchange not in ("N", "A") or test_issue != "N" or etf != "N":
            continue
        if not _is_common_stock(security_name, act_symbol):
            continue
        exchange_label = "NYSE" if exchange == "N" else "NYSE American"
        rows.append(RawSymbol(_normalize_for_yfinance(act_symbol), security_name.strip(), exchange_label))
    return rows


def upsert_symbols(db, rows: list[RawSymbol]) -> None:
    seen_symbols = {row.symbol for row in rows}
    existing = {s.symbol: s for s in db.query(Symbol).all()}
    now = datetime.utcnow()

    for row in rows:
        current = existing.get(row.symbol)
        if current is None:
            db.add(
                Symbol(
                    symbol=row.symbol,
                    name=row.name,
                    exchange=row.exchange,
                    is_active=True,
                    updated_at=now,
                )
            )
        elif current.name != row.name or current.exchange != row.exchange or not current.is_active:
            current.name = row.name
            current.exchange = row.exchange
            current.is_active = True
            current.updated_at = now

    # Symbole, die nicht mehr in der Boersen-Liste stehen, gelten als
    # delisted - Zeile bleibt (Historie/Referenzen), taucht aber nicht mehr
    # in der Suche auf.
    for symbol, existing_row in existing.items():
        if symbol not in seen_symbols and existing_row.is_active:
            existing_row.is_active = False
            existing_row.updated_at = now

    db.commit()
    new_count = len(seen_symbols - set(existing))
    delisted_count = len([s for s in existing if s not in seen_symbols])
    print(
        f"Symbole verarbeitet: {len(rows)} (davon neu: {new_count}, "
        f"als delisted markiert: {delisted_count})"
    )


def enrich_symbols(db, limit: int | None, sleep_seconds: float) -> None:
    # is_active=True: nicht mehr gelistete/gefilterte Symbole (siehe
    # upsert_symbols) tauchen ohnehin nie in der Suche auf - kein Grund,
    # dafuer yfinance-Requests zu verbrauchen.
    query = (
        db.query(Symbol)
        .filter(Symbol.enriched_at.is_(None), Symbol.is_active.is_(True))
        .order_by(Symbol.symbol)
    )
    if limit:
        query = query.limit(limit)
    pending = query.all()

    if not pending:
        print("Keine unangereicherten Symbole gefunden.")
        return

    print(f"Reichere {len(pending)} Symbole an (Ctrl+C jederzeit sicher abbrechbar, Fortschritt bleibt erhalten)...")

    for index, row in enumerate(pending, start=1):
        try:
            info = yf.Ticker(row.symbol).info or {}
            sector = info.get("sector")
            industry = info.get("industry")
            long_name = info.get("longName") or info.get("shortName")

            row.sector = sector
            row.industry = industry
            if long_name:
                row.name = long_name
            row.enriched_at = datetime.utcnow()
            row.updated_at = datetime.utcnow()
            db.commit()

            print(f"[{index}/{len(pending)}] {row.symbol}: sector={sector!r} industry={industry!r}")
        except Exception as exc:  # yfinance kann diverse Fehler werfen (Rate-Limit, Netzwerk, unbekanntes Symbol)
            db.rollback()
            row.enriched_at = datetime.utcnow()
            db.commit()
            print(f"[{index}/{len(pending)}] {row.symbol}: Fehler ({exc}) - als angereichert markiert, sector bleibt leer")

        if sleep_seconds:
            time.sleep(sleep_seconds)


def retry_failed(db) -> int:
    """Setzt enriched_at fuer Symbole zurueck, deren letzter Anreicherungs-
    versuch fehlgeschlagen ist (enriched_at gesetzt, aber sector weiterhin
    leer) - macht sie beim naechsten --enrich-Lauf erneut versuchbar."""
    failed = (
        db.query(Symbol)
        .filter(Symbol.enriched_at.isnot(None), Symbol.sector.is_(None))
        .all()
    )
    for row in failed:
        row.enriched_at = None
    db.commit()
    return len(failed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enrich", action="store_true", help="Branchen per yfinance nachladen (langsam)")
    parser.add_argument("--limit", type=int, default=None, help="Max. Anzahl Symbole pro --enrich-Lauf")
    parser.add_argument("--sleep", type=float, default=0.5, help="Pause zwischen yfinance-Requests (Sekunden)")
    parser.add_argument(
        "--retry-failed", action="store_true", help="Vorherige Anreicherungs-Fehlschlaege erneut versuchbar machen"
    )
    parser.add_argument(
        "--skip-import", action="store_true", help="Listing-Import ueberspringen (nur mit --enrich sinnvoll)"
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        if args.retry_failed:
            count = retry_failed(db)
            print(f"{count} zuvor fehlgeschlagene Symbole fuer erneuten Versuch zurueckgesetzt.")

        if not args.skip_import:
            print("Lade NASDAQ-Trader-Symbolverzeichnisse...")
            rows = fetch_nasdaq_listed() + fetch_other_listed()
            print(f"{len(rows)} Stammaktien nach Filterung gefunden.")
            upsert_symbols(db, rows)

        if args.enrich:
            enrich_symbols(db, limit=args.limit, sleep_seconds=args.sleep)
    finally:
        db.close()


if __name__ == "__main__":
    main()
