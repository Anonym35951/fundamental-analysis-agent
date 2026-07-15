"""Gemeinsame Existenz-/Aktiv-Prüfung für Symbole beim Start einer Analyse
(EVOLVING.md EV-014): `start_single_analysis`/`start_full_analysis`/
`start_custom_analysis`/`run_definition` normalisierten Symbole bisher nur
Groß-/Kleinschreibung, ohne sie gegen die `symbols`-Tabelle zu prüfen -
unbekannte oder delistete Symbole scheiterten dadurch erst tief im Agenten
mit kryptischen Fehlern statt einer frühen, verständlichen 422-Antwort.

WICHTIG (Deploy-Reihenfolge, siehe EVOLVING.md EV-014): Dieser Helfer darf
in Produktion erst scharf geschaltet werden, NACHDEM per
GET /admin/symbols/stats (EV-012) nachgewiesen wurde, dass die
Produktions-`symbols`-Tabelle das volle Universum enthält - sonst würden
gültige Symbole abgelehnt, nur weil der Import dort noch nicht lief."""
from fastapi import HTTPException
from sqlalchemy.orm import Session

from api.models.symbol import Symbol


def _normalize_for_lookup(symbol: str) -> str:
    """Gleiche Punkt→Bindestrich-Normalisierung wie
    scripts/import_symbols.py::_normalize_for_yfinance. Aktienklassen-
    Symbole wie "BRK.B" werden am Markt oft mit Punkt geschrieben, in der
    `symbols`-Tabelle aber im Bindestrich-Stil ("BRK-B") gespeichert (siehe
    dortigen Kommentar) - ohne diese Angleichung würde "BRK.B" hier
    fälschlich als unbekannt abgelehnt."""
    return symbol.replace(".", "-")


def ensure_known_symbol(db: Session, symbol: str) -> str:
    """Wirft HTTPException(422), wenn `symbol` (bereits vom Aufrufer
    .strip().upper()-normalisiert) in der `symbols`-Tabelle nicht existiert
    oder delistet ist. Gibt andernfalls die kanonische, in der DB
    gespeicherte Schreibweise zurück (z. B. "BRK-B" für Eingabe "BRK.B") -
    Aufrufer sollen `symbol = ensure_known_symbol(db, symbol)` verwenden,
    damit die restliche Anfrage (Job-Erstellung, Agent-Aufruf, Historie) mit
    derselben Schreibweise arbeitet wie der Rest der App (yfinance/SEC-EDGAR
    erwarten den Bindestrich-Stil, siehe scripts/import_symbols.py).

    Greift bewusst NICHT ein, wenn die Tabelle insgesamt leer ist (z. B.
    lokale Entwicklung ohne durchgeführten Import, oder eine Produktions-DB
    kurz vor dem ersten Sync) - in diesem Zustand existiert schlicht noch
    kein geprüftes Universum, und ein blockierender 422 für JEDES Symbol
    wäre schlimmer als die (bereits vor EV-014 bestehende) fehlende
    Prüfung; das Symbol wird dann unverändert zurückgegeben."""
    if db.query(Symbol.id).first() is None:
        return symbol

    lookup_symbol = _normalize_for_lookup(symbol)
    row = db.query(Symbol).filter(Symbol.symbol.ilike(lookup_symbol)).first()

    if row is None:
        raise HTTPException(
            status_code=422,
            detail=f"Symbol {symbol} ist nicht im unterstützten NYSE/Nasdaq-Universum.",
        )
    if not row.is_active:
        raise HTTPException(
            status_code=422,
            detail=f"Symbol {symbol} ist nicht mehr gelistet.",
        )
    return row.symbol
