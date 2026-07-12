from datetime import datetime, timezone


def utcnow() -> datetime:
    """Naiver UTC-Zeitstempel, äquivalent zum deprecated datetime.utcnow().

    Alle DateTime-Spalten im Schema sind naiv (kein timezone=True) - ein
    Wechsel auf timezone-aware Werte würde jeden Vergleich mit einem aus der
    DB gelesenen Wert brechen (TypeError: can't compare offset-naive and
    offset-aware datetimes). Dieser Helper ersetzt datetime.utcnow() 1:1,
    ohne das Naiv/Aware-Verhalten zu ändern.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)
