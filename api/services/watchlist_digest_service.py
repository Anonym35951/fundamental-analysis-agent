import logging
from datetime import datetime, timedelta
from api.utils.time import utcnow

from sqlalchemy.orm import Session

from api.models.favorite import Favorite
from api.models.product_event import ProductEvent
from api.models.user import User
from api.services.email_service import send_email_safely, send_weekly_watchlist_digest_email
from api.services.event_service import log_event

logger = logging.getLogger(__name__)

DIGEST_LOOKBACK_DAYS = 7


def _recent_filing_entries_by_symbol(db: Session) -> dict[str, dict]:
    """Liest die in den letzten DIGEST_LOOKBACK_DAYS Tagen von
    filing_alert_service.check_new_filings geloggten "filing_alert_sent"-
    Events aus product_events — statt eine zweite eigene SEC-Abfrage zu
    machen, wird der bereits vom 4-A-Worker ermittelte Zustand
    wiederverwendet."""
    since = utcnow() - timedelta(days=DIGEST_LOOKBACK_DAYS)
    events = (
        db.query(ProductEvent)
        .filter(
            ProductEvent.event_type == "filing_alert_sent",
            ProductEvent.created_at >= since,
        )
        .all()
    )

    entries_by_symbol: dict[str, dict] = {}
    for event in events:
        metadata = event.event_metadata or {}
        symbol = metadata.get("symbol")
        if not symbol:
            continue
        # Bei mehreren Events fürs selbe Symbol in der Woche zählt die
        # jeweils neueste Meldung (created_at ist über die Query bereits
        # aufsteigend genug beisammen — ein einfaches Überschreiben reicht,
        # da Events chronologisch entstehen).
        entries_by_symbol[symbol] = {
            "symbol": symbol,
            "form": metadata.get("form"),
            "filing_date": metadata.get("filing_date"),
        }

    return entries_by_symbol


def send_weekly_digests(db: Session) -> int:
    """Verschickt an jeden Nutzer mit mindestens einem Favoriten, der diese
    Woche eine neue 10-K/10-Q-Meldung hatte, eine wöchentliche
    Rückblick-Mail. Nutzer ohne relevante Neuigkeiten erhalten keine Mail
    (kein Spam).

    Returns:
        Anzahl verschickter Digest-E-Mails.
    """
    recent_entries = _recent_filing_entries_by_symbol(db)
    if not recent_entries:
        return 0

    symbols_with_news = list(recent_entries.keys())

    favorites = (
        db.query(Favorite.user_id, Favorite.symbol)
        .filter(Favorite.symbol.in_(symbols_with_news))
        .all()
    )

    symbols_by_user: dict[int, list[str]] = {}
    for user_id, symbol in favorites:
        symbols_by_user.setdefault(user_id, []).append(symbol)

    if not symbols_by_user:
        return 0

    users = db.query(User).filter(User.id.in_(symbols_by_user.keys())).all()

    digests_sent = 0
    for user in users:
        user_symbols = symbols_by_user.get(user.id, [])
        if not user_symbols:
            continue

        filing_entries = [recent_entries[symbol] for symbol in user_symbols]

        send_email_safely(
            send_weekly_watchlist_digest_email,
            to_email=user.email,
            filing_entries=filing_entries,
        )
        digests_sent += 1

        log_event(
            db,
            "watchlist_digest_sent",
            user_id=user.id,
            metadata={"symbol_count": len(filing_entries)},
        )

    return digests_sent
