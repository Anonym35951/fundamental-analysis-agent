import logging
import time
from datetime import datetime

from sqlalchemy.orm import Session

from agent.data_sources.sec_source import SecSource
from api.core.config import settings
from api.models.favorite import Favorite
from api.models.filing_alert_state import FilingAlertState
from api.models.user import User
from api.services.email_service import send_email_safely, send_new_filing_alert_email
from api.services.event_service import log_event

logger = logging.getLogger(__name__)

# Kurze Pause zwischen SEC-Abfragen pro Symbol — rücksichtsvoll gegenüber der
# SEC-API, auch wenn die realistische Favoriten-Symbolzahl weit unter deren
# Fair-Use-Grenze (~10 req/s) bleibt.
SEC_REQUEST_DELAY_SECONDS = 0.3


def _parse_filing_date(raw: str | None):
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None


def check_new_filings(db: Session) -> int:
    """Prüft für jedes favorisierte Symbol, ob seit dem letzten Check eine
    neue 10-K/10-Q-Meldung bei der SEC eingegangen ist, und benachrichtigt
    alle Nutzer, die dieses Symbol favorisiert haben — 100% Information,
    null Beratung (siehe [[comanalysis-positioning]]).

    Der allererste Check für ein neu favorisiertes Symbol legt nur den
    Ausgangszustand fest und verschickt bewusst KEINE Benachrichtigung —
    sonst würde jede längst bekannte, alte Meldung beim ersten Lauf als
    "neu" gemeldet.

    Returns:
        Anzahl verschickter Benachrichtigungs-E-Mails (nicht Anzahl neuer
        Filings — ein Filing kann mehrere Favoriten-Nutzer treffen).
    """
    symbols = [
        row[0]
        for row in db.query(Favorite.symbol).distinct().all()
    ]
    if not symbols:
        return 0

    sec_source = SecSource(user_agent=settings.EMAIL_FROM)
    alerts_sent = 0

    for index, symbol in enumerate(symbols):
        if index > 0:
            time.sleep(SEC_REQUEST_DELAY_SECONDS)

        try:
            filing = sec_source.get_latest_filing(symbol)
        except Exception:
            logger.exception("Filing-Check fehlgeschlagen für %s", symbol)
            continue

        if "error" in filing:
            # Kein Absturz für einzelne Symbole (z. B. Foreign Private
            # Issuer ohne 10-K/10-Q, oder SEC vorübergehend nicht erreichbar).
            logger.info("Filing-Check ohne Ergebnis für %s: %s", symbol, filing["error"])
            continue

        state = db.query(FilingAlertState).filter(FilingAlertState.symbol == symbol).first()
        accession_number = filing["accession_number"]

        if state is None:
            db.add(
                FilingAlertState(
                    symbol=symbol,
                    last_seen_accession_number=accession_number,
                    last_seen_form=filing["form"],
                    last_seen_filing_date=_parse_filing_date(filing["filing_date"]),
                )
            )
            db.commit()
            continue

        if state.last_seen_accession_number == accession_number:
            state.last_checked_at = datetime.utcnow()
            db.commit()
            continue

        # Neue Meldung seit dem letzten Check.
        state.last_seen_accession_number = accession_number
        state.last_seen_form = filing["form"]
        state.last_seen_filing_date = _parse_filing_date(filing["filing_date"])
        db.commit()

        favoriting_users = (
            db.query(User)
            .join(Favorite, Favorite.user_id == User.id)
            .filter(Favorite.symbol == symbol)
            .all()
        )

        for user in favoriting_users:
            send_email_safely(
                send_new_filing_alert_email,
                to_email=user.email,
                symbol=symbol,
                form=filing["form"],
                filing_date=filing["filing_date"],
            )
            alerts_sent += 1

        log_event(
            db,
            "filing_alert_sent",
            metadata={
                "symbol": symbol,
                "form": filing["form"],
                "filing_date": filing["filing_date"],
                "recipient_count": len(favoriting_users),
            },
        )

    return alerts_sent
