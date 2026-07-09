"""Einmaliges manuelles Setzen von email_verified=True für einen bestehenden User.

Nutzung: python scripts/verify_email.py <email>

Notfall-Werkzeug fuer den Fall, dass die Bestaetigungsmail nicht ankommt
(z. B. SMTP-Fehlkonfiguration) und ein Nutzer sich sonst nicht einloggen
kann - der Login-Endpoint (api/routes/auth.py) verlangt email_verified=True.
Laeuft gegen DATABASE_URL aus der aktiven Umgebung (.env lokal, bzw.
Render-Env-Var wenn dort ausgefuehrt).
"""
import sys

from api.core.database import SessionLocal
from api.models.user import User


def main(email: str) -> None:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if user is None:
            print(f"Kein User mit E-Mail {email} gefunden.")
            sys.exit(1)
        if user.email_verified:
            print(f"{email} ist bereits verifiziert.")
            return
        user.email_verified = True
        db.commit()
        print(f"{email} ist jetzt verifiziert.")
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Nutzung: python scripts/verify_email.py <email>")
        sys.exit(1)
    main(sys.argv[1])
