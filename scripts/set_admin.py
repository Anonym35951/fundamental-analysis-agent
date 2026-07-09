"""Einmaliges Setzen von plan="admin" für einen bestehenden User.

Nutzung: python scripts/set_admin.py <email>

Wird gebraucht, weil der Admin-Endpoint (api/routes/admin_customers.py:
update_customer_plan) selbst schon einen Admin-Account voraussetzt
(Henne-Ei-Problem beim allerersten Admin). Läuft gegen DATABASE_URL aus der
aktiven Umgebung (.env lokal, bzw. Render-Env-Var wenn dort ausgeführt).
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
        if user.plan == "admin":
            print(f"{email} ist bereits admin.")
            return
        user.plan = "admin"
        db.commit()
        print(f"{email} ist jetzt admin.")
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Nutzung: python scripts/set_admin.py <email>")
        sys.exit(1)
    main(sys.argv[1])
