from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def client_ip(request: Request) -> str:
    """Client-IP hinter dem Render-Proxy.

    X-Forwarded-For wird von jedem Hop APPENDED (nicht prepended): der
    letzte Eintrag ist immer die IP, die der unmittelbar vorgelagerte,
    vertrauenswürdige Proxy (Render) tatsächlich gesehen hat. Ein Client
    kann beliebige Werte VOR seiner eigenen IP einschleusen (z. B.
    "1.2.3.4" als kompletten Header senden), aber nicht den von Render
    angehängten, echten letzten Eintrag fälschen.
    Frueher wurde hier der ERSTE Eintrag genutzt - der ist jedoch exakt der
    vom Client frei waehlbare Wert und liess sich nutzen, um pro Request
    einen neuen Rate-Limit-Bucket vorzutaeuschen (Login-/Registrierungs-
    Bruteforce trotz "5/minute"-Limit)."""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
        if parts:
            return parts[-1]
    return get_remote_address(request)


limiter = Limiter(key_func=client_ip)
