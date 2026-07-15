import asyncio
import logging
from datetime import date, datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from api.core.rate_limit import limiter

from api.routes.custom_analysis import router as custom_analysis_router
from api.routes.analyze import router as analyze_router
from api.routes.favorites import router as favorites_router
from api.routes.full_analysis import router as full_analysis_router
from api.routes.metric_routes import router as metric_router
from api.routes.auth import router as auth_router
from api.routes.billing import router as billing_router
from api.routes.stripe_webhook import router as stripe_webhook_router
from api.routes.admin_stats import router as admin_stats_router
from api.routes.admin_customers import router as admin_customers_router
from api.routes.admin_symbols import router as admin_symbols_router
from api.routes.status import router as status_router
from api.routes.support import router as support_router

from api.core.config import settings
from api.core.database import SessionLocal
from api.core.logging_config import setup_logging
from api.services.filing_alert_service import check_new_filings
from api.services.symbol_sync_service import sync_symbols_on_startup, symbol_sync_worker
from api.services.user_service import downgrade_expired_past_due_users
from api.services.watchlist_digest_service import send_weekly_digests

setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Error-Tracking: aktiviert sich nur, wenn SENTRY_DSN gesetzt ist (Render-Env).
if settings.SENTRY_DSN:
    import sentry_sdk

    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        traces_sample_rate=0.1,
        send_default_pii=False,
    )
    logger.info("Sentry error tracking enabled")

# In Produktion enumerieren /docs, /redoc und /openapi.json sonst die
# komplette API-Oberfläche (inkl. aller /admin/*-Routen) für jeden Besucher.
_is_production = settings.ENVIRONMENT == "production"
app = FastAPI(
    title="ComAnalysis API",
    version="0.1",
    docs_url=None if _is_production else "/docs",
    redoc_url=None if _is_production else "/redoc",
    openapi_url=None if _is_production else "/openapi.json",
)

app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "detail": (
                "Zu viele Anfragen. Bitte warte einen Moment und versuche es "
                "erneut."
            )
        },
    )


# allow_credentials=False + explizite Methoden/Headers statt "*": Auth läuft
# ausschließlich über den Authorization-Bearer-Header (localStorage-Token im
# Frontend), nie über Cookies — Credentials/Wildcards wären hier unnötig breit.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# 🔹 Root-Endpoints
@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


# 🔹 Background Job
async def downgrade_worker():
    while True:
        db = SessionLocal()
        try:
            downgraded = downgrade_expired_past_due_users(db)
            if downgraded > 0:
                logger.info("Auto-downgraded users: %s", downgraded)
        except Exception:
            logger.exception("Error in downgrade_worker")
        finally:
            db.close()

        await asyncio.sleep(60)  # alle 60 Sekunden prüfen


# 🔹 Background Job: neue SEC-Filings (10-K/10-Q) für Favoriten erkennen
async def filing_alert_worker():
    while True:
        db = SessionLocal()
        try:
            alerts_sent = check_new_filings(db)
            if alerts_sent > 0:
                logger.info("Filing alerts sent: %s", alerts_sent)
        except Exception:
            logger.exception("Error in filing_alert_worker")
        finally:
            db.close()

        # 10-K/10-Q-Meldungen sind kein minutengenaues Ereignis — ein
        # 6-Stunden-Takt (identisch mit der Cache-TTL in
        # SecSource.get_latest_filing) reicht für eine zeitnahe, aber
        # SEC-schonende Benachrichtigung.
        await asyncio.sleep(6 * 60 * 60)


# 🔹 Background Job: wöchentlicher Watchlist-Digest (Montags)
async def watchlist_digest_worker():
    last_digest_date: date | None = None

    while True:
        now = datetime.now(timezone.utc)

        if now.weekday() == 0 and last_digest_date != now.date():
            db = SessionLocal()
            try:
                digests_sent = send_weekly_digests(db)
                logger.info("Weekly watchlist digests sent: %s", digests_sent)
                last_digest_date = now.date()
            except Exception:
                logger.exception("Error in watchlist_digest_worker")
            finally:
                db.close()

        # Kein präzises Timing nötig — ein verpasster/verspäteter
        # Wochen-Digest ist keine Korrektheitsfrage (anders als der frühere
        # monatliche Quota-Reset), stündliches Prüfen reicht.
        await asyncio.sleep(60 * 60)


# Kein zeitgesteuerter Worker mehr fürs Monatslimit: try_consume_monthly_
# request_quota (api/services/user_service.py) resettet Free-User "on-read"
# anhand von usage_period_start. Ein zeitgesteuerter Worker konnte am
# Monatsersten ausfallen, wenn die App gerade deployed/schlafend war
# (Render) — Free-User blieben dann bis zum nächsten Trigger gesperrt.
# api/jobs/reset_monthly_usage.py bleibt als optionales manuelles/Cron-Tool
# bestehen, ist für die Korrektheit aber nicht mehr erforderlich.


@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(downgrade_worker())
    asyncio.create_task(filing_alert_worker())
    asyncio.create_task(watchlist_digest_worker())
    # EV-010: einmaliger Fire-and-Forget-Check, ob die symbols-Tabelle noch
    # den 23-Zeilen-Migrations-Seed hat (statt des vollen NYSE+NASDAQ-
    # Universums) - läuft im Hintergrund, blockiert den Serverstart nicht.
    asyncio.create_task(sync_symbols_on_startup(SessionLocal))
    # EV-011: wöchentlicher Sync, läuft UNABHÄNGIG vom Schwellwert oben,
    # damit neue Listings/Delistings auch nach dem ersten erfolgreichen
    # Import weiterhin erkannt werden.
    asyncio.create_task(symbol_sync_worker(SessionLocal))


# 🔹 Router einbinden
# WICHTIG: custom_analysis_router muss vor analyze_router registriert werden,
# da analyze_router eine Wildcard-Route "/{mode}/start" besitzt, die sonst
# "/analyze/custom/start" abfangen würde.
app.include_router(auth_router)
app.include_router(custom_analysis_router)
app.include_router(analyze_router)
app.include_router(favorites_router)
app.include_router(full_analysis_router)
app.include_router(metric_router)
app.include_router(billing_router)
app.include_router(stripe_webhook_router)
app.include_router(admin_stats_router)
app.include_router(admin_customers_router)
app.include_router(admin_symbols_router)
app.include_router(status_router)
app.include_router(support_router)