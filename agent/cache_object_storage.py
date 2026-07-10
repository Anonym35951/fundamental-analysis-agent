"""Best-effort Object-Storage-Backing für den lokalen Datei-Cache.

Render-Web-Services auf dem Free-Plan haben kein Persistent Disk (Paid-
Feature, siehe LAUNCH.md P0-3b) — der lokale Cache unter CACHE_DIR überlebt
daher keinen Deploy/Neustart. Ein S3-kompatibler Objektspeicher dient als
persistente Zweitkopie: bei jedem Cache-Write spiegeln wir die Datei dorthin
(`persist`), bei einem lokalen Cache-Miss versuchen wir zuerst, sie von dort
zu laden (`warm`) — die lokale mtime wird dabei auf den Objekt-Zeitstempel
gesetzt, damit die bestehende TTL-Prüfung in DataLoader/SecSource
unverändert greift, ohne die TTL-Logik zu duplizieren.

Konfiguriert für Backblaze B2 (S3-kompatible API, 10 GB/Monat kostenlos,
kein Kreditkarten-Zwang für private Buckets — im Unterschied zu Cloudflare
R2, das eine Zahlungsmethode zur Aktivierung verlangt). Funktioniert dank
generischem boto3-S3-Client mit jedem S3-kompatiblen Anbieter, der einen
CACHE_S3_ENDPOINT_URL nach dem Schema `s3.<region>.<anbieter>.com` anbietet.

Ohne die vier CACHE_S3_*-Env-Vars ist diese Klasse ein reines No-op —
lokale Dev-Umgebungen und CI sind unberührt.
"""

import logging
import os

logger = logging.getLogger(__name__)

_ENV_VARS = (
    "CACHE_S3_ENDPOINT_URL",
    "CACHE_S3_ACCESS_KEY_ID",
    "CACHE_S3_SECRET_ACCESS_KEY",
    "CACHE_S3_BUCKET_NAME",
)

# Kurze Timeouts: ein Object-Storage-Ausfall/eine langsame Verbindung darf
# eine Analyse-Anfrage nicht spürbar verzögern — im Zweifel lieber wie ein
# normaler Cache-Miss verhalten und live nachladen.
_CONNECT_TIMEOUT_SECONDS = 5
_READ_TIMEOUT_SECONDS = 10


def _normalize_endpoint(raw: str) -> str:
    return raw if raw.startswith("http://") or raw.startswith("https://") else f"https://{raw}"


def _region_from_endpoint(endpoint: str) -> str:
    """Backblaze-B2-Endpoints haben das Schema s3.<region>.backblazeb2.com
    (z. B. s3.us-west-004.backblazeb2.com) - Region wird daraus abgeleitet,
    damit keine fünfte Env-Var nötig ist. Fällt auf "auto" zurück, falls das
    Schema nicht passt (z. B. bei R2-Endpoints, die "auto" ignorieren)."""
    host = endpoint.split("//", 1)[-1]
    parts = host.split(".")
    if len(parts) >= 3 and parts[0] == "s3":
        return parts[1]
    return "auto"


def _build_client():
    if not all(os.environ.get(v) for v in _ENV_VARS):
        return None, None, None

    import boto3
    from botocore.client import Config
    from botocore.exceptions import ClientError

    endpoint = _normalize_endpoint(os.environ["CACHE_S3_ENDPOINT_URL"])
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["CACHE_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["CACHE_S3_SECRET_ACCESS_KEY"],
        region_name=_region_from_endpoint(endpoint),
        config=Config(
            connect_timeout=_CONNECT_TIMEOUT_SECONDS,
            read_timeout=_READ_TIMEOUT_SECONDS,
            retries={"max_attempts": 1},
        ),
    )
    return client, os.environ["CACHE_S3_BUCKET_NAME"], ClientError


class ObjectStorageCacheSync:
    def __init__(self):
        try:
            self._client, self._bucket, self._client_error_cls = _build_client()
        except Exception as e:
            logger.warning(f"Object-Storage-Cache-Sync konnte nicht initialisiert werden, bleibt deaktiviert: {e}")
            self._client, self._bucket, self._client_error_cls = None, None, None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def warm(self, local_path: str, key: str) -> None:
        """Lädt `key` in `local_path`, falls die lokale Datei fehlt.
        Kein Objekt vorhanden -> stiller No-op (normaler Cache-Miss, live abrufen)."""
        if not self.enabled or os.path.exists(local_path):
            return
        try:
            head = self._client.head_object(Bucket=self._bucket, Key=key)
            self._client.download_file(self._bucket, key, local_path)
            timestamp = head["LastModified"].timestamp()
            os.utime(local_path, (timestamp, timestamp))
        except self._client_error_cls:
            pass
        except Exception as e:
            logger.warning(f"Cache-Warmup für {key} fehlgeschlagen: {e}")

    def persist(self, local_path: str, key: str) -> None:
        """Spiegelt eine frisch geschriebene lokale Cache-Datei in den Objektspeicher."""
        if not self.enabled:
            return
        try:
            self._client.upload_file(local_path, self._bucket, key)
        except Exception as e:
            logger.warning(f"Cache-Upload für {key} fehlgeschlagen: {e}")
