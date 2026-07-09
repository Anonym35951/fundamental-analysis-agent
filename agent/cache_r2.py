"""Best-effort Cloudflare-R2-Backing für den lokalen Datei-Cache.

Render-Web-Services auf dem Free-Plan haben kein Persistent Disk (das ist
ein Paid-Feature, siehe LAUNCH.md P0-3b) — der lokale Cache unter CACHE_DIR
überlebt daher keinen Deploy/Neustart. Cloudflare R2 (S3-kompatibler
Objektspeicher, 10 GB/Monat kostenlos, keine Egress-Kosten) dient als
persistente Zweitkopie: bei jedem Cache-Write spiegeln wir die Datei nach
R2 (`persist`), bei einem lokalen Cache-Miss versuchen wir zuerst, sie von
dort zu laden (`warm`) — die lokale mtime wird dabei auf den R2-Zeitstempel
gesetzt, damit die bestehende TTL-Prüfung in DataLoader/SecSource
unverändert greift, ohne die TTL-Logik zu duplizieren.

Ohne die vier R2_*-Env-Vars ist diese Klasse ein reines No-op — lokale
Dev-Umgebungen und CI sind unberührt.
"""

import logging
import os

logger = logging.getLogger(__name__)

_ENV_VARS = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME")

# Kurze Timeouts: ein R2-Ausfall/eine langsame Verbindung darf eine
# Analyse-Anfrage nicht spürbar verzögern — im Zweifel lieber wie ein
# normaler Cache-Miss verhalten und live nachladen.
_CONNECT_TIMEOUT_SECONDS = 5
_READ_TIMEOUT_SECONDS = 10


def _build_client():
    if not all(os.environ.get(v) for v in _ENV_VARS):
        return None, None, None

    import boto3
    from botocore.client import Config
    from botocore.exceptions import ClientError

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(
            connect_timeout=_CONNECT_TIMEOUT_SECONDS,
            read_timeout=_READ_TIMEOUT_SECONDS,
            retries={"max_attempts": 1},
        ),
    )
    return client, os.environ["R2_BUCKET_NAME"], ClientError


class R2CacheSync:
    def __init__(self):
        try:
            self._client, self._bucket, self._client_error_cls = _build_client()
        except Exception as e:
            logger.warning(f"R2-Cache-Sync konnte nicht initialisiert werden, bleibt deaktiviert: {e}")
            self._client, self._bucket, self._client_error_cls = None, None, None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def warm(self, local_path: str, key: str) -> None:
        """Lädt `key` von R2 nach `local_path`, falls die lokale Datei fehlt.
        Kein Objekt in R2 -> stiller No-op (normaler Cache-Miss, live abrufen)."""
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
            logger.warning(f"R2-Cache-Warmup für {key} fehlgeschlagen: {e}")

    def persist(self, local_path: str, key: str) -> None:
        """Spiegelt eine frisch geschriebene lokale Cache-Datei nach R2."""
        if not self.enabled:
            return
        try:
            self._client.upload_file(local_path, self._bucket, key)
        except Exception as e:
            logger.warning(f"R2-Cache-Upload für {key} fehlgeschlagen: {e}")
