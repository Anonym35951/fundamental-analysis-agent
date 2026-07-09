import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Zentrales Logging-Setup für die API.

    Loggt nach stdout (Render sammelt stdout ein). Level per LOG_LEVEL-Env
    steuerbar; idempotent, damit Uvicorn-Reloads keine doppelten Handler
    anhängen.
    """
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level)
