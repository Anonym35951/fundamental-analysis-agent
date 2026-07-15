"""Sicherer Wrapper um SecSource.get_reporting_currency (EVOLVING.md
EV-021): wird von den Analyse-Job-Workern (analyze.py/full_analysis.py/
custom_analysis.py) im Hintergrund-Thread aufgerufen, NIE synchron im
Request-Handler (ein SEC-Netzwerk-Roundtrip würde sonst jeden Analyse-Start
spürbar verlangsamen). Ein Fehler bei der Währungs-Ermittlung darf einen
laufenden Analyse-Job niemals zum Absturz bringen - die Kennzahlen selbst
sind wichtiger als das Currency-Label, daher wird hier jede Exception
abgefangen und als `None` (unbestimmbar) behandelt."""
import logging

from agent.AgentAction import AgentAction

logger = logging.getLogger(__name__)


def resolve_reporting_currency(action: AgentAction, symbol: str) -> str | None:
    try:
        return action.dataloader.sec_source.get_reporting_currency(symbol)
    except Exception:
        logger.exception("Reporting-Currency-Ermittlung fehlgeschlagen: symbol=%s", symbol)
        return None
