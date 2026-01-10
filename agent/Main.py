# agent/Main.py

from agent.ActionModule import AgentAction
from agent.AgentOrchestrator import AgentOrchestrator
import logging
logging.getLogger("agent").setLevel(logging.CRITICAL)


def main():
    symbol = input("Enter stock symbol: ").strip().upper()
    if not symbol:
        print("❌ Kein Symbol eingegeben.")
        return

    action = AgentAction()
    orchestrator = AgentOrchestrator(action)

    # ruft alle Analysen auf (annual + quarterly wo sinnvoll)
    orchestrator.run_full_analysis(symbol)


if __name__ == "__main__":
    main()