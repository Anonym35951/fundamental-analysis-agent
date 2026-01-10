import time
from datetime import datetime

class AgentOrchestrator:
    def __init__(self, action_module):
        self.action = action_module

        self.analysis_map = {
            "Wachstumswerte": {"func": self.action.analyze_wachstumswerte, "frequencies": ["annual", "quarterly"]},
            "Dividendenwerte": {"func": self.action.analyze_dividend_companies, "frequencies": ["annual"]},
            "Average Grower": {"func": self.action.analyze_average_grower, "frequencies": ["annual"]},
            "Typische Zykliker": {"func": self.action.analyze_typical_cyclers, "frequencies": ["annual", "quarterly"]},
            "Zyklische Turnarounds": {"func": self.action.analyze_cycler_turnarounds, "frequencies": ["annual", "quarterly"]},
            "Optionality": {"func": self.action.analyze_optionality, "frequencies": ["annual"]},
            "Asset Play": {"func": self.action.analyze_asset_play, "frequencies": ["annual"]},
        }

    def run_full_analysis_collect(self, symbol: str) -> dict:
        symbol = symbol.strip().upper()
        started = time.time()

        results = []

        for analysis_name, config in self.analysis_map.items():
            func = config["func"]
            for freq in config["frequencies"]:
                # robust: frequency nur übergeben, wenn die Methode es hat
                try:
                    if "frequency" in func.__code__.co_varnames:
                        data = func(symbol, frequency=freq)
                    else:
                        data = func(symbol)

                except Exception as e:
                    data = {"symbol": symbol, "frequency": freq, "error": str(e)}

                results.append({
                    "analysis": analysis_name,
                    "frequency": freq,
                    "data": data
                })

        duration_ms = int((time.time() - started) * 1000)

        return {
            "symbol": symbol,
            "results": results,
            "meta": {
                "started_at": datetime.utcnow().isoformat() + "Z",
                "duration_ms": duration_ms,
                "count": len(results)
            }
        }