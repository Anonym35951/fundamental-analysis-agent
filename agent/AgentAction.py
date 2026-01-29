import math
from datetime import datetime, timedelta

import pandas as pd
from scipy.stats import false_discovery_control
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from agent.DataLoader import DataLoader
from agent.Model import Model
import logging

class AgentAction:
    def __init__(self, symbol = None):
        self.logger = logging.getLogger(__name__)
        self.dataloader = DataLoader()
        self.symbol = symbol
        self.model = Model()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def analyze_dividend_companies(self, symbol):
        """
        Analyzes a company for dividend investment suitability based on specified criteria.
        Args:
            symbol (str): Stock ticker symbol (e.g., 'AAPL').
        Returns:
            dict: Contains results of dividend analysis, including criteria checks and overall assessment.
                  Example:
                  {
                      "symbol": "AAPL",
                      "dividend_yield": {"value": 5.2, "meets_criterion": True},
                      "earnings_growth_vs_inflation": {
                          "annual_aagr": 6.0,
                          "quarterly_aqgr": 5.5,
                          "inflation": 3.2,
                          "meets_criterion": True,
                          "message": "Annual growth of 6.0% and/or quarterly growth of 5.5% exceeds inflation of 3.2%."
                      },
                      "payout_ratio": {"value": 60, "meets_criterion": True, "message": "Ausschüttungsquote ≤ 75%."},
                      "interest_coverage_ratio": {"value": 10.0, "meets_criterion": True},
                      "net_debt_to_ebitda": {"value": 1.5, "meets_criterion": True},
                      "ev_to_ebit": {"value": 15.0, "meets_criterion": True},
                      "overall_assessment": "Suitable",
                      "message": "All criteria met."
                  }
                  or on error:
                  {
                      "symbol": symbol,
                      "error": "Error description"
                  }
        """
        try:
            # Validierung des Symbols
            if not isinstance(symbol, str):
                return {"symbol": symbol, "error": "Symbol muss ein String sein."}

            result = {"symbol": symbol}
            all_criteria_met = True
            messages = []

            # 1. Dividend Yield (≥ 5%)
            dividend_data = self.model.dataloader.get_dividend_data(symbol)
            if "error" in dividend_data:
                return {"symbol": symbol, "error": dividend_data["error"]}
            if "dividend_yield" not in dividend_data or dividend_data["dividend_yield"] is None:
                dividend_yield = 0
                messages.append(f"Keine Dividenden für {symbol} verfügbar.")
            else:
                dividend_yield = dividend_data["dividend_yield"]
            dividend_yield_met = dividend_yield >= 5
            result["dividend_yield"] = {
                "value": round(dividend_yield, 2),
                "meets_criterion": dividend_yield_met
            }
            if not dividend_yield_met:
                all_criteria_met = False
                messages.append(f"Dividend yield {dividend_yield}% is below 5%.")

            # 2. Earnings Growth vs. Inflation
            annual_growth = self.model.compare_avg_annual_growth_to_inflation(symbol, None, None)
            quarterly_growth = self.model.compare_avg_quarterly_growth_to_inflation(symbol, None, None)

            earnings_growth_met = False
            annual_aagr = None
            quarterly_aqgr = None
            inflation = None
            actual_start_date = None
            actual_end_date = None

            if "error" in annual_growth or "error" in quarterly_growth:
                error_msg = []
                if "error" in annual_growth:
                    error_msg.append(f"Annual growth error: {annual_growth['error']}")
                if "error" in quarterly_growth:
                    error_msg.append(f"Quarterly growth error: {quarterly_growth['error']}")
                return {"symbol": symbol, "error": "; ".join(error_msg)}

            if "aagr" in annual_growth:
                annual_aagr = annual_growth["aagr"]
                inflation = annual_growth["total_inflation"]
                actual_start_date = annual_growth.get("actual_start_date")
                actual_end_date = annual_growth.get("actual_end_date")
                if bool(annual_growth["outperforms_inflation"]):  # Explizite bool-Konvertierung
                    earnings_growth_met = True
            elif "net_incomes" in annual_growth:
                messages.append(annual_growth["message"])

            if "aqgr" in quarterly_growth:
                quarterly_aqgr = quarterly_growth["aqgr"]
                if inflation is None:
                    inflation = quarterly_growth["total_inflation"]
                if actual_start_date is None:
                    actual_start_date = quarterly_growth.get("actual_start_date")
                if actual_end_date is None:
                    actual_end_date = quarterly_growth.get("actual_end_date")
                if bool(quarterly_growth["outperforms_inflation"]):  # Explizite bool-Konvertierung
                    earnings_growth_met = True
            elif "net_incomes" in quarterly_growth:
                messages.append(quarterly_growth["message"])

            result["earnings_growth_vs_inflation"] = {
                "annual_aagr": annual_aagr,
                "quarterly_aqgr": quarterly_aqgr,
                "inflation": inflation,
                "actual_start_date": actual_start_date,
                "actual_end_date": actual_end_date,
                "meets_criterion": earnings_growth_met,
                "message": (
                    f"Annual growth of {annual_aagr or 'N/A'}% and/or quarterly growth of {quarterly_aqgr or 'N/A'}% "
                    f"{'exceeds' if earnings_growth_met else 'does not exceed'} inflation of {inflation or 'N/A'}%.")
            }
            if not earnings_growth_met:
                all_criteria_met = False
                messages.append(
                    f"Earnings growth (Annual: {annual_aagr or 'N/A'}%, Quarterly: {quarterly_aqgr or 'N/A'}%) "
                    f"does not exceed inflation of {inflation or 'N/A'}%."
                )

            # 3. Payout Ratio (≤ 75%, up to 100% if not debt-financed)
            payout_ratio_data = self.model.analyze_payout_ratio(symbol)
            if "error" in payout_ratio_data:
                return {"symbol": symbol, "error": payout_ratio_data["error"]}
            payout_ratio = float(payout_ratio_data["payout_ratio"])  # Bereits float, sicher
            payout_ratio_met = payout_ratio <= 75  # Erzeugt Python bool
            if 75 < payout_ratio <= 100:
                net_debt_ebitda_data = self.model.calculate_net_debt_to_ebitda(symbol)
                if isinstance(net_debt_ebitda_data, dict) and "error" in net_debt_ebitda_data:
                    return {"symbol": symbol, "error": net_debt_ebitda_data["error"]}
                net_debt_ebitda = float(net_debt_ebitda_data)  # Konvertiere zu Python float
                payout_ratio_met = bool(net_debt_ebitda <= 2 and net_debt_ebitda != float('inf'))  # Explizite bool-Konvertierung
            result["payout_ratio"] = {
                "value": round(payout_ratio, 2),
                "meets_criterion": payout_ratio_met,
                "message": payout_ratio_data.get("message") or payout_ratio_data.get("warning", "")
            }
            if not payout_ratio_met:
                all_criteria_met = False
                if payout_ratio > 100:
                    messages.append(f"Payout ratio {payout_ratio}% exceeds 100%, indicating unsustainable dividends.")
                elif 75 < payout_ratio <= 100:
                    messages.append(
                        f"Payout ratio {payout_ratio}% exceeds 75% but is ≤ 100% and probably debt-financed (Net Debt/EBITDA > 2).")
                elif payout_ratio < 0:
                    messages.append(f"Payout ratio {payout_ratio}% is negative, indicating potential data issues.")

            # 4. Interest Coverage Ratio (≥ 3)
            interest_coverage = self.model.calculate_interest_coverage_ratio(symbol)
            if "error" in interest_coverage:
                return {"symbol": symbol, "error": interest_coverage["error"]}
            interest_coverage_value = float(interest_coverage["interest_coverage_ratio"])
            interest_coverage_met = bool(interest_coverage_value >= 3)  # Explizite bool-Konvertierung
            result["interest_coverage_ratio"] = {
                "value": round(interest_coverage_value, 2),
                "meets_criterion": interest_coverage_met,
                "date": interest_coverage.get("date")
            }
            if not interest_coverage_met:
                all_criteria_met = False
                messages.append(
                    f"Annual interest coverage ratio {interest_coverage_value} is below 3 for {interest_coverage.get('date', 'unknown date')}.")

            # 5. Net Debt/EBITDA (≤ 2)
            net_debt_ebitda_data = self.model.calculate_net_debt_to_ebitda(symbol)
            if isinstance(net_debt_ebitda_data, dict) and "error" in net_debt_ebitda_data:
                return {"symbol": symbol, "error": net_debt_ebitda_data["error"]}
            net_debt_ebitda = float(net_debt_ebitda_data) if not isinstance(net_debt_ebitda_data, dict) else float('inf')
            net_debt_ebitda_met = bool(net_debt_ebitda <= 2 and net_debt_ebitda != float('inf'))
            result["net_debt_to_ebitda"] = {
                "value": net_debt_ebitda if net_debt_ebitda != float('inf') else "inf",
                "meets_criterion": net_debt_ebitda_met,
                "message": "Net Debt/EBITDA is infinite due to zero EBITDA." if net_debt_ebitda == float('inf') else ""
            }
            if not net_debt_ebitda_met:
                all_criteria_met = False
                if net_debt_ebitda == float('inf'):
                    messages.append("Net Debt/EBITDA is infinite due to zero EBITDA, indicating high financial risk.")
                else:
                    messages.append(f"Net Debt/EBITDA {net_debt_ebitda} exceeds 2, indicating high leverage.")

            # 6. EV/EBIT (≤ 20)
            ev_ebit_data = self.model.calculate_ev_to_ebit(symbol)
            if "error" in ev_ebit_data:
                return {"symbol": symbol, "error": ev_ebit_data["error"]}
            ev_ebit = ev_ebit_data["ev_to_ebit"]
            ev_ebit_met = ev_ebit != "inf" and float(ev_ebit) <= 20
            result["ev_to_ebit"] = {
                "value": ev_ebit,
                "meets_criterion": ev_ebit_met,
                "message": ev_ebit_data.get("message", ""),
                "date": ev_ebit_data.get("date")
            }
            if not ev_ebit_met:
                all_criteria_met = False
                date_str = ev_ebit_data.get("date", "unknown date")
                if ev_ebit == "inf":
                    messages.append(
                        f"EV/EBIT is infinite due to zero or negative EBIT for {date_str}, indicating valuation issues.")
                else:
                    messages.append(f"EV/EBIT {ev_ebit} exceeds 20 for {date_str}, indicating high valuation.")

            # 7. Dividend History (Informative Only)
            dividend_history_data = self.model.analyze_dividend_history(symbol)
            if "error" in dividend_history_data:
                result["dividend_history"] = {"error": dividend_history_data["error"]}
            else:
                result["dividend_history"] = {
                    "years_with_dividends": dividend_history_data["years_with_dividends"],
                    "years_with_increases": dividend_history_data["years_with_increases"],
                    "cagr": dividend_history_data["cagr"],
                    "cagr_period_years": dividend_history_data["cagr_period_years"],
                    "message": (
                        f"Dividends paid for {dividend_history_data['years_with_dividends']} years, "
                        f"with increases in {dividend_history_data['years_with_increases']} years. "
                        f"CAGR over {dividend_history_data['cagr_period_years']} years: "
                        f"{dividend_history_data['cagr']}%"
                        if not pd.isna(dividend_history_data["cagr"])
                        else "CAGR not available due to insufficient data or zero starting value."
                    )
                }

            # Overall Assessment
            result["overall_assessment"] = "Dividend Safe" if all_criteria_met else "Dividend Risky"
            result["message"] = "All criteria met." if all_criteria_met else ("; ".join(messages) if messages else "Unknown issues detected.")

            # ---------- 8) CRV-Analyse (Add-on, kein Abbruchkriterium) ----------
            try:
                crv_result = self.calculate_crv_by_sector_multiples(symbol)
                result["crv"] = crv_result
            except Exception as e:
                result["crv"] = {
                    "error": f"CRV-Analyse fehlgeschlagen: {str(e)}"
                }

            return result
        except Exception as e:
            return {"symbol": symbol, "error": f"Error in overall analysis of dividend company {symbol}: {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def analyze_average_grower(self, symbol, use_cache=True):
        """
        Analysiert eine Aktie auf ihre Eignung als Average Grower basierend auf spezifizierten Kriterien.

        - Price/EBIT: Kauf unter historischem Durchschnitt
        - Price/Tangible Book Value: Vergleich mit historischem Median
        - Dividendenrendite: Ansehnlich, Fokus auf Reinvestition
        - Free Cashflow: Nachhaltig (aktueller Wert ≥ 5% des Marktwerts)
        """
        try:
            # ✅ Einheitliches Error-Payload (damit Frontend nie "leer" ist)
            def fail(msg: str):
                return {
                    "symbol": symbol,
                    "overall_assessment": "Nicht analysierbar",
                    "message": msg,
                    "error": msg,
                }

            # Validierung des Symbols
            if not isinstance(symbol, str):
                return fail("Symbol muss ein String sein.")

            result = {"symbol": symbol}
            all_criteria_met = True
            messages = []

            # 1. Price/EBIT: Kauf unter historischem Durchschnitt
            ebit_bandwidth = self.model.evaluate_ebit_bandwidth(symbol, min_years=10.0, use_cache=use_cache)
            if "error" in ebit_bandwidth:
                return fail(ebit_bandwidth["error"])
            if "ebit" not in ebit_bandwidth or "Price_EBIT" not in ebit_bandwidth["ebit"]:
                return fail(f"Ungültige historische P/EBIT-Daten für {symbol}")

            historical_ebit_mean = ebit_bandwidth["ebit"]["Price_EBIT"].mean()

            current_ebit_data = self.model.calculate_price_to_ebit(symbol, use_cache=use_cache, frequency="quarterly")
            if "error" in current_ebit_data:
                return fail(current_ebit_data["error"])
            if "price_to_ebit" not in current_ebit_data:
                return fail(f"Ungültige aktuelle P/EBIT-Daten für {symbol}")

            current_ebit = current_ebit_data["price_to_ebit"]
            if pd.isna(current_ebit) or (isinstance(current_ebit, str) and current_ebit == "inf"):
                return fail(f"Ungültiger aktueller P/EBIT-Wert für {symbol}: {current_ebit}")

            price_ebit_met = bool(current_ebit < historical_ebit_mean)
            result["price_ebit"] = {
                "value": round(current_ebit, 2),
                "historical_mean": round(historical_ebit_mean, 2),
                "meets_criterion": price_ebit_met,
                "message": f"P/EBIT {current_ebit} {'unter' if price_ebit_met else 'über'} historischem Durchschnitt {round(historical_ebit_mean, 2)}."
            }
            if not price_ebit_met:
                all_criteria_met = False
                messages.append(
                    f"P/EBIT {current_ebit} über historischem Durchschnitt {round(historical_ebit_mean, 2)}.")

            # 2. Price/Tangible Book Value: Vergleich mit historischem Median
            tbv_bandwidth = self.model.evaluate_tbv_bandwidth(symbol, min_years=10.0, use_cache=use_cache)
            if "error" in tbv_bandwidth:
                return fail(tbv_bandwidth["error"])
            if "pb" not in tbv_bandwidth or "Price_TangibleBookValue" not in tbv_bandwidth["pb"]:
                return fail(f"Ungültige historische P/TBV-Daten für {symbol}")

            historical_tbv_median = tbv_bandwidth["pb"]["Price_TangibleBookValue"].median()

            if "current" not in tbv_bandwidth or "pb_ratio" not in tbv_bandwidth["current"]:
                return fail(f"Ungültige aktuelle P/TBV-Daten für {symbol}")

            current_tbv = tbv_bandwidth["current"]["pb_ratio"]
            if pd.isna(current_tbv) or (isinstance(current_tbv, str) and current_tbv == "inf"):
                return fail(f"Ungültiger aktueller P/TBV-Wert für {symbol}: {current_tbv}")

            price_tbv_met = bool(current_tbv < historical_tbv_median and 0 < current_tbv <= 1.5)
            result["price_tbv"] = {
                "value": round(current_tbv, 2),
                "historical_median": round(historical_tbv_median, 2),
                "meets_criterion": price_tbv_met,
                "message": f"P/TBV {current_tbv} {'unter' if price_tbv_met else 'über'} historischem Median {round(historical_tbv_median, 2)} {'und in Kaufzone [1.0, 1.5]' if price_tbv_met else ''}."
            }
            if not price_tbv_met:
                all_criteria_met = False
                messages.append(f"P/TBV {current_tbv} über historischem Median {round(historical_tbv_median, 2)}.")

            # 3. Dividendenrendite: Ansehnlich, Fokus auf Reinvestition
            dividend_data = self.dataloader.get_dividend_data(symbol, use_cache=use_cache)
            if "error" in dividend_data:
                return fail(dividend_data["error"])
            if (
                    "dividend_rate" not in dividend_data
                    or "latest_dividend" not in dividend_data
                    or dividend_data["dividend_rate"] is None
            ):
                return fail(f"Ungültige Dividendeninformationen für {symbol}")

            dividend_yield = dividend_data.get("dividend_yield", 0)
            dividend_yield_met = bool(dividend_yield > 2)

            reinvested_profit_data = self.dataloader.get_reinvested_profit(symbol, frequency="annual",
                                                                           use_cache=use_cache)
            if "error" in reinvested_profit_data:
                return fail(reinvested_profit_data["error"])
            reinvested_profit = reinvested_profit_data["reinvested_profit"]

            profit_data = self.dataloader.get_company_profits(symbol, use_cache=use_cache, frequency="annual")
            if "error" in profit_data:
                return fail(profit_data["error"])
            latest_net_income = profit_data["latest_net_income"]

            reinvested_profit_met = bool(reinvested_profit > 0.1 * latest_net_income)

            message_suffix = " (unter 2%)" if not dividend_yield_met else ""
            if "warning" in dividend_data and dividend_yield < 5:
                message_suffix += " – nicht als Zinssubstitut geeignet."
            elif "message" in dividend_data and dividend_yield >= 5:
                message_suffix += " – geeignet als Zinssubstitut."
            if not reinvested_profit_met:
                message_suffix += " – kein positiver reinvestierter Gewinn (unter 10% des Nettoeinkommens)."

            result["dividend_yield"] = {
                "value": round(dividend_yield, 2),
                "meets_criterion": bool(dividend_yield_met and reinvested_profit_met),
                "message": f"Ansehnliche Rendite {dividend_yield}%, Reinvestition bevorzugt{message_suffix}."
            }
            if not dividend_yield_met or not reinvested_profit_met:
                all_criteria_met = False
                if not dividend_yield_met:
                    messages.append(f"Dividendenrendite {dividend_yield}% nicht ansehnlich (unter 2%).")
                if not reinvested_profit_met:
                    messages.append(
                        f"Reinvestierter Gewinn {reinvested_profit} USD nicht positiv (unter 10% des Nettoeinkommens {round(0.1 * latest_net_income, 2)} USD)."
                    )

            # 4. Free Cashflow: Nachhaltig (aktueller Wert ≥ 5% des Marktwerts)
            fcf_data = self.dataloader.get_free_cashflow(symbol, frequency="annual")
            if "error" in fcf_data:
                return fail(fcf_data["error"])

            free_cashflow = fcf_data["free_cashflow"]
            if pd.isna(free_cashflow) or free_cashflow < 0:
                return fail(f"Ungültiger Free Cashflow für {symbol}: {free_cashflow}")

            market_cap_data = self.dataloader.get_market_cap(symbol, use_cache=use_cache)
            if "error" in market_cap_data:
                return fail(market_cap_data["error"])

            market_cap = market_cap_data["market_cap"]
            if pd.isna(market_cap) or market_cap <= 0:
                return fail(f"Ungültige Marktkapitalisierung für {symbol}: {market_cap}")

            free_cashflow_threshold = 0.05 * market_cap
            free_cashflow_met = bool(free_cashflow >= free_cashflow_threshold)
            result["free_cashflow"] = {
                "value": round(free_cashflow, 4),
                "threshold": round(free_cashflow_threshold, 4),
                "meets_criterion": free_cashflow_met,
                "message": f"Free Cashflow {round(free_cashflow, 4)} {'erfüllt' if free_cashflow_met else 'erfüllt nicht'} die Schwelle von {round(free_cashflow_threshold, 4)} (5% Marktwert)."
            }
            if not free_cashflow_met:
                all_criteria_met = False
                messages.append(
                    f"Free Cashflow {round(free_cashflow, 4)} unter 5% Marktwert {round(free_cashflow_threshold, 4)}."
                )

            # Overall Assessment
            result["overall_assessment"] = "Average Grower" if all_criteria_met else "Not an Average Grower"
            result["message"] = "Alle Kriterien erfüllt." if all_criteria_met else (
                "; ".join(messages) if messages else f"Kein spezifisches Problem identifiziert für {symbol}."
            )

            try:
                crv_result = self.calculate_crv_by_sector_multiples(symbol)
                result["crv"] = crv_result
            except Exception as e:
                result["crv"] = {"error": f"CRV-Analyse fehlgeschlagen: {str(e)}"}

            return result

        except Exception as e:
            self.logger.error(f"Fehler bei der Analyse von {symbol}: {str(e)}", exc_info=True)
            return {
                "symbol": symbol,
                "overall_assessment": "Nicht analysierbar",
                "message": f"Fehler bei der Analyse des Average Growers {symbol}: {str(e)}",
                "error": f"Fehler bei der Analyse des Average Growers {symbol}: {str(e)}",
            }

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def analyze_wachstumswerte(self, symbol: str, frequency: str = "annual", use_cache: bool = True) -> dict:
        """
        Analysiert eine Aktie auf ihre Eignung als Wachstumswert basierend auf spezifizierten Kriterien:
        - Gewinnwachstum: > 15%, ideal > 20%
        - PEG-Ratio: < 1 sehr attraktiv, 1-2 attraktiv
        - Bruttomarge: Steigend über 3 Perioden (Preissetzungsmacht)
        - ROIC: Hoch (> 15%, Monopole oft > 20%)
        - Reinvestitionsrate: Hoher Anteil für Expansion (> 50%)
        - EV/Sales: Relevant für Wachstumsphasen (< 10)

        Args:
            symbol (str): Aktiensymbol (z. B. 'KO').
            frequency (str): 'annual' oder 'quarterly' für Datenfrequenz (Standard: 'annual').
            use_cache (bool): Ob der Cache verwendet werden soll (Standard: True).

        Returns:
            dict: Analyseergebnis mit Bewertung der Kriterien.
                  Beispiel:
                  {
                      "symbol": "KO",
                      "profit_growth": {"value": 18.5, "meets_criterion": True, "message": "Gewinnwachstum von 18.5% erfüllt Kriterium (>15%)"},
                      "peg_ratio": {"value": 1.2, "meets_criterion": True, "message": "PEG-Ratio von 1.2 ist attraktiv (1-2)"},
                      "gross_margin": {"trend": "steigend", "meets_criterion": True, "message": "Bruttomarge steigt über 3 Jahre"},
                      "roic": {"value": 20.0, "meets_criterion": True, "message": "ROIC von 20.0% ist hoch (>15%)"},
                      "reinvestment_rate": {"value": 60.0, "meets_criterion": True, "message": "Reinvestitionsrate von 60.0% ist hoch (>50%)"},
                      "ev_to_sales": {"value": 5.0, "meets_criterion": True, "message": "EV/Sales von 5.0 ist akzeptabel (<10)"},
                      "overall_assessment": "Wachstumswert",
                      "message": "Alle Kriterien erfüllt"
                  }
                  oder bei Fehler:
                  {
                      "symbol": symbol,
                      "error": "Fehlerbeschreibung"
                  }
        """
        try:
            # Validierung der Eingaben
            if not isinstance(symbol, str) or not symbol or not symbol.strip():
                return {"symbol": symbol, "error": "Symbol muss ein nicht-leerer String sein."}
            if frequency.lower() not in ["annual", "quarterly"]:
                return {"symbol": symbol,
                        "error": f"Ungültige Frequenz '{frequency}'. Erlaubt sind 'annual' oder 'quarterly'."}
            if not isinstance(use_cache, bool):
                return {"symbol": symbol, "error": "use_cache muss ein Boolean sein."}

            result = {
                "symbol": symbol,
                "profit_growth": {"value": None, "meets_criterion": False, "message": ""},
                "peg_ratio": {"value": None, "meets_criterion": False, "message": ""},
                "gross_margin": {"trend": None, "meets_criterion": False, "message": ""},
                "roic": {"value": None, "meets_criterion": False, "message": ""},
                "reinvestment_rate": {"value": None, "meets_criterion": False, "message": ""},
                "ev_to_sales": {"value": None, "meets_criterion": False, "message": ""},
                "overall_assessment": "Kein Wachstumswert",
                "message": ""
            }
            all_criteria_met = True
            messages = []

            # 1. Gewinnwachstum
            profit_growth_result = (
                self.model.calculate_avg_annual_profit_growth(symbol, start_date=None, end_date=None,use_cache=use_cache)
                if frequency == "annual"
                else self.model.calculate_avg_quarterly_profit_growth(symbol, start_date=None, end_date=None,use_cache=use_cache)
            )
            if "error" in profit_growth_result:
                result["profit_growth"][
                    "message"] = f"Fehler beim Abrufen des {'jährlichen' if frequency == 'annual' else 'quartalsweisen'} Gewinnwachstums: {profit_growth_result['error']}"
                all_criteria_met = False
                messages.append(result["profit_growth"]["message"])
            elif "net_incomes" in profit_growth_result:
                result["profit_growth"]["message"] = f"Negative Nettogewinne: {profit_growth_result['message']}"
                all_criteria_met = False
                messages.append(result["profit_growth"]["message"])
            else:
                aagr = profit_growth_result["avg_growth"]
                if aagr is None or not isinstance(aagr, (int, float)):
                    result["profit_growth"][
                        "message"] = f"Fehler: Ungültiger {'AAGR' if frequency == 'annual' else 'AQGR'}-Wert."
                    all_criteria_met = False
                    messages.append(result["profit_growth"]["message"])
                else:
                    result["profit_growth"]["value"] = aagr
                    result["profit_growth"]["meets_criterion"] = bool(aagr > 15)
                    result["profit_growth"]["is_ideal"] = bool(aagr > 20)
                    result["profit_growth"]["message"] = (
                        f"{'Jährliches' if frequency == 'annual' else 'Quartalsweises'} Gewinnwachstum von {aagr:.2f}% "
                        f"{'erfüllt' if aagr > 15 else 'erfüllt nicht'} Kriterium (>15%)"
                        f"{', ideal (>20%)' if aagr > 20 else ''}"
                    )
                    if aagr > 100:
                        result["profit_growth"][
                            "message"] += " Warnung: Extrem hohes Gewinnwachstum könnte auf Datenanomalien hinweisen."
                    if not result["profit_growth"]["meets_criterion"]:
                        all_criteria_met = False
                    messages.append(result["profit_growth"]["message"])

            # 2. PEG-Ratio
            peg_result = self.model.calculate_peg_ratio(symbol, use_cache=use_cache)
            if "error" in peg_result:
                result["peg_ratio"]["message"] = f"Fehler beim Abrufen der PEG-Ratio: {peg_result['error']}"
                all_criteria_met = False
                messages.append(result["peg_ratio"]["message"])
            else:
                peg = peg_result["peg_ratio"]
                if peg is None or not isinstance(peg, (int, float)):
                    result["peg_ratio"]["message"] = "Fehler: Ungültiger PEG-Ratio-Wert."
                    all_criteria_met = False
                    messages.append(result["peg_ratio"]["message"])
                else:
                    result["peg_ratio"]["value"] = peg
                    result["peg_ratio"]["meets_criterion"] = bool(peg < 2)
                    result["peg_ratio"]["message"] = (
                        f"PEG-Ratio von {peg:.2f} ist "
                        f"{'sehr attraktiv (<1)' if peg < 1 else 'akzeptabel (1-2)' if peg < 2 else 'nicht attraktiv (>=2)'}"
                    )
                    if peg > 10:
                        result["peg_ratio"][
                            "message"] += " Warnung: Sehr hoher PEG-Ratio könnte auf Datenanomalien hinweisen."
                    if not result["peg_ratio"]["meets_criterion"]:
                        all_criteria_met = False
                    messages.append(result["peg_ratio"]["message"])

            # 3) Bruttomarge (steigend über 3 Perioden)
            result.setdefault("gross_margin", {"trend": None, "meets_criterion": False, "value": None, "message": ""})
            financials = self.dataloader.get_stock_financials(symbol, frequency, use_cache=use_cache)
            if isinstance(financials, dict) and "error" in financials:
                msg = f"Fehler beim Abrufen der Finanzdaten: {financials['error']}"
                result["gross_margin"]["message"] = msg
                all_criteria_met = False
                messages.append(msg)
            elif not isinstance(financials, pd.DataFrame) or financials.empty:
                msg = f"Keine Finanzdaten für {symbol} ({frequency}) gefunden."
                result["gross_margin"]["message"] = msg
                all_criteria_met = False
                messages.append(msg)
            else:
                gp_label = next((l for l in ["Gross Profit", "GrossProfit", "Gross profit"] if l in financials.index),
                                None)
                rev_label = next(
                    (l for l in ["Total Revenue", "Revenue", "Total revenue", "TotalRevenue"] if l in financials.index),
                    None)
                if not gp_label or not rev_label:
                    msg = "Fehlende Daten für Bruttomarge (Gross Profit oder Total Revenue)."
                    result["gross_margin"]["message"] = msg
                    all_criteria_met = False
                    messages.append(msg)
                else:
                    gp = financials.loc[gp_label].astype(float)
                    rev = financials.loc[rev_label].astype(float)
                    idx = gp.index.intersection(rev.index)
                    try:
                        cols = sorted(idx, key=lambda c: pd.to_datetime(c))
                    except Exception:
                        msg = "Warnung: Sortierung der Perioden fehlgeschlagen, Analyse der Bruttomarge übersprungen."
                        result["gross_margin"]["message"] = msg
                        all_criteria_met = False
                        messages.append(msg)
                    else:
                        rev = rev[cols].replace(0, pd.NA)
                        gm = ((gp[cols] / rev) * 100).replace([float("inf"), float("-inf")], pd.NA).dropna()
                        if len(gm) < 3:
                            msg = f"Nicht genügend Daten für 3 {('Jahre' if frequency == 'annual' else 'Quartale')} Bruttomarge."
                            result["gross_margin"]["message"] = msg
                            all_criteria_met = False
                            messages.append(msg)
                        else:
                            last3 = gm.tail(3).tolist()  # älter -> neuer
                            if any((m < 0) or (m > 100) for m in last3):
                                msg = "Ungültige Bruttomargenwerte (negativ oder >100%)."
                                result["gross_margin"]["message"] = msg
                                all_criteria_met = False
                                messages.append(msg)
                            else:
                                trend_up = all(last3[i] <= last3[i + 1] for i in range(len(last3) - 1))
                                result["gross_margin"]["trend"] = "steigend" if trend_up else "nicht steigend"
                                result["gross_margin"]["meets_criterion"] = bool(trend_up)
                                result["gross_margin"]["value"] = [round(float(m), 2) for m in last3]
                                result["gross_margin"]["message"] = (
                                    f"Bruttomarge {gm.iloc[-1]:.2f}% – "
                                    f"{'steigend' if trend_up else 'nicht steigend'} über die letzten 3 Perioden "
                                    f"({', '.join(f'{m:.1f}%' for m in last3)})."
                                )
                                if not trend_up:
                                    all_criteria_met = False
                                    messages.append(result["gross_margin"]["message"])

            # 4. ROIC
            roic_result = self.model.calculate_ROIC(symbol, frequency, use_cache=use_cache)
            if "error" in roic_result:
                result["roic"]["message"] = f"Fehler beim Berechnen des ROIC: {roic_result['error']}"
                all_criteria_met = False
                messages.append(result["roic"]["message"])
            else:
                roic = roic_result["roic"]
                result["roic"]["value"] = roic
                result["roic"]["meets_criterion"] = bool(roic > 10)  # Explizites Casting zu bool
                result["roic"]["message"] = (
                    f"ROIC von {roic:.2f}% {'ist hoch (>10%)' if roic > 10 else 'ist nicht hoch (<=10%)'}"
                    f"{', typisch für Monopole (>20%)' if roic > 10 else ''}"
                )
                if not result["roic"]["meets_criterion"]:
                    all_criteria_met = False
                    messages.append(result["roic"]["message"])

            # 5. Reinvestitionsrate
            reinvested_profit_result = self.dataloader.get_reinvested_profit(symbol, frequency, use_cache=use_cache)
            if "error" in reinvested_profit_result:
                result["reinvestment_rate"][
                    "message"] = f"Fehler beim Abrufen des reinvestierten Gewinns: {reinvested_profit_result['error']}"
                all_criteria_met = False
                messages.append(result["reinvestment_rate"]["message"])
            else:
                reinvested_profit = reinvested_profit_result["reinvested_profit"]
                net_income = roic_result.get("net_income")
                if net_income is None or net_income == 0:
                    result["reinvestment_rate"][
                        "message"] = "Net Income fehlt oder ist 0, Reinvestitionsrate kann nicht berechnet werden."
                    all_criteria_met = False
                    messages.append(result["reinvestment_rate"]["message"])
                else:
                    reinvestment_rate = (reinvested_profit / net_income) * 100
                    result["reinvestment_rate"]["value"] = reinvestment_rate
                    result["reinvestment_rate"]["meets_criterion"] = bool(
                        reinvestment_rate > 50)  # Explizites Casting zu bool
                    result["reinvestment_rate"]["message"] = (
                        f"Reinvestitionsrate von {reinvestment_rate:.2f}% "
                        f"{'ist hoch (>50%)' if reinvestment_rate > 50 else 'ist nicht hoch (<=50%)'}"
                    )
                    if not result["reinvestment_rate"]["meets_criterion"]:
                        all_criteria_met = False
                        messages.append(result["reinvestment_rate"]["message"])

            # 6. EV/Sales
            result.setdefault("ev_to_sales", {"value": None, "meets_criterion": False, "message": ""})
            ev_to_sales_result = self.model.calculate_ev_to_sales(symbol, frequency="annual")
            if "error" in ev_to_sales_result:
                result["ev_to_sales"]["message"] = f"Fehler beim Abrufen von EV/Sales: {ev_to_sales_result['error']}"
                all_criteria_met = False
                messages.append(result["ev_to_sales"]["message"])
            else:
                ev_to_sales = ev_to_sales_result["ev_to_sales"]
                result["ev_to_sales"]["value"] = ev_to_sales
                result["ev_to_sales"]["meets_criterion"] = bool(ev_to_sales < 10)
                result["ev_to_sales"]["message"] = (
                    f"EV/Sales von {ev_to_sales:.2f} {'ist akzeptabel (<10)' if ev_to_sales < 10 else 'ist nicht akzeptabel (>=10)'}"
                )
                if not result["ev_to_sales"]["meets_criterion"]:
                    all_criteria_met = False
                messages.append(result["ev_to_sales"]["message"])

            # ---------- 7) CRV-Analyse (Add-on, kein Abbruchkriterium) ----------
            try:
                crv_result = self.calculate_crv_by_sector_multiples(symbol)
                result["crv"] = crv_result
            except Exception as e:
                result["crv"] = {
                    "error": f"CRV-Analyse fehlgeschlagen: {str(e)}"
                }

            # Gesamtbewertung
            result["overall_assessment"] = "Wachstumswert" if all_criteria_met else "Kein Wachstumswert"
            ideal_criteria = (
                    result["profit_growth"]["value"] is not None and result["profit_growth"]["value"] > 20 and
                    result["peg_ratio"]["value"] is not None and result["peg_ratio"]["value"] < 1
            )
            result["message"] = (
                "Alle Kriterien erfüllt, idealer Wachstumswert" if all_criteria_met and ideal_criteria else
                "Alle Kriterien erfüllt" if all_criteria_met else
                "; ".join(messages) if messages else f"Kein spezifisches Problem identifiziert für {symbol}."
            )

            return result

        except Exception as e:
            return {"symbol": symbol,
                    "error": f"Fehler bei der Analyse des Wachstumswerts {symbol} ({frequency}): {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def analyze_typical_cyclers(self, symbol: str, frequency: str = "annual", min_history_years: float = 10.0,
                                use_cache: bool = True) -> dict:
        """
    Analysiert eine Aktie auf Eignung als typischer Zykliker basierend auf den Kriterien aus 'Typische Zykliker.pdf'.

    Zykliker werden als eigene Kategorie betrachtet, die oft mit Wachstumsunternehmen verwechselt werden. Der Unterschied liegt darin, dass Zykliker wiederkehrende Trends unterliegen, während Wachstumsaktien von langjährigen Trends profitieren. Der Umgang mit Kauf und Verkauf von Zyklikern unterscheidet sich von anderen Kategorien.

    Kriterien:
    - ROE: Hoch in guten Zeiten (≥15%).
    - Cashflow-Marge: Hoher Puffer (≥15%).
    - KGV: Niedrig oder negativ in konjunkturellen Tiefs (≤12 oder <0).
    - Vorräte/Umsatz: Warnsignal bei >100%.
    - P/TBV: Kaufzone [1.0–1.5], Verkaufszone [3–4].
    - EV/Sales: Korrelation mit Branchenzyklen (aktuell als Platzhalter; zukünftig historische Analyse).
    - EV/EBITDA: Tiefpunkte für Kauf (≤10).
    - Historische Bandbreitenbewegung: P/TBV und P/EBIT in Kaufzone (synchronisiert).
    - CRV: Chance-Risiko-Verhältnis ≥3:1.

    Args:
        symbol (str): Stock ticker symbol (e.g., 'AAPL').
        frequency (str): 'annual' für jährliche Daten, 'quarterly' für quartalsweise Daten (default: 'annual').
        min_history_years (float): Minimale Historie in Jahren für Bandbreitenanalyse (default: 10.0).
        use_cache (bool): Ob der Cache verwendet werden soll (default: True).
        """
        try:
            if not isinstance(symbol, str):
                return {"symbol": symbol, "error": "Symbol muss ein String sein."}

            result = {"symbol": symbol, "frequency": frequency}
            all_criteria_met = True
            messages = []

            # 1. ROE – hoch in guten Zeiten (≥15%) → bei Kauf in Tiefphase muss ROE niedrig sein!
            roe_data = self.model.calculate_roe(symbol, frequency=frequency)

            if "error" in roe_data:
                # Echter Datenfehler → kein valider ROE → Buy unmöglich
                result["roe"] = {
                    "value": None,
                    "in_high_phase": None,
                    "meets_criterion": False,
                    "message": f"ROE nicht berechenbar: {roe_data['error']}"
                }
                messages.append(f"ROE fehlt oder ungültig: {roe_data['error']}")
                all_criteria_met = False

            else:
                roe = roe_data["ROE"]
                roe_too_high = roe >= 0.15  # ≥15% → bereits Hochphase → KEIN Kauf
                roe_ok_for_buy = roe < 0.15  # <15% → typisch für Tiefphase → gut!

                meets_criterion = roe_ok_for_buy  # ← entscheidend!

                result["roe"] = {
                    "value": roe,
                    "in_high_phase": bool(roe_too_high),
                    "meets_criterion": bool(meets_criterion),
                    "message": (
                        f"ROE {roe:.2%} → "
                        f"{'Hochphase (≥15%) – bereits teuer, KEIN KAUF' if roe_too_high else 'Tiefphase (<15%) – ideal für Zykliker-Einstieg'}"
                    )
                }

                if not meets_criterion:
                    all_criteria_met = False
                    messages.append("ROE ≥15% → Hochphase erreicht, Kaufkriterium NICHT erfüllt")
                else:
                    messages.append("ROE <15% → perfekte Tiefphase, unterstützt starke Kaufthese")

            # 2. Cashflow-Marge – starker Puffer in guten Zeiten (≥15%)
            cfm_data = self.model.calculate_cashflow_margin(symbol, frequency=frequency)
            if "error" in cfm_data:
                result["cashflow_margin"] = {
                    "value": None,
                    "meets_criterion": False,
                    "message": f"Cashflow-Marge nicht berechenbar: {cfm_data['error']}"
                }
                all_criteria_met = False
                messages.append(f"Cashflow-Marge fehlt: {cfm_data['error']}")
            else:
                cfm = cfm_data["cashflow_margin"]  # bereits in Prozent, gerundet
                cfm_met = cfm >= 15.0
                result["cashflow_margin"] = {
                    "value": cfm,
                    "meets_criterion": bool(cfm_met),
                    "message": f"Cashflow-Marge {cfm:.1f}% → {'starker Puffer (≥15%) – typisch für profitable Zykliker' if cfm_met else 'schwacher Puffer – anfällig in Abschwung'}"
                }
                if not cfm_met:
                    all_criteria_met = False
                    messages.append(f"Cashflow-Marge {cfm:.1f}% < 15% – kein ausreichender Puffer für Zyklustiefs")

            # 3. KGV – niedrig oder negativ in konjunkturellen Tiefs (≤12 oder <0)
            kgv_data = self.model.calculate_KGV(symbol)
            if isinstance(kgv_data, dict) and "error" in kgv_data:
                result["kgv"] = {
                    "value": None,
                    "meets_criterion": False,
                    "message": f"KGV nicht berechenbar: {kgv_data['error']}"
                }
                all_criteria_met = False
                messages.append(f"KGV fehlt: {kgv_data['error']}")
            else:
                kgv = kgv_data  # float oder float('inf') oder negativ
                is_negative = kgv < 0
                is_low = 0 < kgv <= 12.0
                kgv_met = is_negative or is_low

                # Formatierung für Ausgabe
                if kgv == float('inf'):
                    kgv_display = "inf"
                elif kgv < 0:
                    kgv_display = f"{kgv:.2f}"
                else:
                    kgv_display = f"{kgv:.2f}"

                result["kgv"] = {
                    "value": kgv if not isinstance(kgv, float) or not math.isinf(kgv) else None,
                    "meets_criterion": bool(kgv_met),
                    "message": (
                        f"KGV {kgv_display} → "
                        f"{'in Tiefphase (negativ oder ≤12) – typische Kaufgelegenheit bei Zyklikern' if kgv_met else 'zu hoch – keine klassische Zyklus-Tiefphase'}"
                    )
                }
                if not kgv_met:
                    all_criteria_met = False
                    messages.append(f"KGV {kgv_display} > 12 und positiv – keine klare Tiefphase erkennbar")

            # 4. Vorräte/Umsatz – Warnsignal bei >100%
            inv_data = self.model.calculate_inventory_to_revenue_ratio(symbol, frequency=frequency)

            if "error" in inv_data:
                if "möglicherweise Dienstleister" in inv_data.get("error", ""):
                    result["inventory_to_revenue"] = {
                        "value": None,
                        "meets_criterion": True,
                        "message": "Keine Vorräte (Dienstleister) → kein Warnsignal → Kriterium erfüllt"
                    }
                else:
                    result["inventory_to_revenue"] = {
                        "value": None,
                        "meets_criterion": False,
                        "message": f"Vorräte/Umsatz nicht berechenbar: {inv_data['error']}"
                    }
                    all_criteria_met = False
                    messages.append(f"Vorräte/Umsatz fehlt: {inv_data['error']}")

            else:
                ratio_percent = inv_data["inventory_to_revenue_ratio"]  # bereits in %
                inv_met = ratio_percent <= 100.0

                result["inventory_to_revenue"] = {
                    "value": ratio_percent,  # bewusst in Prozent (konsistent zur Cashflow-Marge)
                    "meets_criterion": bool(inv_met),
                    "message": (
                        f"Vorräte/Umsatz {ratio_percent:.1f}% → "
                        f"{'kein Warnsignal (≤100%) – gesund' if inv_met else 'Warnsignal (>100%) – Lageraufbau!'}"
                    )
                }

                if not inv_met:
                    all_criteria_met = False
                    messages.append(f"Vorräte/Umsatz {ratio_percent:.1f}% > 100% → Warnsignal!")

            # 5. P/TBV – Kernindikator für Zykliker: Kaufzone [1.0–1.5] + historische Regression-to-the-Mean
            # Wir nutzen evaluate_tbv_bandwidth für Historie + Touches
            # Aber den aktuellen P/TBV holen wir aus der robustesten Quelle: get_current_tbv_and_price
            tbv_eval = self.model.evaluate_tbv_bandwidth(symbol, min_years=min_history_years, use_cache=use_cache)

            tbv_per_share, current_price, current_pb = self.model.get_current_tbv_and_price(symbol)

            # Normalize current_pb (falls irgendwo "inf" als String auftaucht)
            if current_pb == "inf":
                current_pb = float("inf")

            if isinstance(tbv_per_share, dict) and "error" in tbv_per_share:
                result["price_to_tbv"] = {
                    "value": None,
                    "buy_zone": False,
                    "sell_zone": False,
                    "meets_criterion": False,
                    "historical_touches": 0,
                    "history_years": 0,
                    "targets": {"WC": 0, "BUY": 0, "SELL": 0},
                    "message": f"P/TBV nicht berechenbar: {tbv_per_share['error']}"
                }
                all_criteria_met = False
                messages.append(f"P/TBV fehlt: {tbv_per_share['error']}")
                history_years = 0
                touches = 0
                signal = "unknown"
            else:
                # Historische Analyse (nur wenn möglich)
                if not isinstance(tbv_eval, dict) or "error" in tbv_eval:
                    history_years = 0
                    touches = 0
                    signal = "unknown"
                    has_history = False
                    has_regression_proof = False
                else:
                    history_years = tbv_eval["meta"]["history_years"]
                    touches = tbv_eval["zones"]["touches_tbv≈1x"]
                    signal = tbv_eval["signal"]
                    has_history = history_years >= min_history_years
                    has_regression_proof = touches >= 2

                # Aktuelle Bewertung – robust gegen None/NaN
                has_current_pb = isinstance(current_pb, (int, float)) and not math.isnan(current_pb)
                in_buy_zone = bool(has_current_pb and 1.0 <= current_pb <= 1.5)
                in_sell_zone = bool(has_current_pb and current_pb >= 3.0)

                tbv_met = bool(in_buy_zone and has_history and has_regression_proof)

                if isinstance(tbv_per_share, (int, float)) and tbv_per_share > 0:
                    targets = {
                        "WC": round(0.90 * tbv_per_share, 2),
                        "BUY": round(1.15 * tbv_per_share, 2),
                        "SELL": round(3.00 * tbv_per_share, 2)
                    }
                else:
                    targets = {"WC": 0.0, "BUY": 0.0, "SELL": 0.0}

                pb_display = (
                    "inf" if (has_current_pb and current_pb == float("inf")) else
                    (f"{current_pb:.2f}" if has_current_pb else "N/A")
                )

                result["price_to_tbv"] = {
                    "value": round(current_pb, 3) if (has_current_pb and current_pb != float("inf")) else (
                        "inf" if has_current_pb else None),
                    "buy_zone": bool(in_buy_zone),
                    "sell_zone": bool(in_sell_zone),
                    "meets_criterion": bool(tbv_met),
                    "historical_touches": touches,
                    "history_years": round(history_years, 1),
                    "signal": signal,
                    "targets": targets,
                    "current_price": round(current_price, 2) if isinstance(current_price, (int, float)) else None,
                    "tbv_per_share": round(tbv_per_share, 4) if isinstance(tbv_per_share, (int, float)) else 0.0,
                    "message": (
                        f"P/TBV {pb_display} ({'in Kaufzone' if in_buy_zone else 'überbewertet' if in_sell_zone else 'neutral'}) – "
                        f"Historie: {history_years:.1f} Jahre, {touches}x ≈TBV → "
                        f"{'STARKES KAUF-SIGNAL (echter Zykliker in Tiefphase)' if tbv_met else 'kein Kauf: fehlende Historie, kein Touch oder außerhalb Zone'}"
                    )
                }

                if not tbv_met:
                    all_criteria_met = False
                    if not in_buy_zone:
                        if has_current_pb:
                            messages.append(
                                f"P/TBV {pb_display} außerhalb Kaufzone [1.0–1.5] → {'Überbewertung' if in_sell_zone else 'zu teuer'}"
                            )
                        else:
                            messages.append("P/TBV aktuell nicht verfügbar → Kaufzone nicht prüfbar")
                    if not has_regression_proof:
                        messages.append(f"Nur {touches}x ≈TBV in Historie → kein klarer Zykliker")
                    if not has_history:
                        messages.append(f"Nur {history_years:.1f} Jahre Historie < {min_history_years} → unzureichend")

            # 6. EV/EBITDA – Tiefpunkt nur bei Vergleich mit historischem Durchschnitt!
            # Methode: Aktuelles EV/EBITDA vs. Median der letzten 10+ Jahre
            current_ev_data = self.model.calculate_ev_to_ebitda(symbol, use_cache=use_cache, frequency=frequency)
            historical_df = self.model.calculate_historical_ev_to_ebitda(symbol, use_cache=use_cache)

            # --- Fehlerbehandlung ---
            if isinstance(current_ev_data, dict) and "error" in current_ev_data:
                result["ev_to_ebitda"] = {
                    "value": None,
                    "current": None,
                    "historical_median": None,
                    "percentile": None,
                    "meets_criterion": False,
                    "message": f"EV/EBITDA nicht berechenbar: {current_ev_data['error']}"
                }
                all_criteria_met = False
                messages.append(f"EV/EBITDA fehlt: {current_ev_data['error']}")

            else:
                current_val = current_ev_data.get("ev_to_ebitda") if isinstance(current_ev_data, dict) else None

                # Normalize inf (String/Float) → None (für Perzentil-Berechnung nicht sinnvoll)
                if current_val == "inf" or current_val == float("inf"):
                    current_val = None

                # Default-Werte
                hist_median = None
                percentile = None
                ev_met = False
                note = "keine Historie"

                if historical_df is None or getattr(historical_df, "empty", True) or len(historical_df) < 5:
                    note = "Zu wenig historische Daten für Bandbreitenvergleich"
                else:
                    # Nur positive, sinnvolle Werte verwenden
                    valid = historical_df["EV_EBITDA"]
                    valid = valid[valid > 0]

                    if len(valid) < 5:
                        note = "Zu wenige gültige historische EV/EBITDA-Werte"
                    else:
                        hist_median = float(valid.median())

                        # Perzentil nur möglich, wenn aktueller Wert numerisch ist
                        if current_val is None:
                            note = "Aktueller EV/EBITDA-Wert ungültig (inf/None) – Perzentil nicht berechenbar"
                        else:
                            percentile_rank = (valid < float(current_val)).mean()
                            percentile = round(percentile_rank * 100, 1)  # 0.0 bleibt 0.0

                            # Kriterium: Aktuell im unteren 35. Perzentil des historischen Verlaufs
                            ev_met = percentile <= 35.0

                            note = (f"EV/EBITDA {float(current_val):.1f} im {percentile} Perzentil "
                                    f"(Median: {hist_median:.1f}) → {'Tiefphase!' if ev_met else 'noch nicht günstig'}")

                result["ev_to_ebitda"] = {
                    "value": round(float(current_val), 2) if current_val is not None else "inf",
                    "current": round(float(current_val), 2) if current_val is not None else None,
                    "historical_median": round(hist_median, 2) if hist_median is not None else None,
                    "percentile": percentile,
                    "meets_criterion": bool(ev_met),
                    "message": (
                        f"EV/EBITDA {round(float(current_val), 2) if current_val is not None else 'N/A'} → {note}"
                    )
                }

                if not ev_met:
                    all_criteria_met = False
                    messages.append(
                        f"EV/EBITDA {round(float(current_val), 2) if current_val is not None else 'N/A'} "
                        f"nicht im unteren Bereich (Perzentil: {percentile if percentile is not None else 'N/A'}%) – "
                        f"keine klare Tiefphase"
                    )

            # 7. Price / Free Cash Flow – Bewertung der Liquidität & Nachhaltigkeit (annual)
            pfcf_data = self.model.calculate_price_to_freeCashflow(
                symbol,
                use_cache=use_cache,
                frequency="annual"  # explizit annual → einheitlich & zyklusgerecht
            )

            if isinstance(pfcf_data, dict) and "error" in pfcf_data:
                result["price_to_fcf"] = {
                    "value": None,
                    "meets_criterion": False,
                    "message": f"P/FCF nicht berechenbar: {pfcf_data['error']}"
                }
                all_criteria_met = False
                messages.append(f"P/FCF fehlt: {pfcf_data['error']}")

            else:
                p_fcf_raw = pfcf_data.get("price_to_freeCashflow") if isinstance(pfcf_data, dict) else None
                date = pfcf_data.get("date") if isinstance(pfcf_data, dict) else None

                # Jahr robust extrahieren (falls date None oder kein String)
                date_str = str(date) if date is not None else "unbekannt"
                year_str = date_str[-4:] if len(date_str) >= 4 else "unbekannt"

                if p_fcf_raw == "inf" or p_fcf_raw is None:
                    p_fcf_value = None
                    p_fcf_met = False
                    note = "Kein positiver Free Cash Flow (annual) → hohes Risiko"
                    detail = "Negativer oder null FCF im letzten Geschäftsjahr"
                else:
                    p_fcf_value = float(p_fcf_raw)

                    # Strenge, aber zyklusgerechte Bewertung für annual P/FCF
                    if p_fcf_value <= 12:
                        p_fcf_met = True
                        note = "exzellente Cash-Generierung – stark unterbewertet"
                        detail = f"P/FCF {p_fcf_value:.1f}x (annual {year_str}) → Top-Qualität"
                    elif p_fcf_value <= 18:
                        p_fcf_met = True
                        note = "gute nachhaltige Liquidität – attraktiv"
                        detail = f"P/FCF {p_fcf_value:.1f}x (annual {year_str}) → solide Bewertung"
                    elif p_fcf_value <= 25:
                        p_fcf_met = False  # weicher Filter: ab 25 wird gewarnt
                        note = "durchschnittlich – noch akzeptabel, aber Vorsicht"
                        detail = f"P/FCF {p_fcf_value:.1f}x (annual {year_str}) → teurer als Durchschnitt"
                    else:
                        p_fcf_met = False
                        note = "sehr teuer – schwache Cash-Rendite relativ zum Kurs"
                        detail = f"P/FCF {p_fcf_value:.1f}x (annual {year_str}) → hohes Risiko"

                pfcf_display = f"{p_fcf_value:.1f}" if p_fcf_value is not None else "∞"

                result["price_to_fcf"] = {
                    "value": p_fcf_value,
                    "meets_criterion": bool(p_fcf_met),
                    "message": f"P/FCF {pfcf_display} → {note}",
                    "detail": detail,
                    "frequency": "annual",
                    "date": date_str
                }

                if not p_fcf_met:
                    all_criteria_met = False
                    messages.append(f"P/FCF {pfcf_display} > 18 → Cash-Generierung nicht überzeugend (annual)")
                else:
                    messages.append(f"P/FCF {p_fcf_value:.1f}x (annual) → starke Liquiditätsbewertung")

            # ---------- 8) CRV-Analyse (Add-on, KEIN Ausschlusskriterium) ----------
            try:
                crv_result = self.calculate_crv_by_sector_multiples(symbol)
                result["crv"] = crv_result
            except Exception as e:
                result["crv"] = {
                    "error": f"CRV-Analyse fehlgeschlagen: {str(e)}"
                }

            # Gesamtbewertung
            result[
                "overall_assessment"] = "Typical Cycler – Buy" if all_criteria_met else "Typical Cycler – Watch/Avoid"
            result["message"] = "Alle Kriterien erfüllt – attraktive Zyklusphase." if all_criteria_met else (
                "; ".join(messages) if messages else "Unklare Bewertung."
            )

            return result

        except Exception as e:
            return {"symbol": symbol, "error": f"Fehler in analyze_typical_cyclers: {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def analyze_cycler_turnarounds(self, symbol: str,frequency: str = "annual") -> dict:
        """
        Analysiert einen zyklischen Turnaround-Kandidaten basierend auf den Kriterien
        aus 'Turnarounds (Infos aus Buch).pdf'.

        Fokus:
        - Hohe Verschuldung ist erlaubt und typisch
        - Überlebensfähigkeit & Refinanzierungsfähigkeit sind entscheidend
        - Warnsignale (z. B. Vorräte/Umsatz) werden klar hervorgehoben
        - Wertvolle Vermögenswerte dienen als qualitativer Puffer

        Returns:
            dict: Strukturierte Turnaround-Analyse
        """
        try:
            result = {"symbol": symbol}
            messages = []
            all_criteria_met = True  # bewusst weich interpretiert


            # 1. Verschuldungsgrad – Net Debt / EBITDA
            nd_ebitda_data = self.model.calculate_net_debt_to_ebitda(symbol, frequency=frequency)

            if isinstance(nd_ebitda_data, dict) and "error" in nd_ebitda_data:
                result["net_debt_to_ebitda"] = {
                    "value": None,
                    "risk_level": "unknown",
                    "meets_criterion": False,
                    "message": f"Net Debt/EBITDA nicht berechenbar: {nd_ebitda_data['error']}"
                }
                messages.append(f"Net Debt/EBITDA fehlt: {nd_ebitda_data['error']}")
            else:
                nd_ebitda = nd_ebitda_data

                if nd_ebitda == float("inf"):
                    risk_level = "critical"
                    meets_criterion = False
                    message = (
                        "Net Debt/EBITDA unendlich (EBITDA ≤ 0) → "
                        "existenzbedrohende Turnaround-Situation"
                    )
                    all_criteria_met = False
                elif nd_ebitda >= 6.0:
                    risk_level = "high"
                    meets_criterion = True
                    message = (
                        f"Net Debt/EBITDA {nd_ebitda:.1f} → sehr hohe Verschuldung, "
                        "typisch für Turnarounds"
                    )
                elif nd_ebitda >= 3.0:
                    risk_level = "elevated"
                    meets_criterion = True
                    message = (
                        f"Net Debt/EBITDA {nd_ebitda:.1f} → erhöhte Verschuldung, "
                        "Turnaround-fähig"
                    )
                else:
                    risk_level = "low"
                    meets_criterion = True
                    message = (
                        f"Net Debt/EBITDA {nd_ebitda:.1f} → geringe Verschuldung, "
                        "ungewöhnlich defensiv für Turnaround"
                    )

                result["net_debt_to_ebitda"] = {
                    "value": nd_ebitda if nd_ebitda != float("inf") else None,
                    "risk_level": risk_level,
                    "meets_criterion": bool(meets_criterion),
                    "message": message
                }

                if risk_level == "critical":
                    messages.append("EBITDA ≤ 0 → akutes Insolvenzrisiko")


            # 2. Liquidität / Refinanzierungsfähigkeit – Debt to Equity
            dte_data = self.model.calculate_debt_to_equity(symbol, frequency=frequency)

            if isinstance(dte_data, dict) and "error" in dte_data:
                result["debt_to_equity"] = {
                    "value": None,
                    "risk_level": "unknown",
                    "meets_criterion": False,
                    "message": f"Debt-to-Equity nicht berechenbar: {dte_data['error']}"
                }
                messages.append(f"Debt-to-Equity fehlt: {dte_data['error']}")
            else:
                dte = dte_data["debt_to_equity"]

                if dte >= 4.0:
                    risk_level = "critical"
                    meets_criterion = False
                    message = (
                        f"Debt-to-Equity {dte:.2f} → extrem hohe Verschuldung, "
                        "Refinanzierung kritisch"
                    )
                    all_criteria_met = False
                elif dte >= 2.0:
                    risk_level = "high"
                    meets_criterion = True
                    message = (
                        f"Debt-to-Equity {dte:.2f} → hohe Verschuldung, "
                        "Refinanzierung notwendig"
                    )
                else:
                    risk_level = "moderate"
                    meets_criterion = True
                    message = (
                        f"Debt-to-Equity {dte:.2f} → tragbare Verschuldung, "
                        "Refinanzierung realistisch"
                    )

                result["debt_to_equity"] = {
                    "value": dte,
                    "risk_level": risk_level,
                    "meets_criterion": bool(meets_criterion),
                    "message": message
                }

                if risk_level == "critical":
                    messages.append("Debt-to-Equity extrem hoch → Refinanzierungsrisiko")


            # 3. Vorräte / Umsatz – operatives Warnsignal
            inv_data = self.model.calculate_inventory_to_revenue_ratio(symbol, frequency=frequency)

            if "error" in inv_data:
                result["inventory_to_revenue"] = {
                    "value": None,
                    "meets_criterion": False,
                    "message": f"Vorräte/Umsatz nicht berechenbar: {inv_data['error']}"
                }
                messages.append(f"Vorräte/Umsatz fehlt: {inv_data['error']}")
            else:
                ratio = inv_data["inventory_to_revenue_ratio"]
                inv_ok = ratio <= 100.0

                result["inventory_to_revenue"] = {
                    "value": ratio / 100.0,
                    "meets_criterion": bool(inv_ok),
                    "message": (
                        f"Vorräte/Umsatz {ratio:.1f}% → "
                        f"{'kein Warnsignal' if inv_ok else 'Warnsignal: Überbestand'}"
                    )
                }

                if not inv_ok:
                    all_criteria_met = False
                    messages.append("Vorräte wachsen schneller als Umsatz → operatives Risiko")

            # 4. Wertvolle Vermögenswerte – qualitativer Puffer
            result["valuable_assets"] = {
                "meets_criterion": None,
                "message": (
                    " Mögliche wertvolle Vermögenswerte (z. B. Immobilien, Patente) selbst prüfen. "
                    "Können Turnaround stützen – qualitative Einzelfallprüfung erforderlich"
                )
            }

        # ---------- CRV-Analyse (Add-on, kein hartes Kriterium) ----------
            try:
                crv_result = self.calculate_crv_by_sector_multiples(symbol)
                result["crv"] = crv_result
            except Exception as e:
                result["crv"] = {
                    "error": f"CRV-Analyse fehlgeschlagen: {str(e)}"
                }

            # Gesamtbewertung
            if all_criteria_met:
                result["overall_assessment"] = "Turnaround Candidate"
                result["message"] = "Turnaround-Risiken vorhanden, aber Überlebensfähigkeit plausibel."
            else:
                result["overall_assessment"] = "High-Risk Turnaround / Avoid"
                result["message"] = "; ".join(messages) if messages else "Hohe Turnaround-Risiken."

            return result

        except Exception as e:
            return {
                "symbol": symbol,
                "error": f"Fehler in analyze_cycler_turnarounds: {str(e)}"
            }

    def analyze_optionality(self, symbol: str, frequency: str = "annual", use_cache: bool = True) -> dict:
        """
        Analysiert Optionalitäten (asymmetrisches Chancen-Risiko-Profil).

        Kriterien:
        - Geringe Verschuldung (Net Debt / EBITDA)
        - Hohe Cash-Reserven relativ zur Marktkapitalisierung
        - Qualitative Katalysatoren (manuell zu bewerten)

        Returns:
            dict mit Detailbewertung und Gesamturteil
        """
        try:
            result = {"symbol": symbol}
            messages = []
            all_criteria_met = True

            # 1. Verschuldungsgrad – Net Debt / EBITDA
            nd_ebitda = self.model.calculate_net_debt_to_ebitda(symbol, frequency=frequency)

            if isinstance(nd_ebitda, dict) and "error" in nd_ebitda:
                result["net_debt_to_ebitda"] = {
                    "value": None,
                    "meets_criterion": False,
                    "message": f"Net Debt/EBITDA nicht berechenbar: {nd_ebitda['error']}"
                }
                all_criteria_met = False
                messages.append("Verschuldung nicht bewertbar")
            else:
                nd_value = nd_ebitda
                nd_ok = nd_value <= 1.5

                result["net_debt_to_ebitda"] = {
                    "value": nd_value,
                    "meets_criterion": bool(nd_ok),
                    "message": (
                        f"Net Debt/EBITDA {nd_value:.2f} → "
                        f"{'geringe Verschuldung – guter Puffer' if nd_ok else 'erhöhte Verschuldung – eingeschränkte Optionalität'}"
                    )
                }

                if not nd_ok:
                    all_criteria_met = False
                    messages.append(f"Net Debt/EBITDA {nd_value:.2f} > 1.5")

            # 2. Cash / Market Cap – strategische Flexibilität
            cash_mc = self.model.calculate_cash_to_market_cap(
                symbol, frequency=frequency, use_cache=use_cache
            )

            if isinstance(cash_mc, dict) and "error" in cash_mc:
                result["cash_to_market_cap"] = {
                    "value": None,
                    "meets_criterion": False,
                    "message": f"Cash/MarketCap nicht berechenbar: {cash_mc['error']}"
                }
                all_criteria_met = False
                messages.append("Cash-Position nicht bewertbar")
            else:
                cash_ratio = cash_mc["cash_to_market_cap"]
                cash_ok = cash_ratio >= 0.20

                result["cash_to_market_cap"] = {
                    "value": round(cash_ratio, 3),
                    "meets_criterion": bool(cash_ok),
                    "cash": cash_mc["cash"],
                    "market_cap": cash_mc["market_cap"],
                    "message": (
                        f"Cash/MarketCap {cash_ratio:.1%} → "
                        f"{'hohe Optionalität & Flexibilität' if cash_ok else 'begrenzte finanzielle Optionalität'}"
                    )
                }

                if not cash_ok:
                    all_criteria_met = False
                    messages.append(f"Cash/MarketCap {cash_ratio:.1%} < 20%")

            # 3. Katalysatoren – bewusst qualitativ
            result["catalysts"] = {
                "meets_criterion": None,
                "message": (
                    "Mögliche Katalysatoren (z. B. Zulassungen, Spin-offs, "
                    "Restrukturierungen, politische Entscheidungen) "
                    "müssen qualitativ beurteilt werden."
                )
            }

            # ---------- CRV-Analyse (Add-on, kein hartes Kriterium) ----------
            try:
                crv_result = self.calculate_crv_by_sector_multiples(symbol)
                result["crv"] = crv_result
            except Exception as e:
                result["crv"] = {
                    "error": f"CRV-Analyse fehlgeschlagen: {str(e)}"
                }

            # Gesamtbewertung
            result["overall_assessment"] = (
                "Optionality Candidate" if all_criteria_met else "Limited Optionality"
            )

            result["message"] = (
                "Finanziell robuste Struktur mit hoher Optionalität."
                if all_criteria_met
                else "; ".join(messages) if messages else "Optionalität begrenzt."
            )

            return result

        except Exception as e:
            return {
                "symbol": symbol,
                "error": f"Fehler in analyze_optionality: {str(e)}"
            }

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def analyze_asset_play(self, symbol: str, frequency: str = "annual", use_cache: bool = True, ptbv_threshold: float = 1.2) -> dict:
        """
        Analysiert eine Aktie als Asset Play nach Lynch:
        - Niedriges P/TBV
        - Positive Net Current Assets
        - Qualitative Bewertung der Vermögenswerte (manuell)
        """

        result = {"symbol": symbol, "frequency": frequency}
        messages = []
        all_criteria_met = True

        try:
            # 1) Kurs / Tangible Book Value (zentraler Filter)
            tbv_per_share, current_price, ptbv = self.model.get_current_tbv_and_price(symbol)

            if isinstance(tbv_per_share, dict) and "error" in tbv_per_share:
                result["price_to_tangible_book"] = {
                    "value": None,
                    "meets_criterion": False,
                    "message": f"P/TBV nicht berechenbar: {tbv_per_share['error']}"
                }
                all_criteria_met = False
                messages.append(f"P/TBV fehlt: {tbv_per_share['error']}")
            else:
                ptbv_met = ptbv <= ptbv_threshold

                result["price_to_tangible_book"] = {
                    "value": ptbv,
                    "threshold": ptbv_threshold,
                    "meets_criterion": bool(ptbv_met),
                    "tbv_per_share": round(tbv_per_share, 4),
                    "current_price": round(current_price, 2),
                    "message": (
                        f"P/TBV {ptbv:.2f} → "
                        f"{'unterbewertet (Asset Play)' if ptbv_met else 'nicht günstig genug'}"
                    )
                }

                if not ptbv_met:
                    all_criteria_met = False
                    messages.append(f"P/TBV {ptbv:.2f} > {ptbv_threshold}")

            # 2) Net Current Assets (Net-Net-Puffer)
            nca_data = self.model.calculate_current_netCurrentAssets(
                symbol, frequency=frequency, use_cache=use_cache
            )

            if isinstance(nca_data, dict) and "error" in nca_data:
                result["net_current_assets"] = {
                    "value": None,
                    "meets_criterion": False,
                    "message": f"Net Current Assets nicht berechenbar: {nca_data['error']}"
                }
                all_criteria_met = False
                messages.append(f"NCA fehlt: {nca_data['error']}")
            else:
                nca = nca_data["net_current_assets"]
                nca_met = nca > 0

                result["net_current_assets"] = {
                    "value": nca,
                    "current_assets": nca_data["current_assets"],
                    "current_liabilities": nca_data["current_liabilities"],
                    "date": nca_data["date"],
                    "meets_criterion": bool(nca_met),
                    "message": (
                        f"Net Current Assets {nca:,.0f} → "
                        f"{'positiv (Substanzpuffer)' if nca_met else 'negativ (kein Net-Net)'}"
                    )
                }

                if not nca_met:
                    all_criteria_met = False
                    messages.append("Net Current Assets ≤ 0")

            # 3) Wert der Vermögenswerte (qualitativ!)
            result["asset_value_quality"] = {
                "meets_criterion": None,
                "message": (
                    "Qualitative Prüfung erforderlich: Immobilien, Rohstoffreserven, "
                    "Beteiligungen, Patente oder stille Reserven (10-K / Notes prüfen)."
                )
            }

            # ---------- CRV-Analyse (Add-on, kein hartes Kriterium) ----------
            try:
                crv_result = self.calculate_crv_by_sector_multiples(symbol)
                result["crv"] = crv_result
            except Exception as e:
                result["crv"] = {
                    "error": f"CRV-Analyse fehlgeschlagen: {str(e)}"
                }

            # Gesamtbewertung
            result["overall_assessment"] = (
                "Asset Play – Candidate" if all_criteria_met else "Asset Play – Watch/Avoid"
            )

            result["message"] = (
                "Unterbewertet mit Substanzpuffer – klassischer Asset Play."
                if all_criteria_met
                else "; ".join(messages) if messages else "Unklare Bewertung."
            )

            return result

        except Exception as e:
            return {"symbol": symbol, "error": f"Fehler in analyze_asset_play: {str(e)}"}

    def calculate_crv_by_sector_multiples(self, symbol: str) -> dict:
        """
        Ermittelt sektorabhängige relevante Multiples für ein Unternehmen,
        berechnet für jedes Multiple das CRV auf Basis historischer Daten
        und gibt die Ergebnisse strukturiert zurück.

        Returns:
            {
                "symbol": "AAPL",
                "sectors": [...],
                "multiples_used": [...],
                "crv_results": {
                    "Price_FreeCashflow": {...},
                    "EV_EBITDA": {...}
                }
            }
            oder {"error": "...", "symbol": symbol}
        """

        # ---------- 1) Branchen ermitteln ----------
        sectors = self.COMPANY_SECTORS.get(symbol)
        if not sectors:
            return {
                "error": f"Keine Branchen-Zuordnung für {symbol} vorhanden.",
                "symbol": symbol
            }

        # ---------- 2) Relevante Multiples sammeln ----------
        multiples = set()
        for sector in sectors:
            sector_multiples = self.BRANCH_MULTIPLES_MAP.get(sector)
            if sector_multiples:
                multiples.update(sector_multiples)

        if not multiples:
            return {
                "error": f"Keine relevanten Multiples für die Branchen von {symbol} definiert.",
                "symbol": symbol,
                "sectors": sectors
            }

        # ---------- 3) CRV je Multiple berechnen ----------
        crv_results = {}

        for multiple in sorted(multiples):
            try:
                historical_df = None
                use_cache = True

                if multiple == "EV_Sales":
                    historical_df = self.model.calculate_historical_ev_sales(
                        symbol=symbol, start_date=None, end_date=None, use_cache=use_cache
                    )

                elif multiple == "EV_EBIT":
                    historical_df = self.model.calculate_historical_ev_to_ebit(
                        symbol=symbol, start_date=None, end_date=None, use_cache=use_cache
                    )

                elif multiple == "EV_EBITDA":
                    historical_df = self.model.calculate_historical_ev_to_ebitda(
                        symbol=symbol, start_date=None, end_date=None, use_cache=use_cache
                    )

                elif multiple == "Price_Book":
                    historical_df = self.model.calculate_historical_price_to_book(
                        symbol=symbol, start_date=None, end_date=None, use_cache=use_cache
                    )

                elif multiple == "Price_Sales":
                    historical_df = self.model.calculate_historical_price_to_sales(
                        symbol=symbol, start_date=None, end_date=None, use_cache=use_cache
                    )

                elif multiple == "Price_EBIT":
                    historical_df = self.model.calculate_historical_price_to_ebit(
                        symbol=symbol, start_date=None, end_date=None, use_cache=use_cache
                    )

                elif multiple == "Price_NetCurrentAssets":
                    historical_df = self.model.calculate_historical_price_netCurrentAssets(
                        symbol=symbol, start_date=None, end_date=None, use_cache=use_cache
                    )

                elif multiple == "Price_OperatingCashflow":
                    historical_df = self.model.calculate_historical_price_OperatingCashflow(
                        symbol=symbol, start_date=None, end_date=None, use_cache=use_cache
                    )

                elif multiple == "Price_FreeCashflow":
                    historical_df = self.model.calculate_historical_Price_FreeCashflow(
                        symbol=symbol, start_date=None, end_date=None, use_cache=use_cache
                    )

                elif multiple == "Price_TangibleBookValue":
                    historical_df = self.model.calculate_historical_price_to_TangibleBookValue(
                        symbol=symbol, start_date=None, end_date=None, use_cache=use_cache
                    )

                else:
                    crv_results[multiple] = {
                        "error": f"Multiple {multiple} ist nicht implementiert."
                    }
                    continue

                # ---------- Validierung der historischen Daten ----------
                if historical_df is None or getattr(historical_df, "empty", True):
                    crv_results[multiple] = {
                        "error": "Keine historischen Daten verfügbar."
                    }
                    continue

                # ---------- CRV berechnen ----------
                crv = self.model.calculate_crv(symbol, historical_df)

                if isinstance(crv, dict) and "error" in crv:
                    crv_results[multiple] = crv
                    continue

                # ---------- CRV bewerten ----------
                crv_positive = (
                        crv.get("crv_conservative") is not None
                        and crv["crv_conservative"] >= 3
                )

                crv_results[multiple] = {
                    **crv,
                    "crv_positive": bool(crv_positive)
                }

            except Exception as e:
                crv_results[multiple] = {
                    "error": f"Unerwarteter Fehler bei CRV-Berechnung: {str(e)}"
                }

        # ---------- 4) Gesamtergebnis ----------
        return {
            "symbol": symbol,
            "sectors": sectors,
            "multiples_used": sorted(multiples),
            "crv_results": crv_results
        }

    COMPANY_SECTORS = {
        "ILMN": ["Biotech", "Genomics"],
        "GOOGL": ["Internet Services", "Advertising", "Technology"],
        "TSLA": ["EV", "Energy", "Robotics", "AI"],
        "AMD": ["Semiconductors"],
        "PYPL": ["FinTech", "Digital Payments"],
        "NVDA": ["Semiconductors", "Tech"],
        "NKE": ["Sporting Goods"],
        "UNH": ["Healthcare"],
        "XPEV": ["EV", "AI", "Robotics"],
        "OCGN": ["Biotech", "Healthcare", "Pharma"],
        "UAA": ["Sporting Goods"],
        "BABA": ["E-Commerce", "Cloud Computing", "Digital Media", "FinTech", "Logistics"],
        "LUMN": ["Telecommunications"],
        "TTWO": ["Gaming", "Game Publisher"],
        "BIDU": ["Search Engine", "Cloud", "E-Commerce"],
        "JD": ["AI", "Robotics", "E-Commerce"],
        "CRSP": ["Pharma", "Biotech"],
        "NVO": ["Pharma"],
        "NFLX": ["Media", "Film", "Streaming"],
        "AAPL": ["Tech", "Digital Media"],
        "MO": ["Tabak"],
        "BYD": ["EV"],
        "SAP": ["Cloud Computing"]
    }

    # ActionModule – Branchen → relevante historische Multiples für CRV
    BRANCH_MULTIPLES_MAP = {
        "BioTech": [
            "Price_TangibleBookValue",
            "Price_NetCurrentAssets",
        ],
        "Pharma": [
            "Price_TangibleBookValue",
            "Price_NetCurrentAssets",
        ],
        "Genomik": [
            "Price_TangibleBookValue",
            "Price_NetCurrentAssets",
        ],

        "Healthcare": [
            "EV_EBIT",
            "Price_OperatingCashflow",
        ],

        "Tech": [
            "Price_FreeCashflow",
            "EV_EBITDA",
        ],

        "AI": [
            "EV_Sales",
            "Price_Sales",
        ],
        "Robots": [
            "EV_Sales",
            "Price_Sales",
        ],

        "Chips": [
            "EV_EBITDA",
            "EV_EBIT",
        ],
        "Halbleiter": [
            "EV_EBITDA",
            "EV_EBIT",
        ],

        "EV": [
            "EV_Sales",
            "EV_EBITDA",
        ],
        "Energy": [
            "EV_Sales",
            "EV_EBITDA",
        ],

        "FinTech": [
            "Price_FreeCashflow",
            "EV_EBITDA",
        ],
        "Digital Payment": [
            "Price_FreeCashflow",
            "EV_EBITDA",
        ],

        "E-Commerce": [
            "EV_Sales",
            "Price_FreeCashflow",
        ],
        "Digital Media": [
            "EV_Sales",
            "Price_FreeCashflow",
        ],

        "Logistik": [
            "EV_EBITDA",
            "Price_TangibleBookValue",
        ],

        "Telekommunikation": [
            "EV_EBITDA",
            "Price_FreeCashflow",
        ],

        "Sportartikelhersteller": [
            "Price_EBIT",
            "Price_FreeCashflow",
        ],

        "Game Publisher": [
            "Price_FreeCashflow",
            "EV_EBIT",
        ],

        "Film": [
            "EV_EBITDA",
            "Price_FreeCashflow",
        ],
        "Streaming": [
            "EV_EBITDA",
            "Price_FreeCashflow",
        ],
        "Tabak": [
            "Price_FreeCashflow",
            "EV_EBITDA",
            "Price_OperatingCashflow",
        ],
    }

    # ActionModule – Multiple → Model-Methode
    MULTIPLE_METHOD_MAP = {
        "EV_Sales": "calculate_historical_ev_sales",
        "EV_EBIT": "calculate_historical_ev_to_ebit",
        "EV_EBITDA": "calculate_historical_ev_to_ebitda",

        "Price_Book": "calculate_historical_price_to_book",
        "Price_Sales": "calculate_historical_price_to_sales",
        "Price_EBIT": "calculate_historical_price_to_ebit",
        "Price_NetCurrentAssets": "calculate_historical_price_netCurrentAssets",
        "Price_OperatingCashflow": "calculate_historical_price_OperatingCashflow",
        "Price_FreeCashflow": "calculate_historical_Price_FreeCashflow",
        "Price_TangibleBookValue": "calculate_historical_price_to_TangibleBookValue",
    }