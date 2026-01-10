import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from typing import Optional
from dateutil.relativedelta import relativedelta
from agent.DataLoader import DataLoader
from agent.DataPreprocessor import DataPreprocessor
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

class Model:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.dataloader = DataLoader()
        self.preprocessor = DataPreprocessor()

    PRICE_MULTIPLE_COLUMNS = ["Price_Book","Price_Sales", "Price_EBIT", "Price_NetCurrentAssets", "Price_OperatingCashflow",
                              "Price_FreeCashflow", "Price_TangibleBookValue"]
    EV_MULTIPLE_COLUMNS = ["EV_Sales", "EV_EBIT", "EV_EBITDA"]
    HISTORICAL_MULTIPLE_COLUMNS = PRICE_MULTIPLE_COLUMNS + EV_MULTIPLE_COLUMNS

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def calculate_KGV(self, symbol):
        """Berechnet das Kurs-Gewinn-Verhältnis (KGV).

        Negative KGV-Werte (bei negativem Gewinn) werden explizit zurückgegeben,
        da sie für Zykliker-Analysen essenziell sind.

        Args:
            symbol (str): Aktiensymbol (z.B. 'AAPL').

        Returns:
            float: KGV (positiv oder negativ), gerundet auf 2 Dezimalstellen,
                   oder float('inf') bei EPS = 0.
            dict: Fehlerdetails bei Problemen.
        """
        try:
            financials = self.dataloader.get_stock_financials(symbol)
            if isinstance(financials, dict) and "error" in financials:
                return financials

            shares = self.dataloader.get_shares_outstanding(symbol)
            if isinstance(shares, dict) and "error" in shares:
                return shares

            current_price = self.dataloader.get_current_price_per_share(symbol)
            if isinstance(current_price, dict) and "error" in current_price:
                return current_price

            if "Net Income" not in financials.index:
                return {"error": f"Keine Nettogewinn-Daten für {symbol} gefunden."}

            net_income = financials.loc["Net Income"].iloc[0]

            if shares <= 0:
                return {"error": f"Ungültige Aktienanzahl für {symbol}: {shares}"}

            eps = net_income / shares

            # EPS == 0 → mathematisch nicht definiert
            if eps == 0:
                return float("inf")

            # Positives oder negatives KGV explizit zulassen
            pe_ratio = current_price / eps
            return round(pe_ratio, 2)

        except Exception as e:
            return {"error": f"Fehler beim Berechnen des KGV für {symbol}: {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_tangible_book_value(self, balance_sheet):
        """
        Berechnet den materiellen Buchwert (TBV) nach pragmatischem Lynch-Ansatz.

        Priorität:
        1) 'Tangible Book Value' (falls direkt verfügbar)
        2) Stockholders Equity - Goodwill
           (keine zusätzlichen Intangible-Abzüge, um Doppelzählungen zu vermeiden)

        Returns:
            float oder dict mit 'error'
        """
        try:
            # 1) Direkter TBV, falls vorhanden (best case)
            if "Tangible Book Value" in balance_sheet.index:
                tbv = balance_sheet.loc["Tangible Book Value"].iloc[0]
                return float(tbv)

            # 2) Fallback: Equity - Goodwill
            if "Stockholders Equity" not in balance_sheet.index:
                return {"error": "Stockholders Equity nicht in Bilanz vorhanden – TBV nicht berechenbar."}

            equity = balance_sheet.loc["Stockholders Equity"].iloc[0]

            goodwill = 0.0
            if "Goodwill" in balance_sheet.index:
                goodwill = balance_sheet.loc["Goodwill"].iloc[0]

            tbv = equity - goodwill
            return float(tbv)

        except Exception as e:
            return {"error": f"Fehler bei TBV-Berechnung: {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def calculate_net_debt_to_ebitda(self, symbol, frequency="annual"):
        """
        Berechnet das Verhältnis von Nettoschulden zu EBITDA.

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            frequency (str): 'annual' für jährliche Daten, 'quarterly' für quartalsweise Daten.

        Returns:
            float: Das Net Debt/EBITDA-Verhältnis, gerundet auf 2 Dezimalstellen.
            dict: Fehlerdetails, falls Daten nicht verfügbar sind.
        """
        try:
            net_debt_data = self.dataloader.get_net_debt_data(symbol, frequency=frequency)
            ebitda_data = self.dataloader.get_ebitda_data(symbol, frequency=frequency)

            if "error" in net_debt_data:
                return net_debt_data
            if "error" in ebitda_data:
                return ebitda_data

            net_debt = net_debt_data['net_debt']
            ebitda = ebitda_data['ebitda']

            if ebitda == 0:
                return float('inf')  # Unendlich, wenn EBITDA = 0
            ratio = net_debt / ebitda
            return round(ratio, 2)
        except Exception as e:
            return {"error": f"Fehler beim Berechnen von Net Debt/EBITDA für {symbol}: {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def analyze_payout_ratio(self, symbol):
        """
        Analysiert die Ausschüttungsquote und warnt, wenn sie > 100% ist.

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').

        Returns:
            dict: Ausschüttungsquote und ggf. eine Warnung.
        """
        try:
            payout_data = self.dataloader.get_payout_ratio_data_annual(symbol)
            if "error" in payout_data:
                return payout_data

            payout_ratio = payout_data['payout_ratio_eps']
            if payout_ratio > 75:
                return {
                    "payout_ratio": payout_ratio,
                    "warning": "Ausschüttungsquote über 75% – Nachhaltigkeit prüfen."
                }
            else:
                return {
                    "payout_ratio": payout_ratio,
                    "message": "Ausschüttungsquote ≤ 100%."
                }
        except Exception as e:
            return {"error": f"Fehler beim Analysieren der Ausschüttungsquote für {symbol}: {str(e)}"}


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def analyze_dividend_history(self, symbol):
        """
        Analysiert die Dividendenhistorie eines Unternehmens und berechnet relevante Metriken, inkl. CAGR.
        Robust gegenüber Cache-Rückgaben (Series vs. DataFrame).
        """
        try:
            # 1) Dividendenhistorie abrufen
            dividend_history = self.dataloader.get_dividend_history(symbol)
            if isinstance(dividend_history, dict) and "error" in dividend_history:
                return dividend_history

            dividends = dividend_history.get("dividends_history")
            if dividends is None:
                return {"error": "Keine Dividendenhistorie verfügbar."}

            # 2) Robust re-hydrieren: Series -> DataFrame mit Spalte 'dividend'
            if isinstance(dividends, pd.Series):
                dividends = dividends.to_frame(name="dividend")
            elif isinstance(dividends, pd.DataFrame):
                if "dividend" not in dividends.columns:
                    # Falls aus Cache ohne Spaltennamen kommt, nimm erste Spalte als 'dividend'
                    first_col = dividends.columns[0]
                    dividends = dividends.rename(columns={first_col: "dividend"})
            else:
                return {"error": "Unerwartetes Format der Dividendenhistorie."}

            # Index säubern
            try:
                dividends.index = pd.to_datetime(dividends.index)
            except Exception:
                pass  # falls bereits DatetimeIndex

            # Leere Daten abfangen
            if dividends.empty or "dividend" not in dividends.columns:
                return {"error": "Keine Dividendenhistorie verfügbar."}

            # 3) Jährliche Dividenden summieren
            annual_dividends = dividends["dividend"].resample("YE").sum().astype(float)

            # 4) Metriken
            years_with_dividends = int((annual_dividends > 0).sum())
            increases = int(((annual_dividends.diff() > 0) &
                             (annual_dividends > 0) &
                             (annual_dividends.shift(1) > 0)).sum())

            available_years = int(len(annual_dividends))

            # 5) Dynamische CAGR-Berechnung
            cagr = float("nan")
            n = available_years - 1 if available_years >= 2 else 0
            if available_years >= 30:
                n = 30
                ending_value = float(annual_dividends.iloc[-1])
                beginning_value = float(annual_dividends.iloc[-n - 1])  # 31 Jahre zurück
            elif n >= 1:
                ending_value = float(annual_dividends.iloc[-1])
                beginning_value = float(annual_dividends.iloc[0])
            else:
                ending_value = beginning_value = None

            if n >= 1 and beginning_value and beginning_value > 0:
                cagr = round(((ending_value / beginning_value) ** (1 / n) - 1) * 100, 2)

            # 6) Ergebnis
            return {
                "years_with_dividends": years_with_dividends,
                "years_with_increases": increases,
                "cagr_period_years": n if n >= 1 else 0,
                "cagr": cagr
            }

        except Exception as e:
            return {"error": f"Fehler beim Analysieren der Dividendenhistorie für {symbol}: {str(e)}"}


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def calculate_historical_dividend_yield_average(self, symbol, years=10):
        """
        Berechnet den durchschnittlichen historischen Dividendenrendite.

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            years (int): Anzahl der Jahre für den Durchschnitt (Standard: 10).

        Returns:
            float: Durchschnittliche Rendite, gerundet auf 2 Dezimalstellen, oder None bei Fehlern.
        """
        try:
            dividend_history = self.dataloader.get_dividend_history(symbol)
            if "error" in dividend_history:
                return None
            dividends = dividend_history['dividends_history']

            stock_data = self.dataloader.get_stock_data(symbol, period=f"{years}y")
            if "error" in stock_data:
                return None
            prices = stock_data['Close']

            annual_dividends = dividends.resample('A').sum()
            year_end_prices = prices.resample('A').last()

            yields = (annual_dividends / year_end_prices) * 100
            average_yield = yields.dropna().mean()
            return round(average_yield, 2)
        except Exception as e:
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def determine_buy_sell_points(self, symbol):
        """
        Bestimmt Kauf-/Verkaufsempfehlungen basierend auf der Dividendenrendite.

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').

        Returns:
            str: "buy", "sell" oder "hold".
        """
        try:
            dividend_data = self.dataloader.get_dividend_data(symbol)
            if "error" in dividend_data:
                return dividend_data

            current_yield = dividend_data['dividend_yield']
            average_yield = self.calculate_historical_dividend_yield_average(symbol)

            if average_yield is None:
                return {"error": "Historische Daten für Dividendenrendite nicht verfügbar."}

            if current_yield > average_yield * 1.2:
                return "buy"
            elif current_yield < average_yield * 0.8:
                return "sell"
            else:
                return "hold"
        except Exception as e:
            return {"error": f"Fehler bei der Bestimmung der Kauf-/Verkaufszeitpunkte für {symbol}: {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def calculate_book_value_per_share(self, symbol):
        """Berechnet den materiellen Buchwert pro Aktie für ein gegebenes Aktiensymbol."""
        balance_sheet = self.dataloader.get_balance_sheet(symbol)
        if "error" in balance_sheet:
            return balance_sheet
        tangible_book_value = self.get_tangible_book_value(balance_sheet)
        if "error" in tangible_book_value:
            return tangible_book_value
        shares = self.dataloader.get_shares_outstanding(symbol)
        if "error" in shares:
            return shares
        return tangible_book_value / shares

    def get_current_tbv_and_price(self, symbol):
        """
        Gibt (tbv_per_share, current_price, pb_ratio) zurück
        – mit periodenkonsistenten Shares und robustem Fehlerhandling
        """

        # 1) Bilanz holen (quarterly = näher an 'aktuell')
        bs = self.dataloader.get_balance_sheet(symbol, frequency="quarterly")
        if isinstance(bs, dict):
            return bs, None, None

        # 2) TBV berechnen (Lynch-pragmatisch)
        tbv_total = self.get_tangible_book_value(bs)
        if isinstance(tbv_total, dict):
            return tbv_total, None, None

        # 3) Shares bevorzugt aus Bilanz
        shares = None
        if "Common Stock Shares Outstanding" in bs.index:
            shares = bs.loc["Common Stock Shares Outstanding"].iloc[0]

        # Fallback: yfinance info
        if not shares or shares <= 0:
            shares = self.dataloader.get_shares_outstanding(symbol)
            if isinstance(shares, dict):
                return shares, None, None

        if not isinstance(shares, (int, float)) or shares <= 0:
            return {"error": "Ungültige Anzahl ausstehender Aktien."}, None, None

        tbv_per_share = tbv_total / shares if tbv_total > 0 else 0.0

        # 4) Aktuellen Preis holen
        price_result = self.dataloader.get_current_price_per_share(symbol)
        if isinstance(price_result, dict):
            return price_result, None, None

        current_price = float(price_result)

        # 5) P/TBV
        pb_ratio = current_price / tbv_per_share if tbv_per_share > 0 else float("inf")
        pb_ratio = round(pb_ratio, 3)

        return tbv_per_share, current_price, pb_ratio

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def calculate_kuv(self, symbol: str, frequency: str = "annual") -> dict:
        """
        Berechnet das Kurs-Umsatz-Verhältnis (KUV).
        Für frequency='annual' wird der TTM-Umsatz (Summe der letzten vier Quartale) verwendet.
        Für frequency='quarterly' wird der Umsatz des neuesten Quartals verwendet.

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            frequency (str): 'annual' für TTM-Daten, 'quarterly' für quartalsweise Daten.

        Returns:
            dict: Enthält das KUV, gerundet auf 2 Dezimalstellen, oder Fehlerdetails.
                  Beispiel: {"KUV": 5.25, "symbol": "AAPL", "frequency": "annual"}
        """
        try:
            # Daten abrufen
            current_price = self.dataloader.get_current_price_per_share(symbol)
            shares = self.dataloader.get_shares_outstanding(symbol)

            # Umsatzdaten abrufen
            if frequency == "annual":
                # TTM-Umsatz: Summe der letzten vier Quartale
                quarterly_financials = self.dataloader.get_stock_financials(symbol, frequency="quarterly")
                if isinstance(quarterly_financials, dict) and "error" in quarterly_financials:
                    return quarterly_financials
                if "Total Revenue" not in quarterly_financials.index:
                    return {"error": f"Keine Umsatzdaten für {symbol} (quarterly) gefunden."}
                if len(quarterly_financials.loc["Total Revenue"]) < 4:
                    return {"error": f"Nicht genügend Quartalsdaten für {symbol} zur TTM-Berechnung."}
                revenue = quarterly_financials.loc["Total Revenue"].iloc[:4].sum()
            else:  # frequency == "quarterly"
                financials = self.dataloader.get_stock_financials(symbol, frequency="quarterly")
                if isinstance(financials, dict) and "error" in financials:
                    return financials
                if "Total Revenue" not in financials.index:
                    return {"error": f"Keine Umsatzdaten für {symbol} ({frequency}) gefunden."}
                revenue = financials.loc["Total Revenue"].iloc[0]

            # Fehlerprüfung für Daten
            if isinstance(current_price, dict) and "error" in current_price:
                return current_price
            if isinstance(shares, dict) and "error" in shares:
                return shares

            # Prüfen auf ungültige Werte
            if revenue <= 0:
                return {"error": f"Ungültiger Umsatz für {symbol} ({frequency}): {revenue}."}
            if shares <= 0:
                return {"error": f"Ungültige Anzahl ausstehender Aktien für {symbol}: {shares}."}
            if current_price <= 0:
                return {"error": f"Ungültiger Aktienkurs für {symbol}: {current_price}."}

            # KUV berechnen
            market_cap = current_price * shares
            kuv = market_cap / revenue
            return {"KUV": round(kuv, 2), "symbol": symbol, "frequency": frequency}
        except Exception as e:
            return {"error": f"Fehler beim Berechnen des KUV für {symbol}: {str(e)}"}


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def calculate_roe(self, symbol: str, frequency: str = "annual") -> dict:
        """
        Berechnet die Eigenkapitalrendite (ROE).
        Formel: ROE = Nettogewinn / Eigenkapital.
        Für frequency='annual' wird der TTM-Nettogewinn (Summe der letzten vier Quartale) verwendet.
        Für frequency='quarterly' wird der Nettogewinn des neuesten Quartals verwendet.
        Eigenkapital ist der neueste verfügbare Wert aus der Bilanz unter 'Stockholders Equity'.

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            frequency (str): 'annual' für TTM-Daten, 'quarterly' für quartalsweise Daten.

        Returns:
            dict: Enthält das ROE, gerundet auf 2 Dezimalstellen, oder Fehlerdetails.
                  Beispiel: {"ROE": 0.25, "symbol": "AAPL", "frequency": "annual"}
        """
        try:
            # Nettogewinn abrufen
            if frequency == "annual":
                # TTM-Nettogewinn: Summe der letzten vier Quartale
                quarterly_financials = self.dataloader.get_stock_financials(symbol, frequency="quarterly")
                if isinstance(quarterly_financials, dict) and "error" in quarterly_financials:
                    return quarterly_financials
                net_income = None
                if "Net Income" in quarterly_financials.index:
                    net_income = quarterly_financials.loc["Net Income"].iloc[:4].sum()
                elif "Net Income Common Stockholders" in quarterly_financials.index:
                    net_income = quarterly_financials.loc["Net Income Common Stockholders"].iloc[:4].sum()
                else:
                    return {
                        "error": f"Das Unternehmen {symbol} liefert keine Nettogewinn-Daten (weder 'Net Income' noch 'Net Income Common Stockholders') für TTM-Berechnung (quarterly)."}
                if len(quarterly_financials.loc[
                           "Net Income" if "Net Income" in quarterly_financials.index else "Net Income Common Stockholders"]) < 4:
                    return {"error": f"Nicht genügend Quartalsdaten für {symbol} zur TTM-Berechnung."}
                print(f"TTM Net Income für {symbol}: {net_income}")  # Debugging-Ausgabe
            else:  # frequency == "quarterly"
                financials = self.dataloader.get_stock_financials(symbol, frequency="quarterly")
                if isinstance(financials, dict) and "error" in financials:
                    return financials
                net_income = None
                if "Net Income" in financials.index:
                    net_income = financials.loc["Net Income"].iloc[0]
                elif "Net Income Common Stockholders" in financials.index:
                    net_income = financials.loc["Net Income Common Stockholders"].iloc[0]
                else:
                    return {
                        "error": f"Das Unternehmen {symbol} liefert keine Nettogewinn-Daten (weder 'Net Income' noch 'Net Income Common Stockholders') für {frequency}."}
                print(f"Quarterly Net Income für {symbol}: {net_income}")  # Debugging-Ausgabe

            # Eigenkapital abrufen
            balance_sheet = self.dataloader.get_balance_sheet(symbol, frequency=frequency)
            if isinstance(balance_sheet, dict) and "error" in balance_sheet:
                return balance_sheet
            if "Stockholders Equity" not in balance_sheet.index:
                return {
                    "error": f"Keine Eigenkapital-Daten für {symbol} ({frequency}) gefunden. Verfügbare Labels: {list(balance_sheet.index)}"}
            equity = balance_sheet.loc["Stockholders Equity"].iloc[0]
            print(f"Stockholders Equity für {symbol} ({frequency}): {equity}")  # Debugging-Ausgabe

            # Prüfen auf ungültige Werte
            if pd.isna(net_income):
                return {"error": f"Ungültiger Nettogewinn für {symbol} ({frequency}): {net_income}."}
            if equity <= 0:
                return {"error": f"Ungültiges Eigenkapital für {symbol} ({frequency}): {equity}."}

            # ROE berechnen
            roe = net_income / equity
            return {"ROE": round(roe, 2), "symbol": symbol, "frequency": frequency}
        except Exception as e:
            return {"error": f"Fehler beim Berechnen des ROE für {symbol}: {str(e)}"}


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def calculate_debt_to_equity(self, symbol: str, frequency: str = "annual") -> dict:
        """
        Berechnet den Verschuldungsgrad (Debt-to-Equity Ratio).
        Formel: Total Liabilities / Stockholders Equity.
        Verwendet die neuesten verfügbaren Bilanzdaten basierend auf der Frequenz.
        Unterstützt mehrere Bezeichnungen für Verbindlichkeiten, um mehr Unternehmen abzudecken.

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            frequency (str): 'annual' für jährliche Daten, 'quarterly' für quartalsweise Daten.

        Returns:
            dict: Enthält den Verschuldungsgrad, gerundet auf 2 Dezimalstellen, oder Fehlerdetails.
                  Beispiel: {"debt_to_equity": 1.5, "symbol": "AAPL", "frequency": "annual"}
        """
        try:
            # Bilanzdaten abrufen
            balance_sheet = self.dataloader.get_balance_sheet(symbol, frequency=frequency)
            if isinstance(balance_sheet, dict) and "error" in balance_sheet:
                return balance_sheet

            # Verbindlichkeiten abrufen (versuche mehrere Bezeichnungen)
            total_liabilities = None
            liability_labels = [
                "Total Liabilities Net Minority Interest",  # Häufigste Bezeichnung
                "Total Liabilities",  # Standardbezeichnung
                "TotalLiabilities"  # Variante ohne Leerzeichen
            ]
            for label in liability_labels:
                if label in balance_sheet.index:
                    total_liabilities = balance_sheet.loc[label].iloc[0]
                    break

            if total_liabilities is None:
                return {
                    "error": f"Keine Verbindlichkeitsdaten für {symbol} ({frequency}) gefunden. Verfügbare Labels: {list(balance_sheet.index)}"
                }

            # Eigenkapital abrufen
            if "Stockholders Equity" not in balance_sheet.index:
                return {
                    "error": f"Keine Eigenkapital-Daten für {symbol} ({frequency}) gefunden. Verfügbare Labels: {list(balance_sheet.index)}"
                }
            equity = balance_sheet.loc["Stockholders Equity"].iloc[0]

            # Prüfen auf ungültige Werte
            if pd.isna(total_liabilities):
                return {"error": f"Ungültige Verbindlichkeitsdaten für {symbol} ({frequency}): {total_liabilities}."}
            if total_liabilities < 0:
                return {
                    "error": f"Negative Verbindlichkeiten für {symbol} ({frequency}): {total_liabilities}. Dies ist unrealistisch."
                }
            if pd.isna(equity):
                return {"error": f"Ungültiges Eigenkapital für {symbol} ({frequency}): {equity}."}
            if equity <= 0:
                return {
                    "error": f"Ungültiges Eigenkapital für {symbol} ({frequency}): {equity}. Eigenkapital muss größer als 0 sein."
                }

            # Verschuldungsgrad berechnen
            debt_to_equity = total_liabilities / equity
            return {"debt_to_equity": round(debt_to_equity, 2), "symbol": symbol, "frequency": frequency}
        except Exception as e:
            return {"error": f"Fehler beim Berechnen des Verschuldungsgrads für {symbol}: {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def calculate_peg_ratio(self, symbol, start_date=None, end_date=None, use_cache=True):
        """
        Berechnet die PEG-Ratio eines Unternehmens, entweder durch direkte Abfrage von Yahoo Finance
        oder durch manuelle Berechnung basierend auf KGV und jährlicher Gewinnwachstumsrate.

        Die PEG-Ratio wird als (P/E Ratio) / (erwartete jährliche EPS-Wachstumsrate) definiert.

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            start_date (str, optional): Startdatum im Format 'YYYY-MM-DD' für die Wachstumsberechnung.
            end_date (str, optional): Enddatum im Format 'YYYY-MM-DD' für die Wachstumsberechnung.
            use_cache (bool): Verwendung des Caches (Standard: True).

        Returns:
            dict: Enthält die PEG-Ratio, die Methode der Berechnung und ggf. zusätzliche Daten.
        """
        try:
            # Schritt 1: Direkte PEG-Ratio von DataLoader abrufen
            peg_data = self.dataloader.get_peg_ratio(symbol, use_cache=use_cache)
            if "error" not in peg_data:
                return {
                    "peg_ratio": peg_data["peg_ratio"],
                    "symbol": symbol,
                    "method": peg_data["method"]
                }

            # Schritt 2: Manuelle Berechnung, falls direkte Abfrage fehlschlägt
            pe_ratio = self.calculate_KGV(symbol)
            if isinstance(pe_ratio, dict) and "error" in pe_ratio:
                return pe_ratio

            growth_data = self.calculate_avg_annual_profit_growth(symbol, start_date, end_date, use_cache=use_cache)
            if "error" in growth_data:
                return growth_data

            if "avg_growth" not in growth_data:
                return {
                    "error": f"Kann PEG-Ratio für {symbol} nicht berechnen: {growth_data.get('message', 'Unbekannter Fehler bei der Wachstumsberechnung.')}",
                    "symbol": symbol
                }

            growth_rate = growth_data["avg_growth"]
            if growth_rate <= 0:
                return {
                    "error": f"Ungültige Gewinnwachstumsrate für {symbol}: {growth_rate}. Wachstum muss positiv sein.",
                    "symbol": symbol
                }

            peg_ratio = pe_ratio / growth_rate
            if not isinstance(peg_ratio, (int, float)):
                return {
                    "error": f"Ungültiger PEG-Ratio für {symbol}: {peg_ratio}",
                    "symbol": symbol
                }

            return {
                "peg_ratio": round(float(peg_ratio), 2),
                "symbol": symbol,
                "method": "manual_calculation",
                "pe_ratio": round(float(pe_ratio), 2),
                "growth_rate": round(float(growth_rate), 2)
            }

        except Exception as e:
            return {
                "error": f"Fehler beim Berechnen der PEG-Ratio für {symbol}: {str(e)}",
                "symbol": symbol
            }

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def calculate_interest_coverage_ratio(self, symbol, frequency="annual"):
        """
        Berechnet die Zinsdeckungsrate (EBIT / Zinsaufwendungen) eines Unternehmens.

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            frequency (str): Zeitraum, entweder 'annual' oder 'quarterly' (Standard: 'annual').

        Returns:
            dict: Enthält die Zinsdeckungsrate.
                  Beispiel:
                  {
                      "interest_coverage_ratio": 5.0,
                      "symbol": "AAPL",
                      "frequency": "annual",
                      "date": "2024-12-31"
                  }
                  oder bei Fehler:
                  {
                      "error": "Fehlerbeschreibung",
                      "symbol": "AAPL"
                  }
        """
        try:
            # Hole EBIT-Daten
            ebit_data = self.dataloader.get_ebit_data(symbol, frequency=frequency)
            if isinstance(ebit_data, dict) and "error" in ebit_data:
                return ebit_data

            # Hole Zinsaufwendungen
            interest_expense_data = self.dataloader.get_interest_expense_data(symbol, frequency=frequency)
            if isinstance(interest_expense_data, dict) and "error" in interest_expense_data:
                return interest_expense_data

            # Extrahiere Werte
            ebit = ebit_data["ebit"]
            interest_expense = interest_expense_data["interest_expense"]

            # Prüfe auf ungültige Werte
            if interest_expense == 0:
                error = {
                    "error": f"Zinsaufwendungen für {symbol} ({frequency}) sind 0, Zinsdeckungsrate kann nicht berechnet werden.",
                    "symbol": symbol
                }
                return error
            if ebit <= 0:
                error = {
                    "error": f"EBIT für {symbol} ({frequency}) ist {ebit}, Zinsdeckungsrate kann nicht berechnet werden (EBIT muss positiv sein).",
                    "symbol": symbol
                }
                return error

            # Berechne die Zinsdeckungsrate
            interest_coverage_ratio = ebit / interest_expense
            interest_coverage_ratio = round(interest_coverage_ratio, 2)

            # Erstelle Rückgabe-Daten
            data = {
                "interest_coverage_ratio": float(interest_coverage_ratio),
                "symbol": symbol,
                "frequency": frequency,
                "date": ebit_data["date"]
            }

            return data

        except Exception as e:
            error = {
                "error": f"Fehler beim Berechnen der Zinsdeckungsrate für {symbol} ({frequency}): {str(e)}",
                "symbol": symbol
            }
            return error

    def calculate_cashflow_margin(self, symbol: str, frequency: str = "annual") -> dict:
        """
        Berechnet die Cashflow-Marge (Operating Cashflow / Umsatz) für ein gegebenes Aktiensymbol.

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            frequency (str): 'annual' für jährliche Daten, 'quarterly' für quartalsweise Daten.

        Returns:
            dict: Enthält die Cashflow-Marge (in Prozent), Symbol, Frequenz und Datum.
                  Beispiel:
                  {
                      "cashflow_margin": 25.0,
                      "symbol": "AAPL",
                      "frequency": "annual",
                      "date": "2024-12-31"
                  }
                  oder bei Fehler:
                  {
                      "error": "Fehlerbeschreibung",
                      "symbol": symbol
                  }
        """
        if frequency not in ["annual", "quarterly"]:
            return {"error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.", "symbol": symbol}

        try:
            # Operativen Cashflow abrufen
            cashflow_result = self.dataloader.get_operating_cashflow(symbol, frequency)
            if "error" in cashflow_result:
                return {"error": f"Fehler beim Abrufen des operativen Cashflows: {cashflow_result['error']}",
                        "symbol": symbol}

            # Umsatz abrufen
            revenue_result = self.dataloader.get_revenue(symbol, frequency)
            if "error" in revenue_result:
                return {"error": f"Fehler beim Abrufen des Umsatzes: {revenue_result['error']}", "symbol": symbol}

            # Werte extrahieren
            operating_cashflow = cashflow_result["operating_cashflow"]
            revenue = revenue_result["revenue"]
            date = cashflow_result["date"]

            # Datumskonsistenz prüfen
            if date != revenue_result["date"]:
                return {
                    "error": f"Datumsinkonsistenz für {symbol} ({frequency}): Cashflow-Datum {date} != Umsatz-Datum {revenue_result['date']}",
                    "symbol": symbol
                }

            # Division durch Null prüfen
            if revenue == 0:
                return {
                    "error": f"Umsatz für {symbol} ({frequency}) ist 0, Cashflow-Marge kann nicht berechnet werden.",
                    "symbol": symbol}

            # Cashflow-Marge berechnen (in Prozent)
            cashflow_margin = (operating_cashflow / revenue) * 100
            cashflow_margin = round(cashflow_margin, 2)

            # Prüfen auf ungültige Werte
            if pd.isna(cashflow_margin):
                return {"error": f"Ungültige Cashflow-Marge für {symbol} ({frequency}): {cashflow_margin}",
                        "symbol": symbol}

            # Ergebnis-Dictionary erstellen
            result = {
                "cashflow_margin": float(cashflow_margin),
                "symbol": symbol,
                "frequency": frequency,
                "date": date
            }

            return result

        except Exception as e:
            return {"error": f"Fehler beim Berechnen der Cashflow-Marge für {symbol} ({frequency}): {str(e)}",
                    "symbol": symbol}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def calculate_inventory_to_revenue_ratio(self, symbol: str, frequency: str = "annual") -> dict:
        """
        Berechnet die Vorräte/Umsatz-Kennzahl (in Prozent) für ein gegebenes Aktiensymbol.

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            frequency (str): 'annual' für jährliche Daten, 'quarterly' für quartalsweise Daten.

        Returns:
            dict: Enthält die Vorräte/Umsatz-Kennzahl (in Prozent), Symbol, Frequenz und Datum.
                  Beispiel:
                  {
                      "inventory_to_revenue_ratio": 10.0,
                      "symbol": "AAPL",
                      "frequency": "annual",
                      "date": "2024-12-31"
                  }
                  oder bei Fehler:
                  {
                      "error": "Fehlerbeschreibung",
                      "symbol": symbol
                  }
        """
        if frequency not in ["annual", "quarterly"]:
            return {"error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.", "symbol": symbol}

        try:
            # Vorräte abrufen
            inventory_result = self.dataloader.get_inventory(symbol, frequency)
            if "error" in inventory_result:
                if "Keine Vorräte" in inventory_result["error"]:
                    return {
                        "error": f"Keine Vorräte für {symbol} ({frequency}) vorhanden, möglicherweise Dienstleister.",
                        "symbol": symbol}
                return {"error": f"Fehler beim Abrufen der Vorräte: {inventory_result['error']}", "symbol": symbol}

            # Umsatz abrufen
            revenue_result = self.dataloader.get_revenue(symbol, frequency)
            if "error" in revenue_result:
                return {"error": f"Fehler beim Abrufen des Umsatzes: {revenue_result['error']}", "symbol": symbol}

            # Werte extrahieren
            inventory = inventory_result["inventory"]
            revenue = revenue_result["revenue"]
            date = inventory_result["date"]

            # Datumskonsistenz prüfen
            if date != revenue_result["date"]:
                return {
                    "error": f"Datumsinkonsistenz für {symbol} ({frequency}): Vorräte-Datum {date} != Umsatz-Datum {revenue_result['date']}",
                    "symbol": symbol
                }

            # Division durch Null prüfen
            if revenue == 0:
                return {
                    "error": f"Umsatz für {symbol} ({frequency}) ist 0, Vorräte/Umsatz kann nicht berechnet werden.",
                    "symbol": symbol}

            # Vorräte/Umsatz berechnen (in Prozent)
            ratio = (inventory / revenue) * 100
            ratio = round(ratio, 2)

            # Prüfen auf ungültige Werte
            if pd.isna(ratio):
                return {"error": f"Ungültige Vorräte/Umsatz-Kennzahl für {symbol} ({frequency}): {ratio}",
                        "symbol": symbol}

            # Ergebnis-Dictionary erstellen
            result = {
                "inventory_to_revenue_ratio": float(ratio),
                "symbol": symbol,
                "frequency": frequency,
                "date": date
            }

            return result

        except Exception as e:
            return {"error": f"Fehler beim Berechnen der Vorräte/Umsatz-Kennzahl für {symbol} ({frequency}): {str(e)}",
                    "symbol": symbol}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def calculate_cash_to_market_cap(self, symbol: str, frequency: str = "annual", use_cache: bool = True) -> dict:
        """
        Berechnet das Verhältnis von Cash & Cash Equivalents zur Marktkapitalisierung.

        Formel:
            Cash-to-Market-Cap = Cash & Equivalents / Market Cap

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL')
            frequency (str): 'annual' oder 'quarterly' (Cash-Seite)
            use_cache (bool): Cache verwenden (default: True)

        Returns:
            dict:
                {
                  "cash_to_market_cap": float,
                  "cash": float,
                  "market_cap": float,
                  "symbol": symbol,
                  "frequency": frequency,
                  "date": "<cash-date>"
                }
            oder bei Fehler:
                {"error": "...", "symbol": symbol}
        """
        try:
            # 1. Cash & Equivalents abrufen
            cash_data = self.dataloader.get_cash_and_equivalents(symbol, frequency=frequency, use_cache=use_cache)

            if isinstance(cash_data, dict) and "error" in cash_data:
                return cash_data

            cash_value = cash_data.get("cash_and_equivalents")
            cash_date = cash_data.get("date")

            if cash_value is None or cash_value < 0:
                return {
                    "error": f"Ungültiger Cash-Wert für {symbol}: {cash_value}",
                    "symbol": symbol
                }

            # 2. Marktkapitalisierung abrufen
            market_cap_data = self.dataloader.get_market_cap(symbol, use_cache=use_cache)
            if isinstance(market_cap_data, dict) and "error" in market_cap_data:
                return market_cap_data

            market_cap = market_cap_data.get("market_cap")

            if market_cap is None or market_cap <= 0:
                return {
                    "error": f"Ungültige Marktkapitalisierung für {symbol}: {market_cap}",
                    "symbol": symbol
                }

            # 3. Verhältnis berechnen
            cash_to_market_cap = round(cash_value / market_cap, 4)

            return {
                "cash_to_market_cap": cash_to_market_cap,
                "cash": float(cash_value),
                "market_cap": float(market_cap),
                "symbol": symbol,
                "frequency": frequency,
                "date": cash_date
            }

        except Exception as e:
            return {
                "error": f"Fehler bei calculate_cash_to_market_cap für {symbol}: {str(e)}",
                "symbol": symbol
            }

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def calculate_ev_to_sales(self, symbol: str, frequency: str = "annual", use_cache: bool = True) -> dict:
        """
        Berechnet das EV/Sales Multiple (Enterprise Value to Sales Ratio) für ein Unternehmen.

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            frequency (str): 'annual' für jährliche Daten, 'quarterly' für quartalsweise Daten.
            use_cache (bool): Ob der Cache verwendet werden soll (Standard: True).

        Returns:
            dict: Enthält das EV/Sales Multiple, Symbol und Frequenz, oder Fehlerdetails.
        """
        data_type = f"ev_to_sales_{frequency}"
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, data_type)
            if cached_data is not None:
                return cached_data

        try:
            if frequency not in ["annual", "quarterly"]:
                return {"error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.",
                        "symbol": symbol}

            ev_data = self.dataloader.get_enterprise_value(symbol, frequency)
            if "error" in ev_data:
                return ev_data

            revenue_data = self.dataloader.get_revenue(symbol, frequency)
            if "error" in revenue_data:
                return revenue_data

            enterprise_value = ev_data["enterprise_value"]
            revenue = revenue_data["revenue"]
            if revenue <= 0:
                return {"error": f"Ungültiger Umsatz für {symbol} ({frequency}): {revenue}.", "symbol": symbol}

            ev_to_sales = enterprise_value / revenue
            result = {
                "ev_to_sales": float(ev_to_sales),
                "symbol": symbol,
                "frequency": frequency
            }

            if use_cache:
                self.dataloader._cache_data(result, symbol, data_type)
            return result

        except Exception as e:
            return {"error": f"Fehler bei der Berechnung des EV/Sales Multiples für {symbol}: {str(e)}",
                    "symbol": symbol}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def calculate_price_to_freeCashflow(self, symbol: str, use_cache: bool = True, frequency: str = "annual"):
        """
        Berechnet das Price/FreeCashflow-Verhältnis (Preis pro Aktie / FreeCashflow pro Aktie).

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (Standard: True).
            frequency (str): Zeitraum, entweder 'annual' oder 'quarterly' (Standard: 'annual').

        Returns:
            dict: Enthält das Price/FreeCashflow-Verhältnis, Symbol, Frequenz, Datum und ggf. eine Nachricht.
                  Beispiel:
                  {
                      "price_to_freeCashflow": 15.0,
                      "symbol": "AAPL",
                      "frequency": "annual",
                      "date": "2024-12-31"
                  }
                  oder bei ungültigem FreeCashflow:
                  {
                      "price_to_freeCashflow": "inf",
                      "symbol": "AAPL",
                      "frequency": "annual",
                      "date": "2024-12-31",
                      "message": "Price/FreeCashflow ist unendlich aufgrund eines null oder negativen FreeCashflow."
                  }
                  oder bei Fehler:
                  {
                      "error": "Fehlerbeschreibung",
                      "symbol": "AAPL"
                  }
        """
        cache_key = f"{symbol}_price_to_freeCashflow_{frequency}"
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None and "error" not in cached_data:
                return cached_data

        try:
            # Prüfen, ob Frequenz gültig ist
            if frequency not in ["annual", "quarterly"]:
                return {"error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.",
                        "symbol": symbol}

            # Aktuellen Preis pro Aktie abrufen
            price_data = self.dataloader.get_current_price_per_share(symbol)
            if isinstance(price_data, dict) and "error" in price_data:
                return {"error": price_data["error"], "symbol": symbol}
            current_price = price_data
            if not isinstance(current_price, (int, float)) or current_price <= 0:
                return {"error": f"Ungültiger aktueller Preis für {symbol}: {current_price}", "symbol": symbol}

            # FreeCashflow-Daten abrufen
            freeCashflow_data = self.dataloader.get_free_cashflow(symbol, frequency=frequency)
            if "error" in freeCashflow_data:
                return {"error": freeCashflow_data["error"], "symbol": symbol}
            freeCashflow = freeCashflow_data["free_cashflow"]
            date = freeCashflow_data.get("date")
            if not isinstance(freeCashflow, (int, float)):
                return {"error": f"Ungültiger FreeCashflow-Wert für {symbol}: {freeCashflow}", "symbol": symbol}

            # Prüfen auf ungültigen FreeCashflow
            shares = self.dataloader.get_shares_outstanding(symbol)
            if isinstance(shares, dict) and "error" in shares:
                return {"error": shares["error"], "symbol": symbol}
            if shares is None or not isinstance(shares, (int, float)) or shares <= 0:
                return {"error": f"Ungültige Anzahl ausstehender Aktien für {symbol}: {shares}", "symbol": symbol}
            freeCashflow_per_share = freeCashflow / shares
            if freeCashflow_per_share <= 0:
                data = {
                    "price_to_freeCashflow": "inf",
                    "symbol": symbol,
                    "frequency": frequency,
                    "date": date,
                    "message": "Price/FreeCashflow ist unendlich aufgrund eines null oder negativen FreeCashflow."
                }
                if use_cache:
                    self.dataloader._cache_data(data, symbol, cache_key)
                return data

            # Price/FreeCashflow berechnen
            price_to_freeCashflow = round(current_price / freeCashflow_per_share, 2)
            data = {
                "price_to_freeCashflow": price_to_freeCashflow,
                "symbol": symbol,
                "frequency": frequency,
                "date": date
            }
            if use_cache:
                self.dataloader._cache_data(data, symbol, cache_key)
            return data

        except Exception as e:
            return {"error": f"Fehler beim Berechnen von Price/FreeCashflow für {symbol} ({frequency}): {str(e)}",
                    "symbol": symbol}


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def calculate_annual_inflation_rate(self, current_date_str=None, target_date_str=None):
        """
        Berechnet die jährliche Inflationsrate basierend auf einem angegebenen oder dem aktuellsten verfügbaren Monat
        im Vergleich zum Vorjahresmonat.

        Args:
            current_date_str (str, optional): Das aktuelle Datum im Format 'YYYY-MM-DD'.
                                             Wenn None, wird das aktuelle Datum verwendet.
            target_date_str (str, optional): Das Datum des Monats, für den die Inflation berechnet werden soll
                                            im Format 'YYYY-MM-DD'. Wenn None, wird der aktuellste verfügbare
                                            Monat basierend auf current_date_str verwendet.

        Returns:
            dict: Enthält die Inflationsrate und die verwendeten CPI-Daten.
        """
        try:
            # Aktuelles Datum (heute)
            today = datetime.now()

            # Wenn kein Datum angegeben, aktuelles Datum verwenden
            if current_date_str is None:
                current_date = today
            else:
                current_date = datetime.strptime(current_date_str, '%Y-%m-%d')

            # Prüfe, ob current_date in der Zukunft liegt
            if current_date > today:
                return {
                    "error": f"Das Datum {current_date_str} liegt in der Zukunft. Inflationsdaten sind nicht verfügbar."}

            # Bestimme den Zielmonat (benutzerdefinierter Monat oder aktuellster verfügbarer Monat)
            if target_date_str is not None:
                target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
                # Prüfe, ob target_date in der Zukunft liegt
                if target_date > today:
                    return {
                        "error": f"Das Ziel-Datum {target_date_str} liegt in der Zukunft. Inflationsdaten sind nicht verfügbar."}
                # Bestimme den letzten Tag des Monats von target_date
                next_month = target_date.replace(day=28) + timedelta(days=4)
                latest_available_date = next_month - timedelta(days=next_month.day)
            else:
                # Bestimme den neuesten verfügbaren Monat basierend auf dem Veröffentlichungszyklus
                if current_date.day < 15:  # Vor der Mitte des Monats -> Daten des vorherigen Monats verfügbar
                    latest_available_date = current_date.replace(day=1) - timedelta(days=1)
                else:  # Nach der Mitte des Monats -> Daten des aktuellen Monats verfügbar
                    latest_available_date = current_date.replace(day=1) - timedelta(days=1)

            latest_available_month = latest_available_date.strftime('%Y-%m')

            # Vorjahresmonat
            previous_year_date = latest_available_date - relativedelta(years=1)
            previous_year_month = previous_year_date.strftime('%Y-%m')

            # Dynamischer Zeitraum: Stelle sicher, dass der Vorjahresmonat abgedeckt ist
            earliest_needed_date = previous_year_date - timedelta(days=365)
            start_date = earliest_needed_date.strftime('%Y-%m-%d')
            end_date = current_date.strftime('%Y-%m-%d')
            cpi_data = self.dataloader.get_inflation_data(start_date=start_date, end_date=end_date)

            if isinstance(cpi_data, dict) and "error" in cpi_data:
                return cpi_data

            # CPI-Daten nach Datum sortieren
            cpi_data_sorted = sorted(cpi_data, key=lambda x: x['date'])

            # Neueste verfügbare CPI-Daten finden
            latest_cpi_entry = None
            for entry in reversed(cpi_data_sorted):
                entry_date = datetime.strptime(entry['date'], '%Y-%m-%d')
                if entry_date.strftime('%Y-%m') == latest_available_month:
                    latest_cpi_entry = entry
                    break
            if not latest_cpi_entry:
                return {"error": f"Keine CPI-Daten für den erwarteten Monat {latest_available_month} verfügbar."}

            # Vorjahresmonat finden
            previous_cpi_entry = None
            for entry in reversed(cpi_data_sorted):
                entry_date = datetime.strptime(entry['date'], '%Y-%m-%d')
                if entry_date.strftime('%Y-%m') <= previous_year_month:
                    previous_cpi_entry = entry
                    break
            if not previous_cpi_entry:
                return {"error": f"Keine CPI-Daten für {previous_year_month} oder früher verfügbar."}

            # Inflationsrate berechnen
            cpi_current = latest_cpi_entry['value']
            cpi_previous = previous_cpi_entry['value']
            inflation_rate = ((cpi_current - cpi_previous) / cpi_previous) * 100
            inflation_rate = round(inflation_rate, 2)

            return {
                "current_cpi": cpi_current,
                "previous_cpi": cpi_previous,
                "current_date": latest_cpi_entry['date'],
                "previous_date": previous_cpi_entry['date'],
                "inflation_rate": inflation_rate,
                "target_month": latest_available_month
            }

        except Exception as e:
            return {"error": f"Fehler bei der Berechnung der jährlichen Inflationsrate: {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def calculate_total_inflation_for_period(self, start_date, end_date, use_cache=True):
        """
        Berechnet die tatsächliche kumulative Inflation für einen gegebenen Zeitraum basierend auf CPI-Daten.

        Args:
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (z. B. '2020-02-01').
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (z. B. '2025-02-01').
            use_cache (bool): Verwendung des Caches (Standard: True).

        Returns:
            dict: Enthält die tatsächliche Inflation in Prozent oder eine Fehlermeldung.
        """
        try:
            # Konvertiere Datumsstrings zu datetime-Objekten
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)

            # Prüfe, ob der Zeitraum sinnvoll ist
            if end <= start:
                return {"error": "Enddatum muss nach Startdatum liegen."}

            # Hole Inflationsdaten für den Zeitraum
            cpi_data = self.dataloader.get_inflation_data(use_cache=use_cache, start_date=start_date, end_date=end_date)
            if isinstance(cpi_data, dict) and "error" in cpi_data:
                return cpi_data

            if not cpi_data:
                return {"error": f"Keine Inflationsdaten für den Zeitraum {start_date} bis {end_date} verfügbar."}

            # Konvertiere in DataFrame und sortiere nach Datum
            cpi_df = pd.DataFrame(cpi_data)
            cpi_df['date'] = pd.to_datetime(cpi_df['date'])
            cpi_df = cpi_df.sort_values('date')

            # Filtere Daten für den genauen Zeitraum
            cpi_df = cpi_df[(cpi_df['date'] >= start) & (cpi_df['date'] <= end)]

            if len(cpi_df) < 2:
                return {
                    "error": f"Nicht genügend Inflationsdaten für den Zeitraum {start_date} bis {end_date} (mindestens 2 Monate erforderlich)."}

            # Nimm den ersten und letzten CPI-Wert im Zeitraum
            cpi_start = cpi_df.iloc[0]['value']  # CPI am Startdatum
            cpi_end = cpi_df.iloc[-1]['value']  # CPI am Enddatum

            if cpi_start == 0:
                return {"error": "CPI-Wert am Startdatum ist 0, kann Inflationsrate nicht berechnen."}

            # Berechne die tatsächliche kumulative Inflation
            total_inflation = ((cpi_end / cpi_start) - 1) * 100
            total_inflation = round(total_inflation, 2)

            return {
                "total_inflation": total_inflation,
                "start_date": start_date,
                "end_date": end_date,
                "message": f"Die tatsächliche Inflation von {start_date} bis {end_date} beträgt {total_inflation}%."
            }

        except Exception as e:
            return {
                "error": f"Fehler bei der Berechnung der Inflation für den Zeitraum {start_date} bis {end_date}: {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def calculate_avg_quarterly_profit_growth(self, symbol, start_date, end_date, use_cache=True):
        """
        Berechnet die durchschnittliche quartalsweise Gewinnwachstumsrate (AQGR) basierend auf allen verfügbaren
        quartalsweisen Nettogewinndaten, sofern keine negativen Werte vorliegen. Bei negativen Werten werden die Daten
        aufgelistet und eine Warnung ausgegeben.

        Die Methode ignoriert die Parameter start_date und end_date, da sie immer alle verfügbaren Daten verwendet.
        Die Daten werden absteigend sortiert (neueste zuerst).

        Args:
            symbol (str): Das Aktien-Symbol (z. B. 'AAPL').
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (wird ignoriert, nur für Kompatibilität).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (wird ignoriert, nur für Kompatibilität).
            use_cache (bool): Verwendung des Caches (Standard: True).

        Returns:
            dict: Enthält die AQGR oder eine Liste der Nettogewinne mit Warnung, den analysierten Zeitraum
                  und eine Fehlermeldung, falls zutreffend.
                  Beispiel bei Erfolg mit AQGR:
                  {
                      "avg_growth": 2.5,
                      "symbol": "AAPL",
                      "actual_start_date": "2024-03-31",
                      "actual_end_date": "2025-03-31",
                      "frequency": "quarterly",
                      "message": "Durchschnittliche quartalsweise Gewinnwachstumsrate (AQGR) für AAPL: 2.5% von 2024-03-31 bis 2025-03-31 basierend auf 5 Berichten."
                  }
                  Beispiel bei negativen Werten:
                  {
                      "net_incomes": [{"date": "2025-03-31", "value": -300}, {"date": "2024-12-31", "value": -350}, ...],
                      "symbol": "LCID",
                      "actual_start_date": "2024-03-31",
                      "actual_end_date": "2025-03-31",
                      "frequency": "quarterly",
                      "message": "Das Unternehmen LCID weist durchgängig negative Nettogewinne auf: -300, -350, ... Mio. USD. Eine Investition in diesen Wert ist mit hohem Risiko behaftet, da das Unternehmen Verluste macht."
                  }
                  oder bei Fehler:
                  {
                      "error": "Fehlerbeschreibung"
                  }
        """
        try:
            # Schritt 1: Finanzdaten abrufen
            financials = self.dataloader.get_stock_financials(symbol, frequency="quarterly", use_cache=use_cache)
            if isinstance(financials, dict) and "error" in financials:
                return financials

            # Schritt 2: Prüfen, ob Nettogewinn-Daten vorhanden sind
            if "Net Income" not in financials.index:
                return {"error": f"Keine Nettogewinn-Daten für {symbol} verfügbar.", "symbol": symbol}

            # Schritt 3: Nettogewinndaten extrahieren, bereinigen und absteigend sortieren
            net_incomes = financials.loc["Net Income"].dropna()
            net_incomes.index = pd.to_datetime(net_incomes.index)
            net_incomes = net_incomes.sort_index(ascending=False)  # Neueste Daten zuerst

            # Schritt 4: Prüfen, ob genügend Datenpunkte vorhanden sind (mindestens 2 für Wachstumsrate)
            if len(net_incomes) < 2:
                return {"error": f"Nicht genügend Nettogewinndaten für {symbol} (mindestens 2 Berichte erforderlich)."}

            # Schritt 5: Prüfen auf negative Werte
            has_negative_values = (net_incomes <= 0).any()

            if has_negative_values:
                # Schritt 6a: Daten auflisten und Warnung ausgeben
                net_income_list = [
                    {"date": date.strftime('%Y-%m-%d'), "value": value}
                    for date, value in zip(net_incomes.index, net_incomes.values)
                ]
                message = (f"Das Unternehmen {symbol} weist durchgängig negative Nettogewinne auf: "
                           f"{', '.join(f'{d['value']:.0f}' for d in net_income_list)} Mio. USD. "
                           f"Eine Investition in diesen Wert ist mit hohem Risiko behaftet, da das Unternehmen Verluste macht.")
                return {
                    "net_incomes": net_income_list,
                    "symbol": symbol,
                    "actual_start_date": net_incomes.index[-1].strftime('%Y-%m-%d'),
                    "actual_end_date": net_incomes.index[0].strftime('%Y-%m-%d'),
                    "frequency": "quarterly",
                    "message": message
                }
            else:
                # Schritt 6b: Wachstumsraten zwischen aufeinanderfolgenden Quartalen berechnen (keine negativen Werte)
                growth_rates = []
                for i in range(len(net_incomes) - 1):
                    current_value = net_incomes.iloc[i]  # Aktueller Nettogewinn (neuere Daten)
                    next_value = net_incomes.iloc[i + 1]  # Nächster Nettogewinn (ältere Daten)
                    if current_value <= 0 or next_value <= 0:
                        return {
                            "error": f"Ungültige Daten für Wachstumsrate-Berechnung für {symbol} (Wert ≤ 0 erkannt)."
                        }
                    growth_rate = ((current_value / next_value) - 1) * 100  # Wachstumsrate in Prozent
                    growth_rates.append(growth_rate)

                # Schritt 7: Durchschnittliche Wachstumsrate berechnen
                avg_growth = sum(growth_rates) / len(growth_rates)
                avg_growth = round(avg_growth, 2)
                message = (f"Durchschnittliche quartalsweise Gewinnwachstumsrate (AQGR) für {symbol}: {avg_growth}% "
                           f"von {net_incomes.index[-1].strftime('%Y-%m-%d')} bis {net_incomes.index[0].strftime('%Y-%m-%d')} "
                           f"basierend auf {len(net_incomes)} Berichten.")

                # Schritt 8: Ergebnis zurückgeben
                return {
                    "avg_growth": avg_growth,
                    "symbol": symbol,
                    "actual_start_date": net_incomes.index[-1].strftime('%Y-%m-%d'),
                    "actual_end_date": net_incomes.index[0].strftime('%Y-%m-%d'),
                    "frequency": "quarterly",
                    "message": message
                }

        except Exception as e:
            return {"error": f"Fehler bei der Berechnung des quartalsweisen Gewinnwachstums für {symbol}: {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def calculate_avg_annual_profit_growth(self, symbol, start_date, end_date, use_cache=True):
        """
        Berechnet die durchschnittliche jährliche Gewinnwachstumsrate (AAGR) basierend auf allen verfügbaren
        jährlichen Nettogewinndaten, aber nur, wenn alle Werte positiv sind. Bei Vorhandensein eines negativen
        Werts werden die Daten aufgelistet und eine Warnung ausgegeben.

        Die Methode ignoriert die Parameter start_date und end_date, da sie immer alle verfügbaren Daten verwendet.
        Die Daten werden absteigend sortiert (neueste zuerst).

        Args:
            symbol (str): Das Aktien-Symbol (z. B. 'AAPL').
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (wird ignoriert, nur für Kompatibilität).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (wird ignoriert, nur für Kompatibilität).
            use_cache (bool): Verwendung des Caches (Standard: True).

        Returns:
            dict: Enthält die AAGR oder eine Liste der Nettogewinne mit Warnung, den analysierten Zeitraum
                  und eine Fehlermeldung, falls zutreffend.
                  Beispiel bei Erfolg mit AAGR:
                  {
                      "avg_growth": 10.0,
                      "symbol": "AAPL",
                      "actual_start_date": "2021-12-31",
                      "actual_end_date": "2024-12-31",
                      "frequency": "annual",
                      "message": "Durchschnittliche jährliche Gewinnwachstumsrate (AAGR) für AAPL: 10.0% von 2021-12-31 bis 2024-12-31 basierend auf 4 Berichten."
                  }
                  Beispiel bei negativen Werten:
                  {
                      "net_incomes": [{"date": "2024-12-31", "value": -500}, {"date": "2023-12-31", "value": -600}, ...],
                      "symbol": "LCID",
                      "actual_start_date": "2021-12-31",
                      "actual_end_date": "2024-12-31",
                      "frequency": "annual",
                      "message": "Das Unternehmen LCID weist negative Nettogewinne auf: -500, -600, ... Mio. USD. Eine Investition in diesen Wert ist mit hohem Risiko behaftet, da das Unternehmen Verluste macht. Ein Vergleich mit der Inflation ist nicht sinnvoll."
                  }
                  oder bei Fehler:
                  {
                      "error": "Fehlerbeschreibung"
                  }
        """
        try:
            # Schritt 1: Finanzdaten abrufen
            financials = self.dataloader.get_stock_financials(symbol, frequency="annual", use_cache=use_cache)
            if isinstance(financials, dict) and "error" in financials:
                return financials

            # Schritt 2: Prüfen, ob Nettogewinn-Daten vorhanden sind
            if "Net Income" not in financials.index:
                return {"error": f"Keine Nettogewinn-Daten für {symbol} verfügbar.", "symbol": symbol}

            # Schritt 3: Nettogewinndaten extrahieren, bereinigen und absteigend sortieren
            net_incomes = financials.loc["Net Income"].dropna()
            net_incomes.index = pd.to_datetime(net_incomes.index)
            net_incomes = net_incomes.sort_index(ascending=False)  # Neueste Daten zuerst

            # Schritt 4: Prüfen, ob genügend Datenpunkte vorhanden sind (mindestens 2 für Wachstumsrate)
            if len(net_incomes) < 2:
                return {"error": f"Nicht genügend Nettogewinndaten für {symbol} (mindestens 2 Berichte erforderlich)."}

            # Schritt 5: Prüfen auf negative Werte
            has_negative_values = (net_incomes <= 0).any()

            if has_negative_values:
                # Schritt 6a: Daten auflisten und Warnung ausgeben, wenn mindestens ein Wert negativ ist
                net_income_list = [
                    {"date": date.strftime('%Y-%m-%d'), "value": value / 1000000}  # Umrechnung in Mio. USD
                    for date, value in zip(net_incomes.index, net_incomes.values)
                ]
                message = (f"Das Unternehmen {symbol} weist negative Nettogewinne auf: "
                           f"{', '.join(f'{d['value']:.0f}' for d in net_income_list)} Mio. USD. "
                           f"Eine Investition in diesen Wert ist mit hohem Risiko behaftet, da das Unternehmen Verluste macht. "
                           f"Ein Vergleich mit der Inflation ist nicht sinnvoll.")
                return {
                    "net_incomes": net_income_list,
                    "symbol": symbol,
                    "actual_start_date": net_incomes.index[-1].strftime('%Y-%m-%d'),
                    "actual_end_date": net_incomes.index[0].strftime('%Y-%m-%d'),
                    "frequency": "annual",
                    "message": message
                }
            else:
                # Schritt 6b: Wachstumsraten zwischen aufeinanderfolgenden Jahren berechnen (alle Werte positiv)
                growth_rates = []
                for i in range(len(net_incomes) - 1):
                    current_value = net_incomes.iloc[i]  # Aktueller Nettogewinn (neuere Daten)
                    next_value = net_incomes.iloc[i + 1]  # Nächster Nettogewinn (ältere Daten)
                    growth_rate = ((current_value / next_value) - 1) * 100  # Wachstumsrate in Prozent
                    growth_rates.append(growth_rate)

                # Schritt 7: Durchschnittliche Wachstumsrate berechnen
                avg_growth = sum(growth_rates) / len(growth_rates)
                avg_growth = round(avg_growth, 2)

                # Schritt 8: Tatsächlichen Zeitraum bestimmen
                actual_start_date = net_incomes.index[-1].strftime('%Y-%m-%d')  # Älteste Datum
                actual_end_date = net_incomes.index[0].strftime('%Y-%m-%d')  # Neueste Datum

                # Schritt 9: Ergebnis zurückgeben
                return {
                    "avg_growth": avg_growth,
                    "symbol": symbol,
                    "actual_start_date": actual_start_date,
                    "actual_end_date": actual_end_date,
                    "frequency": "annual",
                    "message": (f"Durchschnittliche jährliche Gewinnwachstumsrate (AAGR) für {symbol}: {avg_growth}% "
                                f"von {actual_start_date} bis {actual_end_date} basierend auf {len(net_incomes)} Berichten.")
                }

        except Exception as e:
            return {"error": f"Fehler bei der Berechnung des jährlichen Gewinnwachstums für {symbol}: {str(e)}"}

    def compare_avg_quarterly_growth_to_inflation(self, symbol, start_date, end_date, use_cache=True):
        """
        Vergleicht das durchschnittliche quartalsweise Gewinnwachstum (AQGR) mit der kumulativen Inflation
        über denselben Zeitraum, sofern die AQGR berechnet wurde. Bei negativen Nettogewinnen wird keine
        Berechnung durchgeführt und eine Warnung zurückgegeben.

        Args:
            symbol (str): Das Aktien-Symbol (z. B. 'AAPL').
            start_date (str): Startdatum im Format 'YYYY-MM-DD'.
            end_date (str): Enddatum im Format 'YYYY-MM-DD'.
            use_cache (bool): Verwendung des Caches (Standard: True).

        Returns:
            dict: Enthält die AQGR, die kumulative Inflation, eine Bewertung und eine Nachricht bei positiven Gewinnen.
                  Bei negativen Gewinnen wird die Datenauflistung und Warnung zurückgegeben.
                  Beispiel bei Erfolg:
                  {
                      "symbol": "AAPL",
                      "aqgr": 5.0,
                      "total_inflation": 3.2,
                      "outperforms_inflation": True,
                      "message": "Das quartalsweise Gewinnwachstum von 5.0% übersteigt die Inflation von 3.2%."
                  }
                  Beispiel bei negativen Gewinnen:
                  {
                      "net_incomes": [{"date": "2025-03-31", "value": -300}, ...],
                      "symbol": "LCID",
                      "message": "Das Unternehmen LCID weist durchgängig negative Nettogewinne auf: ... Ein Vergleich mit der Inflation ist nicht sinnvoll."
                  }
                  Beispiel bei Fehler:
                  {
                      "error": "Fehlerbeschreibung"
                  }
        """
        try:
            # AQGR oder Datenauflistung abrufen
            growth_result = self.calculate_avg_quarterly_profit_growth(symbol, start_date, end_date, use_cache=use_cache)
            if "error" in growth_result:
                return growth_result

            # Prüfen, ob AQGR berechnet wurde (positive Gewinne) oder Datenauflistung (negative Gewinne)
            if "avg_growth" in growth_result:
                # Positive Gewinne: AQGR vorhanden, Inflation berechnen und vergleichen
                actual_start_date = growth_result["actual_start_date"]
                actual_end_date = growth_result["actual_end_date"]

                # Kumulative Inflation berechnen
                inflation_result = self.calculate_total_inflation_for_period(actual_start_date, actual_end_date,
                                                                             use_cache=use_cache)
                if "error" in inflation_result:
                    return inflation_result

                # AQGR und Inflation vergleichen
                aqgr = growth_result["avg_growth"]
                total_inflation = inflation_result["total_inflation"]
                outperforms_inflation = bool(aqgr > total_inflation)

                # Ergebnis zusammenstellen
                return {
                    "symbol": symbol,
                    "aqgr": aqgr,
                    "total_inflation": total_inflation,
                    "outperforms_inflation": outperforms_inflation,
                    "message": (f"Das quartalsweise Gewinnwachstum von {aqgr}% "
                                f"{'übersteigt' if outperforms_inflation else 'unterliegt'} "
                                f"der Inflation von {total_inflation}%.")
                }
            elif "net_incomes" in growth_result:
                # Negative Gewinne: Datenauflistung und Warnung weitergeben
                return growth_result

            else:
                return {"error": f"Unerwartetes Ergebnis von calculate_quarterly_profit_growth für {symbol}."}

        except Exception as e:
            return {
                "error": f"Fehler beim Vergleich des quartalsweisen Gewinnwachstums mit der Inflation für {symbol}: {str(e)}"}

    def compare_avg_annual_growth_to_inflation(self, symbol, start_date, end_date, use_cache=True):
        """
        Vergleicht die durchschnittliche jährliche Gewinnwachstumsrate (AAGR) mit der kumulativen Inflation
        über denselben Zeitraum, sofern die AAGR berechnet wurde. Bei negativen Nettogewinnen wird keine
        Berechnung durchgeführt und eine Warnung zurückgegeben.

        Args:
            symbol (str): Das Aktien-Symbol (z. B. 'AAPL').
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (wird ignoriert, nur für Kompatibilität).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (wird ignoriert, nur für Kompatibilität).
            use_cache (bool): Verwendung des Caches (Standard: True).

        Returns:
            dict: Enthält die AAGR, die kumulative Inflation, eine Bewertung und eine Nachricht bei positiven Gewinnen.
                  Bei negativen Gewinnen wird die Datenauflistung und Warnung zurückgegeben.
                  Beispiel bei Erfolg:
                  {
                      "symbol": "AAPL",
                      "aagr": 10.0,
                      "total_inflation": 3.2,
                      "outperforms_inflation": True,
                      "message": "Die jährliche Gewinnwachstumsrate von 10.0% übersteigt die Inflation von 3.2%."
                  }
                  Beispiel bei negativen Gewinnen:
                  {
                      "net_incomes": [{"date": "2024-12-31", "value": -500}, ...],
                      "symbol": "LCID",
                      "message": "Das Unternehmen LCID weist durchgängig negative Nettogewinne auf: ... Ein Vergleich mit der Inflation ist nicht sinnvoll."
                  }
                  Beispiel bei Fehler:
                  {
                      "error": "Fehlerbeschreibung"
                  }
        """
        try:
            # AAGR oder Datenauflistung abrufen
            growth_result = self.calculate_avg_annual_profit_growth(symbol, start_date, end_date, use_cache=use_cache)
            if "error" in growth_result:
                return growth_result

            # Prüfen, ob AAGR berechnet wurde (positive Gewinne) oder Datenauflistung (negative Gewinne)
            if "avg_growth" in growth_result:
                # Positive Gewinne: AAGR vorhanden, Inflation berechnen und vergleichen
                actual_start_date = growth_result["actual_start_date"]
                actual_end_date = growth_result["actual_end_date"]

                # Kumulative Inflation berechnen
                inflation_result = self.calculate_total_inflation_for_period(actual_start_date, actual_end_date,
                                                                             use_cache=use_cache)
                if "error" in inflation_result:
                    return inflation_result

                # AAGR und Inflation vergleichen
                aagr = growth_result["avg_growth"]
                total_inflation = inflation_result["total_inflation"]
                outperforms_inflation = bool(aagr > total_inflation)

                # Ergebnis zusammenstellen
                return {
                    "symbol": symbol,
                    "aagr": aagr,
                    "total_inflation": total_inflation,
                    "outperforms_inflation": outperforms_inflation,
                    "message": (f"Die jährliche Gewinnwachstumsrate von {aagr}% "
                                f"{'übersteigt' if outperforms_inflation else 'unterliegt'} "
                                f"der Inflation von {total_inflation}%.")
                }
            elif "net_incomes" in growth_result:
                # Negative Gewinne: Datenauflistung und Warnung weitergeben
                return growth_result
            else:
                return {"error": f"Unerwartetes Ergebnis von calculate_annual_profit_growth für {symbol}."}

        except Exception as e:
            return {
                "error": f"Fehler beim Vergleich des jährlichen Gewinnwachstums mit der Inflation für {symbol}: {str(e)}"}

    from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
    import pandas as pd

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def calculate_current_netCurrentAssets(self, symbol: str, frequency: str = "annual",
                                           use_cache: bool = True) -> dict:
        """
        Berechnet aktuelle Net Current Assets = Total Current Assets - Total Current Liabilities
        aus der Bilanz (yfinance balance_sheet).

        Returns:
            dict:
              {
                "net_current_assets": float,
                "current_assets": float,
                "current_liabilities": float,
                "labels_used": {"assets": str, "liabilities": str},
                "symbol": symbol,
                "frequency": frequency,
                "date": "<latest-period>"
              }
            oder {"error": "...", "symbol": symbol}
        """
        if frequency not in ["annual", "quarterly"]:
            return {"error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.", "symbol": symbol}

        cache_key = f"{symbol}_net_current_assets_{frequency}"
        if use_cache:
            cached = self.dataloader._load_cached_data(symbol, cache_key)
            if cached is not None and isinstance(cached, dict) and "error" not in cached:
                return cached

        try:
            bs = self.dataloader.get_balance_sheet(symbol, frequency=frequency, use_cache=use_cache)
            if isinstance(bs, dict) and "error" in bs:
                return {"error": bs["error"], "symbol": symbol}

            if bs is None or getattr(bs, "empty", True):
                return {"error": f"Keine Bilanzdaten ({frequency}) für {symbol} verfügbar.", "symbol": symbol}

            # yfinance-Labels variieren; daher tolerant suchen
            assets_labels = [
                "Total Current Assets",
                "Current Assets",
                "TotalCurrentAssets",
                "CurrentAssets",
            ]
            liab_labels = [
                "Total Current Liabilities",
                "Current Liabilities",
                "TotalCurrentLiabilities",
                "CurrentLiabilities",
            ]

            current_assets = None
            assets_label_used = None
            for lbl in assets_labels:
                if lbl in bs.index:
                    current_assets = bs.loc[lbl].iloc[0]
                    assets_label_used = lbl
                    break

            current_liabilities = None
            liab_label_used = None
            for lbl in liab_labels:
                if lbl in bs.index:
                    current_liabilities = bs.loc[lbl].iloc[0]
                    liab_label_used = lbl
                    break

            if current_assets is None or current_liabilities is None:
                return {
                    "error": (
                        f"Net Current Assets nicht berechenbar für {symbol} ({frequency}). "
                        f"Gefunden: assets_label={assets_label_used}, liab_label={liab_label_used}. "
                        f"Verfügbare Labels: {list(bs.index)}"
                    ),
                    "symbol": symbol
                }

            if pd.isna(current_assets) or pd.isna(current_liabilities):
                return {"error": f"Ungültige Werte für Umlaufdaten {symbol} ({frequency}).", "symbol": symbol}

            nca = float(current_assets) - float(current_liabilities)

            # Datum aus Spalten (neueste Periode)
            date_col = bs.columns[0] if len(bs.columns) > 0 else None
            date = str(date_col.date()) if hasattr(date_col, "date") else str(
                date_col) if date_col is not None else None

            out = {
                "net_current_assets": float(nca),
                "current_assets": float(current_assets),
                "current_liabilities": float(current_liabilities),
                "labels_used": {"assets": assets_label_used, "liabilities": liab_label_used},
                "symbol": symbol,
                "frequency": frequency,
                "date": date
            }

            if use_cache:
                self.dataloader._cache_data(out, symbol, cache_key)

            return out

        except Exception as e:
            return {"error": f"Fehler bei NetCurrentAssets für {symbol} ({frequency}): {str(e)}", "symbol": symbol}

    def calculate_historical_market_cap(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Berechnet die historische Marktkapitalisierung für ein Aktiensymbol basierend auf den exakten fiscalDateEnding-Daten.

        Args:
            symbol (str): Aktiensymbol (z. B. 'ILMN').
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (optional).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (optional).
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (Default: True).

        Returns:
            Optional[pd.DataFrame]: DataFrame mit Spalten 'MarketCap', 'commonStockSharesOutstanding', 'Close', indiziert nach fiscalDateEnding, oder None bei Fehler.
        """
        cache_key = f"historical_market_cap_{symbol}"

        # 1. Cache prüfen
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache-Daten für Marktkapitalisierung von {symbol} geladen.")
                if isinstance(cached_data, dict):
                    df = pd.DataFrame({
                        "MarketCap": cached_data["MarketCap"],
                        "commonStockSharesOutstanding": cached_data["commonStockSharesOutstanding"],
                        "Close": cached_data["Close"]
                    }, index=cached_data["index"])
                    df.index = pd.to_datetime(df.index)
                    return df
                return cached_data

        try:
            # 2. Fundamentaldaten (für commonStockSharesOutstanding und fiscalDateEnding) abrufen
            fundamentals = self.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            if "error" in fundamentals:
                self.logger.error(f"Fehler bei Fundamentaldaten für {symbol}: {fundamentals['error']}")
                return None

            balance_sheet = fundamentals.get("balance_sheet")
            if balance_sheet is None or "commonStockSharesOutstanding" not in balance_sheet.columns:
                self.logger.error(f"Keine Balance Sheet-Daten oder commonStockSharesOutstanding für {symbol} verfügbar.")
                return None

            # Zeitraum einschränken
            df_shares = balance_sheet[["commonStockSharesOutstanding"]].copy()
            if start_date:
                df_shares = df_shares[df_shares.index >= pd.to_datetime(start_date)]
            if end_date:
                df_shares = df_shares[df_shares.index <= pd.to_datetime(end_date)]
            if df_shares.empty:
                self.logger.error(f"Keine commonStockSharesOutstanding-Daten für {symbol} im angegebenen Zeitraum.")
                return None

            # 3. Historische Kursdaten abrufen
            stock_data = self.dataloader.get_max_historical_stock_data(symbol, use_cache=True)
            if stock_data is None or "Close" not in stock_data.columns:
                self.logger.error(f"Keine historischen Kursdaten oder Close-Spalte für {symbol} verfügbar.")
                return None

            # 4. Kursdaten für exakte fiscalDateEnding-Daten auswählen
            df_prices = stock_data[["Close"]].reindex(df_shares.index, method="ffill")
            if df_prices.isna().all().any():
                self.logger.error(f"Keine gültigen Kursdaten für fiscalDateEnding-Daten von {symbol}.")
                return None

            # 5. Marktkapitalisierung berechnen
            df = df_shares.join(df_prices, how="inner")
            df["MarketCap"] = df["commonStockSharesOutstanding"] * df["Close"]
            df = df[["MarketCap", "commonStockSharesOutstanding", "Close"]].dropna()

            if df.empty:
                self.logger.error(f"Keine gültigen Marktkapitalisierungs-Daten für {symbol} nach Berechnung.")
                return None

            # 6. Cache als Dictionary speichern (Index als String)
            cache_data = {
                "MarketCap": {date.strftime("%Y-%m-%d"): value for date, value in df["MarketCap"].items()},
                "commonStockSharesOutstanding": {date.strftime("%Y-%m-%d"): value for date, value in df["commonStockSharesOutstanding"].items()},
                "Close": {date.strftime("%Y-%m-%d"): value for date, value in df["Close"].items()},
                "index": df.index.strftime("%Y-%m-%d").tolist()
            }
            self.dataloader._cache_data(cache_data, symbol, cache_key)
            self.logger.info(f"Marktkapitalisierungs-Daten für {symbol} erfolgreich berechnet und gecacht.")
            return df

        except Exception as e:
            self.logger.error(f"Fehler bei Marktkapitalisierungs-Berechnung für {symbol}: {e}")
            return None

    def calculate_historical_ev(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Berechnet den historischen Enterprise Value (EV) für ein Aktiensymbol.

        Args:
            symbol (str): Aktiensymbol (z. B. 'ILMN').
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (optional).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (optional).
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (default: True).

        Returns:
            Optional[pd.DataFrame]: DataFrame mit EV-Werten, indiziert nach Datum, oder None bei Fehler.
        """
        cache_key = f"historical_ev_{symbol}"

        # 1. Cache prüfen
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache-Daten für EV von {symbol} geladen.")
                if isinstance(cached_data, dict) and "EV" in cached_data:
                    s = pd.Series(cached_data["EV"], dtype="float64")
                    s.index = pd.to_datetime(s.index)
                    s = s.sort_index()
                    return s.to_frame(name="EV")
                return cached_data

        try:
            # 2. Fundamentaldaten abrufen
            fundamentals = self.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            if "error" in fundamentals:
                self.logger.error(f"Fehler bei Fundamentaldaten für {symbol}: {fundamentals['error']}")
                return None

            balance_sheet = fundamentals.get("balance_sheet")
            if balance_sheet is None:
                self.logger.error(f"Keine Balance Sheet-Daten für {symbol} verfügbar.")
                return None

            # Spaltenprüfung für balance_sheet
            required_columns = {"totalLiabilities", "cashAndCashEquivalentsAtCarryingValue"}
            missing_columns = required_columns - set(balance_sheet.columns)
            if missing_columns:
                self.logger.error(f"Fehlende Spalten für {symbol} in balance_sheet: {missing_columns}")
                return None

            # 3. Marktkapitalisierung abrufen
            market_cap_df = self.calculate_historical_market_cap(symbol, start_date, end_date, use_cache=True)
            if market_cap_df is None or market_cap_df.empty:
                self.logger.error(f"Keine Marktkapitalisierungs-Daten für {symbol} verfügbar.")
                return None

            # 4. Daten zusammenführen
            df = balance_sheet[list(required_columns)].copy()
            df = df.join(market_cap_df[["MarketCap"]], how="inner")  # Nur gemeinsame Zeitpunkte
            if df.empty:
                self.logger.error(f"Keine übereinstimmenden Daten für {symbol} nach Join.")
                return None

            # Zeitraum einschränken
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            if df.empty:
                self.logger.error(f"Keine Daten für {symbol} im angegebenen Zeitraum.")
                return None

            # 5. EV berechnen
            df["EV"] = df["MarketCap"] + df["totalLiabilities"] - df["cashAndCashEquivalentsAtCarryingValue"]
            df = df[["EV"]].dropna()

            if df.empty:
                self.logger.error(f"Keine gültigen EV-Daten für {symbol} nach Berechnung.")
                return None

            # 6. Cache als Dictionary speichern (Index als String)
            cache_data = {
                "EV": {date.strftime("%Y-%m-%d"): value for date, value in df["EV"].items()},
                "index": df.index.strftime("%Y-%m-%d").tolist()
            }
            self.dataloader._cache_data(cache_data, symbol, cache_key)
            self.logger.info(f"EV-Daten für {symbol} erfolgreich berechnet und gecacht.")
            return df

        except Exception as e:
            self.logger.error(f"Fehler bei EV-Berechnung für {symbol}: {e}")
            return None


    def calculate_historical_sales(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None,
                                  use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Berechnet die historischen Umsatzerlöse (Sales) für ein Aktiensymbol.

        Args:
            symbol (str): Aktiensymbol (z. B. 'ILMN').
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (optional).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (optional).
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (default: True).

        Returns:
            Optional[pd.DataFrame]: DataFrame mit Umsatzerlösen (Spalte 'Sales'), indiziert nach Datum, oder None bei Fehler.
        """
        cache_key = f"historical_sales_{symbol}"

        # 1. Cache prüfen
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache-Daten für Umsatzerlöse von {symbol} geladen.")
                if isinstance(cached_data, dict):
                    try:
                        df = pd.DataFrame({
                            "Sales": list(cached_data["Sales"].values())
                        }, index=pd.to_datetime(cached_data["index"]))
                        self.logger.info(f"Rekonstruiertes sales_data für {symbol}: Spalten={df.columns.tolist()}, Index={df.index[:5].tolist()}")
                        return df
                    except Exception as e:
                        self.logger.error(f"Fehler beim Rekonstruieren des Cache für {symbol}: {e}")
                        return None
                return cached_data

        try:
            # 2. Fundamentaldaten abrufen
            fundamentals = self.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            if "error" in fundamentals:
                self.logger.error(f"Fehler bei Fundamentaldaten für {symbol}: {fundamentals['error']}")
                return None

            income_statement = fundamentals.get("income_statement")
            if income_statement is None:
                self.logger.error(f"Keine Income Statement-Daten für {symbol} verfügbar.")
                return None

            self.logger.info(f"income_statement-Spalten für {symbol}: {income_statement.columns.tolist()}")
            self.logger.info(f"income_statement-Daten (erste Zeilen):\n{income_statement.head()}")

            # Spaltenprüfung
            required_columns = {"totalRevenue"}
            missing_columns = required_columns - set(income_statement.columns)
            if missing_columns:
                self.logger.error(f"Fehlende Spalten für {symbol} in income_statement: {missing_columns}")
                return None

            # Zeitraum einschränken
            df = income_statement[["totalRevenue"]].copy()
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            if df.empty:
                self.logger.error(f"Keine Income Statement-Daten für {symbol} im angegebenen Zeitraum.")
                return None

            # 3. Sales-Spalte umbenennen
            df = df.rename(columns={"totalRevenue": "Sales"}).dropna()

            if df.empty:
                self.logger.error(f"Keine gültigen Umsatzerlöse für {symbol} nach Abruf.")
                return None

            # 4. Cache als Dictionary speichern
            cache_data = {
                "Sales": {date.strftime("%Y-%m-%d"): value for date, value in df["Sales"].items()},
                "index": df.index.strftime("%Y-%m-%d").tolist()
            }
            self.dataloader._cache_data(cache_data, symbol, cache_key)
            self.logger.info(f"Umsatzerlöse für {symbol} erfolgreich abgerufen und gecacht.")
            return df

        except Exception as e:
            self.logger.error(f"Fehler bei Abruf der Umsatzerlöse für {symbol}: {e}")
            return None

    def calculate_historical_ev_sales(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Berechnet das historische EV/Sales-Multiple für ein Aktiensymbol.

        Args:
            symbol (str): Aktiensymbol (z. B. 'ILMN').
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (optional).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (optional).
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (default: True).

        Returns:
            Optional[pd.DataFrame]: DataFrame mit EV/Sales-Multiples, indiziert nach Datum, oder None bei Fehler.
        """
        cache_key = f"historical_ev_sales_{symbol}"

        # 1. Cache prüfen
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache-Daten für EV/Sales von {symbol} geladen.")
                if isinstance(cached_data, dict) and "EV_Sales" in cached_data:
                    s = pd.Series(cached_data["EV_Sales"], dtype="float64")
                    s.index = pd.to_datetime(s.index)
                    s = s.sort_index()
                    df = s.to_frame(name="EV_Sales")
                    return df
                return cached_data

        try:
            # 2. EV-Daten abrufen
            ev_df = self.calculate_historical_ev(symbol, start_date, end_date, use_cache=True)
            if ev_df is None or ev_df.empty:
                self.logger.error(f"Keine EV-Daten für {symbol} verfügbar.")
                return None

            # 3. Sales-Daten abrufen
            sales_df = self.calculate_historical_sales(symbol, start_date, end_date, use_cache=True)
            if sales_df is None or sales_df.empty:
                self.logger.error(f"Keine Sales-Daten für {symbol} verfügbar.")
                return None

            # 4. Daten zusammenführen
            df = ev_df.join(sales_df, how="inner")
            if df.empty:
                self.logger.error(f"Keine übereinstimmenden Daten für {symbol} nach Join.")
                return None

            # 5. EV/Sales berechnen
            df["EV_Sales"] = df["EV"] / df["Sales"]
            df = df[["EV_Sales"]].dropna()

            if df.empty:
                self.logger.error(f"Keine gültigen EV/Sales-Daten für {symbol} nach Berechnung.")
                return None

            # Division durch Null prüfen
            if np.isinf(df["EV_Sales"]).any():
                self.logger.warning(f"Ungültige EV/Sales-Werte (unendlich) für {symbol} erkannt.")
                df = df[~np.isinf(df["EV_Sales"])]

            if df.empty:
                self.logger.error(f"Keine gültigen EV/Sales-Daten für {symbol} nach Entfernung ungültiger Werte.")
                return None

            # 6. Cache als Dictionary speichern (Index als String)
            cache_data = {
                "EV_Sales": {date.strftime("%Y-%m-%d"): value for date, value in df["EV_Sales"].items()},
                "index": df.index.strftime("%Y-%m-%d").tolist()
            }
            self.dataloader._cache_data(cache_data, symbol, cache_key)
            self.logger.info(f"EV/Sales-Daten für {symbol} erfolgreich berechnet und gecacht.")
            return df

        except Exception as e:
            self.logger.error(f"Fehler bei Berechnung des EV/Sales-Multiples für {symbol}: {e}")
            return None


    def calculate_historical_ebit(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None,
                                  use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Berechnet die historischen EBIT-Werte (Earnings Before Interest and Taxes) für ein Aktiensymbol.

        Args:
            symbol (str): Aktiensymbol (z. B. 'ILMN').
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (optional).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (optional).
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (default: True).

        Returns:
            Optional[pd.DataFrame]: DataFrame mit EBIT-Werten, indiziert nach Datum, oder None bei Fehler.
        """
        cache_key = f"historical_ebit_{symbol}"

        # 1. Cache prüfen
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache-Daten für EBIT von {symbol} geladen.")
                if isinstance(cached_data, dict):
                    try:
                        # Korrekte Rekonstruktion des DataFrames
                        df = pd.DataFrame({
                            "EBIT": list(cached_data["EBIT"].values())
                        }, index=pd.to_datetime(cached_data["index"]))
                        self.logger.info(
                            f"Rekonstruiertes ebit_data für {symbol}: Spalten={df.columns.tolist()}, Index={df.index[:5].tolist()}")
                        return df
                    except Exception as e:
                        self.logger.error(f"Fehler beim Rekonstruieren des Cache für {symbol}: {e}")
                        return None
                return cached_data

        try:
            # 2. Fundamentaldaten abrufen
            fundamentals = self.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            if "error" in fundamentals:
                self.logger.error(f"Fehler bei Fundamentaldaten für {symbol}: {fundamentals['error']}")
                return None

            income_statement = fundamentals.get("income_statement")
            if income_statement is None:
                self.logger.error(f"Keine Income Statement-Daten für {symbol} verfügbar.")
                return None

            # Spaltenprüfung
            required_columns = {"operatingIncome"}
            missing_columns = required_columns - set(income_statement.columns)
            if missing_columns:
                self.logger.error(f"Fehlende Spalten für {symbol} in income_statement: {missing_columns}")
                return None

            # Zeitraum einschränken
            df = income_statement[["operatingIncome"]].copy()
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            if df.empty:
                self.logger.error(f"Keine Income Statement-Daten für {symbol} im angegebenen Zeitraum.")
                return None

            # 3. EBIT-Spalte umbenennen
            df = df.rename(columns={"operatingIncome": "EBIT"}).dropna()

            if df.empty:
                self.logger.error(f"Keine gültigen EBIT-Werte für {symbol} nach Abruf.")
                return None

            # 4. Cache als Dictionary speichern (Index als String)
            cache_data = {
                "EBIT": {date.strftime("%Y-%m-%d"): value for date, value in df["EBIT"].items()},
                "index": df.index.strftime("%Y-%m-%d").tolist()
            }
            self.dataloader._cache_data(cache_data, symbol, cache_key)
            self.logger.info(f"EBIT-Werte für {symbol} erfolgreich abgerufen und gecacht.")
            return df

        except Exception as e:
            self.logger.error(f"Fehler bei Abruf der EBIT-Werte für {symbol}: {e}")
            return None

    def calculate_historical_ev_to_ebit(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Berechnet das historische EV/EBIT-Multiple für ein Aktiensymbol.

        Args:
            symbol (str): Aktiensymbol (z. B. 'ILMN').
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (optional).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (optional).
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (default: True).

        Returns:
            Optional[pd.DataFrame]: DataFrame mit EV/EBIT-Multiples, indiziert nach Datum, oder None bei Fehler.
        """
        cache_key = f"historical_ev_to_ebit_{symbol}"

        # 1. Cache prüfen
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache-Daten für EV/EBIT von {symbol} geladen.")
                if isinstance(cached_data, dict):
                    # Korrigierte DataFrame-Konstruktion
                    df = pd.DataFrame({
                        "EV_EBIT": list(cached_data["EV_EBIT"].values())
                    }, index=pd.to_datetime(cached_data["index"]))
                    return df
                return cached_data

        try:
            # 2. EV-Daten abrufen
            ev_df = self.calculate_historical_ev(symbol, start_date, end_date, use_cache=True)
            if ev_df is None or ev_df.empty:
                self.logger.error(f"Keine EV-Daten für {symbol} verfügbar.")
                return None

            # 3. EBIT-Daten abrufen
            ebit_df = self.calculate_historical_ebit(symbol, start_date, end_date, use_cache=True)
            if ebit_df is None or ebit_df.empty:
                self.logger.error(f"Keine EBIT-Daten für {symbol} verfügbar.")
                return None

            # 4. Daten zusammenführen
            df = ev_df.join(ebit_df, how="inner")
            if df.empty:
                self.logger.error(f"Keine übereinstimmenden Daten für {symbol} nach Join.")
                return None

            # 5. EV/EBIT berechnen
            df["EV_EBIT"] = df["EV"] / df["EBIT"]
            df = df[["EV_EBIT"]].dropna()

            if df.empty:
                self.logger.error(f"Keine gültigen EV/EBIT-Daten für {symbol} nach Berechnung.")
                return None

            # Division durch Null und ungültige Werte prüfen
            if np.isinf(df["EV_EBIT"]).any():
                self.logger.warning(f"Ungültige EV/EBIT-Werte (unendlich) für {symbol} erkannt.")
                df = df[~np.isinf(df["EV_EBIT"])]

            # Negative EBIT-Werte können zu negativen Multiples führen
            if (df["EV_EBIT"] < 0).any():
                self.logger.warning(f"Negative EV/EBIT-Werte für {symbol} erkannt, möglicherweise aufgrund negativer EBIT-Werte.")

            if df.empty:
                self.logger.error(f"Keine gültigen EV/EBIT-Daten für {symbol} nach Entfernung ungültiger Werte.")
                return None

            # 6. Cache als Dictionary speichern (Index als String)
            cache_data = {
                "EV_EBIT": {date.strftime("%Y-%m-%d"): value for date, value in df["EV_EBIT"].items()},
                "index": df.index.strftime("%Y-%m-%d").tolist()
            }
            self.dataloader._cache_data(cache_data, symbol, cache_key)
            self.logger.info(f"EV/EBIT-Daten für {symbol} erfolgreich berechnet und gecacht.")
            return df

        except Exception as e:
            self.logger.error(f"Fehler bei Berechnung des EV/EBIT-Multiples für {symbol}: {e}")
            return None

    def calculate_historical_ebitda(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None,
                                    use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Berechnet die historischen EBITDA-Werte (Earnings Before Interest, Taxes, Depreciation, and Amortization) für ein Aktiensymbol.

        Args:
            symbol (str): Aktiensymbol (z. B. 'ILMN').
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (optional).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (optional).
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (default: True).

        Returns:
            Optional[pd.DataFrame]: DataFrame mit Spalten 'EBIT', 'depreciationAndAmortization' und 'EBITDA', indiziert nach Datum, oder None bei Fehler.
        """
        cache_key = f"historical_ebitda_{symbol}"

        # 1. Cache prüfen
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache-Daten für EBITDA von {symbol} geladen.")
                if isinstance(cached_data, dict):
                    df = pd.DataFrame({
                        "EBIT": list(cached_data["EBIT"].values()),
                        "depreciationAndAmortization": list(cached_data["depreciationAndAmortization"].values()),
                        "EBITDA": list(cached_data["EBITDA"].values())
                    }, index=pd.to_datetime(cached_data["index"]))
                    return df
                return cached_data

        try:
            # 2. Fundamentaldaten abrufen
            fundamentals = self.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            if "error" in fundamentals:
                self.logger.error(f"Fehler bei Fundamentaldaten für {symbol}: {fundamentals['error']}")
                return None

            income_statement = fundamentals.get("income_statement")
            if income_statement is None:
                self.logger.error(f"Keine Income Statement-Daten für {symbol} verfügbar.")
                return None

            # Spaltenprüfung
            required_columns = {"operatingIncome", "depreciationAndAmortization"}
            missing_columns = required_columns - set(income_statement.columns)
            if missing_columns:
                self.logger.error(f"Fehlende Spalten für {symbol} in income_statement: {missing_columns}")
                return None

            # 3. EBITDA berechnen
            df = income_statement[["operatingIncome", "depreciationAndAmortization"]].copy()
            df["EBITDA"] = df["operatingIncome"] + df["depreciationAndAmortization"]
            df = df.rename(columns={"operatingIncome": "EBIT"})[
                ["EBIT", "depreciationAndAmortization", "EBITDA"]].dropna()

            if df.empty:
                self.logger.error(f"Keine gültigen EBITDA-Werte für {symbol} nach Berechnung.")
                return None

            # Zeitraum einschränken
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            if df.empty:
                self.logger.error(f"Keine EBITDA-Daten für {symbol} im angegebenen Zeitraum.")
                return None

            # 4. Cache als Dictionary speichern (Index als String)
            cache_data = {
                "EBIT": {date.strftime("%Y-%m-%d"): value for date, value in df["EBIT"].items()},
                "depreciationAndAmortization": {date.strftime("%Y-%m-%d"): value for date, value in
                                                df["depreciationAndAmortization"].items()},
                "EBITDA": {date.strftime("%Y-%m-%d"): value for date, value in df["EBITDA"].items()},
                "index": df.index.strftime("%Y-%m-%d").tolist()
            }
            self.dataloader._cache_data(cache_data, symbol, cache_key)
            self.logger.info(f"EBITDA-Werte für {symbol} erfolgreich berechnet und gecacht.")
            return df

        except Exception as e:
            self.logger.error(f"Fehler bei Abruf der EBITDA-Werte für {symbol}: {e}")
            return None

    def calculate_historical_ev_to_ebitda(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Berechnet das historische EV/EBITDA-Multiple für ein Aktiensymbol.

        Args:
            symbol (str): Aktiensymbol (z. B. 'ILMN').
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (optional).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (optional).
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (default: True).

        Returns:
            Optional[pd.DataFrame]: DataFrame mit EV/EBITDA-Multiples, indiziert nach Datum, oder None bei Fehler.
        """
        cache_key = f"historical_ev_to_ebitda_{symbol}"

        # 1. Cache prüfen
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache-Daten für EV/EBITDA von {symbol} geladen.")
                if isinstance(cached_data, dict):
                    df = pd.DataFrame({
                        "EV_EBITDA": list(cached_data["EV_EBITDA"].values())
                    }, index=pd.to_datetime(cached_data["index"]))
                    return df
                return cached_data

        try:
            # 2. EV-Daten abrufen
            ev_df = self.calculate_historical_ev(symbol, start_date, end_date, use_cache=True)
            if ev_df is None or ev_df.empty:
                self.logger.error(f"Keine EV-Daten für {symbol} verfügbar.")
                return None

            # 3. EBITDA-Daten abrufen
            ebitda_df = self.calculate_historical_ebitda(symbol, start_date, end_date, use_cache=True)
            if ebitda_df is None or ebitda_df.empty:
                self.logger.error(f"Keine EBITDA-Daten für {symbol} verfügbar.")
                return None

            # 4. Daten zusammenführen
            df = ev_df.join(ebitda_df[["EBITDA"]], how="inner")
            if df.empty:
                self.logger.error(f"Keine übereinstimmenden Daten für {symbol} nach Join.")
                return None

            # 5. EV/EBITDA berechnen
            df["EV_EBITDA"] = df["EV"] / df["EBITDA"]
            df = df[["EV_EBITDA"]].dropna()

            if df.empty:
                self.logger.error(f"Keine gültigen EV/EBITDA-Daten für {symbol} nach Berechnung.")
                return None

            # Division durch Null und ungültige Werte prüfen
            if np.isinf(df["EV_EBITDA"]).any():
                self.logger.warning(f"Ungültige EV/EBITDA-Werte (unendlich) für {symbol} erkannt.")
                df = df[~np.isinf(df["EV_EBITDA"])]

            # Negative EBITDA-Werte können zu negativen Multiples führen
            if (df["EV_EBITDA"] < 0).any():
                self.logger.warning(
                    f"Negative EV/EBITDA-Werte für {symbol} erkannt, möglicherweise aufgrund negativer EBITDA-Werte.")

            # Zeitraum einschränken
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            if df.empty:
                self.logger.error(f"Keine EV/EBITDA-Daten für {symbol} im angegebenen Zeitraum.")
                return None

            # 6. Cache als Dictionary speichern (Index als String)
            cache_data = {
                "EV_EBITDA": {date.strftime("%Y-%m-%d"): value for date, value in df["EV_EBITDA"].items()},
                "index": df.index.strftime("%Y-%m-%d").tolist()
            }
            self.dataloader._cache_data(cache_data, symbol, cache_key)
            self.logger.info(f"EV/EBITDA-Daten für {symbol} erfolgreich berechnet und gecacht.")
            return df

        except Exception as e:
            self.logger.error(f"Fehler bei Berechnung des EV/EBITDA-Multiples für {symbol}: {e}")
            return None

    def calculate_historical_price_to_book(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Berechnet das historische Price/Book-Multiple für ein Aktiensymbol, wobei der Aktienkurs und die Anzahl der ausstehenden Aktien exakt auf die Tage der Bilanzdaten abgestimmt werden.
        Gibt zusätzlich Price, totalAssets, totalLiabilities und commonStockSharesOutstanding zurück.

        Args:
            symbol (str): Aktiensymbol (z. B. 'ILMN').
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (optional).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (optional).
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (default: True).

        Returns:
            Optional[pd.DataFrame]: DataFrame mit Price/Book-Multiple, Price, totalAssets, totalLiabilities und commonStockSharesOutstanding,
                                    indiziert nach fiscalDateEnding, oder None bei Fehler.
        """
        cache_key = f"historical_price_to_book_{symbol}"

        # 1. Cache prüfen
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache-Daten für Price/Book von {symbol} geladen.")
                if isinstance(cached_data, dict):
                    df = pd.DataFrame({
                        "Price_Book": list(cached_data["Price_Book"].values()),
                        "Price": list(cached_data["Price"].values()),
                        "totalAssets": list(cached_data["totalAssets"].values()),
                        "totalLiabilities": list(cached_data["totalLiabilities"].values()),
                        "commonStockSharesOutstanding": list(cached_data["commonStockSharesOutstanding"].values())
                    }, index=pd.to_datetime(cached_data["index"]))
                    return df
                return cached_data

        try:
            # 2. Fundamentaldaten abrufen (Balance Sheet)
            fundamentals = self.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            if "error" in fundamentals:
                self.logger.error(f"Fehler bei Fundamentaldaten für {symbol}: {fundamentals['error']}")
                return None

            balance_sheet = fundamentals.get("balance_sheet")
            if balance_sheet is None:
                self.logger.error(f"Keine Balance Sheet-Daten für {symbol} verfügbar.")
                return None

            # Benötigte Spalten prüfen
            required_columns = {"totalAssets", "totalLiabilities", "commonStockSharesOutstanding"}
            missing_columns = required_columns - set(balance_sheet.columns)
            if missing_columns:
                self.logger.error(f"Fehlende Spalten in balance_sheet für {symbol}: {missing_columns}")
                return None

            # 3. Aktienkursdaten abrufen (täglich, um exakte Tage zu treffen)
            price_data = self.dataloader.get_max_historical_stock_data(symbol, start_date=start_date, end_date=end_date, interval="1d", use_cache=True)
            if price_data is None or price_data.empty:
                self.logger.error(f"Keine Aktienkursdaten für {symbol} verfügbar.")
                return None

            # 4. Book Value berechnen
            df = balance_sheet[["totalAssets", "totalLiabilities", "commonStockSharesOutstanding"]].copy()
            df["Book_Value"] = df["totalAssets"] - df["totalLiabilities"]
            if df["Book_Value"].isna().all():
                self.logger.error(f"Keine gültigen Book Value-Daten für {symbol}.")
                return None

            if df["commonStockSharesOutstanding"].isna().all() or (df["commonStockSharesOutstanding"] <= 0).any():
                self.logger.error(f"Ungültige oder fehlende Shares Outstanding-Daten für {symbol}.")
                return None

            # 5. Aktienkurs auf Bilanzdaten joinen
            price_data.index = pd.to_datetime(price_data.index)
            price_data = price_data[["Close"]].reindex(df.index, method="ffill")
            if price_data.isna().all().any():
                self.logger.error(f"Keine passenden Aktienkursdaten für die Bilanzdaten von {symbol}.")
                return None
            df["Price"] = price_data["Close"]

            # 6. Price/Book berechnen
            df["Price_Book"] = df["Price"] / (df["Book_Value"] / df["commonStockSharesOutstanding"])
            df = df[["Price_Book", "Price", "totalAssets", "totalLiabilities", "commonStockSharesOutstanding"]].replace([np.inf, -np.inf], np.nan).dropna()

            if df.empty:
                self.logger.error(f"Keine gültigen Price/Book-Daten für {symbol} nach Berechnung.")
                return None

            # Warnung für negative Multiples
            if (df["Price_Book"] < 0).any():
                self.logger.warning(f"Negative Price/Book-Werte für {symbol} erkannt, möglicherweise aufgrund negativer Book Values.")

            # Zeitraum einschränken
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            if df.empty:
                self.logger.error(f"Keine Price/Book-Daten für {symbol} im angegebenen Zeitraum.")
                return None

            # 7. Cache als Dictionary speichern
            cache_data = {
                "Price_Book": {date.strftime("%Y-%m-%d"): value for date, value in df["Price_Book"].items()},
                "Price": {date.strftime("%Y-%m-%d"): value for date, value in df["Price"].items()},
                "totalAssets": {date.strftime("%Y-%m-%d"): value for date, value in df["totalAssets"].items()},
                "totalLiabilities": {date.strftime("%Y-%m-%d"): value for date, value in df["totalLiabilities"].items()},
                "commonStockSharesOutstanding": {date.strftime("%Y-%m-%d"): value for date, value in df["commonStockSharesOutstanding"].items()},
                "index": df.index.strftime("%Y-%m-%d").tolist()
            }
            self.dataloader._cache_data(cache_data, symbol, cache_key)
            self.logger.info(f"Price/Book-Daten für {symbol} erfolgreich berechnet und gecacht.")
            return df

        except Exception as e:
            self.logger.error(f"Fehler bei Berechnung des Price/Book-Multiples für {symbol}: {e}")
            return None

    def calculate_historical_price_to_sales(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Berechnet das historische Price-to-Sales-Multiple für ein Aktiensymbol.
        Gibt Price_Sales, Price, Sales und commonStockSharesOutstanding zurück.

        Args:
            symbol (str): Aktiensymbol (z. B. 'ILMN').
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (optional).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (optional).
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (default: True).

        Returns:
            Optional[pd.DataFrame]: DataFrame mit Price/Sales-Multiple, Price, Sales und commonStockSharesOutstanding,
                                    indiziert nach fiscalDateEnding, oder None bei Fehler.
        """
        cache_key = f"historical_price_to_sales_{symbol}"

        # 1. Cache prüfen
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache-Daten für Price/Sales von {symbol} geladen.")
                if isinstance(cached_data, dict):
                    try:
                        df = pd.DataFrame({
                            "Price_Sales": list(cached_data["Price_Sales"].values()),
                            "Price": list(cached_data["Price"].values()),
                            "Sales": list(cached_data["Sales"].values()),
                            "commonStockSharesOutstanding": list(cached_data["commonStockSharesOutstanding"].values())
                        }, index=pd.to_datetime(cached_data["index"]))
                        self.logger.info(f"Rekonstruiertes price_to_sales für {symbol}: Spalten={df.columns.tolist()}, Index={df.index[:5].tolist()}")
                        return df
                    except Exception as e:
                        self.logger.error(f"Fehler beim Rekonstruieren des Cache für {symbol}: {e}")
                        return None
                return cached_data

        try:
            # 2. Umsatzdaten abrufen
            sales_data = self.calculate_historical_sales(symbol, start_date=start_date, end_date=end_date, use_cache=use_cache)
            if sales_data is None:
                self.logger.error(f"Keine Umsatzdaten für {symbol} verfügbar.")
                return None

            self.logger.info(f"sales_data-Spalten für {symbol}: {sales_data.columns.tolist()}")
            self.logger.info(f"sales_data-Daten (erste Zeilen):\n{sales_data.head()}")

            # Benötigte Spalten prüfen
            if "Sales" not in sales_data.columns:
                self.logger.error(f"Spalte 'Sales' fehlt in sales_data für {symbol}. Gefundene Spalten: {sales_data.columns.tolist()}")
                return None

            # 3. Startdatum anpassen, falls nicht angegeben
            if start_date is None:
                start_date = sales_data.index.min().strftime("%Y-%m-%d")
                self.logger.info(f"Kein start_date angegeben, verwende {start_date} basierend auf sales_data.")

            # 4. Fundamentaldaten für commonStockSharesOutstanding abrufen
            fundamentals = self.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=use_cache)
            if "error" in fundamentals:
                self.logger.error(f"Fehler bei Fundamentaldaten für {symbol}: {fundamentals['error']}")
                return None

            balance_sheet = fundamentals.get("balance_sheet")
            if balance_sheet is None:
                self.logger.error(f"Keine Balance Sheet-Daten für {symbol} verfügbar.")
                return None

            # Benötigte Spalten prüfen
            required_columns = {"commonStockSharesOutstanding"}
            missing_columns = required_columns - set(balance_sheet.columns)
            if missing_columns:
                self.logger.error(f"Fehlende Spalten in balance_sheet für {symbol}: {missing_columns}")
                return None

            # 5. Aktienkursdaten abrufen (täglich, um exakte Tage zu treffen)
            price_data = self.dataloader.get_max_historical_stock_data(symbol, start_date=start_date, end_date=end_date, interval="1d", use_cache=use_cache)
            if price_data is None or price_data.empty:
                self.logger.error(f"Keine Aktienkursdaten für {symbol} verfügbar.")
                return None

            # 6. Daten kombinieren
            df = sales_data[["Sales"]].copy()
            df["commonStockSharesOutstanding"] = balance_sheet["commonStockSharesOutstanding"].reindex(df.index, method="ffill")
            if df["Sales"].isna().all():
                self.logger.error(f"Keine gültigen Umsatzdaten für {symbol}.")
                return None

            if df["commonStockSharesOutstanding"].isna().all() or (df["commonStockSharesOutstanding"] <= 0).any():
                self.logger.error(f"Ungültige oder fehlende Shares Outstanding-Daten für {symbol}.")
                return None

            # 7. Aktienkurs auf Umsatzdaten joinen
            price_data.index = pd.to_datetime(price_data.index)
            price_data = price_data[["Close"]].reindex(df.index, method="ffill")
            if price_data.isna().all().any():
                self.logger.error(f"Keine passenden Aktienkursdaten für die Umsatzdaten von {symbol}.")
                return None
            df["Price"] = price_data["Close"]

            # 8. Price/Sales berechnen
            df["Price_Sales"] = df["Price"] / (df["Sales"] / df["commonStockSharesOutstanding"])
            df = df[["Price_Sales", "Price", "Sales", "commonStockSharesOutstanding"]].replace([np.inf, -np.inf], np.nan).dropna()

            if df.empty:
                self.logger.error(f"Keine gültigen Price/Sales-Daten für {symbol} nach Berechnung.")
                return None

            # Warnung für negative Multiples
            if (df["Price_Sales"] < 0).any():
                self.logger.warning(f"Negative Price/Sales-Werte für {symbol} erkannt, möglicherweise aufgrund negativer Umsätze.")

            # Zeitraum einschränken
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            if df.empty:
                self.logger.error(f"Keine Price/Sales-Daten für {symbol} im angegebenen Zeitraum.")
                return None

            # 9. Cache als Dictionary speichern
            cache_data = {
                "Price_Sales": {date.strftime("%Y-%m-%d"): value for date, value in df["Price_Sales"].items()},
                "Price": {date.strftime("%Y-%m-%d"): value for date, value in df["Price"].items()},
                "Sales": {date.strftime("%Y-%m-%d"): value for date, value in df["Sales"].items()},
                "commonStockSharesOutstanding": {date.strftime("%Y-%m-%d"): value for date, value in df["commonStockSharesOutstanding"].items()},
                "index": df.index.strftime("%Y-%m-%d").tolist()
            }
            self.dataloader._cache_data(cache_data, symbol, cache_key)
            self.logger.info(f"Price/Sales-Daten für {symbol} erfolgreich berechnet und gecacht.")
            return df

        except Exception as e:
            self.logger.error(f"Fehler bei Berechnung des Price/Sales-Multiples für {symbol}: {e}")
            return None

    def calculate_historical_price_to_ebit(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Berechnet das historische Price-to-EBIT-Multiple für ein Aktiensymbol.
        Gibt Price_EBIT, Price, EBIT und commonStockSharesOutstanding zurück.

        Args:
            symbol (str): Aktiensymbol (z. B. 'ILMN').
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (optional).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (optional).
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (default: True).

        Returns:
            Optional[pd.DataFrame]: DataFrame mit Price/EBIT-Multiple, Price, EBIT und commonStockSharesOutstanding,
                                    indiziert nach fiscalDateEnding, oder None bei Fehler.
        """
        cache_key = f"historical_price_to_ebit_{symbol}"

        # 1. Cache prüfen
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache-Daten für Price/EBIT von {symbol} geladen.")
                if isinstance(cached_data, dict):
                    try:
                        df = pd.DataFrame({
                            "Price_EBIT": list(cached_data["Price_EBIT"].values()),
                            "Price": list(cached_data["Price"].values()),
                            "EBIT": list(cached_data["EBIT"].values()),
                            "commonStockSharesOutstanding": list(cached_data["commonStockSharesOutstanding"].values())
                        }, index=pd.to_datetime(cached_data["index"]))
                        self.logger.info(f"Rekonstruiertes price_to_ebit für {symbol}: Spalten={df.columns.tolist()}, Index={df.index[:5].tolist()}")
                        return df
                    except Exception as e:
                        self.logger.error(f"Fehler beim Rekonstruieren des Cache für {symbol}: {e}")
                        return None
                return cached_data

        try:
            # 2. EBIT-Daten abrufen
            ebit_data = self.calculate_historical_ebit(symbol, start_date=start_date, end_date=end_date, use_cache=use_cache)
            if ebit_data is None:
                self.logger.error(f"Keine EBIT-Daten für {symbol} verfügbar.")
                return None

            self.logger.info(f"ebit_data-Spalten für {symbol}: {ebit_data.columns.tolist()}")
            self.logger.info(f"ebit_data-Daten (erste Zeilen):\n{ebit_data.head()}")

            # Benötigte Spalten prüfen
            if "EBIT" not in ebit_data.columns:
                self.logger.error(f"Spalte 'EBIT' fehlt in ebit_data für {symbol}. Gefundene Spalten: {ebit_data.columns.tolist()}")
                return None

            # 3. Startdatum anpassen, falls nicht angegeben
            if start_date is None:
                start_date = ebit_data.index.min().strftime("%Y-%m-%d")
                self.logger.info(f"Kein start_date angegeben, verwende {start_date} basierend auf ebit_data.")

            # 4. Fundamentaldaten für commonStockSharesOutstanding abrufen
            fundamentals = self.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=use_cache)
            if "error" in fundamentals:
                self.logger.error(f"Fehler bei Fundamentaldaten für {symbol}: {fundamentals['error']}")
                return None

            balance_sheet = fundamentals.get("balance_sheet")
            if balance_sheet is None:
                self.logger.error(f"Keine Balance Sheet-Daten für {symbol} verfügbar.")
                return None

            # Benötigte Spalten prüfen
            required_columns = {"commonStockSharesOutstanding"}
            missing_columns = required_columns - set(balance_sheet.columns)
            if missing_columns:
                self.logger.error(f"Fehlende Spalten in balance_sheet für {symbol}: {missing_columns}")
                return None

            # 5. Aktienkursdaten abrufen (täglich, um exakte Tage zu treffen)
            price_data = self.dataloader.get_max_historical_stock_data(symbol, start_date=start_date, end_date=end_date, interval="1d", use_cache=use_cache)
            if price_data is None or price_data.empty:
                self.logger.error(f"Keine Aktienkursdaten für {symbol} verfügbar.")
                return None

            # 6. Daten kombinieren
            df = ebit_data[["EBIT"]].copy()
            df["commonStockSharesOutstanding"] = balance_sheet["commonStockSharesOutstanding"].reindex(df.index, method="ffill")
            if df["EBIT"].isna().all():
                self.logger.error(f"Keine gültigen EBIT-Daten für {symbol}.")
                return None

            if df["commonStockSharesOutstanding"].isna().all() or (df["commonStockSharesOutstanding"] <= 0).any():
                self.logger.error(f"Ungültige oder fehlende Shares Outstanding-Daten für {symbol}.")
                return None

            # 7. Aktienkurs auf EBIT-Daten joinen
            price_data.index = pd.to_datetime(price_data.index)
            price_data = price_data[["Close"]].reindex(df.index, method="ffill")
            if price_data.isna().all().any():
                self.logger.error(f"Keine passenden Aktienkursdaten für die EBIT-Daten von {symbol}.")
                return None
            df["Price"] = price_data["Close"]

            # 8. Price/EBIT berechnen
            df["Price_EBIT"] = df["Price"] / (df["EBIT"] / df["commonStockSharesOutstanding"])
            df = df[["Price_EBIT", "Price", "EBIT", "commonStockSharesOutstanding"]].replace([np.inf, -np.inf], np.nan).dropna()

            if df.empty:
                self.logger.error(f"Keine gültigen Price/EBIT-Daten für {symbol} nach Berechnung.")
                return None

            # Warnung für negative Multiples
            if (df["Price_EBIT"] < 0).any():
                self.logger.warning(f"Negative Price/EBIT-Werte für {symbol} erkannt, möglicherweise aufgrund negativer EBIT-Werte.")

            # Zeitraum einschränken
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            if df.empty:
                self.logger.error(f"Keine Price/EBIT-Daten für {symbol} im angegebenen Zeitraum.")
                return None

            # 9. Cache als Dictionary speichern
            cache_data = {
                "Price_EBIT": {date.strftime("%Y-%m-%d"): value for date, value in df["Price_EBIT"].items()},
                "Price": {date.strftime("%Y-%m-%d"): value for date, value in df["Price"].items()},
                "EBIT": {date.strftime("%Y-%m-%d"): value for date, value in df["EBIT"].items()},
                "commonStockSharesOutstanding": {date.strftime("%Y-%m-%d"): value for date, value in df["commonStockSharesOutstanding"].items()},
                "index": df.index.strftime("%Y-%m-%d").tolist()
            }
            self.dataloader._cache_data(cache_data, symbol, cache_key)
            self.logger.info(f"Price/EBIT-Daten für {symbol} erfolgreich berechnet und gecacht.")
            return df

        except Exception as e:
            self.logger.error(f"Fehler bei Berechnung des Price/EBIT-Multiples für {symbol}: {e}")
            return None

    def calculate_historical_netCurrentAssets(self, symbol: str, start_date: Optional[str] = None,
                                              end_date: Optional[str] = None, use_cache: bool = True) -> Optional[
        pd.DataFrame]:
        """
        Berechnet die historischen Netto-Umlaufvermögen (Total Current Assets - Total Current Liabilities) für ein Aktiensymbol.

        Args:
            symbol (str): Aktiensymbol (z. B. 'ILMN').
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (optional).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (optional).
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (default: True).

        Returns:
            Optional[pd.DataFrame]: DataFrame mit NetCurrentAssets-Werten (Spalte 'NetCurrentAssets'),
                                    indiziert nach fiscalDateEnding, oder None bei Fehler.
        """
        cache_key = f"historical_netCurrentAssets_{symbol}"

        # 1. Cache prüfen
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache-Daten für NetCurrentAssets von {symbol} geladen.")
                if isinstance(cached_data, dict):
                    try:
                        df = pd.DataFrame({
                            "NetCurrentAssets": list(cached_data["NetCurrentAssets"].values())
                        }, index=pd.to_datetime(cached_data["index"]))
                        self.logger.info(
                            f"Rekonstruiertes netCurrentAssets für {symbol}: Spalten={df.columns.tolist()}, Index={df.index[:5].tolist()}")
                        return df
                    except Exception as e:
                        self.logger.error(f"Fehler beim Rekonstruieren des Cache für {symbol}: {e}")
                        return None
                return cached_data

        try:
            # 2. Fundamentaldaten abrufen
            fundamentals = self.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            if "error" in fundamentals:
                self.logger.error(f"Fehler bei Fundamentaldaten für {symbol}: {fundamentals['error']}")
                return None

            balance_sheet = fundamentals.get("balance_sheet")
            if balance_sheet is None:
                self.logger.error(f"Keine Balance Sheet-Daten für {symbol} verfügbar.")
                return None

            # Spaltenprüfung
            required_columns = {"totalCurrentAssets", "totalCurrentLiabilities"}
            missing_columns = required_columns - set(balance_sheet.columns)
            if missing_columns:
                self.logger.error(f"Fehlende Spalten für {symbol} in balance_sheet: {missing_columns}")
                return None

            # 3. NetCurrentAssets berechnen
            df = balance_sheet[["totalCurrentAssets", "totalCurrentLiabilities"]].copy()
            df["NetCurrentAssets"] = df["totalCurrentAssets"] - df["totalCurrentLiabilities"]

            # Zeitraum einschränken
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            if df.empty:
                self.logger.error(f"Keine Balance Sheet-Daten für {symbol} im angegebenen Zeitraum.")
                return None

            # Spalten bereinigen und NaN-Werte entfernen
            df = df[["NetCurrentAssets"]].dropna()

            if df.empty:
                self.logger.error(f"Keine gültigen NetCurrentAssets-Werte für {symbol} nach Berechnung.")
                return None

            # 4. Cache als Dictionary speichern
            cache_data = {
                "NetCurrentAssets": {date.strftime("%Y-%m-%d"): value for date, value in
                                     df["NetCurrentAssets"].items()},
                "index": df.index.strftime("%Y-%m-%d").tolist()
            }
            self.dataloader._cache_data(cache_data, symbol, cache_key)
            self.logger.info(f"NetCurrentAssets-Werte für {symbol} erfolgreich berechnet und gecacht.")
            return df

        except Exception as e:
            self.logger.error(f"Fehler bei Berechnung der NetCurrentAssets-Werte für {symbol}: {e}")
            return None

    def calculate_historical_price_netCurrentAssets(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Berechnet das historische Price-to-NetCurrentAssets-Multiple für ein Aktiensymbol.
        Gibt Price_NetCurrentAssets, Price, NetCurrentAssets und commonStockSharesOutstanding zurück.

        Args:
            symbol (str): Aktiensymbol (z. B. 'ILMN').
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (optional).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (optional).
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (default: True).

        Returns:
            Optional[pd.DataFrame]: DataFrame mit Price/NetCurrentAssets-Multiple, Price, NetCurrentAssets und commonStockSharesOutstanding,
                                    indiziert nach fiscalDateEnding, oder None bei Fehler.
        """
        cache_key = f"historical_price_netCurrentAssets_{symbol}"

        # 1. Cache prüfen
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache-Daten für Price/NetCurrentAssets von {symbol} geladen.")
                if isinstance(cached_data, dict):
                    try:
                        df = pd.DataFrame({
                            "Price_NetCurrentAssets": list(cached_data["Price_NetCurrentAssets"].values()),
                            "Price": list(cached_data["Price"].values()),
                            "NetCurrentAssets": list(cached_data["NetCurrentAssets"].values()),
                            "commonStockSharesOutstanding": list(cached_data["commonStockSharesOutstanding"].values())
                        }, index=pd.to_datetime(cached_data["index"]))
                        self.logger.info(
                            f"Rekonstruiertes price_netCurrentAssets für {symbol}: Spalten={df.columns.tolist()}, Index={df.index[:5].tolist()}")
                        return df
                    except Exception as e:
                        self.logger.error(f"Fehler beim Rekonstruieren des Cache für {symbol}: {e}")
                        return None
                return cached_data

        try:
            # 2. NetCurrentAssets-Daten abrufen
            netCurrentAssets_data = self.calculate_historical_netCurrentAssets(symbol, start_date=start_date,
                                                                               end_date=end_date, use_cache=use_cache)
            if netCurrentAssets_data is None:
                self.logger.error(f"Keine NetCurrentAssets-Daten für {symbol} verfügbar.")
                return None

            self.logger.info(f"netCurrentAssets_data-Spalten für {symbol}: {netCurrentAssets_data.columns.tolist()}")
            self.logger.info(f"netCurrentAssets_data-Daten (erste Zeilen):\n{netCurrentAssets_data.head()}")

            # Benötigte Spalten prüfen
            if "NetCurrentAssets" not in netCurrentAssets_data.columns:
                self.logger.error(
                    f"Spalte 'NetCurrentAssets' fehlt in netCurrentAssets_data für {symbol}. Gefundene Spalten: {netCurrentAssets_data.columns.tolist()}")
                return None

            # 3. Startdatum anpassen, falls nicht angegeben
            if start_date is None:
                start_date = netCurrentAssets_data.index.min().strftime("%Y-%m-%d")
                self.logger.info(
                    f"Kein start_date angegeben, verwende {start_date} basierend auf netCurrentAssets_data.")

            # 4. Fundamentaldaten für commonStockSharesOutstanding abrufen
            fundamentals = self.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=use_cache)
            if "error" in fundamentals:
                self.logger.error(f"Fehler bei Fundamentaldaten für {symbol}: {fundamentals['error']}")
                return None

            balance_sheet = fundamentals.get("balance_sheet")
            if balance_sheet is None:
                self.logger.error(f"Keine Balance Sheet-Daten für {symbol} verfügbar.")
                return None

            # Benötigte Spalten prüfen
            required_columns = {"commonStockSharesOutstanding"}
            missing_columns = required_columns - set(balance_sheet.columns)
            if missing_columns:
                self.logger.error(f"Fehlende Spalten in balance_sheet für {symbol}: {missing_columns}")
                return None

            # 5. Aktienkursdaten abrufen (täglich, um exakte Tage zu treffen)
            price_data = self.dataloader.get_max_historical_stock_data(symbol, start_date=start_date, end_date=end_date,
                                                                       interval="1d", use_cache=use_cache)
            if price_data is None or price_data.empty:
                self.logger.error(f"Keine Aktienkursdaten für {symbol} verfügbar.")
                return None

            # 6. Daten kombinieren
            df = netCurrentAssets_data[["NetCurrentAssets"]].copy()
            df["commonStockSharesOutstanding"] = balance_sheet["commonStockSharesOutstanding"].reindex(df.index,
                                                                                                       method="ffill")
            if df["NetCurrentAssets"].isna().all():
                self.logger.error(f"Keine gültigen NetCurrentAssets-Daten für {symbol}.")
                return None

            if df["commonStockSharesOutstanding"].isna().all() or (df["commonStockSharesOutstanding"] <= 0).any():
                self.logger.error(f"Ungültige oder fehlende Shares Outstanding-Daten für {symbol}.")
                return None

            # 7. Aktienkurs auf NetCurrentAssets-Daten joinen
            price_data.index = pd.to_datetime(price_data.index)
            price_data = price_data[["Close"]].reindex(df.index, method="ffill")
            if price_data.isna().all().any():
                self.logger.error(f"Keine passenden Aktienkursdaten für die NetCurrentAssets-Daten von {symbol}.")
                return None
            df["Price"] = price_data["Close"]

            # 8. Price/NetCurrentAssets berechnen
            df["Price_NetCurrentAssets"] = df["Price"] / (df["NetCurrentAssets"] / df["commonStockSharesOutstanding"])
            df = df[["Price_NetCurrentAssets", "Price", "NetCurrentAssets", "commonStockSharesOutstanding"]].replace(
                [np.inf, -np.inf], np.nan).dropna()

            if df.empty:
                self.logger.error(f"Keine gültigen Price/NetCurrentAssets-Daten für {symbol} nach Berechnung.")
                return None

            # Warnung für negative Multiples
            if (df["Price_NetCurrentAssets"] < 0).any():
                self.logger.warning(
                    f"Negative Price/NetCurrentAssets-Werte für {symbol} erkannt, möglicherweise aufgrund negativer NetCurrentAssets-Werte.")

            # Zeitraum einschränken
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            if df.empty:
                self.logger.error(f"Keine Price/NetCurrentAssets-Daten für {symbol} im angegebenen Zeitraum.")
                return None

            # 9. Cache als Dictionary speichern
            cache_data = {
                "Price_NetCurrentAssets": {date.strftime("%Y-%m-%d"): value for date, value in
                                           df["Price_NetCurrentAssets"].items()},
                "Price": {date.strftime("%Y-%m-%d"): value for date, value in df["Price"].items()},
                "NetCurrentAssets": {date.strftime("%Y-%m-%d"): value for date, value in
                                     df["NetCurrentAssets"].items()},
                "commonStockSharesOutstanding": {date.strftime("%Y-%m-%d"): value for date, value in
                                                 df["commonStockSharesOutstanding"].items()},
                "index": df.index.strftime("%Y-%m-%d").tolist()
            }
            self.dataloader._cache_data(cache_data, symbol, cache_key)
            self.logger.info(f"Price/NetCurrentAssets-Daten für {symbol} erfolgreich berechnet und gecacht.")
            return df

        except Exception as e:
            self.logger.error(f"Fehler bei Berechnung des Price/NetCurrentAssets-Multiples für {symbol}: {e}")
            return None

    def calculate_historical_OperatingCashflow(self, symbol: str, start_date: Optional[str] = None,
                                               end_date: Optional[str] = None, use_cache: bool = True) -> Optional[
        pd.DataFrame]:
        """
        Berechnet die historischen Operating Cashflow-Werte für ein Aktiensymbol.

        Args:
            symbol (str): Aktiensymbol (z. B. 'ILMN').
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (optional).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (optional).
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (default: True).

        Returns:
            Optional[pd.DataFrame]: DataFrame mit Operating Cashflow-Werten (Spalte 'OperatingCashflow'),
                                    indiziert nach fiscalDateEnding, oder None bei Fehler.
        """
        cache_key = f"historical_operatingCashflow_{symbol}"

        # 1. Cache prüfen
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache-Daten für OperatingCashflow von {symbol} geladen.")
                if isinstance(cached_data, dict):
                    try:
                        df = pd.DataFrame({
                            "OperatingCashflow": list(cached_data["OperatingCashflow"].values())
                        }, index=pd.to_datetime(cached_data["index"]))
                        self.logger.info(
                            f"Rekonstruiertes OperatingCashflow für {symbol}: Spalten={df.columns.tolist()}, Index={df.index[:5].tolist()}")
                        return df
                    except Exception as e:
                        self.logger.error(f"Fehler beim Rekonstruieren des Cache für {symbol}: {e}")
                        return None
                return cached_data

        try:
            # 2. Fundamentaldaten abrufen
            fundamentals = self.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            if "error" in fundamentals:
                self.logger.error(f"Fehler bei Fundamentaldaten für {symbol}: {fundamentals['error']}")
                return None

            cash_flow = fundamentals.get("cash_flow")
            if cash_flow is None:
                self.logger.error(f"Keine Cashflow-Daten für {symbol} verfügbar.")
                return None

            # Spaltenprüfung
            required_columns = {"operatingCashflow"}
            missing_columns = required_columns - set(cash_flow.columns)
            if missing_columns:
                self.logger.error(f"Fehlende Spalten für {symbol} in cash_flow: {missing_columns}")
                return None

            # 3. OperatingCashflow extrahieren
            df = cash_flow[["operatingCashflow"]].copy()
            df = df.rename(columns={"operatingCashflow": "OperatingCashflow"})

            # Zeitraum einschränken
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            if df.empty:
                self.logger.error(f"Keine Cashflow-Daten für {symbol} im angegebenen Zeitraum.")
                return None

            # Spalten bereinigen und NaN-Werte entfernen
            df = df[["OperatingCashflow"]].dropna()

            if df.empty:
                self.logger.error(f"Keine gültigen OperatingCashflow-Werte für {symbol} nach Berechnung.")
                return None

            # 4. Cache als Dictionary speichern
            cache_data = {
                "OperatingCashflow": {date.strftime("%Y-%m-%d"): value for date, value in
                                      df["OperatingCashflow"].items()},
                "index": df.index.strftime("%Y-%m-%d").tolist()
            }
            self.dataloader._cache_data(cache_data, symbol, cache_key)
            self.logger.info(f"OperatingCashflow-Werte für {symbol} erfolgreich berechnet und gecacht.")
            return df

        except Exception as e:
            self.logger.error(f"Fehler bei Berechnung der OperatingCashflow-Werte für {symbol}: {e}")
            return None

    def calculate_historical_price_OperatingCashflow(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Berechnet das historische Price-to-OperatingCashflow-Multiple für ein Aktiensymbol.
        Gibt Price_OperatingCashflow, Price, OperatingCashflow und commonStockSharesOutstanding zurück.

        Args:
            symbol (str): Aktiensymbol (z. B. 'ILMN').
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (optional).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (optional).
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (default: True).

        Returns:
            Optional[pd.DataFrame]: DataFrame mit Price/OperatingCashflow-Multiple, Price, OperatingCashflow und commonStockSharesOutstanding,
                                    indiziert nach fiscalDateEnding, oder None bei Fehler.
        """
        cache_key = f"historical_price_operatingCashflow_{symbol}"

        # 1. Cache prüfen
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache-Daten für Price/OperatingCashflow von {symbol} geladen.")
                if isinstance(cached_data, dict):
                    try:
                        df = pd.DataFrame({
                            "Price_OperatingCashflow": list(cached_data["Price_OperatingCashflow"].values()),
                            "Price": list(cached_data["Price"].values()),
                            "OperatingCashflow": list(cached_data["OperatingCashflow"].values()),
                            "commonStockSharesOutstanding": list(cached_data["commonStockSharesOutstanding"].values())
                        }, index=pd.to_datetime(cached_data["index"]))
                        self.logger.info(
                            f"Rekonstruiertes price_operatingCashflow für {symbol}: Spalten={df.columns.tolist()}, Index={df.index[:5].tolist()}")
                        return df
                    except Exception as e:
                        self.logger.error(f"Fehler beim Rekonstruieren des Cache für {symbol}: {e}")
                        return None
                return cached_data

        try:
            # 2. OperatingCashflow-Daten abrufen
            operating_cashflow_data = self.calculate_historical_OperatingCashflow(symbol, start_date=start_date,
                                                                                  end_date=end_date,
                                                                                  use_cache=use_cache)
            if operating_cashflow_data is None:
                self.logger.error(f"Keine OperatingCashflow-Daten für {symbol} verfügbar.")
                return None

            self.logger.info(f"operatingCashflow_data-Spalten für {symbol}: {operating_cashflow_data.columns.tolist()}")
            self.logger.info(f"operatingCashflow_data-Daten (erste Zeilen):\n{operating_cashflow_data.head()}")

            # Benötigte Spalten prüfen
            if "OperatingCashflow" not in operating_cashflow_data.columns:
                self.logger.error(
                    f"Spalte 'OperatingCashflow' fehlt in operatingCashflow_data für {symbol}. Gefundene Spalten: {operating_cashflow_data.columns.tolist()}")
                return None

            # 3. Startdatum anpassen, falls nicht angegeben
            if start_date is None:
                start_date = operating_cashflow_data.index.min().strftime("%Y-%m-%d")
                self.logger.info(
                    f"Kein start_date angegeben, verwende {start_date} basierend auf operatingCashflow_data.")

            # 4. Fundamentaldaten für commonStockSharesOutstanding abrufen
            fundamentals = self.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=use_cache)
            if "error" in fundamentals:
                self.logger.error(f"Fehler bei Fundamentaldaten für {symbol}: {fundamentals['error']}")
                return None

            balance_sheet = fundamentals.get("balance_sheet")
            if balance_sheet is None:
                self.logger.error(f"Keine Balance Sheet-Daten für {symbol} verfügbar.")
                return None

            # Benötigte Spalten prüfen
            required_columns = {"commonStockSharesOutstanding"}
            missing_columns = required_columns - set(balance_sheet.columns)
            if missing_columns:
                self.logger.error(f"Fehlende Spalten in balance_sheet für {symbol}: {missing_columns}")
                return None

            # 5. Aktienkursdaten abrufen (täglich, um exakte Tage zu treffen)
            price_data = self.dataloader.get_max_historical_stock_data(symbol, start_date=start_date, end_date=end_date,
                                                                       interval="1d", use_cache=use_cache)
            if price_data is None or price_data.empty:
                self.logger.error(f"Keine Aktienkursdaten für {symbol} verfügbar.")
                return None

            # 6. Daten kombinieren
            df = operating_cashflow_data[["OperatingCashflow"]].copy()
            df["commonStockSharesOutstanding"] = balance_sheet["commonStockSharesOutstanding"].reindex(df.index,
                                                                                                       method="ffill")
            if df["OperatingCashflow"].isna().all():
                self.logger.error(f"Keine gültigen OperatingCashflow-Daten für {symbol}.")
                return None

            if df["commonStockSharesOutstanding"].isna().all() or (df["commonStockSharesOutstanding"] <= 0).any():
                self.logger.error(f"Ungültige oder fehlende Shares Outstanding-Daten für {symbol}.")
                return None

            # 7. Aktienkurs auf OperatingCashflow-Daten joinen
            price_data.index = pd.to_datetime(price_data.index)
            price_data = price_data[["Close"]].reindex(df.index, method="ffill")
            if price_data.isna().all().any():
                self.logger.error(f"Keine passenden Aktienkursdaten für die OperatingCashflow-Daten von {symbol}.")
                return None
            df["Price"] = price_data["Close"]

            # 8. Price/OperatingCashflow berechnen
            df["Price_OperatingCashflow"] = df["Price"] / (df["OperatingCashflow"] / df["commonStockSharesOutstanding"])
            df = df[["Price_OperatingCashflow", "Price", "OperatingCashflow", "commonStockSharesOutstanding"]].replace(
                [np.inf, -np.inf], np.nan).dropna()

            if df.empty:
                self.logger.error(f"Keine gültigen Price/OperatingCashflow-Daten für {symbol} nach Berechnung.")
                return None

            # Warnung für negative Multiples
            if (df["Price_OperatingCashflow"] < 0).any():
                self.logger.warning(
                    f"Negative Price/OperatingCashflow-Werte für {symbol} erkannt, möglicherweise aufgrund negativer OperatingCashflow-Werte.")

            # Zeitraum einschränken
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            if df.empty:
                self.logger.error(f"Keine Price/OperatingCashflow-Daten für {symbol} im angegebenen Zeitraum.")
                return None

            # 9. Cache als Dictionary speichern
            cache_data = {
                "Price_OperatingCashflow": {date.strftime("%Y-%m-%d"): value for date, value in
                                            df["Price_OperatingCashflow"].items()},
                "Price": {date.strftime("%Y-%m-%d"): value for date, value in df["Price"].items()},
                "OperatingCashflow": {date.strftime("%Y-%m-%d"): value for date, value in
                                      df["OperatingCashflow"].items()},
                "commonStockSharesOutstanding": {date.strftime("%Y-%m-%d"): value for date, value in
                                                 df["commonStockSharesOutstanding"].items()},
                "index": df.index.strftime("%Y-%m-%d").tolist()
            }
            self.dataloader._cache_data(cache_data, symbol, cache_key)
            self.logger.info(f"Price/OperatingCashflow-Daten für {symbol} erfolgreich berechnet und gecacht.")
            return df

        except Exception as e:
            self.logger.error(f"Fehler bei Berechnung des Price/OperatingCashflow-Multiples für {symbol}: {e}")
            return None

    def calculate_historical_FreeCashflow(self, symbol: str, start_date: Optional[str] = None,
                                          end_date: Optional[str] = None, use_cache: bool = True) -> Optional[
        pd.DataFrame]:
        """
        Berechnet die historischen Free Cashflow-Werte (Operating Cashflow - Capital Expenditures) für ein Aktiensymbol.
        Gibt FreeCashflow, OperatingCashflow und CapitalExpenditures zurück.

        Args:
            symbol (str): Aktiensymbol (z. B. 'ILMN').
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (optional).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (optional).
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (default: True).

        Returns:
            Optional[pd.DataFrame]: DataFrame mit FreeCashflow, OperatingCashflow und CapitalExpenditures,
                                    indiziert nach fiscalDateEnding, oder None bei Fehler.
        """
        cache_key = f"historical_freeCashflow_{symbol}"

        # 1. Cache prüfen
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache-Daten für FreeCashflow von {symbol} geladen.")
                if isinstance(cached_data, dict):
                    try:
                        df = pd.DataFrame({
                            "FreeCashflow": list(cached_data["FreeCashflow"].values()),
                            "OperatingCashflow": list(cached_data["OperatingCashflow"].values()),
                            "CapitalExpenditures": list(cached_data["CapitalExpenditures"].values())
                        }, index=pd.to_datetime(cached_data["index"]))
                        self.logger.info(
                            f"Rekonstruiertes FreeCashflow für {symbol}: Spalten={df.columns.tolist()}, Index={df.index[:5].tolist()}")
                        return df
                    except Exception as e:
                        self.logger.error(f"Fehler beim Rekonstruieren des Cache für {symbol}: {e}")
                        return None
                return cached_data

        try:
            # 2. Fundamentaldaten abrufen
            fundamentals = self.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            if "error" in fundamentals:
                self.logger.error(f"Fehler bei Fundamentaldaten für {symbol}: {fundamentals['error']}")
                return None

            cash_flow = fundamentals.get("cash_flow")
            if cash_flow is None:
                self.logger.error(f"Keine Cashflow-Daten für {symbol} verfügbar.")
                return None

            # Spaltenprüfung
            required_columns = {"operatingCashflow", "capitalExpenditures"}
            missing_columns = required_columns - set(cash_flow.columns)
            if missing_columns:
                self.logger.error(f"Fehlende Spalten für {symbol} in cash_flow: {missing_columns}")
                return None

            # 3. FreeCashflow berechnen
            df = cash_flow[["operatingCashflow", "capitalExpenditures"]].copy()
            df["FreeCashflow"] = df["operatingCashflow"] - df["capitalExpenditures"]
            df = df.rename(
                columns={"operatingCashflow": "OperatingCashflow", "capitalExpenditures": "CapitalExpenditures"})

            # Warnung für negative FreeCashflow-Werte
            if (df["FreeCashflow"] < 0).any():
                self.logger.warning(
                    f"Negative FreeCashflow-Werte für {symbol} erkannt, möglicherweise aufgrund hoher Capital Expenditures.")

            # Zeitraum einschränken
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            if df.empty:
                self.logger.error(f"Keine Cashflow-Daten für {symbol} im angegebenen Zeitraum.")
                return None

            # Spalten bereinigen und NaN-Werte entfernen
            df = df[["FreeCashflow", "OperatingCashflow", "CapitalExpenditures"]].dropna()

            if df.empty:
                self.logger.error(f"Keine gültigen FreeCashflow-Werte für {symbol} nach Berechnung.")
                return None

            # 4. Cache als Dictionary speichern
            cache_data = {
                "FreeCashflow": {date.strftime("%Y-%m-%d"): value for date, value in
                                 df["FreeCashflow"].items()},
                "OperatingCashflow": {date.strftime("%Y-%m-%d"): value for date, value in
                                      df["OperatingCashflow"].items()},
                "CapitalExpenditures": {date.strftime("%Y-%m-%d"): value for date, value in
                                        df["CapitalExpenditures"].items()},
                "index": df.index.strftime("%Y-%m-%d").tolist()
            }
            self.dataloader._cache_data(cache_data, symbol, cache_key)
            self.logger.info(f"FreeCashflow-Werte für {symbol} erfolgreich berechnet und gecacht.")
            return df

        except Exception as e:
            self.logger.error(f"Fehler bei Berechnung der FreeCashflow-Werte für {symbol}: {e}")
            return None

    def calculate_historical_Price_FreeCashflow(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Berechnet das historische Price-to-FreeCashflow-Multiple für ein Aktiensymbol.
        Gibt Price_FreeCashflow, Price, FreeCashflow, OperatingCashflow, CapitalExpenditures und commonStockSharesOutstanding zurück.

        Args:
            symbol (str): Aktiensymbol (z. B. 'ILMN').
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (optional).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (optional).
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (default: True).

        Returns:
            Optional[pd.DataFrame]: DataFrame mit Price/FreeCashflow-Multiple, Price, FreeCashflow, OperatingCashflow,
                                    CapitalExpenditures und commonStockSharesOutstanding, indiziert nach fiscalDateEnding,
                                    oder None bei Fehler.
        """
        cache_key = f"historical_price_freeCashflow_{symbol}"

        # 1. Cache prüfen
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache-Daten für Price/FreeCashflow von {symbol} geladen.")
                if isinstance(cached_data, dict):
                    try:
                        df = pd.DataFrame({
                            "Price_FreeCashflow": list(cached_data["Price_FreeCashflow"].values()),
                            "Price": list(cached_data["Price"].values()),
                            "FreeCashflow": list(cached_data["FreeCashflow"].values()),
                            "OperatingCashflow": list(cached_data["OperatingCashflow"].values()),
                            "CapitalExpenditures": list(cached_data["CapitalExpenditures"].values()),
                            "commonStockSharesOutstanding": list(cached_data["commonStockSharesOutstanding"].values())
                        }, index=pd.to_datetime(cached_data["index"]))
                        self.logger.info(
                            f"Rekonstruiertes price_freeCashflow für {symbol}: Spalten={df.columns.tolist()}, Index={df.index[:5].tolist()}")
                        return df
                    except Exception as e:
                        self.logger.error(f"Fehler beim Rekonstruieren des Cache für {symbol}: {e}")
                        return None
                return cached_data

        try:
            # 2. FreeCashflow-Daten abrufen
            free_cashflow_data = self.calculate_historical_FreeCashflow(symbol, start_date=start_date,
                                                                        end_date=end_date, use_cache=use_cache)
            if free_cashflow_data is None:
                self.logger.error(f"Keine FreeCashflow-Daten für {symbol} verfügbar.")
                return None

            self.logger.info(f"freeCashflow_data-Spalten für {symbol}: {free_cashflow_data.columns.tolist()}")
            self.logger.info(f"freeCashflow_data-Daten (erste Zeilen):\n{free_cashflow_data.head()}")

            # Benötigte Spalten prüfen
            required_columns = {"FreeCashflow", "OperatingCashflow", "CapitalExpenditures"}
            missing_columns = required_columns - set(free_cashflow_data.columns)
            if missing_columns:
                self.logger.error(
                    f"Fehlende Spalten in freeCashflow_data für {symbol}: Erwartet {required_columns}, erhalten {free_cashflow_data.columns.tolist()}")
                return None

            # 3. Startdatum anpassen, falls nicht angegeben
            if start_date is None:
                start_date = free_cashflow_data.index.min().strftime("%Y-%m-%d")
                self.logger.info(
                    f"Kein start_date angegeben, verwende {start_date} basierend auf freeCashflow_data.")

            # 4. Fundamentaldaten für commonStockSharesOutstanding abrufen
            fundamentals = self.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=use_cache)
            if "error" in fundamentals:
                self.logger.error(f"Fehler bei Fundamentaldaten für {symbol}: {fundamentals['error']}")
                return None

            balance_sheet = fundamentals.get("balance_sheet")
            if balance_sheet is None:
                self.logger.error(f"Keine Balance Sheet-Daten für {symbol} verfügbar.")
                return None

            # Benötigte Spalten prüfen
            required_balance_columns = {"commonStockSharesOutstanding"}
            missing_balance_columns = required_balance_columns - set(balance_sheet.columns)
            if missing_balance_columns:
                self.logger.error(f"Fehlende Spalten in balance_sheet für {symbol}: {missing_balance_columns}")
                return None

            # 5. Aktienkursdaten abrufen (täglich, um exakte Tage zu treffen)
            price_data = self.dataloader.get_max_historical_stock_data(symbol, start_date=start_date, end_date=end_date,
                                                                       interval="1d", use_cache=use_cache)
            if price_data is None or price_data.empty:
                self.logger.error(f"Keine Aktienkursdaten für {symbol} verfügbar.")
                return None

            # 6. Daten kombinieren
            df = free_cashflow_data[["FreeCashflow", "OperatingCashflow", "CapitalExpenditures"]].copy()
            df["commonStockSharesOutstanding"] = balance_sheet["commonStockSharesOutstanding"].reindex(df.index,
                                                                                                       method="ffill")
            if df["FreeCashflow"].isna().all():
                self.logger.error(f"Keine gültigen FreeCashflow-Daten für {symbol}.")
                return None

            if df["commonStockSharesOutstanding"].isna().all() or (df["commonStockSharesOutstanding"] <= 0).any():
                self.logger.error(f"Ungültige oder fehlende Shares Outstanding-Daten für {symbol}.")
                return None

            # 7. Aktienkurs auf FreeCashflow-Daten joinen
            price_data.index = pd.to_datetime(price_data.index)
            price_data = price_data[["Close"]].reindex(df.index, method="ffill")
            if price_data.isna().all().any():
                self.logger.error(f"Keine passenden Aktienkursdaten für die FreeCashflow-Daten von {symbol}.")
                return None
            df["Price"] = price_data["Close"]

            # 8. Price/FreeCashflow berechnen
            df["Price_FreeCashflow"] = df["Price"] / (df["FreeCashflow"] / df["commonStockSharesOutstanding"])
            df = df[["Price_FreeCashflow", "Price", "FreeCashflow", "OperatingCashflow", "CapitalExpenditures",
                     "commonStockSharesOutstanding"]].replace([np.inf, -np.inf], np.nan).dropna()

            if df.empty:
                self.logger.error(f"Keine gültigen Price/FreeCashflow-Daten für {symbol} nach Berechnung.")
                return None

            # Warnung für negative Multiples
            if (df["Price_FreeCashflow"] < 0).any():
                self.logger.warning(
                    f"Negative Price/FreeCashflow-Werte für {symbol} erkannt, möglicherweise aufgrund negativer FreeCashflow-Werte.")

            # Zeitraum einschränken
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            if df.empty:
                self.logger.error(f"Keine Price/FreeCashflow-Daten für {symbol} im angegebenen Zeitraum.")
                return None

            # 9. Cache als Dictionary speichern
            cache_data = {
                "Price_FreeCashflow": {date.strftime("%Y-%m-%d"): value for date, value in
                                       df["Price_FreeCashflow"].items()},
                "Price": {date.strftime("%Y-%m-%d"): value for date, value in df["Price"].items()},
                "FreeCashflow": {date.strftime("%Y-%m-%d"): value for date, value in
                                 df["FreeCashflow"].items()},
                "OperatingCashflow": {date.strftime("%Y-%m-%d"): value for date, value in
                                      df["OperatingCashflow"].items()},
                "CapitalExpenditures": {date.strftime("%Y-%m-%d"): value for date, value in
                                        df["CapitalExpenditures"].items()},
                "commonStockSharesOutstanding": {date.strftime("%Y-%m-%d"): value for date, value in
                                                 df["commonStockSharesOutstanding"].items()},
                "index": df.index.strftime("%Y-%m-%d").tolist()
            }
            self.dataloader._cache_data(cache_data, symbol, cache_key)
            self.logger.info(f"Price/FreeCashflow-Daten für {symbol} erfolgreich berechnet und gecacht.")
            return df

        except Exception as e:
            self.logger.error(f"Fehler bei Berechnung des Price/FreeCashflow-Multiples für {symbol}: {e}")
            return None

    def calculate_historical_TangibleBookValue(self, symbol: str, start_date: Optional[str] = None,
                                              end_date: Optional[str] = None, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Berechnet den historischen materiellen Buchwert (Total Assets - Intangible Assets - Goodwill - Total Liabilities)
        für ein Aktiensymbol und gibt die Komponenten totalAssets, intangibleAssets, goodwill, totalLiabilities zurück.

        Args:
            symbol (str): Aktiensymbol (z. B. 'ILMN').
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (optional).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (optional).
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (default: True).

        Returns:
            Optional[pd.DataFrame]: DataFrame mit TangibleBookValue, totalAssets, intangibleAssets, goodwill, totalLiabilities,
                                    indiziert nach fiscalDateEnding, oder None bei Fehler.
        """
        cache_key = f"historical_TangibleBookValue_{symbol}"

        # 1. Cache prüfen
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache-Daten für TangibleBookValue von {symbol} geladen.")
                if isinstance(cached_data, dict):
                    try:
                        df = pd.DataFrame({
                            "TangibleBookValue": list(cached_data["TangibleBookValue"].values()),
                            "totalAssets": list(cached_data["totalAssets"].values()),
                            "intangibleAssets": list(cached_data["intangibleAssets"].values()),
                            "goodwill": list(cached_data["goodwill"].values()),
                            "totalLiabilities": list(cached_data["totalLiabilities"].values())
                        }, index=pd.to_datetime(cached_data["index"]))
                        self.logger.info(f"Rekonstruiertes TangibleBookValue für {symbol}: Spalten={df.columns.tolist()}, Index={df.index[:5].tolist()}")
                        return df
                    except Exception as e:
                        self.logger.error(f"Fehler beim Rekonstruieren des Cache für {symbol}: {e}")
                        return None
                return cached_data

        try:
            # 2. Fundamentaldaten abrufen
            fundamentals = self.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            if "error" in fundamentals:
                self.logger.error(f"Fehler bei Fundamentaldaten für {symbol}: {fundamentals['error']}")
                return None

            balance_sheet = fundamentals.get("balance_sheet")
            if balance_sheet is None:
                self.logger.error(f"Keine Balance Sheet-Daten für {symbol} verfügbar.")
                return None

            # Spaltenprüfung
            required_columns = {"totalAssets", "intangibleAssets", "goodwill", "totalLiabilities"}
            missing_columns = required_columns - set(balance_sheet.columns)
            if missing_columns:
                self.logger.error(f"Fehlende Spalten für {symbol} in balance_sheet: {missing_columns}")
                return None

            # 3. TangibleBookValue berechnen
            df = balance_sheet[["totalAssets", "intangibleAssets", "goodwill", "totalLiabilities"]].copy()
            df["TangibleBookValue"] = df["totalAssets"] - df["intangibleAssets"] - df["goodwill"] - df["totalLiabilities"]

            # Zeitraum einschränken
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            if df.empty:
                self.logger.error(f"Keine Balance Sheet-Daten für {symbol} im angegebenen Zeitraum.")
                return None

            # Spalten bereinigen und NaN-Werte entfernen
            df = df[["TangibleBookValue", "totalAssets", "intangibleAssets", "goodwill", "totalLiabilities"]].dropna()

            if df.empty:
                self.logger.error(f"Keine gültigen TangibleBookValue-Werte für {symbol} nach Berechnung.")
                return None

            # 4. Cache als Dictionary speichern
            cache_data = {
                "TangibleBookValue": {date.strftime("%Y-%m-%d"): value for date, value in df["TangibleBookValue"].items()},
                "totalAssets": {date.strftime("%Y-%m-%d"): value for date, value in df["totalAssets"].items()},
                "intangibleAssets": {date.strftime("%Y-%m-%d"): value for date, value in df["intangibleAssets"].items()},
                "goodwill": {date.strftime("%Y-%m-%d"): value for date, value in df["goodwill"].items()},
                "totalLiabilities": {date.strftime("%Y-%m-%d"): value for date, value in df["totalLiabilities"].items()},
                "index": df.index.strftime("%Y-%m-%d").tolist()
            }
            self.dataloader._cache_data(cache_data, symbol, cache_key)
            self.logger.info(f"TangibleBookValue-Werte und Komponenten für {symbol} erfolgreich berechnet und gecacht.")
            return df

        except Exception as e:
            self.logger.error(f"Fehler bei Berechnung der TangibleBookValue-Werte für {symbol}: {e}")
            return None


    def calculate_historical_price_to_TangibleBookValue(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Berechnet das historische Price-to-TangibleBookValue-Multiple für ein Aktiensymbol.
        Gibt Price_TangibleBookValue, Price, TangibleBookValue und commonStockSharesOutstanding zurück.

        Args:
            symbol (str): Aktiensymbol (z. B. 'ILMN').
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (optional).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (optional).
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (default: True).

        Returns:
            Optional[pd.DataFrame]: DataFrame mit Price/TangibleBookValue-Multiple, Price, TangibleBookValue und commonStockSharesOutstanding,
                                    indiziert nach fiscalDateEnding, oder None bei Fehler.
        """
        cache_key = f"historical_price_to_TangibleBookValue_{symbol}"

        # 1. Cache prüfen
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache-Daten für Price/TangibleBookValue von {symbol} geladen.")
                if isinstance(cached_data, dict):
                    try:
                        df = pd.DataFrame({
                            "Price_TangibleBookValue": list(cached_data["Price_TangibleBookValue"].values()),
                            "Price": list(cached_data["Price"].values()),
                            "TangibleBookValue": list(cached_data["TangibleBookValue"].values()),
                            "commonStockSharesOutstanding": list(cached_data["commonStockSharesOutstanding"].values())
                        }, index=pd.to_datetime(cached_data["index"]))
                        self.logger.info(f"Rekonstruiertes price_to_TangibleBookValue für {symbol}: Spalten={df.columns.tolist()}, Index={df.index[:5].tolist()}")
                        return df
                    except Exception as e:
                        self.logger.error(f"Fehler beim Rekonstruieren des Cache für {symbol}: {e}")
                        return None
                return cached_data

        try:
            # 2. TangibleBookValue-Daten abrufen
            tangible_book_data = self.calculate_historical_TangibleBookValue(symbol, start_date=start_date, end_date=end_date, use_cache=use_cache)
            if tangible_book_data is None:
                self.logger.error(f"Keine TangibleBookValue-Daten für {symbol} verfügbar.")
                return None

            self.logger.info(f"tangible_book_data-Spalten für {symbol}: {tangible_book_data.columns.tolist()}")
            self.logger.info(f"tangible_book_data-Daten (erste Zeilen):\n{tangible_book_data.head()}")

            # Benötigte Spalten prüfen
            required_columns = {"TangibleBookValue"}
            missing_columns = required_columns - set(tangible_book_data.columns)
            if missing_columns:
                self.logger.error(f"Fehlende Spalten in tangible_book_data für {symbol}: {missing_columns}")
                return None

            # 3. Startdatum anpassen, falls nicht angegeben
            if start_date is None:
                start_date = tangible_book_data.index.min().strftime("%Y-%m-%d")
                self.logger.info(f"Kein start_date angegeben, verwende {start_date} basierend auf tangible_book_data.")

            # 4. Fundamentaldaten für commonStockSharesOutstanding abrufen
            fundamentals = self.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=use_cache)
            if "error" in fundamentals:
                self.logger.error(f"Fehler bei Fundamentaldaten für {symbol}: {fundamentals['error']}")
                return None

            balance_sheet = fundamentals.get("balance_sheet")
            if balance_sheet is None:
                self.logger.error(f"Keine Balance Sheet-Daten für {symbol} verfügbar.")
                return None

            # Benötigte Spalten prüfen
            required_balance_columns = {"commonStockSharesOutstanding"}
            missing_balance_columns = required_balance_columns - set(balance_sheet.columns)
            if missing_balance_columns:
                self.logger.error(f"Fehlende Spalten in balance_sheet für {symbol}: {missing_balance_columns}")
                return None

            # 5. Aktienkursdaten abrufen (täglich, um exakte Tage zu treffen)
            price_data = self.dataloader.get_max_historical_stock_data(symbol, start_date=start_date, end_date=end_date,
                                                                       interval="1d", use_cache=use_cache)
            if price_data is None or price_data.empty:
                self.logger.error(f"Keine Aktienkursdaten für {symbol} verfügbar.")
                return None

            # 6. Daten kombinieren
            df = tangible_book_data[["TangibleBookValue"]].copy()
            df["commonStockSharesOutstanding"] = balance_sheet["commonStockSharesOutstanding"].reindex(df.index, method="ffill")
            if df["TangibleBookValue"].isna().all():
                self.logger.error(f"Keine gültigen TangibleBookValue-Daten für {symbol}.")
                return None

            if df["commonStockSharesOutstanding"].isna().all() or (df["commonStockSharesOutstanding"] <= 0).any():
                self.logger.error(f"Ungültige oder fehlende Shares Outstanding-Daten für {symbol}.")
                return None

            # 7. Aktienkurs auf TangibleBookValue-Daten joinen
            price_data.index = pd.to_datetime(price_data.index)
            price_data = price_data[["Close"]].reindex(df.index, method="ffill")
            if price_data.isna().all().any():
                self.logger.error(f"Keine passenden Aktienkursdaten für die TangibleBookValue-Daten von {symbol}.")
                return None
            df["Price"] = price_data["Close"]

            # 8. Price/TangibleBookValue berechnen
            df["Price_TangibleBookValue"] = df["Price"] / (df["TangibleBookValue"] / df["commonStockSharesOutstanding"])
            df = df[["Price_TangibleBookValue", "Price", "TangibleBookValue", "commonStockSharesOutstanding"]].replace([np.inf, -np.inf], np.nan).dropna()

            if df.empty:
                self.logger.error(f"Keine gültigen Price/TangibleBookValue-Daten für {symbol} nach Berechnung.")
                return None

            # Warnung für negative Multiples
            if (df["Price_TangibleBookValue"] < 0).any():
                self.logger.warning(f"Negative Price/TangibleBookValue-Werte für {symbol} erkannt, möglicherweise aufgrund negativer TangibleBookValue-Werte.")

            # Zeitraum einschränken
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            if df.empty:
                self.logger.error(f"Keine Price/TangibleBookValue-Daten für {symbol} im angegebenen Zeitraum.")
                return None

            # 9. Cache als Dictionary speichern
            cache_data = {
                "Price_TangibleBookValue": {date.strftime("%Y-%m-%d"): value for date, value in df["Price_TangibleBookValue"].items()},
                "Price": {date.strftime("%Y-%m-%d"): value for date, value in df["Price"].items()},
                "TangibleBookValue": {date.strftime("%Y-%m-%d"): value for date, value in df["TangibleBookValue"].items()},
                "commonStockSharesOutstanding": {date.strftime("%Y-%m-%d"): value for date, value in df["commonStockSharesOutstanding"].items()},
                "index": df.index.strftime("%Y-%m-%d").tolist()
            }
            self.dataloader._cache_data(cache_data, symbol, cache_key)
            self.logger.info(f"Price/TangibleBookValue-Daten für {symbol} erfolgreich berechnet und gecacht.")
            return df

        except Exception as e:
            self.logger.error(f"Fehler bei Berechnung des Price/TangibleBookValue-Multiples für {symbol}: {e}")
            return None

    def calculate_buy_case(self, historical_data):
        """
        Berechnet den Buy-Wert basierend auf dem globalen Tief und dem Median der 3 niedrigsten Werte
        in den folgenden Jahren, wobei jeder Wert aus einem anderen Jahr stammen muss.

        Args:
            historical_data (pd.DataFrame): DataFrame mit Datumsindex (z. B. fiscalDateEnding) und Multiple-Spalten.

        Returns:
            dict: Enthält das globale Tief und den Buy-Wert, oder ein Fehlerdictionary bei Problemen.
        """
        # Prüfe, ob historische Daten vorhanden sind
        if historical_data is None or historical_data.empty:
            return {'error': 'Keine historischen Daten vorhanden'}

        # Finde die passende Multiple-Spalte aus der Liste
        multiple_column = next((col for col in self.HISTORICAL_MULTIPLE_COLUMNS if col in historical_data.columns),
                               None)
        if multiple_column is None:
            return {'error': 'Keine passende Multiple-Spalte gefunden'}

        # Normalisiere den Index auf tz-naive Timestamps
        historical_data.index = pd.to_datetime(historical_data.index).tz_localize(None)

        # Schritt 1: Globales Tief finden
        global_min_idx = historical_data[multiple_column].idxmin()
        global_min = historical_data.loc[global_min_idx, multiple_column]
        global_min_date = historical_data.loc[global_min_idx].name
        global_min_year = global_min_date.year

        # >>> Änderung: Daten vor dem globalen Tief-Jahr verwenden
        data_before_min = historical_data[historical_data.index.year < global_min_year].copy()

        # Schritt 2: Finde für jedes Jahr den niedrigsten Wert (vor dem Minimum)
        yearly_mins = data_before_min.groupby(data_before_min.index.year)[multiple_column].min()
        if len(yearly_mins) < 3:
            # Fallback: Nimm die globalen 3 Jahres-Minima über den GESAMTEN Datensatz
            yearly_mins_all = historical_data.groupby(historical_data.index.year)[multiple_column].min()
            if len(yearly_mins_all) < 3:
                # selbst global nicht genug verschiedene Jahre
                return {
                    'error': 'Nicht genug verschiedene Jahre für 3 Werte',
                    'buy_fallback_algo_used': True
                }

            top_3_years_global = yearly_mins_all.nsmallest(3).index
            lowest_3_values_global = [
                historical_data[historical_data.index.year == y][multiple_column].min() for y in top_3_years_global
            ]
            buy_value_fallback = round(pd.Series(lowest_3_values_global).median(), 2)

            return {
                'global_min': float(global_min),
                'buy_value': float(buy_value_fallback),
                'buy_fallback_algo_used': True
            }

        # Schritt 3: Wähle die 3 niedrigsten Werte aus verschiedenen Jahren
        top_3_years = yearly_mins.nsmallest(3).index
        lowest_3_values = [
            data_before_min[data_before_min.index.year == year][multiple_column].min()
            for year in top_3_years
        ]

        # Berechne den Median der 3 niedrigsten Werte und runde auf zwei Dezimalstellen
        buy_value = round(pd.Series(lowest_3_values).median(), 2)

        return {
            'global_min': global_min,
            'buy_value': buy_value
        }

    def calculate_worst_case(self, historical_data):
        """
        Berechnet den Worst-Case-Wert, indem der Buy-Wert aus calculate_buy_case durch 1,2 geteilt wird.

        Args:
            historical_data (pd.DataFrame): DataFrame mit Datumsindex und Multiple-Spalten.

        Returns:
            float: Der Worst-Case-Wert, oder ein Fehlerdictionary bei Problemen.
        """
        # Prüfe, ob historische Daten vorhanden sind
        if historical_data is None or historical_data.empty:
            return {'error': 'Keine historischen Daten vorhanden'}

        # Berechne den Buy-Wert
        buy_result = self.calculate_buy_case(historical_data)
        if isinstance(buy_result, dict) and 'error' in buy_result:
            return {'error': f"Fehler beim Abruf des Buy-Werts: {buy_result['error']}"}
        if buy_result is None or 'buy_value' not in buy_result:
            return {'error': 'Fehler beim Abruf des Buy-Werts (unbekannter Grund)'}

        # Hole den Buy-Wert und teile durch 1,2
        buy_value = buy_result['buy_value']
        if buy_value is None or not isinstance(buy_value, (int, float)) or buy_value <= 0:
            return {'error': 'Ungültiger Buy-Wert: Muss eine positive Zahl sein'}

        worst_case_value = round(buy_value / 1.2, 2)

        return worst_case_value

    def calculate_sell_case(self, historical_data):
        """
        Berechnet den Sell-Wert basierend auf dem Median der 3 höchsten Werte des Multiples
        aus 3 verschiedenen Jahren.

        Args:
            historical_data (pd.DataFrame): DataFrame mit Datumsindex (z. B. fiscalDateEnding) und Multiple-Spalten.

        Returns:
            float: Der Sell-Wert, oder ein Fehlerdictionary bei Problemen.
        """
        # Prüfe, ob historische Daten vorhanden sind
        if historical_data is None or historical_data.empty:
            return {'error': 'Keine historischen Daten vorhanden'}

        # Finde die passende Multiple-Spalte aus der Liste
        multiple_column = next((col for col in self.HISTORICAL_MULTIPLE_COLUMNS if col in historical_data.columns),
                               None)
        if multiple_column is None:
            return {'error': 'Keine gültige Multiple-Spalte vorhanden'}

        # Normalisiere den Index auf tz-naive Timestamps
        historical_data.index = pd.to_datetime(historical_data.index).tz_localize(None)

        # Schritt 1: Finde für jedes Jahr den höchsten P/S-Wert
        yearly_maxs = historical_data.groupby(historical_data.index.year)[multiple_column].max()
        if len(yearly_maxs) < 3:
            return {'error': 'Nicht genug verschiedene Jahre für 3 Werte (mindestens 3 Jahre erforderlich)'}

        # Schritt 2: Wähle die 3 höchsten Werte aus verschiedenen Jahren
        top_3_years = yearly_maxs.nlargest(3).index
        highest_3_values = [historical_data[historical_data.index.year == year][multiple_column].max() for year in
                            top_3_years]

        # Berechne den Median der 3 höchsten Werte und runde auf zwei Dezimalstellen
        sell_value = round(pd.Series(highest_3_values).median(), 2)

        return sell_value

    def calculate_fairValue_case(self, buy_value, sell_value):
        """
        Berechnet den fairen Wert als arithmetisches Mittel aus Buy-Wert und Sell-Wert.

        Args:
            buy_value (float): Der Buy-Wert.
            sell_value (float): Der Sell-Wert.

        Returns:
            float: Der faire Wert, oder ein Fehlerdictionary bei Problemen.
        """
        # Prüfe, ob buy_value und sell_value vorhanden sind
        if buy_value is None or sell_value is None:
            return {'error': 'Buy-Wert oder Sell-Wert fehlt'}

        # Prüfe, ob die Werte numerisch sind
        if not isinstance(buy_value, (int, float)) or not isinstance(sell_value, (int, float)):
            return {'error': 'Buy-Wert oder Sell-Wert ist nicht numerisch'}

        # Berechne das arithmetische Mittel und runde auf zwei Dezimalstellen
        fair_value = round((buy_value + sell_value) / 2, 2)

        return fair_value

    def calculate_course_target_PriceMultiples(self, historical_data: pd.DataFrame, symbol: str):
        # Globale Listen sind bereits definiert und können direkt genutzt werden

        # Schritt 1a: Prüfe, ob historical_data None ist oder leer ist
        if not isinstance(historical_data, pd.DataFrame):
            self.logger.error("Übergebene historical_data ist kein pandas DataFrame.")
            return {'error': 'Übergebene historical_data ist kein pandas DataFrame'}
        if historical_data is None:
            self.logger.error("Übergebene historical_data ist None.")
            return {'error': 'Übergebene historical_data ist None'}
        if historical_data.empty:
            self.logger.error("Übergebene historical_data ist leer.")
            return {'error': 'Übergebene historical_data ist leer'}

        # Schritt 1b: Prüfe, ob der DataFrame ein historisches Multiple enthält
        price_multiple_column = next(
            (col for col in self.HISTORICAL_MULTIPLE_COLUMNS if col in historical_data.columns),
            None)

        if price_multiple_column is None:
            self.logger.error("Kein gültiges Price-Multiple in den Spalten des DataFrames gefunden.")
            return {'error': 'Kein gültiges Price-Multiple in den Spalten des DataFrames gefunden'}

        self.logger.info(f"Erkannt, dass ein historisches Multiple vorliegt: {price_multiple_column}")

        # Schritt 2a: Hole die Szenario-Multiples
        buy_result = self.calculate_buy_case(historical_data)
        if isinstance(buy_result, dict) and 'error' in buy_result:
            self.logger.error(f"Fehler beim Abruf des Buy-Werts: {buy_result['error']}")
            return {'error': f"Fehler beim Abruf des Buy-Werts: {buy_result['error']}"}
        buy_multiple = buy_result['buy_value']

        wc_multiple = self.calculate_worst_case(historical_data)
        if isinstance(wc_multiple, dict) and 'error' in wc_multiple:
            self.logger.error(f"Fehler beim Abruf des Worst-Case-Werts: {wc_multiple['error']}")
            return {'error': f"Fehler beim Abruf des Worst-Case-Werts: {wc_multiple['error']}"}

        sell_multiple = self.calculate_sell_case(historical_data)
        if isinstance(sell_multiple, dict) and 'error' in sell_multiple:
            self.logger.error(f"Fehler beim Abruf des Sell-Werts: {sell_multiple['error']}")
            return {'error': f"Fehler beim Abruf des Sell-Werts: {sell_multiple['error']}"}

        fv_multiple = self.calculate_fairValue_case(buy_multiple, sell_multiple)
        if isinstance(fv_multiple, dict) and 'error' in fv_multiple:
            self.logger.error(f"Fehler beim Abruf des Fair-Value-Werts: {fv_multiple['error']}")
            return {'error': f"Fehler beim Abruf des Fair-Value-Werts: {fv_multiple['error']}"}

        self.logger.info(
            f"Szenario-Multiples - WC: {wc_multiple}, BUY: {buy_multiple}, FV: {fv_multiple}, SELL: {sell_multiple}")

        # Schritt 2b: Berechne die Kennzahl pro Aktie basierend auf dem Multiple
        shares_outstanding = self.dataloader.get_shares_outstanding(symbol)
        if isinstance(shares_outstanding, dict) and "error" in shares_outstanding:
            self.logger.error(f"Fehler beim Abruf der Aktienzahl: {shares_outstanding['error']}")
            return {'error': f"Fehler beim Abruf der Aktienzahl: {shares_outstanding['error']}"}
        if shares_outstanding <= 0:
            self.logger.error(f"Ungültige Aktienzahl für {symbol}: {shares_outstanding}")
            return {'error': f"Ungültige Aktienzahl für {symbol}: {shares_outstanding}"}

        metric_per_share = 0.0
        if price_multiple_column == "Price_Sales":
            revenue = self.dataloader.get_revenue(symbol, frequency="quarterly")["revenue"]
            if isinstance(revenue, dict) and "error" in revenue:
                self.logger.error(f"Fehler beim Abruf von revenue: {revenue['error']}")
                return {'error': f"Fehler beim Abruf von revenue: {revenue['error']}"}
            metric_per_share = revenue / shares_outstanding if revenue else 0.0
        elif price_multiple_column == "Price_EBIT":
            ebit_data = self.dataloader.get_ebit_data(symbol, frequency="quarterly")
            if isinstance(ebit_data, dict) and "error" in ebit_data:
                self.logger.error(f"Fehler beim Abruf von EBIT: {ebit_data['error']}")
                return {'error': f"Fehler beim Abruf von EBIT: {ebit_data['error']}"}
            metric_per_share = ebit_data["ebit"] / shares_outstanding
        elif price_multiple_column == "Price_NetCurrentAssets":
            balance_sheet = self.dataloader.get_balance_sheet(symbol, frequency="quarterly")
            if isinstance(balance_sheet, dict) and "error" in balance_sheet:
                self.logger.error(f"Fehler beim Abruf der Bilanz: {balance_sheet['error']}")
                return {'error': f"Fehler beim Abruf der Bilanz: {balance_sheet['error']}"}
            total_assets = balance_sheet.loc["Total Assets"].iloc[0] if "Total Assets" in balance_sheet.index else 0
            total_liabilities = balance_sheet.loc["Total Liabilities"].iloc[
                0] if "Total Liabilities" in balance_sheet.index else 0
            net_current_assets = total_assets - total_liabilities
            metric_per_share = net_current_assets / shares_outstanding
        elif price_multiple_column == "Price_OperatingCashflow":
            operating_cashflow = self.dataloader.get_operating_cashflow(symbol, frequency="quarterly")[
                "operating_cashflow"]
            metric_per_share = operating_cashflow / shares_outstanding
        elif price_multiple_column == "Price_FreeCashflow":
            free_cashflow = self.dataloader.get_free_cashflow(symbol, frequency="quarterly")["free_cashflow"]
            metric_per_share = free_cashflow / shares_outstanding
        elif price_multiple_column == "Price_TangibleBookValue":
            balance_sheet = self.dataloader.get_balance_sheet(symbol, frequency="quarterly")
            if isinstance(balance_sheet, dict) and "error" in balance_sheet:
                self.logger.error(f"Fehler beim Abruf der Bilanz: {balance_sheet['error']}")
                return {'error': f"Fehler beim Abruf der Bilanz: {balance_sheet['error']}"}
            total_assets = balance_sheet.loc["Total Assets"].iloc[0] if "Total Assets" in balance_sheet.index else 0
            intangible_assets = balance_sheet.loc["Intangible Assets"].iloc[
                0] if "Intangible Assets" in balance_sheet.index else 0
            total_liabilities = balance_sheet.loc["Total Liabilities"].iloc[
                0] if "Total Liabilities" in balance_sheet.index else 0
            tangible_book_value = total_assets - intangible_assets - total_liabilities
            metric_per_share = tangible_book_value / shares_outstanding
        elif price_multiple_column == "Price_Book":
            # Berechne Book Value per Share aus den vorhandenen Daten
            book_value = (historical_data["totalAssets"] - historical_data["totalLiabilities"]).mean()
            metric_per_share = book_value / shares_outstanding if book_value > 0 else 0.0

        if metric_per_share <= 0:
            self.logger.error(f"Ungültige Kennzahl pro Aktie für {price_multiple_column}: {metric_per_share}")
            return {'error': f"Ungültige Kennzahl pro Aktie für {price_multiple_column}: {metric_per_share}"}

        self.logger.info(f"Berechnete Kennzahl pro Aktie: {metric_per_share}")

        # Schritt 2c: Berechne die Kursziele
        wc_price = wc_multiple * metric_per_share
        buy_price = buy_multiple * metric_per_share
        fv_price = fv_multiple * metric_per_share
        sell_price = sell_multiple * metric_per_share

        self.logger.info(f"Kursziele - WC: {wc_price}, BUY: {buy_price}, FV: {fv_price}, SELL: {sell_price}")

        # Rückgabe der Kursziele mit Rundung auf zwei Dezimalstellen
        return {
            "worst_case_price": round(wc_price, 2),
            "buy_price": round(buy_price, 2),
            "fair_value_price": round(fv_price, 2),
            "sell_price": round(sell_price, 2)
        }

    def calculate_course_target_EVMultiples(self, symbol: str, historical_data: pd.DataFrame):
        """
        Kursziele aus EV-Multiples (genau eine Spalte in historical_data, z.B. EV_Sales ODER EV_EBIT ODER EV_EBITDA).
        - Szenariofaktoren (WC/BUY/FV/SELL) werden aus der vollen Quartalshistorie dieses Multiples abgeleitet.
        - Bewertet wird mit aktuellen Quartalswerten (Revenue/EBIT/EBITDA, Net Debt, Shares).
        """
        # --- Eingaben prüfen ---
        if not isinstance(historical_data, pd.DataFrame) or historical_data.empty:
            return {"error": "Ungültige oder leere historische Daten"}


        # finde genau die eine Multiple-Spalte, die enthalten ist
        multiple_col = next((c for c in Model.EV_MULTIPLE_COLUMNS if c in historical_data.columns), None)
        if multiple_col is None:
            return {"error": f"Keine passende EV-Multiple-Spalte gefunden. Erwartet eine von {Model.EV_MULTIPLE_COLUMNS}"}

        # --- Schritt 1: Szenariofaktoren aus kompletter Historie (DataFrame unverändert übergeben) ---
        buy_result = self.calculate_buy_case(historical_data)
        if isinstance(buy_result, dict) and "error" in buy_result:
            return {"error": buy_result["error"]}
        buy_value = buy_result["buy_value"]

        wc_value = self.calculate_worst_case(historical_data)
        if isinstance(wc_value, dict) and "error" in wc_value:
            return {"error": wc_value["error"]}

        sell_value = self.calculate_sell_case(historical_data)
        if isinstance(sell_value, dict) and "error" in sell_value:
            return {"error": sell_value["error"]}

        fv_value = self.calculate_fairValue_case(buy_value, sell_value)
        if isinstance(fv_value, dict) and "error" in fv_value:
            return {"error": fv_value["error"]}

        scenario = {"WC": float(wc_value), "BUY": float(buy_value), "FV": float(fv_value), "SELL": float(sell_value)}

        # --- Schritt 2: aktuelle Quartalsgrößen holen (Basis passend zum Multiple) ---
        if multiple_col == "EV_Sales":
            rev = self.dataloader.get_revenue(symbol, frequency="quarterly")
            if isinstance(rev, dict) and "error" in rev:
                return rev
            base = float(rev["revenue"])
            base_name = "Umsatz (Revenue)"
        elif multiple_col == "EV_EBIT":
            ebit = self.dataloader.get_ebit_data(symbol, frequency="quarterly")
            if isinstance(ebit, dict) and "error" in ebit:
                return ebit
            base = float(ebit["ebit"])
            base_name = "EBIT"
        else:  # "EV_EBITDA"
            ebitda = self.dataloader.get_ebitda_data(symbol, frequency="quarterly")
            if isinstance(ebitda, dict) and "error" in ebitda:
                return ebitda
            base = float(ebitda["ebitda"])
            base_name = "EBITDA"

        # Validität + Negativ-Check (inkl. NaN/Inf)
        if base is None or not np.isfinite(base):
            return {"error": f"{base_name} fehlt oder ist ungültig"}
        if base < 0:
            return {"error": f"Da negative {base_name} keine Berechnung der Kursziele möglich"}
        if base == 0:
            return {"error": f"{base_name} ist 0 – keine aussagekräftige Berechnung möglich"}

        # --- Schritt 3: EV-Ziele je Szenario ---
        ev_targets = {k: base * v for k, v in scenario.items()}

        # --- Schritt 4: EV -> Equity -> Kurs je Aktie (alles: aktuelles Quartal) ---
        nd = self.dataloader.get_net_debt_data(symbol, frequency="quarterly")
        if isinstance(nd, dict) and "error" in nd: return nd
        net_debt = float(nd["net_debt"])

        shares = self.dataloader.get_shares_outstanding(symbol)
        if isinstance(shares, dict) and "error" in shares: return shares
        if not shares or shares <= 0: return {"error": f"Ungültige Aktienzahl: {shares}"}

        minority_interest = self.dataloader.get_minority_interest(symbol, frequency="quarterly")
        if isinstance(minority_interest, dict) and "error" in minority_interest:
            return minority_interest
        minority_interest = float(minority_interest["minority_interest"])
        if minority_interest < 0:
            return {"error": f"Negativer Minderheitenanteil ({minority_interest}) für {symbol} nicht erlaubt"}

        preferred_stock = self.dataloader.get_preferred_stock(symbol, frequency="quarterly")
        if isinstance(preferred_stock, dict) and "error" in preferred_stock:
            return preferred_stock
        preferred_stock = float(preferred_stock["preferred_stock"])
        if preferred_stock < 0:
            return {"error": f"Negativer Vorzugsaktienwert ({preferred_stock}) für {symbol} nicht erlaubt"}

        # Neue Validierung auf nan
        for value, name in [(net_debt, "Net Debt"), (shares, "Shares"), (minority_interest, "Minority Interest"),
                            (preferred_stock, "Preferred Stock")]:
            if not np.isfinite(value):
                return {"error": f"{name} enthält ungültige Werte (z.B. nan oder inf)"}

        result = {multiple_col: {}}
        for scen_name, ev_goal in ev_targets.items():
            equity_goal = float(ev_goal) - net_debt - minority_interest - preferred_stock
            if not np.isfinite(equity_goal):
                return {"error": f"Equity Goal für Szenario {scen_name} ist ungültig (nan oder inf)"}
            price = equity_goal / float(shares)
            result[multiple_col][scen_name] = round(float(price), 2)

        return result

    def calculate_percentiles(self, data, multiple_column, decimals=2):
        """
        Berechnet 12 gleich große Perzentil-Bereiche basierend auf der Differenz von
        Höchstwert und Tiefstwert eines Multiples, mit einer zusätzlichen Range von 0 bis zum Minimum.

        Args:
            self: Instanz der Model-Klasse
            data (pd.DataFrame): DataFrame mit Spalten [multiple_column, ...]
            multiple_column (str): Name des zu analysierenden Multiples (z. B. 'Price_TangibleBookValue')
            decimals (int): Anzahl der Dezimalstellen für die Rundung der Grenzwerte (default: 2)

        Returns:
            dict: Enthält die 12 Bereiche (Grenzwerte), die Bereichspaare und das analysierte Multiple

        Raises:
            ValueError: Wenn data kein DataFrame ist, multiple_column nicht existiert oder keine gültigen Daten vorliegen
        """

        # Fehlerbehandlung: Prüfe, ob data ein DataFrame ist
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Der Parameter 'data' muss ein pandas DataFrame sein.")

        # Fehlerbehandlung: Prüfe, ob multiple_column existiert
        if multiple_column not in data.columns:
            raise ValueError(f"Die Spalte '{multiple_column}' existiert nicht im DataFrame.")

        # Fehlerbehandlung: Prüfe auf fehlende Werte
        if data[multiple_column].isnull().all():
            raise ValueError(f"Keine gültigen Daten in der Spalte '{multiple_column}'.")

        # Entferne fehlende Werte für die Berechnung
        valid_data = data[multiple_column].dropna()
        if valid_data.empty:
            raise ValueError(f"Keine gültigen Daten nach Entfernen von NaN-Werten in '{multiple_column}'.")

        # Höchstwert und Tiefstwert des Multiples
        max_value = valid_data.max()
        min_value = valid_data.min()

        # Fehlerbehandlung: Prüfe auf identische Werte (keine Differenz)
        if max_value == min_value:
            raise ValueError(f"Kein Bereich möglich: Höchstwert und Tiefstwert von '{multiple_column}' sind identisch.")

        # Differenz und Intervallgröße für 12 Bereiche
        difference = max_value - min_value
        interval_size = difference / 12

        # Bereiche als Liste von Grenzwerten (12 Intervalle, beginnend bei min_value)
        percentiles = [round(min_value + i * interval_size, decimals) for i in
                       range(13)]  # 13 Grenzwerte für 12 Bereiche

        # Füge 0.0 als untersten Grenzwert hinzu
        percentiles = [0.0] + percentiles

        # Bereichspaare für Verweildauer-Analyse (inklusive der untersten Range)
        ranges = [(percentiles[i], percentiles[i + 1]) for i in range(13)]

        return {
            'multiple': multiple_column,
            'percentiles': percentiles,
            'ranges': ranges
        }

    def calculate_DurationInRange(self, data: pd.DataFrame, multiple_column: str, ranges: list) -> dict:
        """
        Zählt die Tage, an denen der Wert des Multiples in einer der angegebenen Ranges liegt,
        und berechnet den Anteil der Verweildauer an der Gesamtdauer.

        Args:
            self: Instanz der Model-Klasse
            data (pd.DataFrame): DataFrame mit der Spalte [multiple_column]
            multiple_column (str): Name des zu analysierenden Multiples (z. B. 'EV_EBITDA')
            ranges (list): Liste von Tupeln mit den Grenzwerten der 12 Ranges (aus calculate_percentiles)

        Returns:
            dict: Dictionary mit den Ranges als Schlüssel und Werten für Anzahl der Tage und Anteile

        Raises:
            ValueError: Wenn data kein DataFrame ist, multiple_column nicht existiert, keine gültigen Daten vorliegen,
                        oder die Ranges nicht konsistent sind (untere Grenze >= obere Grenze).
        """
        # Fehlerbehandlung: Prüfe, ob data ein DataFrame ist
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Der Parameter 'data' muss ein pandas DataFrame sein.")

        # Fehlerbehandlung: Prüfe, ob multiple_column existiert
        if multiple_column not in data.columns:
            raise ValueError(f"Die Spalte '{multiple_column}' existiert nicht im DataFrame.")

        # Fehlerbehandlung: Prüfe auf fehlende Werte
        if data[multiple_column].isnull().all():
            raise ValueError(f"Keine gültigen Daten in der Spalte '{multiple_column}'.")

        # Entferne fehlende Werte für die Berechnung
        valid_data = data[multiple_column].dropna()
        if valid_data.empty:
            raise ValueError(f"Keine gültigen Daten nach Entfernen von NaN-Werten in '{multiple_column}'.")

        # Validierung der Ranges: Prüfe, ob untere Grenze < obere Grenze (außer letzter Bereich)
        if not all(r[0] < r[1] for r in ranges[:-1]) or ranges[-1][0] >= ranges[-1][1]:
            raise ValueError("Die Ranges sind inkonsistent: Untere Grenze muss kleiner als obere Grenze sein.")
        # Zusätzlicher Check auf Lücken zwischen Bereichen
        for i in range(len(ranges) - 1):
            if ranges[i][1] != ranges[i + 1][0]:
                raise ValueError(f"Lücke zwischen Bereichen: {ranges[i][1]} != {ranges[i + 1][0]}")

        # Definiere Bin-Grenzen (inklusive der oberen Grenze des letzten Bereichs)
        bins = [r[0] for r in ranges] + [ranges[-1][1]]
        range_labels = [f"{r[0]:.3f}-{r[1]:.3f}" for r in ranges]

        # Weise Werte den Ranges zu und zähle mit pd.cut
        duration_series = pd.cut(valid_data, bins=bins, labels=range_labels, include_lowest=True)
        duration_counts = duration_series.value_counts().reindex(range_labels, fill_value=0).to_dict()

        # Berechne Gesamtdauer und Anteile
        total_days = len(valid_data)
        if total_days < 2:
            raise ValueError("Zu wenige Datenpunkte für sinnvolle Anteile.")
        duration_shares = {key: value / total_days for key, value in duration_counts.items()}

        # Kombiniere Ergebnisse in einem Dictionary
        result = {
            "counts": duration_counts,
            "shares": duration_shares,
            "total_days": total_days
        }

        return result

    def calculate_probability(self, data: pd.DataFrame, multiple_column: str, ranges: list) -> dict:
        """
        Berechnet die kumulierte Wahrscheinlichkeit für einen Abstieg eines Multiples basierend auf der Verweildauer.

        Args:
            self: Instanz der Model-Klasse
            data (pd.DataFrame): DataFrame mit der Spalte [multiple_column]
            multiple_column (str): Name des zu analysierenden Multiples (z. B. 'EV_EBITDA')
            ranges (list): Liste von Tupeln mit den Grenzwerten der 12 Ranges (aus calculate_percentiles)

        Returns:
            dict: Dictionary mit Ranges, probability_down und probability_up

        Raises:
            ValueError: Wenn data kein DataFrame ist, multiple_column nicht existiert, oder keine gültigen Daten vorliegen.
        """
        # Fehlerbehandlung
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Der Parameter 'data' muss ein pandas DataFrame sein.")
        if multiple_column not in data.columns:
            raise ValueError(f"Die Spalte '{multiple_column}' existiert nicht im DataFrame.")
        if data[multiple_column].isnull().all():
            raise ValueError(f"Keine gültigen Daten in der Spalte '{multiple_column}'.")

        # Entferne fehlende Werte
        valid_data = data[multiple_column].dropna()
        if valid_data.empty:
            raise ValueError(f"Keine gültigen Daten nach Entfernen von NaN-Werten in '{multiple_column}'.")

        # Berechne DurationInRange
        duration_result = self.calculate_DurationInRange(data, multiple_column, ranges)
        shares = duration_result["shares"]

        # Definiere Range-Labels
        range_labels = [f"{r[0]:.3f}-{r[1]:.3f}" for r in ranges]

        # Berechne kumulierte Wahrscheinlichkeiten
        cumulative_down = 0.0
        probability_down = {}
        for label in range_labels:
            share = shares.get(label, 0)
            if share > 0:
                cumulative_down += share
            probability_down[label] = cumulative_down

        # Berechne umgekehrte kumulierte Wahrscheinlichkeit (probability_up)
        probability_up = {label: 1.0 - probability_down[label] for label in range_labels}

        return {
            "ranges": range_labels,
            "probability_down": probability_down,
            "probability_up": probability_up
        }

    def calculate_crv(self, symbol: str, historical_data: pd.DataFrame):
        """
        Berechnet das konservative Chancen-Risiko-Verhältnis (CRV) aus Kurszielen
        und dem aktuellen Kurs.

        CRV = Upside_konservativ / Downside
        Upside_konservativ = FV_Kurs - aktueller_Kurs
        Downside           = aktueller_Kurs - WC_Kurs
        """
        # -------- 0) Eingaben prüfen --------
        if not isinstance(historical_data, pd.DataFrame) or historical_data.empty:
            return {"error": "Ungültige oder leere historische Daten"}


        # -------- 1) Multiple-Typ erkennen & Kursziele ermitteln --------
        has_price_multiple = any(col in historical_data.columns for col in Model.PRICE_MULTIPLE_COLUMNS)
        has_ev_multiple = any(col in historical_data.columns for col in Model.EV_MULTIPLE_COLUMNS)

        course_targets = None  # erwartetes Format: {"WC": x, "BUY": y, "FV": z, "SELL": w}

        if has_price_multiple:
            price_targets = self.calculate_course_target_PriceMultiples(historical_data, symbol)
            if price_targets is None or (isinstance(price_targets, dict) and "error" in price_targets):
                return price_targets if isinstance(price_targets, dict) else {
                    "error": "Kursziel-Berechnung (Price) fehlgeschlagen"}
            try:
                course_targets = {
                    "WC": float(price_targets["worst_case_price"]),
                    "BUY": float(price_targets["buy_price"]),
                    "FV": float(price_targets["fair_value_price"]),
                    "SELL": float(price_targets["sell_price"]),
                }
            except Exception:
                return {"error": "Unerwartetes Format der Price-Kursziele"}

        elif has_ev_multiple:
            ev_targets = self.calculate_course_target_EVMultiples(symbol, historical_data)
            if ev_targets is None or (isinstance(ev_targets, dict) and "error" in ev_targets):
                return ev_targets if isinstance(ev_targets, dict) else {
                    "error": "Kursziel-Berechnung (EV) fehlgeschlagen"}
            try:
                ev_key = next(k for k in Model.EV_MULTIPLE_COLUMNS if k in ev_targets)
                scen = ev_targets[ev_key]
                course_targets = {
                    "WC": float(scen["WC"]),
                    "BUY": float(scen["BUY"]),
                    "FV": float(scen["FV"]),
                    "SELL": float(scen["SELL"]),
                }
            except Exception:
                return {"error": "Unerwartetes Format der EV-Kursziele"}
        else:
            return {"error": "Kein unterstütztes Multiple im Datensatz gefunden."}

        # -------- 2) Aktuellen Kurs holen --------
        current_price = self.dataloader.get_current_price_per_share(symbol)
        if isinstance(current_price, dict) and "error" in current_price:
            return current_price
        try:
            current_price = float(current_price)
        except Exception:
            return {"error": f"Ungültiger aktueller Kurs: {current_price}"}
        if not np.isfinite(current_price) or current_price <= 0:
            return {"error": f"Ungültiger aktueller Kurs: {current_price}"}

        # -------- 3) Konservative und aggressive CRV-Berechnung --------
        wc_price = course_targets["WC"]
        fv_price = course_targets["FV"]
        sell_price = course_targets["SELL"]

        if current_price < wc_price:
            downside = current_price
        else:
            downside = current_price - wc_price
        upside_conservative = fv_price - current_price
        upside_aggressive = sell_price - current_price

        upside_conservative = max(upside_conservative, 0.0)  # kein negatives Aufwärtspotenzial
        upside_aggressive = max(upside_aggressive, 0.0)  # kein negatives Aufwärtspotenzial

        if downside <= 0:
            return {
                "error": "Downside ≤ 0 – CRV nicht definiert (aktueller Kurs ≤ Worst-Case-Kurs).",
                "inputs": {
                    "current_price": current_price,
                    "wc_price": wc_price,
                    "fv_price": fv_price,
                    "sell_price": sell_price
                },
                "course_targets": course_targets
            }

        crv_conservative = round(upside_conservative / downside, 2)
        crv_aggressive = round(upside_aggressive / downside, 2)

        return {
            "crv_conservative": crv_conservative,  # konservativ (basierend auf FV)
            "crv_aggressive": crv_aggressive,  # aggressiv (basierend auf SELL)
            "inputs": {
                "current_price": current_price,
                "downside": round(downside, 2),
                "upside_conservative": round(upside_conservative, 2),
                "upside_aggressive": round(upside_aggressive, 2),
            },
            "course_targets": course_targets
        }

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def evaluate_tbv_bandwidth(self, symbol: str, min_years: float = 10.0, use_cache: bool = True) -> dict:
        """
        Bewertet eine Aktie anhand des materiellen Buchwerts (TBV) nach dem
        Bandbreiten/Regression-to-the-Mean-Ansatz aus der Quelle.

        - Historie prüfen (>= min_years)
        - Kaufzonen:   P/TBV in [1.0, 1.5]
        - Überbewertung: P/TBV in [3.0, 4.0]
        - Touches: Wie oft lag P/TBV ~ 1 (hier: in [0.9, 1.1])
        - Zielpreise: WC = 0.9×TBV/Share, BUY = 1.15×TBV/Share, SELL = 3×TBV/Share
        - Signal: buy/neutral/sell (Warnung bei negativem TBV)

        Returns:
            dict mit Kennzahlen, Zonen, Zielpreisen und aktuellem Status oder einem "error"-Feld.
        """
        cache_key = f"tbv_bandwidth_eval_{symbol}"
        if use_cache:
            cached = self.dataloader._load_cached_data(symbol, cache_key)
            if isinstance(cached, dict) and "symbol" in cached and "targets" in cached and "current" in cached:
                # Re-hydrate pb DataFrame, falls ausgelagert gespeichert
                if isinstance(cached.get("pb"), dict) and "index" in cached["pb"]:
                    cached["pb"] = pd.DataFrame({
                        "Price_TangibleBookValue": cached["pb"]["Price_TangibleBookValue"],
                        "Price": cached["pb"]["Price"],
                        "TangibleBookValue": cached["pb"]["TangibleBookValue"],
                        "commonStockSharesOutstanding": cached["pb"]["commonStockSharesOutstanding"],
                    }, index=pd.to_datetime(cached["pb"]["index"]))
                return cached

        # 1) Historische P/TBV-Daten
        pb_df = self.calculate_historical_price_to_TangibleBookValue(symbol, use_cache=use_cache)
        if pb_df is None or pb_df.empty:
            return {"error": f"Keine Price/TangibleBookValue-Daten für {symbol} verfügbar"}

        span_years = (pb_df.index.max() - pb_df.index.min()).days / 365.25
        if span_years < min_years:
            return {"error": f"Unzureichende Historie: {span_years:.2f} Jahre < {min_years:.2f} Jahre"}

        # 2) Zonen auf Basis historischer P/TBV-Reihe (negative Werte ignorieren)
        s = pb_df["Price_TangibleBookValue"].astype(float)
        s = s[~s.isna() & (s >= 0)]

        buy_mask = (s >= 0.0) & (s <= 1.5)
        sell_mask = (s >= 3.0)
        touch_mask = (s >= 0.9) & (s <= 1.1)  # "Preis trifft Buchwert"

        buy_zones = [{"date": d.strftime("%Y-%m-%d"),
                      "price": float(pb_df.loc[d, "Price"]),
                      "pb_ratio": float(s.loc[d])}
                     for d in s.index[buy_mask]]

        sell_zones = [{"date": d.strftime("%Y-%m-%d"),
                       "price": float(pb_df.loc[d, "Price"]),
                       "pb_ratio": float(s.loc[d])}
                      for d in s.index[sell_mask]]

        touch_count = int(touch_mask.sum())  # wie oft ~1× TBV

        # 3) Aktuelle TBV/Share, Kurs & P/TBV
        bs = self.dataloader.get_balance_sheet(symbol, frequency="quarterly", use_cache=use_cache)
        if isinstance(bs, dict) and "error" in bs:
            return {"error": bs["error"]}

        tbv_total = self.get_tangible_book_value(bs)
        shares = self.dataloader.get_shares_outstanding(symbol)
        if isinstance(shares, dict) and "error" in shares:
            return {"error": shares["error"]}

        tbv_per_share = (tbv_total / shares) if shares else 0.0

        price_df = self.dataloader.get_max_historical_stock_data(symbol, interval="1d", use_cache=use_cache)
        if price_df is None or price_df.empty:
            return {"error": f"Keine aktuellen Kursdaten für {symbol} verfügbar"}
        current_price = float(price_df["Close"].iloc[-1])

        current_pb = float("inf") if tbv_per_share == 0 else round(current_price / tbv_per_share, 2)

        # 4) Zielpreise & Signal gemäß Text
        if tbv_per_share <= 0:
            targets = {"WC": 0.0, "BUY": 0.0, "SELL": 0.0}
            signal = "warning"
            message = (f"Warnung: Negativer/Null-Buchwert pro Aktie (TBV/Aktie={tbv_per_share:.2f}). "
                       "P/TBV nicht sinnvoll interpretierbar.")
        else:
            targets = {
                "WC": round(0.90 * tbv_per_share, 2),
                "BUY": round(1.15 * tbv_per_share, 2),
                "SELL": round(3.00 * tbv_per_share, 2),
            }
            if current_pb >= 3.0:
                signal = "sell"
                message = f"Aktie in überbewerteter Zone (P/TBV={current_pb} ≥ 3)."
            elif current_pb <= 1.5:
                signal = "buy"
                message = f"Aktie in Wert-Zone (P/TBV={current_pb} ∈ [1.0, 1.5])."
            else:
                signal = "neutral"
                message = "Außerhalb klarer Kauf-/Verkaufszonen."

        result = {
            "symbol": symbol,
            "pb": pb_df,  # kompletter historischer Kontext
            "zones": {
                "buy_zone_points": buy_zones,
                "sell_zone_points": sell_zones,
                "touches_tbv≈1x": touch_count,
            },
            "targets": targets,  # WC/BUY/SELL-Preisziele (pro Aktie)
            "current": {
                "price": round(current_price, 4),
                "tbv_per_share": round(float(tbv_per_share), 4),
                "pb_ratio": current_pb,
                "shares_outstanding": int(shares) if shares else None,
            },
            "signal": signal,
            "message": message,
            "meta": {
                "history_years": round(span_years, 2),
                "rules": {
                    "zone_point_filters": {
                    "value_zone_pb_min": 0.0,
                    "value_zone_pb_max": 1.5,
                    "overvaluation_min": 3.0
                },
                "touch_band_pb": [0.9, 1.1],
                "targets_pb_multiples": {"WC": 0.90, "BUY": 1.15, "SELL": 3.00},
            },
        }

        }

        if use_cache:
            # kompakt cachen (pb als Listen)
            cache_payload = result.copy()
            cache_payload["pb"] = {
                "Price_TangibleBookValue": pb_df["Price_TangibleBookValue"].tolist(),
                "Price": pb_df["Price"].tolist(),
                "TangibleBookValue": pb_df["TangibleBookValue"].tolist(),
                "commonStockSharesOutstanding": pb_df["commonStockSharesOutstanding"].tolist(),
                "index": pb_df.index.strftime("%Y-%m-%d").tolist(),
            }
            self.dataloader._cache_data(cache_payload, symbol, cache_key)

        return result

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def evaluate_ebit_bandwidth(self, symbol: str, min_years: float = 10.0, use_cache: bool = True) -> dict:
        """
        Bewertet eine Aktie anhand des Price/EBIT-Multiples nach dem Bandbreiten/Regression-to-the-Mean-Ansatz.

        - Historie prüfen (>= min_years)
        - Kaufzonen: P/EBIT in [6.0, 10.0] (typisch für stabile Unternehmen)
        - Überbewertung: P/EBIT in [20.0, 25.0]
        - Touches: Wie oft lag P/EBIT ~ 8 (hier: in [7.5, 8.5])
        - Zielpreise: WC = 7.5×EBIT/Share, BUY = 8.5×EBIT/Share, SELL = 22×EBIT/Share
        - Signal: buy/neutral/sell (Warnung bei negativem EBIT)

        Returns:
            dict mit Kennzahlen, Zonen, Zielpreisen und aktuellem Status oder einem "error"-Feld.
        """
        cache_key = f"ebit_bandwidth_eval_{symbol}"
        if use_cache:
            cached = self.dataloader._load_cached_data(symbol, cache_key)
            if isinstance(cached, dict) and "symbol" in cached and "targets" in cached and "current" in cached:
                # Re-hydrate ebit DataFrame, falls ausgelagert gespeichert
                if isinstance(cached.get("ebit"), dict) and "index" in cached["ebit"]:
                    cached["ebit"] = pd.DataFrame({
                        "Price_EBIT": cached["ebit"]["Price_EBIT"],
                        "Price": cached["ebit"]["Price"],
                        "EBIT": cached["ebit"]["EBIT"],
                    }, index=pd.to_datetime(cached["ebit"]["index"]))
                return cached

        # 1) Historische P/EBIT-Daten
        ebit_df = self.calculate_historical_price_to_ebit(symbol, use_cache=use_cache)
        if ebit_df is None or ebit_df.empty:
            return {"error": f"Keine Price/EBIT-Daten für {symbol} verfügbar"}

        span_years = (ebit_df.index.max() - ebit_df.index.min()).days / 365.25
        if span_years < min_years:
            return {"error": f"Unzureichende Historie: {span_years:.2f} Jahre < {min_years:.2f} Jahre"}

        # 2) Zonen auf Basis historischer P/EBIT-Reihe (negative Werte ignorieren)
        s = ebit_df["Price_EBIT"].astype(float)
        s = s[~s.isna() & (s >= 0)]

        buy_mask = (s >= 6.0) & (s <= 10.0)
        sell_mask = (s >= 20.0)
        touch_mask = (s >= 7.5) & (s <= 8.5)  # "Preis trifft typisches EBIT-Multiple"

        buy_zones = [{"date": d.strftime("%Y-%m-%d"),
                      "price": float(ebit_df.loc[d, "Price"]),
                      "ebit_ratio": float(s.loc[d])}
                     for d in s.index[buy_mask]]

        sell_zones = [{"date": d.strftime("%Y-%m-%d"),
                       "price": float(ebit_df.loc[d, "Price"]),
                       "ebit_ratio": float(s.loc[d])}
                      for d in s.index[sell_mask]]

        touch_count = int(touch_mask.sum())  # wie oft ~8× EBIT

        # 3) Aktuelle EBIT/Share, Kurs & P/EBIT
        ebit_data = self.dataloader.get_ebit_data(symbol, use_cache=use_cache, frequency="quarterly")
        if isinstance(ebit_data, dict) and "error" in ebit_data:
            return {"error": ebit_data["error"]}

        ebit_total = ebit_data["ebit"]
        shares = self.dataloader.get_shares_outstanding(symbol)
        if isinstance(shares, dict) and "error" in shares:
            return {"error": shares["error"]}

        ebit_per_share = (ebit_total / shares) if shares else 0.0

        price_df = self.dataloader.get_max_historical_stock_data(symbol, interval="1d", use_cache=use_cache)
        if price_df is None or price_df.empty:
            return {"error": f"Keine aktuellen Kursdaten für {symbol} verfügbar"}
        current_price = float(price_df["Close"].iloc[-1])

        current_ebit = float("inf") if ebit_per_share == 0 else round(current_price / ebit_per_share, 2)

        # 4) Zielpreise & Signal gemäß Text
        if ebit_per_share <= 0:
            targets = {"WC": 0.0, "BUY": 0.0, "SELL": 0.0}
            signal = "warning"
            message = (f"Warnung: Negativer/Null-EBIT pro Aktie (EBIT/Aktie={ebit_per_share:.2f}). "
                       "P/EBIT nicht sinnvoll interpretierbar.")
        else:
            targets = {
                "WC": round(7.5 * ebit_per_share, 2),
                "BUY": round(8.5 * ebit_per_share, 2),
                "SELL": round(22.0 * ebit_per_share, 2),
            }
            if current_ebit >= 20.0:
                signal = "sell"
                message = f"Aktie in überbewerteter Zone (P/EBIT={current_ebit} ≥ 20)."
            elif current_ebit <= 10.0:
                signal = "buy"
                message = f"Aktie in Wert-Zone (P/EBIT={current_ebit} ∈ [6.0, 10.0])."
            else:
                signal = "neutral"
                message = "Außerhalb klarer Kauf-/Verkaufszonen."

        result = {
            "symbol": symbol,
            "ebit": ebit_df,  # kompletter historischer Kontext
            "zones": {
                "buy_zone_points": buy_zones,
                "sell_zone_points": sell_zones,
                "touches_ebit≈8x": touch_count,
            },
            "targets": targets,  # WC/BUY/SELL-Preisziele (pro Aktie)
            "current": {
                "price": round(current_price, 4),
                "ebit_per_share": round(float(ebit_per_share), 4),
                "ebit_ratio": current_ebit,
                "shares_outstanding": int(shares) if shares else None,
            },
            "signal": signal,
            "message": message,
            "meta": {
                "history_years": round(span_years, 2),
                "rules": {
                    "zone_point_filters": {
                        "value_zone_ebit_min": 6.0,
                        "value_zone_ebit_max": 10.0,
                        "overvaluation_min": 20.0
                    },
                    "touch_band_ebit": [7.5, 8.5],
                    "targets_ebit_multiples": {"WC": 7.5, "BUY": 8.5, "SELL": 22.0},
                },
            },
        }

        if use_cache:
            # kompakt cachen (ebit als Listen)
            cache_payload = result.copy()
            cache_payload["ebit"] = {
                "Price_EBIT": ebit_df["Price_EBIT"].tolist(),
                "Price": ebit_df["Price"].tolist(),
                "EBIT": ebit_df["EBIT"].tolist(),
                "index": ebit_df.index.strftime("%Y-%m-%d").tolist(),
            }
            self.dataloader._cache_data(cache_payload, symbol, cache_key)

        return result


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def calculate_ev_to_ebit(self, symbol: str, use_cache: bool = True, frequency: str = "annual"):
        """
        Berechnet das EV/EBIT-Verhältnis (Enterprise Value / EBIT).
        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (Standard: True).
            frequency (str): 'annual' für jährliche Daten, 'quarterly' für quartalsweise Daten (Standard: 'annual').
        Returns:
            dict: Enthält das EV/EBIT-Verhältnis, Symbol, Frequenz, Datum und ggf. eine Nachricht.
                  Beispiel:
                  {
                      "ev_to_ebit": 15.0,
                      "symbol": "AAPL",
                      "frequency": "annual",
                      "date": "2024-12-31"
                  }
                  oder bei ungültigem EBIT:
                  {
                      "ev_to_ebit": "inf",
                      "symbol": "AAPL",
                      "frequency": "annual",
                      "date": "2024-12-31",
                      "message": "EV/EBIT ist unendlich aufgrund eines null oder negativen EBIT."
                  }
                  oder bei Fehler:
                  {
                      "error": "Fehlerbeschreibung",
                      "symbol": "AAPL"
                  }
        """
        cache_key = f"{symbol}_ev_to_ebit_{frequency}"
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None and "error" not in cached_data:
                return cached_data

        try:
            # Prüfen, ob Frequenz gültig ist
            if frequency not in ["annual", "quarterly"]:
                return {"error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.",
                        "symbol": symbol}

            # Enterprise Value abrufen
            ev_data = self.dataloader.get_enterprise_value(symbol, use_cache=use_cache, frequency=frequency)
            if "error" in ev_data:
                return {"error": ev_data["error"], "symbol": symbol}
            enterprise_value = ev_data["enterprise_value"]
            if not isinstance(enterprise_value, (int, float)) or enterprise_value < 0:
                return {"error": f"Ungültiger Enterprise Value für {symbol}: {enterprise_value}", "symbol": symbol}

            # EBIT abrufen
            ebit_data = self.dataloader.get_ebit_data(symbol, use_cache=use_cache, frequency=frequency)
            if "error" in ebit_data:
                return {"error": ebit_data["error"], "symbol": symbol}
            ebit = ebit_data["ebit"]
            date = ebit_data.get("date")
            if not isinstance(ebit, (int, float)):
                return {"error": f"Ungültiger EBIT-Wert für {symbol}: {ebit}", "symbol": symbol}

            # Prüfen auf ungültigen EBIT
            if ebit <= 0:
                data = {
                    "ev_to_ebit": "inf",
                    "symbol": symbol,
                    "frequency": frequency,
                    "date": date,
                    "message": "EV/EBIT ist unendlich aufgrund eines null oder negativen EBIT."
                }
                if use_cache:
                    self.dataloader._cache_data(data, symbol, cache_key)
                return data

            # EV/EBIT berechnen
            ev_to_ebit = round(enterprise_value / ebit, 2)
            data = {
                "ev_to_ebit": ev_to_ebit,
                "symbol": symbol,
                "frequency": frequency,
                "date": date
            }
            if use_cache:
                self.dataloader._cache_data(data, symbol, cache_key)
            return data

        except Exception as e:
            return {"error": f"Fehler beim Berechnen von EV/EBIT für {symbol} ({frequency}): {str(e)}",
                    "symbol": symbol}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def calculate_ev_to_ebitda(self, symbol: str, use_cache: bool = True, frequency: str = "annual") -> dict:
        """
        Berechnet das EV/EBITDA-Verhältnis (Enterprise Value / EBITDA).
        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (Standard: True).
            frequency (str): 'annual' für jährliche Daten, 'quarterly' für quartalsweise Daten (Standard: 'annual').
        Returns:
            dict: Enthält das EV/EBITDA-Verhältnis, Symbol, Frequenz, Datum und ggf. eine Nachricht.
        """
        cache_key = f"{symbol}_ev_to_ebitda_{frequency}"
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None and "error" not in cached_data:
                return cached_data

        try:
            if frequency not in ["annual", "quarterly"]:
                return {"error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.",
                        "symbol": symbol}

            # Enterprise Value abrufen
            ev_data = self.dataloader.get_enterprise_value(symbol, use_cache=use_cache, frequency=frequency)
            if "error" in ev_data:
                return {"error": ev_data["error"], "symbol": symbol}
            enterprise_value = ev_data["enterprise_value"]
            if not isinstance(enterprise_value, (int, float)) or enterprise_value < 0:
                return {"error": f"Ungültiger Enterprise Value für {symbol}: {enterprise_value}", "symbol": symbol}

            # EBITDA + Datum aus stock_financials abrufen
            financials = self.dataloader.get_stock_financials(symbol, frequency=frequency)
            if isinstance(financials, dict) and "error" in financials:
                return {"error": financials["error"], "symbol": symbol}

            if "EBITDA" not in financials.index:
                return {"error": f"Keine EBITDA-Daten für {symbol} ({frequency}) gefunden.", "symbol": symbol}

            ebitda = financials.loc["EBITDA"].iloc[0]
            if not isinstance(ebitda, (int, float)):
                return {"error": f"Ungültiger EBITDA-Wert für {symbol}: {ebitda}", "symbol": symbol}

            # Datum aus Spalten (neueste Periode)
            date_col = financials.columns[0]
            date = str(date_col.date()) if hasattr(date_col, 'date') else str(date_col)

            # Prüfen auf ungültiges EBITDA
            if ebitda <= 0:
                data = {
                    "ev_to_ebitda": "inf",
                    "symbol": symbol,
                    "frequency": frequency,
                    "date": date,
                    "message": "EV/EBITDA ist unendlich aufgrund eines null oder negativen EBITDA."
                }
                if use_cache:
                    self.dataloader._cache_data(data, symbol, cache_key)
                return data

            # EV/EBITDA berechnen
            ev_to_ebitda = round(enterprise_value / ebitda, 2)
            data = {
                "ev_to_ebitda": ev_to_ebitda,
                "symbol": symbol,
                "frequency": frequency,
                "date": date
            }
            if use_cache:
                self.dataloader._cache_data(data, symbol, cache_key)
            return data

        except Exception as e:
            return {"error": f"Fehler beim Berechnen von EV/EBITDA für {symbol} ({frequency}): {str(e)}",
                    "symbol": symbol}


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def calculate_price_to_ebit(self, symbol: str, use_cache: bool = True, frequency: str = "annual"):
        """
        Berechnet das Price/EBIT-Verhältnis (Preis pro Aktie / EBIT pro Aktie).
        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            use_cache (bool): Ob Cache-Daten verwendet werden sollen (Standard: True).
            frequency (str): Zeitraum, entweder 'annual' oder 'quarterly' (Standard: 'quarterly').
        Returns:
            dict: Enthält das Price/EBIT-Verhältnis, Symbol, Frequenz, Datum und ggf. eine Nachricht.
                  Beispiel:
                  {
                      "price_to_ebit": 15.0,
                      "symbol": "AAPL",
                      "frequency": "quarterly",
                      "date": "2024-12-31"
                  }
                  oder bei ungültigem EBIT:
                  {
                      "price_to_ebit": "inf",
                      "symbol": "AAPL",
                      "frequency": "quarterly",
                      "date": "2024-12-31",
                      "message": "Price/EBIT ist unendlich aufgrund eines null oder negativen EBIT."
                  }
                  oder bei Fehler:
                  {
                      "error": "Fehlerbeschreibung",
                      "symbol": "AAPL"
                  }
        """
        cache_key = f"{symbol}_price_to_ebit_{frequency}"
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, cache_key)
            if cached_data is not None and "error" not in cached_data:
                return cached_data
        try:
            # Prüfen, ob Frequenz gültig ist
            if frequency not in ["annual", "quarterly"]:
                return {"error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.",
                        "symbol": symbol}
            # Aktuellen Preis pro Aktie abrufen
            price_data = self.dataloader.get_current_price_per_share(symbol)
            if isinstance(price_data, dict) and "error" in price_data:
                return {"error": price_data["error"], "symbol": symbol}
            current_price = price_data
            if not isinstance(current_price, (int, float)) or current_price <= 0:
                return {"error": f"Ungültiger aktueller Preis für {symbol}: {current_price}", "symbol": symbol}
            # EBIT-Daten abrufen
            ebit_data = self.dataloader.get_ebit_data(symbol, use_cache=use_cache, frequency=frequency)
            if "error" in ebit_data:
                return {"error": ebit_data["error"], "symbol": symbol}
            ebit = ebit_data["ebit"]
            date = ebit_data.get("date")
            if not isinstance(ebit, (int, float)):
                return {"error": f"Ungültiger EBIT-Wert für {symbol}: {ebit}", "symbol": symbol}
            # Prüfen auf ungültigen EBIT
            shares = self.dataloader.get_shares_outstanding(symbol)  # Entferne use_cache, wenn nicht unterstützt
            if isinstance(shares, dict) and "error" in shares:
                return {"error": shares["error"], "symbol": symbol}
            if shares is None or not isinstance(shares, (int, float)) or shares <= 0:
                return {"error": f"Ungültige Anzahl ausstehender Aktien für {symbol}: {shares}", "symbol": symbol}
            ebit_per_share = ebit / shares
            if ebit_per_share <= 0:
                data = {
                    "price_to_ebit": "inf",
                    "symbol": symbol,
                    "frequency": frequency,
                    "date": date,
                    "message": "Price/EBIT ist unendlich aufgrund eines null oder negativen EBIT."
                }
                if use_cache:
                    self.dataloader._cache_data(data, symbol, cache_key)
                return data
            # Price/EBIT berechnen
            price_to_ebit = round(current_price / ebit_per_share, 2)
            data = {
                "price_to_ebit": price_to_ebit,
                "symbol": symbol,
                "frequency": frequency,
                "date": date
            }
            if use_cache:
                self.dataloader._cache_data(data, symbol, cache_key)
            return data
        except Exception as e:
            return {"error": f"Fehler beim Berechnen von Price/EBIT für {symbol} ({frequency}): {str(e)}",
                    "symbol": symbol}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def calculate_ROIC(self, symbol: str, frequency: str = "annual", use_cache: bool = True) -> dict:
        """
        Berechnet die Rendite auf investiertes Kapital (ROIC) für ein gegebenes Aktiensymbol.

        Args:
            symbol (str): Aktiensymbol (z. B. 'KO').
            frequency (str): 'annual' für jährliche Daten (Standard: 'annual').
            use_cache (bool): Ob der Cache verwendet werden soll (Standard: True).

        Returns:
            dict: Enthält den ROIC in Prozent sowie zugrunde liegende Daten.
                  Beispiel:
                  {
                      "roic": 12.34,
                      "symbol": "KO",
                      "frequency": "annual",
                      "date": "2024-12-31",
                      "net_income": 987654321.0,
                      "invested_capital": 1234567890.0
                  }
                  oder bei Fehler:
                  {
                      "error": "Fehlerbeschreibung",
                      "symbol": symbol
                  }
        """
        if frequency not in ["annual", "quarterly"]:
            return {"error": f"Ungültige Frequenz: {frequency}. Nur 'annual' wird unterstützt.", "symbol": symbol}

        # Cache-Daten-Typ für den Schlüssel
        data_type = f"roic_{frequency}"

        # Versuche, Daten aus dem Cache zu laden
        if use_cache:
            cached_data = self.dataloader._load_cached_data(symbol, data_type)
            if cached_data is not None:
                return cached_data

        try:
            # Net Income aus Finanzdaten abrufen
            financials = self.dataloader.get_stock_financials(symbol, frequency)
            if isinstance(financials, dict) and "error" in financials:
                return financials

            if not isinstance(financials, pd.DataFrame) or financials.empty:
                raise ValueError(f"Keine Finanzdaten für {symbol} ({frequency}) gefunden.")

            net_income = None
            income_labels = [
                "Net Income Common Stockholders",
                "Net Income",
                "Net Income Applicable To Common Shares"
            ]
            for label in income_labels:
                if label in financials.index:
                    net_income = financials.loc[label].iloc[0]
                    if pd.notna(net_income):
                        break
            if net_income is None or pd.isna(net_income):
                return {
                    "error": f"Kein Net Income für {symbol} ({frequency}) gefunden. Verfügbare Labels: {list(financials.index)}",
                    "symbol": symbol
                }

            # Investiertes Kapital abrufen
            invested_capital_result = self.dataloader.get_invested_capital(symbol, frequency)
            if isinstance(invested_capital_result, dict) and "error" in invested_capital_result:
                return invested_capital_result

            invested_capital = invested_capital_result["invested_capital"]
            if invested_capital <= 0:
                return {
                    "error": f"Investiertes Kapital für {symbol} ({frequency}) ist <= 0: {invested_capital}",
                    "symbol": symbol
                }

            # ROIC berechnen
            roic = (net_income / invested_capital) * 100
            roic = round(roic, 2)

            # Datum extrahieren (aus Finanzdaten oder investiertem Kapital)
            latest_date = financials.columns[0]
            if not isinstance(latest_date, pd.Timestamp):
                latest_date = pd.to_datetime(latest_date)

            # Ergebnis-Dictionary erstellen
            result = {
                "roic": float(roic),
                "symbol": symbol,
                "frequency": frequency,
                "date": latest_date.strftime("%Y-%m-%d"),
                "net_income": float(net_income),
                "invested_capital": float(invested_capital)
            }

            # Ergebnis im Cache speichern
            if use_cache:
                self.dataloader._cache_data(result, symbol, data_type)

            return result

        except Exception as e:
            return {
                "error": f"Fehler beim Berechnen des ROIC für {symbol} ({frequency}): {str(e)}",
                "symbol": symbol
            }

    def print_balance_sheet(self, symbol: str, dates: Optional[list] = None) -> Optional[pd.DataFrame]:
        """
        Gibt das gesamte Balance Sheet für ein Aktiensymbol aus und speichert es als CSV.

        Args:
            symbol (str): Aktiensymbol (z. B. 'ILMN').
            dates (list): Liste von Daten (z. B. ['2015-03-31', '2005-03-31']) für die Ausgabe (optional).

        Returns:
            Optional[pd.DataFrame]: Balance Sheet DataFrame oder None bei Fehler.
        """
        try:
            # Fundamentaldaten abrufen
            fundamentals = self.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            if "error" in fundamentals:
                self.logger.error(f"Fehler bei Fundamentaldaten für {symbol}: {fundamentals['error']}")
                return None

            balance_sheet = fundamentals.get("balance_sheet")
            if balance_sheet is None:
                self.logger.error(f"Keine Balance Sheet-Daten für {symbol} verfügbar.")
                return None

            # Alle Spalten ausgeben
            self.logger.info(f"Balance Sheet Spalten für {symbol}: {balance_sheet.columns.tolist()}")
            print(f"\nBalance Sheet Spalten für {symbol}:")
            print(balance_sheet.columns.tolist())

            # Spezifische Datenpunkte ausgeben, falls angegeben
            if dates:
                balance_sheet_subset = balance_sheet.loc[balance_sheet.index.isin(pd.to_datetime(dates))]
                if not balance_sheet_subset.empty:
                    print(f"\nBalance Sheet für {symbol} an spezifischen Daten ({dates}):")
                    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                        print(balance_sheet_subset)
                else:
                    self.logger.warning(f"Keine Balance Sheet-Daten für {symbol} an den angegebenen Daten: {dates}")

            # Gesamtes Balance Sheet als CSV speichern
            balance_sheet.to_csv(f"{symbol}_balance_sheet.csv")
            self.logger.info(f"Balance Sheet für {symbol} als {symbol}_balance_sheet.csv gespeichert.")

            return balance_sheet

        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen des Balance Sheets für {symbol}: {e}")
            return None

    def identify_elliott_waves(self, symbol):
        """Identifiziert Elliott-Wellen-Muster (vereinfachte Logik)."""
        data = self.dataloader.get_stock_data(symbol)
        if "error" in data:
            return data
        data = self.preprocessor.preprocess_stock_data(data)
        if "error" in data:
            return data
        data = self.preprocessor.calculate_technical_indicators(data)
        if "error" in data:
            return data
        data = self.preprocessor.calculate_fibonacci_retracements(data)
        if "error" in data:
            return data
        data = self.preprocessor.identify_elliott_waves(data)
        if "error" in data:
            return data
        return data

    