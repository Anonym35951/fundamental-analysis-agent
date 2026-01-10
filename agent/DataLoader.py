import logging
from time import sleep
from typing import Optional, Dict
from io import StringIO
import pandas as pd
import yfinance as yf
import requests
import os
import json
from datetime import datetime, timedelta

from dateutil.relativedelta import relativedelta
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

class DataLoader:
    def __init__(self, user_agent="gecen.efe1308@gmail.com"):
        self.user_agent = user_agent
        self.ticker_cache = {}
        self.price_cache = {}
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.api_key = "UJN306OQ5D0F9M3I"  # Ersetze mit deinem Alpha Vantage API-Schlüssel --> muss neu Subscription anmelden für neuen Key
        self.base_url = "https://www.alphavantage.co/query?"
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_stock_data(self, symbol, period="20y", interval="1d"):
        """Ruft historische Kursdaten von Yahoo Finance ab."""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period, interval=interval)
            if hist.empty:
                raise ValueError(f"Keine historischen Daten für {symbol} gefunden.")
            # Resample für 4h-Chart, falls nötig
            if interval == "4h":
                hist = hist.resample("4H").agg({
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum"
                }).dropna()
            return hist
        except Exception as e:
            return {"error": f"Fehler beim Abrufen der Kursdaten für {symbol}: {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_stock_financials(self, symbol, frequency="annual", use_cache=True):
        """Ruft Finanzdaten eines Unternehmens von Yahoo Finance ab.
        frequency: 'annual' für jährliche Daten, 'quarterly' für quartalsweise Daten.
        """
        if use_cache:
            cached_data = self._load_cached_data(symbol, f"stock_financials_{frequency}")
            if cached_data is not None:
                if isinstance(cached_data, dict) and "error" in cached_data:
                    return cached_data
                if not isinstance(cached_data, pd.DataFrame):
                    return {"error": f"Cached-Daten für {symbol} ({frequency}) sind kein DataFrame: {str(cached_data)}",
                            "symbol": symbol}
                return cached_data

        try:
            stock = yf.Ticker(symbol)
            if frequency == "quarterly":
                financials = stock.quarterly_financials
            else:
                financials = stock.financials

            if not isinstance(financials, pd.DataFrame):
                raise ValueError(f"Finanzdaten für {symbol} ({frequency}) sind kein DataFrame: {str(financials)}")
            if financials.empty:
                raise ValueError(f"Keine Finanzdaten ({frequency}) für {symbol} gefunden.")
            if use_cache:
                self._cache_data(financials, symbol, f"stock_financials_{frequency}")
            return financials
        except Exception as e:
            return {"error": f"Fehler beim Abrufen der Finanzdaten für {symbol} ({frequency}): {str(e)}",
                    "symbol": symbol}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_invested_capital(self, symbol: str, frequency: str = "annual", use_cache: bool = True) -> dict:
        """
        Berechnet das investierte Kapital (Total Equity + Total Debt) für ein gegebenes Aktiensymbol.

        Args:
            symbol (str): Aktiensymbol (z. B. 'KO').
            frequency (str): 'annual' für jährliche Daten, 'quarterly' für quartalsweise Daten (Standard: 'annual').
            use_cache (bool): Ob der Cache verwendet werden soll (Standard: True).

        Returns:
            dict: Enthält das investierte Kapital in USD.
                  Beispiel:
                  {
                      "invested_capital": 1234567890.0,
                      "symbol": "KO",
                      "frequency": "annual",
                      "date": "2024-12-31",
                      "total_equity": 987654321.0,
                      "total_debt": 246913568.0
                  }
                  oder bei Fehler:
                  {
                      "error": "Fehlerbeschreibung",
                      "symbol": symbol
                  }
        """
        if frequency not in ["annual", "quarterly"]:
            return {"error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.", "symbol": symbol}

        data_type = f"invested_capital_{frequency}"

        if use_cache:
            cached_data = self._load_cached_data(symbol, data_type)
            if cached_data is not None:
                return cached_data

        try:
            balance_sheet = self.get_balance_sheet(symbol, frequency)
            if isinstance(balance_sheet, dict) and "error" in balance_sheet:
                return balance_sheet

            if not isinstance(balance_sheet, pd.DataFrame) or balance_sheet.empty:
                raise ValueError(f"Keine Bilanzdaten für {symbol} ({frequency}) gefunden.")

            total_equity = None
            equity_labels = [
                "Total Stockholders Equity",
                "Total Equity",
                "Shareholders Equity",
                "Common Stock Equity",
                "Total Common Equity"
            ]
            for label in equity_labels:
                if label in balance_sheet.index:
                    total_equity = balance_sheet.loc[label].iloc[0]
                    if pd.notna(total_equity):
                        break
            if total_equity is None or pd.isna(total_equity):
                return {
                    "error": f"Kein Eigenkapital für {symbol} ({frequency}) gefunden. Verfügbare Labels: {list(balance_sheet.index)}",
                    "symbol": symbol
                }

            total_debt = None
            debt_labels = [
                "Total Debt",
                "Long Term Debt",
                "Short Term Debt",
                "Current Debt",
                "Long Term Debt And Capital Lease Obligation",
                "Current Debt And Capital Lease Obligation"
            ]
            if "Total Debt" in balance_sheet.index:
                total_debt = balance_sheet.loc["Total Debt"].iloc[0]
            else:
                long_term_debt = 0.0
                short_term_debt = 0.0
                if "Long Term Debt" in balance_sheet.index and pd.notna(balance_sheet.loc["Long Term Debt"].iloc[0]):
                    long_term_debt = balance_sheet.loc["Long Term Debt"].iloc[0]
                elif "Long Term Debt And Capital Lease Obligation" in balance_sheet.index and pd.notna(
                        balance_sheet.loc["Long Term Debt And Capital Lease Obligation"].iloc[0]):
                    long_term_debt = balance_sheet.loc["Long Term Debt And Capital Lease Obligation"].iloc[0]
                if "Short Term Debt" in balance_sheet.index and pd.notna(balance_sheet.loc["Short Term Debt"].iloc[0]):
                    short_term_debt = balance_sheet.loc["Short Term Debt"].iloc[0]
                elif "Current Debt" in balance_sheet.index and pd.notna(balance_sheet.loc["Current Debt"].iloc[0]):
                    short_term_debt = balance_sheet.loc["Current Debt"].iloc[0]
                elif "Current Debt And Capital Lease Obligation" in balance_sheet.index and pd.notna(
                        balance_sheet.loc["Current Debt And Capital Lease Obligation"].iloc[0]):
                    short_term_debt = balance_sheet.loc["Current Debt And Capital Lease Obligation"].iloc[0]
                total_debt = long_term_debt + short_term_debt

            if total_debt is None or pd.isna(total_debt):
                total_debt = 0.0

            invested_capital = total_equity + total_debt
            invested_capital = round(invested_capital, 2)

            latest_date = balance_sheet.columns[0]
            if not isinstance(latest_date, pd.Timestamp):
                latest_date = pd.to_datetime(latest_date)

            result = {
                "invested_capital": float(invested_capital),
                "symbol": symbol,
                "frequency": frequency,
                "date": latest_date.strftime("%Y-%m-%d"),
                "total_equity": float(total_equity),
                "total_debt": float(total_debt)
            }

            if use_cache:
                self._cache_data(result, symbol, data_type)

            return result

        except Exception as e:
            return {
                "error": f"Fehler beim Berechnen des investierten Kapitals für {symbol} ({frequency}): {str(e)}",
                "symbol": symbol
            }


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_edgar_data(self, cik):
        """Ruft Finanzdaten von der SEC EDGAR-Datenbank ab."""
        try:
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            headers = {"User-Agent": self.user_agent}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Fehler beim Abrufen der EDGAR-Daten für CIK {cik}: {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_max_historical_stock_data(self, symbol: str, use_cache: bool = True, start_date: Optional[str] = None,
                                      end_date: Optional[str] = None, interval: str = "1mo") -> Optional[pd.DataFrame]:
        """
        Ruft historische Kursdaten für ein Aktiensymbol von Yahoo Finance ab.

        Args:
            symbol (str): Aktiensymbol (z. B. 'NVDA').
            use_cache (bool): Ob Cache verwendet werden soll (default: True).
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (optional).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (optional).
            interval (str): Datenintervall ('1d', '1wk', '1mo') (default: '1mo').

        Returns:
            Optional[pd.DataFrame]: DataFrame mit Kursdaten oder None bei Fehler.
        """
        if interval not in ["1d", "1wk", "1mo"]:
            self.logger.error(f"Ungültiges Intervall für {symbol}: {interval}. Erlaubt: '1d', '1wk', '1mo'.")
            raise ValueError(f"Ungültiges Intervall: {interval}. Erlaubt: '1d', '1wk', '1mo'.")

        cache_key = f"historical_stock_prices_{symbol}_{interval}"
        # Prüfe Cache zuerst
        if use_cache:
            cached_data = self._load_cached_data(symbol, cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache-Daten für {symbol} ({interval}) geladen.")
                return cached_data

        try:
            # Verwende yfinance für API-Aufruf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="max" if not start_date else None, start=start_date, end=end_date,
                                interval=interval)
            if df.empty:
                self.logger.warning(f"Keine Kursdaten für {symbol} ({interval}) verfügbar.")
                return None
            # Zeitzone entfernen und Index als DateTime setzen
            df.index = pd.to_datetime(df.index).tz_convert(None)
            # Nur benötigte Spalten behalten
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[[col for col in required_columns if col in df.columns]]
            # Cache die Daten
            self._cache_data(df, symbol, cache_key)
            self.logger.info(f"Kursdaten für {symbol} ({interval}) erfolgreich abgerufen und gecacht.")
            return df
        except ValueError as e:
            self.logger.error(f"Datumsfehler beim Abruf der Kursdaten für {symbol} ({interval}): {e}")
            return None
        except Exception as e:
            self.logger.error(f"API-Fehler beim Abruf der Kursdaten für {symbol} ({interval}): {e}")
            return None


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_fundamental_data(self, symbol: str, frequency: str = "annual", use_cache: bool = True,
                             start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Ruft historische Fundamentaldaten (Bilanz, GuV, Cashflow) für einen Ticker ab.

        Args:
            symbol (str): Ticker-Symbol (z. B. 'AAPL').
            frequency (str): 'annual' oder 'quarterly' (default: 'annual').
            use_cache (bool): Ob Cache verwendet werden soll (default: True).
            start_date (str): Startdatum im Format 'YYYY-MM-DD' (optional).
            end_date (str): Enddatum im Format 'YYYY-MM-DD' (optional).

        Returns:
            Dict[str, Optional[pd.DataFrame]]: Dictionary mit DataFrames für 'income_statement',
                                               'balance_sheet', 'cash_flow' oder None bei Fehler.
        """
        if not symbol or not isinstance(symbol, str):
            self.logger.error("Ungültiger Ticker-Symbol.")
            return {"error": "Ticker muss ein gültiger String sein.", "symbol": symbol}

        if frequency not in ["annual", "quarterly"]:
            self.logger.error(f"Ungültige Frequenz: {frequency}")
            return {"error": "Ungültige Frequenz", "symbol": symbol}

        # Validierung von Datumsformaten
        if start_date:
            try:
                datetime.strptime(start_date, '%Y-%m-%d')
            except ValueError:
                return {"error": "start_date muss im Format 'YYYY-MM-DD' sein.", "symbol": symbol}
        if end_date:
            try:
                datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                return {"error": "end_date muss im Format 'YYYY-MM-DD' sein.", "symbol": symbol}
        if start_date and end_date and datetime.strptime(end_date, '%Y-%m-%d') <= datetime.strptime(start_date, '%Y-%m-%d'):
            return {"error": "end_date muss nach start_date liegen.", "symbol": symbol}

        result = {
            "income_statement": None,
            "balance_sheet": None,
            "cash_flow": None
        }

        # API-Endpunkte und Cache-Schlüssel
        endpoints = [
            ("INCOME_STATEMENT", f"historical_income_statement_{frequency}",
             "annualReports" if frequency == "annual" else "quarterlyReports", "income_statement"),
            ("BALANCE_SHEET", f"historical_balance_sheet_{frequency}",
             "annualReports" if frequency == "annual" else "quarterlyReports", "balance_sheet"),
            ("CASH_FLOW", f"historical_cash_flow_{frequency}",
             "annualReports" if frequency == "annual" else "quarterlyReports", "cash_flow")
        ]

        for function, cache_key, data_key, result_key in endpoints:
            # Cache prüfen
            if use_cache:
                cached_data = self._load_cached_data(symbol, cache_key)
                if cached_data is not None:
                    self.logger.info(f"Daten aus Cache geladen: {cache_key}")
                    result[result_key] = cached_data if isinstance(cached_data, pd.DataFrame) else None
                    continue

            # API-Aufruf
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": self.api_key
            }

            try:
                self.logger.info(f"API-Aufruf: {function} für {symbol}")
                response = requests.get(self.base_url, params=params, headers={"User-Agent": self.user_agent})
                response.raise_for_status()
                data = response.json()

                # Detaillierte Fehlerbehandlung
                if "Error Message" in data:
                    self.logger.error(f"API-Fehler für {function}: {data['Error Message']}")
                    result[result_key] = None
                    continue
                if "Note" in data:
                    self.logger.warning(f"API-Limit erreicht: {data['Note']}")
                    return {"error": "API-Limit erreicht.", "symbol": symbol}
                if data_key not in data or not data[data_key]:
                    self.logger.warning(f"Keine {frequency} Daten für {symbol} unter {function} verfügbar. API-Antwort: {data}")
                    result[result_key] = None
                    continue

                # Daten verarbeiten
                df = pd.DataFrame(data[data_key])
                df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"])
                df.set_index("fiscalDateEnding", inplace=True)
                df = df.apply(pd.to_numeric, errors="coerce")
                # Zeitzonen entfernen
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
                if start_date:
                    df = df[df.index >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df.index <= pd.to_datetime(end_date)]
                if df.empty:
                    self.logger.warning(f"DataFrame für {function} ist leer nach Filterung.")
                    result[result_key] = None
                else:
                    result[result_key] = df
                    if use_cache:
                        self._cache_data(df, symbol, cache_key)

                sleep(12)  # Warte 12 Sekunden, um API-Limit (5/min) zu respektieren

            except Exception as e:
                self.logger.error(f"Fehler bei {function} für {symbol}: {e}. API-Antwort: {data if 'data' in locals() else 'Keine Antwort'}")
                result[result_key] = None
                continue

        if all(value is None for value in result.values()):
            self.logger.error(f"Keine Fundamentaldaten für {symbol} ({frequency}) abgerufen.")
            return {"error": f"Keine Fundamentaldaten für {symbol} ({frequency}) abgerufen.", "symbol": symbol}

        return result


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_peg_ratio(self, symbol, use_cache=True):
        """
        Ruft die PEG-Ratio eines Unternehmens von Yahoo Finance ab oder berechnet sie manuell,
        falls keine direkten Daten verfügbar sind.

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            use_cache (bool): Verwendung des Caches (Standard: True).

        Returns:
            dict: Enthält die PEG-Ratio, die Methode der Berechnung und ggf. zusätzliche Daten.
                  Beispiel:
                  {
                      "peg_ratio": 2.34,
                      "symbol": "AAPL",
                      "method": "direct"
                  }
                  oder bei Berechnung:
                  {
                      "peg_ratio": 4.0,
                      "symbol": "AAPL",
                      "method": "calculated_from_earnings_growth",
                      "trailing_pe": 31.24,
                      "earnings_growth": 7.8
                  }
                  oder bei Fehler:
                  {
                      "error": "Fehlerbeschreibung",
                      "symbol": "AAPL"
                  }
        """
        if use_cache:
            cached_data = self._load_cached_data(symbol, "peg_ratio")
            if cached_data is not None and "error" not in cached_data:
                return cached_data

        try:
            stock = yf.Ticker(symbol)

            # Schritt 1: Direkte PEG-Ratio abrufen
            peg_ratio = stock.info.get("pegRatio")
            if peg_ratio is not None and not pd.isna(peg_ratio):
                try:
                    peg_val = float(peg_ratio)
                except Exception:
                    peg_val = None

                if peg_val is not None and peg_val > 0:
                    data = {
                        "peg_ratio": round(peg_val, 2),
                        "symbol": symbol,
                        "method": "direct"
                    }
                    if use_cache:
                        self._cache_data(data, symbol, "peg_ratio")
                    return data

            # Schritt 2: Fallback auf Berechnung
            trailing_pe = stock.info.get("trailingPE")
            earnings_growth = stock.info.get("earningsGrowth")

            if (
                    trailing_pe is not None
                    and not pd.isna(trailing_pe)
                    and earnings_growth is not None
                    and not pd.isna(earnings_growth)
                    and earnings_growth > 0
            ):
                # Skaliere earningsGrowth von Dezimal (z. B. 0.078) zu Prozent (7.8)
                earnings_growth_percent = earnings_growth * 100
                calculated_peg = trailing_pe / earnings_growth_percent
                data = {
                    "peg_ratio": round(float(calculated_peg), 2),
                    "symbol": symbol,
                    "method": "calculated_from_earnings_growth",
                    "trailing_pe": round(float(trailing_pe), 2),
                    "earnings_growth": round(float(earnings_growth_percent), 2)
                }
                if use_cache:
                    self._cache_data(data, symbol, "peg_ratio")
                return data

            # Schritt 3: Keine ausreichenden Daten gefunden
            error = {
                "error": f"Keine PEG-Ratio-Daten für {symbol} verfügbar",
                "symbol": symbol
            }
            return error

        except Exception as e:
            error = {
                "error": f"Fehler beim Abrufen oder Berechnen der PEG-Ratio für {symbol}: {str(e)}",
                "symbol": symbol
            }
            return error

    def get_book_value(self, symbol):
        """Ruft den Buchwert des Unternehmens ab"""
        try:
            stock = yf.Ticker(symbol)
            book_value = stock.info.get("bookValue")
            if book_value is None:
                raise ValueError(f"Keine Daten zu ausstehenden Aktien für {symbol} gefunden.")
            return book_value
        except Exception as e:
            return {"error": f"Fehler beim Abrufen des Buchwerts für {symbol}: {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_current_price_per_share(self, symbol):
        """Gibt den aktuellen Preis einer Aktie in US-Dollar zurück."""
        try:
            stock = yf.Ticker(symbol)
            current_price = stock.info.get("regularMarketPrice")
            if current_price is None:
                hist = stock.history(period="1d")
                if hist.empty:
                    raise ValueError(f"Keine aktuellen Preisdaten für {symbol} gefunden.")
                current_price = hist["Close"].iloc[-1]
            return current_price
        except Exception as e:
            return {"error": f"Fehler beim Abrufen des Preises für {symbol}: {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_balance_sheet(self, symbol, frequency="annual", use_cache = True):
        """
        Ruft die Bilanzdaten eines Unternehmens ab.
        frequency: 'annual' für jährliche Daten, 'quarterly' für quartalsweise Daten.
        """
        if use_cache:
            cached_data = self._load_cached_data(symbol, f"balance_sheet_{frequency}")
            if cached_data is not None and "error" not in cached_data:
                return cached_data
        try:
            stock = yf.Ticker(symbol)
            if frequency == "quarterly":
                balance_sheet = stock.quarterly_balance_sheet
            else:
                balance_sheet = stock.balance_sheet

            if balance_sheet.empty:
                raise ValueError(f"Keine Bilanzdaten ({frequency}) für {symbol} gefunden.")
            if use_cache:
                self._cache_data(balance_sheet, symbol, f"balance_sheet_{frequency}")
            return balance_sheet
        except Exception as e:
            return {"error": f"Fehler beim Abrufen der Bilanzdaten für {symbol} ({frequency}): {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_market_cap(self, symbol: str, use_cache: bool = True) -> dict:
        """
        Ruft die aktuelle Marktkapitalisierung für ein gegebenes Aktiensymbol ab.

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            use_cache (bool): Ob der Cache verwendet werden soll (Standard: True).

        Returns:
            dict: Enthält die Marktkapitalisierung, Symbol und Datum.
                  Beispiel:
                  {
                      "market_cap": 1234567890000.0,
                      "symbol": "AAPL",
                      "date": "2025-10-13"
                  }
                  oder bei Fehler:
                  {
                      "error": "Fehlerbeschreibung",
                      "symbol": "AAPL"
                  }
        """
        # Cache-Daten-Typ für den Schlüssel
        data_type = "market_cap"

        # Versuche, Daten aus dem Cache zu laden
        if use_cache:
            cached_data = self._load_cached_data(symbol, data_type)
            if cached_data is not None and "error" not in cached_data:
                self.logger.info(f"Cache-Daten für {symbol} (market_cap) geladen.")
                return cached_data

        try:
            # Ticker-Objekt erstellen
            ticker = yf.Ticker(symbol)

            # Marktkapitalisierung direkt abrufen
            market_cap = ticker.info.get("marketCap")
            if market_cap is None or pd.isna(market_cap) or market_cap <= 0:
                # Fallback: Berechnung aus Aktienkurs und ausstehenden Aktien
                current_price = self.get_current_price_per_share(symbol)
                shares = self.get_shares_outstanding(symbol)
                if isinstance(current_price, dict) and "error" in current_price:
                    return {"error": current_price["error"], "symbol": symbol}
                if isinstance(shares, dict) and "error" in shares:
                    return {"error": shares["error"], "symbol": symbol}
                market_cap = current_price * shares
                if market_cap <= 0:
                    raise ValueError(f"Ungültige Marktkapitalisierung für {symbol}: {market_cap}")

            # Datum extrahieren (heutiges Datum, da es sich um aktuelle Daten handelt)
            latest_date = datetime.now().strftime("%Y-%m-%d")

            # Ergebnis-Dictionary erstellen
            result = {
                "market_cap": float(market_cap),
                "symbol": symbol,
                "date": latest_date
            }

            # Ergebnis im Cache speichern
            if use_cache:
                self._cache_data(result, symbol, data_type)
                self.logger.info(f"Marktkapitalisierung für {symbol} erfolgreich abgerufen und gecacht.")

            return result

        except Exception as e:
            error_msg = f"Fehler beim Abrufen der Marktkapitalisierung für {symbol}: {str(e)}"
            if "HTTP Error 404" in str(e):
                error_msg = f"Ungültiges Symbol: {symbol}. {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg, "symbol": symbol}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_cash_and_equivalents(self, symbol: str, frequency: str = "annual", use_cache: bool = True) -> dict:
        """
        Ruft die Cash & Cash Equivalents (bzw. nächstbeste Cash-Position) aus der Bilanz ab.

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL')
            frequency (str): 'annual' oder 'quarterly'
            use_cache (bool): Cache verwenden (default: True)

        Returns:
            dict:
                {
                  "cash_and_equivalents": float,
                  "symbol": symbol,
                  "frequency": frequency,
                  "label_used": "<balance-sheet-label>",
                  "date": "<latest-period>"
                }
            oder bei Fehler:
                {"error": "...", "symbol": symbol}
        """
        if frequency not in ["annual", "quarterly"]:
            return {"error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.", "symbol": symbol}

        cache_key = f"cash_and_equivalents_{frequency}"
        if use_cache:
            cached_data = self._load_cached_data(symbol, cache_key)
            if cached_data is not None and "error" not in cached_data:
                return cached_data

        try:
            balance_sheet = self.get_balance_sheet(symbol, frequency=frequency, use_cache=use_cache)
            if isinstance(balance_sheet, dict) and "error" in balance_sheet:
                return {"error": balance_sheet["error"], "symbol": symbol}

            if balance_sheet is None or getattr(balance_sheet, "empty", True):
                return {"error": f"Keine Bilanzdaten ({frequency}) für {symbol} verfügbar.", "symbol": symbol}

            # mögliche Labels (yfinance variiert je nach Company)
            cash_labels = [
                "Cash And Cash Equivalents",
                "Cash And Cash Equivalents And Short Term Investments",
                "Cash Cash Equivalents And Short Term Investments",
                "Cash And Short Term Investments",
                "Cash",
                "CashAndCashEquivalents",
            ]

            cash_value = None
            label_used = None

            for label in cash_labels:
                if label in balance_sheet.index:
                    cash_value = balance_sheet.loc[label].iloc[0]
                    label_used = label
                    break

            if cash_value is None:
                return {
                    "error": f"Keine Cash-Position für {symbol} ({frequency}) gefunden. "
                             f"Verfügbare Labels: {list(balance_sheet.index)}",
                    "symbol": symbol
                }

            if pd.isna(cash_value):
                return {"error": f"Ungültiger Cash-Wert für {symbol} ({frequency}): {cash_value}", "symbol": symbol}

            # Datum aus Spalten (neueste Periode)
            date_col = balance_sheet.columns[0] if len(balance_sheet.columns) > 0 else None
            date = str(date_col.date()) if hasattr(date_col, "date") else str(
                date_col) if date_col is not None else None

            result = {
                "cash_and_equivalents": float(cash_value),
                "symbol": symbol,
                "frequency": frequency,
                "label_used": label_used,
                "date": date
            }

            if use_cache:
                self._cache_data(result, symbol, cache_key)

            return result

        except Exception as e:
            return {"error": f"Fehler beim Abrufen von Cash & Equivalents für {symbol} ({frequency}): {str(e)}",
                    "symbol": symbol}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_shares_outstanding(self, symbol):
        """Ruft die Anzahl der ausstehenden Aktien ab."""
        try:
            stock = yf.Ticker(symbol)
            shares = stock.info.get("sharesOutstanding")
            if shares is None:
                raise ValueError(f"Keine Daten zu ausstehenden Aktien für {symbol} gefunden.")
            return shares
        except Exception as e:
            return {"error": f"Fehler beim Abrufen der ausstehenden Aktien für {symbol}: {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_dividend_data(self, symbol, use_cache=True):
        """Ruft die Daten für die Dividenden ab."""
        if use_cache:
            cached_data = self._load_cached_data(symbol, "dividend_data")
            if cached_data is not None and "error" not in cached_data:
                return cached_data
        try:
            stock = yf.Ticker(symbol)
            if 'dividendRate' not in stock.info:
                return {"error": f"Dividendenrate für {symbol} nicht verfügbar."}
            if 'regularMarketPrice' not in stock.info:
                raise ValueError(f"Aktueller Marktpreis für {symbol} nicht verfügbar.")
            dividend_rate = stock.info.get('dividendRate')
            current_price = stock.info.get("regularMarketPrice")
            if current_price == 0:
                raise ValueError(f"Marktpreis für {symbol} ist 0, Dividendenrendite kann nicht berechnet werden.")
            dividend_yield = (dividend_rate / current_price) * 100 if dividend_rate else 0
            dividend_yield = round(dividend_yield, 2)
            dividends = stock.dividends
            latest_dividend = dividends.iloc[-1] if not dividends.empty else 0
            data = {
                'dividend_yield': dividend_yield,
                'dividend_rate': dividend_rate,
                'latest_dividend': latest_dividend
            }
            if dividend_yield < 5:
                data['warning'] = "Dividendenrendite unter 5% – nicht als Zinssubstitut geeignet."
            else:
                data['message'] = "Dividendenrendite erfüllt die Anforderung für Zinssubstitute (≥ 5%)."

            if use_cache:
                self._cache_data(data, symbol, "dividend_data")
            return data
        except Exception as e:
            return {"error": f"Fehler beim Abrufen der Dividenden für {symbol}: {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_dividend_history(self, symbol, use_cache=True):
        """Ruft die Rohdaten der Dividenden der letzten 20 Jahre ab (als DataFrame mit Spalte 'dividend')."""
        if use_cache:
            cached_data = self._load_cached_data(symbol, "dividend_history")
            if cached_data is not None and "error" not in cached_data:
                return cached_data
        try:
            stock = yf.Ticker(symbol)
            dividends = stock.dividends
            if dividends.empty:
                raise ValueError(f"Keine Dividendenhistorie für {symbol} gefunden.")

            # auf 20 Jahre (≈80 Einträge) begrenzen und in DataFrame verwandeln
            df = dividends.tail(80).to_frame(name="dividend")
            df.index = pd.to_datetime(df.index)  # sicherheitshalber
            data = {"dividends_history": df}

            if use_cache:
                self._cache_data(data, symbol, "dividend_history")
            return data
        except Exception as e:
            return {"error": f"Fehler beim Abrufen der Dividendenhistorie für {symbol}: {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_paid_dividends(self, symbol: str, frequency: str = "annual", use_cache: bool = True) -> dict:
        """
        Ruft den Gesamtbetrag der ausgezahlten Dividenden für ein gegebenes Aktiensymbol ab,
        basierend auf der Cash Flow-Rechnung von Yahoo Finance.

        Args:
            symbol (str): Aktiensymbol (z. B. 'KO').
            frequency (str): 'annual' für jährliche Daten, 'quarterly' für quartalsweise Daten (Standard: 'annual').
            use_cache (bool): Ob der Cache verwendet werden soll (Standard: True).

        Returns:
            dict: Enthält den Betrag der ausgezahlten Dividenden.
                  Beispiel:
                  {
                      "paid_dividends": -8340000000.0,
                      "symbol": "KO",
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

        # Cache-Daten-Typ für den Schlüssel
        data_type = f"paid_dividends_{frequency}"

        # Versuche, Daten aus dem Cache zu laden
        if use_cache:
            cached_data = self._load_cached_data(symbol, data_type)
            if cached_data is not None:
                return cached_data

        try:
            # Ticker-Objekt erstellen
            stock = yf.Ticker(symbol)

            # Cash Flow-Daten abrufen
            if frequency == "annual":
                cashflow = stock.cashflow
            else:  # quarterly
                cashflow = stock.quarterly_cashflow

            # Prüfen, ob Cash Flow-Daten verfügbar sind
            if not isinstance(cashflow, pd.DataFrame) or cashflow.empty:
                raise ValueError(f"Keine Cash Flow-Daten für {symbol} ({frequency}) gefunden.")

            # Dividenden ausgezahlt ermitteln
            dividends_paid = 0.0  # Standardwert, wenn keine Dividenden gefunden werden
            for label in ["Dividends Paid", "Common Stock Dividends Paid", "Preferred Stock Dividends Paid",
                          "Total Dividends Paid", "Cash Dividends Paid"]:
                if label in cashflow.index:
                    dividends_paid = cashflow.loc[label].iloc[0]
                    if pd.notna(dividends_paid):
                        break

            # Datum extrahieren
            latest_date = cashflow.columns[0]
            if not isinstance(latest_date, pd.Timestamp):
                latest_date = pd.to_datetime(latest_date)

            # Ergebnis-Dictionary erstellen
            result = {
                "paid_dividends": float(dividends_paid),
                "symbol": symbol,
                "frequency": frequency,
                "date": latest_date.strftime("%Y-%m-%d")
            }

            # Ergebnis im Cache speichern, wenn Cache aktiviert
            if use_cache:
                self._cache_data(result, symbol, data_type)

            return result

        except Exception as e:
            return {"error": f"Fehler beim Abrufen der ausgezahlten Dividenden für {symbol} ({frequency}): {str(e)}",
                    "symbol": symbol}


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_net_debt_data(self, symbol, frequency="annual"):
        """
        Ruft Daten für die Berechnung des Nettoschuldenstands ab.
        frequency: 'annual' für jährliche Daten, 'quarterly' für quartalsweise Daten.
        """
        try:
            balance_sheet = self.get_balance_sheet(symbol, frequency=frequency)
            if isinstance(balance_sheet, dict) and "error" in balance_sheet:
                return balance_sheet

            # Versuche, Net Debt direkt zu holen
            if "Net Debt" in balance_sheet.index:
                net_debt = balance_sheet.loc["Net Debt"].iloc[0]
                total_debt = None
                cash = None
            else:
                # Fallback auf Berechnung mit Total Debt und Cash
                if "Total Debt" in balance_sheet.index:
                    total_debt = balance_sheet.loc["Total Debt"].iloc[0]
                elif "Long Term Debt" in balance_sheet.index and "Short Term Debt" in balance_sheet.index:
                    total_debt = balance_sheet.loc["Long Term Debt"].iloc[0] + \
                                 balance_sheet.loc["Short Term Debt"].iloc[0]
                else:
                    raise ValueError(f"Keine Schulden-Daten für {symbol} ({frequency}) gefunden.")

                if "Cash" in balance_sheet.index:
                    cash = balance_sheet.loc["Cash"].iloc[0]
                elif "Cash And Cash Equivalents" in balance_sheet.index:
                    cash = balance_sheet.loc["Cash And Cash Equivalents"].iloc[0]
                else:
                    raise ValueError(f"Keine Cash-Daten für {symbol} ({frequency}) gefunden.")
                net_debt = total_debt - cash

            data = {
                "total_debt": total_debt,
                "cash": cash,
                "net_debt": net_debt,
                "frequency": frequency
            }
            return data
        except Exception as e:
            return {"error": f"Fehler beim Abrufen der Nettoschuldendaten für {symbol} ({frequency}): {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_ebitda_data(self, symbol, frequency="annual"):
        try:
            financials = self.get_stock_financials(symbol, frequency=frequency)
            if isinstance(financials, dict) and "error" in financials:
                return financials
            if "EBITDA" not in financials.index:
                raise ValueError(f"Keine EBITDA-Daten für {symbol} ({frequency}) gefunden.")
            data = {
                "ebitda": financials.loc["EBITDA"].iloc[0],
                "frequency": frequency
            }
            return data
        except Exception as e:
            return {"error": f"Fehler beim Abrufen der EBITDA-Daten für {symbol} ({frequency}): {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_ebit_data(self, symbol, use_cache=True, frequency="annual"):
        """
        Ruft die EBIT-Daten (Earnings Before Interest and Taxes) eines Unternehmens ab.

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            use_cache (bool): Ob der Cache verwendet werden soll (Standard: True).
            frequency (str): Zeitraum, entweder 'annual' oder 'quarterly' (Standard: 'annual').

        Returns:
            dict: Enthält die EBIT-Daten und den Zeitraum.
                  Beispiel:
                  {
                      "ebit": 1234567890.0,
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
        # Cache-Schlüssel mit frequency erstellen
        cache_key = f"{symbol}_ebit_{frequency}"

        if use_cache:
            cached_data = self._load_cached_data(symbol, cache_key)
            if cached_data is not None and "error" not in cached_data:
                return cached_data

        try:
            financials = self.get_stock_financials(symbol, frequency=frequency)
            if isinstance(financials, dict) and "error" in financials:
                return financials

            if "EBIT" not in financials.index:
                error = {
                    "error": f"Keine EBIT-Daten für {symbol} ({frequency}) gefunden.",
                    "symbol": symbol
                }
                return error

            ebit = financials.loc["EBIT"].iloc[0]
            if pd.isna(ebit):
                error = {
                    "error": f"Ungültige EBIT-Daten für {symbol} ({frequency}): {ebit}.",
                    "symbol": symbol
                }
                return error

            # Konvertiere das Datum (pandas.Timestamp) in einen String
            date_str = financials.columns[0].strftime("%Y-%m-%d") if isinstance(financials.columns[0],
                                                                                pd.Timestamp) else str(
                financials.columns[0])

            data = {
                "ebit": float(ebit),
                "symbol": symbol,
                "frequency": frequency,
                "date": date_str
            }
            if use_cache:
                self._cache_data(data, symbol, cache_key)
            return data

        except Exception as e:
            error = {
                "error": f"Fehler beim Abrufen der EBIT-Daten für {symbol} ({frequency}): {str(e)}",
                "symbol": symbol
            }
            return error

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_interest_expense_data(self, symbol, use_cache=True, frequency="annual"):
        """
        Ruft die Zinsaufwendungen (Interest Expenses) eines Unternehmens ab.

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            use_cache (bool): Ob der Cache verwendet werden soll (Standard: True).
            frequency (str): Zeitraum, entweder 'annual' oder 'quarterly' (Standard: 'annual').

        Returns:
            dict: Enthält die Zinsaufwendungen und den Zeitraum.
                  Beispiel:
                  {
                      "interest_expense": 12345678.0,
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
        # Cache-Schlüssel mit frequency erstellen
        cache_key = f"interest_expense_{frequency}"

        if use_cache:
            cached_data = self._load_cached_data(symbol, cache_key)
            if cached_data is not None and "error" not in cached_data:
                return cached_data

        try:
            financials = self.get_stock_financials(symbol, frequency=frequency)
            if isinstance(financials, dict) and "error" in financials:
                # Füge den symbol-Schlüssel zum Fehler-Dictionary hinzu
                financials["symbol"] = symbol
                return financials

            if "Interest Expense" not in financials.index:
                error = {
                    "error": f"Keine Zinsaufwendungen für {symbol} ({frequency}) gefunden.",
                    "symbol": symbol
                }
                return error

            interest_expense = financials.loc["Interest Expense"].iloc[0]
            if pd.isna(interest_expense):
                error = {
                    "error": f"Ungültige Zinsaufwendungen für {symbol} ({frequency}): {interest_expense}.",
                    "symbol": symbol
                }
                return error

            # Konvertiere das Datum (pandas.Timestamp) in einen String
            date_str = financials.columns[0].strftime("%Y-%m-%d") if isinstance(financials.columns[0],
                                                                                pd.Timestamp) else str(
                financials.columns[0])

            data = {
                "interest_expense": abs(float(interest_expense)),  # Verwende abs(), um positive Werte sicherzustellen
                "symbol": symbol,
                "frequency": frequency,
                "date": date_str
            }
            if use_cache:
                self._cache_data(data, symbol, cache_key)
            return data

        except Exception as e:
            error = {
                "error": f"Fehler beim Abrufen der Zinsaufwendungen für {symbol} ({frequency}): {str(e)}",
                "symbol": symbol
            }
            return error

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_free_cashflow(self, symbol: str, frequency: str = "annual") -> dict:
        """
        Ruft den Free Cashflow (FCF) für ein gegebenes Aktiensymbol ab. Prüft zuerst, ob 'Free Cash Flow' verfügbar ist.
        Falls nicht, berechnet FCF aus operativem Cashflow und Capital Expenditures.

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            frequency (str): 'annual' für jährliche Daten, 'quarterly' für quartalsweise Daten.

        Returns:
            dict: Enthält den Free Cashflow, Symbol, Frequenz und Datum.
                  Beispiel:
                  {
                      "free_cashflow": 1234567890.0,
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

        # Cache-Daten-Typ für den Schlüssel
        data_type = f"free_cashflow_{frequency}"

        # Versuche, Daten aus dem Cache zu laden
        cached_data = self._load_cached_data(symbol, data_type)
        if cached_data is not None:
            return cached_data

        try:
            # Ticker-Objekt erstellen
            ticker = yf.Ticker(symbol)

            # Cashflow-Daten abrufen
            if frequency == "annual":
                cashflow_data = ticker.cashflow
            else:  # quarterly
                cashflow_data = ticker.quarterly_cashflow

            if cashflow_data.empty:
                return {"error": f"Keine Cashflow-Daten für {symbol} ({frequency}) gefunden.", "symbol": symbol}

            # Datum extrahieren (für spätere Verwendung)
            latest_date = cashflow_data.columns[0].strftime("%Y-%m-%d")

            # Schritt 1: Prüfen, ob Free Cash Flow direkt verfügbar ist
            free_cashflow_label = "Free Cash Flow"
            if free_cashflow_label in cashflow_data.index:
                free_cashflow = cashflow_data.loc[free_cashflow_label].iloc[0]
                if not pd.isna(free_cashflow):
                    # Ergebnis-Dictionary erstellen
                    result = {
                        "free_cashflow": float(free_cashflow),
                        "symbol": symbol,
                        "frequency": frequency,
                        "date": latest_date
                    }
                    # Ergebnis im Cache speichern
                    self._cache_data(result, symbol, data_type)
                    return result

            # Schritt 2: Fallback auf Berechnung, wenn Free Cash Flow fehlt oder NaN ist
            # Operativen Cashflow ermitteln
            operating_cashflow = None
            for label in ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]:
                if label in cashflow_data.index:
                    operating_cashflow = cashflow_data.loc[label].iloc[0]
                    if not pd.isna(operating_cashflow):
                        break
            if operating_cashflow is None or pd.isna(operating_cashflow):
                return {
                    "error": f"Kein operativer Cashflow für {symbol} ({frequency}) gefunden. Verfügbare Labels: {list(cashflow_data.index)}",
                    "symbol": symbol
                }

            # Capital Expenditures ermitteln
            capex = None
            for label in ["Capital Expenditure", "Purchase Of PPE", "Capital Expenditure Reported"]:
                if label in cashflow_data.index:
                    capex = cashflow_data.loc[label].iloc[0]
                    if not pd.isna(capex):
                        break
            if capex is None or pd.isna(capex):
                # Fallback: Net PPE Purchase And Sale minus Sale Of PPE
                if "Net PPE Purchase And Sale" in cashflow_data.index:
                    net_ppe = cashflow_data.loc["Net PPE Purchase And Sale"].iloc[0]
                    sale_of_ppe = cashflow_data.loc["Sale Of PPE"].iloc[
                        0] if "Sale Of PPE" in cashflow_data.index else 0
                    capex = net_ppe - sale_of_ppe if not (pd.isna(net_ppe) or pd.isna(sale_of_ppe)) else None

            if capex is None or pd.isna(capex):
                return {
                    "error": f"Keine Capital Expenditures für {symbol} ({frequency}) gefunden. Verfügbare Labels: {list(cashflow_data.index)}",
                    "symbol": symbol
                }

            # Free Cashflow berechnen: FCF = Operating Cashflow - CapEx
            # CapEx ist negativ in yfinance, daher addieren
            free_cashflow = operating_cashflow + capex

            # Prüfen auf ungültige Werte
            if pd.isna(free_cashflow):
                return {"error": f"Ungültiger berechneter Free Cashflow für {symbol} ({frequency}): {free_cashflow}.",
                        "symbol": symbol}

            # Ergebnis-Dictionary erstellen
            result = {
                "free_cashflow": float(free_cashflow),
                "symbol": symbol,
                "frequency": frequency,
                "date": latest_date
            }

            # Ergebnis im Cache speichern
            self._cache_data(result, symbol, data_type)
            return result

        except Exception as e:
            return {"error": f"Fehler beim Abrufen des Free Cashflows für {symbol} ({frequency}): {str(e)}",
                    "symbol": symbol}


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_operating_cashflow(self, symbol: str, frequency: str = "annual") -> dict:
        """
        Ruft den operativen Cashflow für ein gegebenes Aktiensymbol ab.

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            frequency (str): 'annual' für jährliche Daten, 'quarterly' für quartalsweise Daten.

        Returns:
            dict: Enthält den operativen Cashflow, Symbol, Frequenz und Datum.
                  Beispiel:
                  {
                      "operating_cashflow": 1234567890.0,
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

        # Cache-Daten-Typ für den Schlüssel
        data_type = f"operating_cashflow_{frequency}"

        # Versuche, Daten aus dem Cache zu laden
        cached_data = self._load_cached_data(symbol, data_type)
        if cached_data is not None:
            return cached_data

        try:
            # Ticker-Objekt erstellen
            ticker = yf.Ticker(symbol)

            # Cashflow-Daten abrufen
            if frequency == "annual":
                cashflow_data = ticker.cashflow
            else:  # quarterly
                cashflow_data = ticker.quarterly_cashflow

            if cashflow_data.empty:
                return {"error": f"Keine Cashflow-Daten für {symbol} ({frequency}) gefunden.", "symbol": symbol}

            # Datum extrahieren
            latest_date = cashflow_data.columns[0].strftime("%Y-%m-%d")

            # Operativen Cashflow ermitteln
            operating_cashflow = None
            for label in ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]:
                if label in cashflow_data.index:
                    operating_cashflow = cashflow_data.loc[label].iloc[0]
                    if not pd.isna(operating_cashflow):
                        break
            if operating_cashflow is None or pd.isna(operating_cashflow):
                return {
                    "error": f"Kein operativer Cashflow für {symbol} ({frequency}) gefunden. Verfügbare Labels: {list(cashflow_data.index)}",
                    "symbol": symbol
                }

            # Ergebnis-Dictionary erstellen
            result = {
                "operating_cashflow": float(operating_cashflow),
                "symbol": symbol,
                "frequency": frequency,
                "date": latest_date
            }

            # Ergebnis im Cache speichern
            self._cache_data(result, symbol, data_type)
            return result

        except Exception as e:
            return {"error": f"Fehler beim Abrufen des operativen Cashflows für {symbol} ({frequency}): {str(e)}",
                    "symbol": symbol}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_revenue(self, symbol: str, frequency: str = "annual") -> dict:
        """
        Ruft den Umsatz (Revenue) für ein gegebenes Aktiensymbol ab.

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            frequency (str): 'annual' für jährliche Daten, 'quarterly' für quartalsweise Daten.

        Returns:
            dict: Enthält den Umsatz, Symbol, Frequenz und Datum.
                  Beispiel:
                  {
                      "revenue": 1234567890.0,
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

        # Cache-Daten-Typ für den Schlüssel
        data_type = f"revenue_{frequency}"

        # Versuche, Daten aus dem Cache zu laden
        cached_data = self._load_cached_data(symbol, data_type)
        if cached_data is not None:
            return cached_data

        try:
            # Ticker-Objekt erstellen
            ticker = yf.Ticker(symbol)

            # Finanzdaten abrufen
            if frequency == "annual":
                financial_data = ticker.financials
            else:  # quarterly
                financial_data = ticker.quarterly_financials

            if financial_data.empty:
                return {"error": f"Keine Finanzdaten für {symbol} ({frequency}) gefunden.", "symbol": symbol}

            # Datum extrahieren
            latest_date = financial_data.columns[0].strftime("%Y-%m-%d")

            # Umsatz ermitteln
            revenue = None
            for label in ["Total Revenue", "Revenue"]:
                if label in financial_data.index:
                    revenue = financial_data.loc[label].iloc[0]
                    if not pd.isna(revenue):
                        break
            if revenue is None or pd.isna(revenue):
                return {
                    "error": f"Kein Umsatz für {symbol} ({frequency}) gefunden. Verfügbare Labels: {list(financial_data.index)}",
                    "symbol": symbol
                }

            # Ergebnis-Dictionary erstellen
            result = {
                "revenue": float(revenue),
                "symbol": symbol,
                "frequency": frequency,
                "date": latest_date
            }

            # Ergebnis im Cache speichern
            self._cache_data(result, symbol, data_type)
            return result

        except Exception as e:
            return {"error": f"Fehler beim Abrufen des Umsatzes für {symbol} ({frequency}): {str(e)}", "symbol": symbol}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_inventory(self, symbol: str, frequency: str = "annual") -> dict:
        """
        Ruft den Wert der Vorräte (Inventories) für ein gegebenes Aktiensymbol ab.

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            frequency (str): 'annual' für jährliche Daten, 'quarterly' für quartalsweise Daten.

        Returns:
            dict: Enthält den Vorratswert, Symbol, Frequenz und Datum.
                  Beispiel:
                  {
                      "inventory": 1234567890.0,
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

        # Cache-Daten-Typ für den Schlüssel
        data_type = f"inventory_{frequency}"

        # Versuche, Daten aus dem Cache zu laden
        cached_data = self._load_cached_data(symbol, data_type)
        if cached_data is not None:
            return cached_data

        try:
            # Ticker-Objekt erstellen
            ticker = yf.Ticker(symbol)

            # Bilanzdaten abrufen
            if frequency == "annual":
                balance_data = ticker.balance_sheet
            else:  # quarterly
                balance_data = ticker.quarterly_balance_sheet

            if balance_data.empty:
                return {"error": f"Keine Bilanzdaten für {symbol} ({frequency}) gefunden.", "symbol": symbol}

            # Datum extrahieren
            latest_date = balance_data.columns[0].strftime("%Y-%m-%d")

            # Vorräte ermitteln
            inventory = None
            for label in ["Inventory", "Total Inventories"]:
                if label in balance_data.index:
                    inventory = balance_data.loc[label].iloc[0]
                    if not pd.isna(inventory):
                        break
            if inventory is None or pd.isna(inventory):
                return {
                    "error": f"Keine Vorräte für {symbol} ({frequency}) gefunden. Verfügbare Labels: {list(balance_data.index)}",
                    "symbol": symbol
                }

            # Ergebnis-Dictionary erstellen
            result = {
                "inventory": float(inventory),
                "symbol": symbol,
                "frequency": frequency,
                "date": latest_date
            }

            # Ergebnis im Cache speichern
            self._cache_data(result, symbol, data_type)
            return result

        except Exception as e:
            return {"error": f"Fehler beim Abrufen der Vorräte für {symbol} ({frequency}): {str(e)}", "symbol": symbol}


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_payout_ratio_data_annual(self, symbol):
        """
        Ruft Daten ab und berechnet die Ausschüttungsquote über das letzte Jahr auf EPS-Basis.
        Formel: (DPS / EPS) * 100%, wobei DPS die Summe der Dividenden pro Aktie im letzten Jahr ist
        und EPS entweder aus stock.info["trailingEps"] oder Net Income / Shares Outstanding berechnet wird.
        Logik:
        1. Prüfen, ob stock.info["payoutRatio"] verfügbar ist. Falls ja, direkt anwenden.
        2. Falls nicht, Fallback auf trailingEPS und Dividendenhistorie.
        3. Falls trailingEPS nicht verfügbar, Berechnung wie bisher mit Net Income.
        """
        try:
            stock = yf.Ticker(symbol)
            shares = self.get_shares_outstanding(symbol)
            if isinstance(shares, dict) and "error" in shares:
                return shares

            # Schritt 1: Prüfen, ob payoutRatio verfügbar ist
            payout_ratio = stock.info.get("payoutRatio")
            if payout_ratio is not None and not pd.isna(payout_ratio) and payout_ratio >= 0:
                return {
                    "payout_ratio_eps": round(payout_ratio * 100, 2),
                    "dps": None,
                    "eps": None,
                    "net_income": None,
                    "shares_outstanding": shares
                }

            # Schritt 2: Fallback auf trailingEps
            trailing_eps = stock.info.get("trailingEps")
            if trailing_eps is not None and not pd.isna(trailing_eps) and trailing_eps > 0:
                dividend_history = self.get_dividend_history(symbol)
                if isinstance(dividend_history, dict) and "error" in dividend_history:
                    return dividend_history
                dividends_history = dividend_history['dividends_history']
                if dividends_history.empty:
                    return {
                        "payout_ratio_eps": 0,
                        "dps": 0,
                        "eps": round(trailing_eps, 2),
                        "net_income": None,
                        "shares_outstanding": shares,
                        "warning": f"Keine Dividendenhistorie für {symbol}, Payout Ratio auf 0 gesetzt"
                    }
                end_date = pd.Timestamp.now(tz="America/New_York")
                start_date = end_date - pd.Timedelta(days=365)
                dps = dividends_history[
                    (dividends_history.index >= start_date) &
                    (dividends_history.index <= end_date)
                    ].sum()
                dps = round(dps, 2)
                payout_ratio_eps = (dps / trailing_eps * 100) if trailing_eps > 0 else 0
                payout_ratio_eps = round(payout_ratio_eps, 2)
                return {
                    "payout_ratio_eps": payout_ratio_eps,
                    "dps": dps,
                    "eps": round(trailing_eps, 2),
                    "net_income": None,
                    "shares_outstanding": shares
                }

            # Schritt 3: Fallback auf bisherige Berechnung
            financials = self.get_stock_financials(symbol)
            if isinstance(financials, dict) and "error" in financials:
                return financials
            if "Net Income" not in financials.index:
                raise ValueError(f"Keine Net Income-Daten für {symbol} gefunden.")
            net_income = financials.loc["Net Income"].iloc[0]
            eps = net_income / shares if shares > 0 else 0
            eps = round(eps, 2)
            dividend_history = self.get_dividend_history(symbol)
            if isinstance(dividend_history, dict) and "error" in dividend_history:
                return dividend_history
            dividends_history = dividend_history['dividends_history']
            if dividends_history.empty:
                return {
                    "payout_ratio_eps": 0,
                    "dps": 0,
                    "eps": eps,
                    "net_income": round(net_income, 2),
                    "shares_outstanding": shares,
                    "warning": f"Keine Dividendenhistorie für {symbol}, Payout Ratio auf 0 gesetzt"
                }
            end_date = pd.Timestamp.now(tz="America/New_York")
            start_date = end_date - pd.Timedelta(days=365)
            dps = dividends_history[
                (dividends_history.index >= start_date) &
                (dividends_history.index <= end_date)
                ].sum()
            dps = round(dps, 2)
            payout_ratio_eps = (dps / eps * 100) if eps > 0 else 0
            payout_ratio_eps = round(payout_ratio_eps, 2)
            return {
                "payout_ratio_eps": payout_ratio_eps,
                "dps": dps,
                "eps": eps,
                "net_income": round(net_income, 2),
                "shares_outstanding": shares
            }
        except Exception as e:
            return {"error": f"Fehler beim Abrufen oder Berechnen der Ausschüttungsquotendaten für {symbol}: {str(e)}"}


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_reinvested_profit(self, symbol: str, frequency: str = "annual", use_cache: bool = True) -> dict:
        """
        Berechnet den reinvestierten Gewinn für ein gegebenes Aktiensymbol,
        definiert als Nettoeinkommen minus ausgezahlte Dividenden.

        Args:
            symbol (str): Aktiensymbol (z. B. 'KO').
            frequency (str): 'annual' für jährliche Daten, 'quarterly' für quartalsweise Daten (Standard: 'annual').
            use_cache (bool): Ob der Cache verwendet werden soll (Standard: True).

        Returns:
            dict: Enthält den reinvestierten Gewinn in USD.
                  Beispiel:
                  {
                      "reinvested_profit": 1234567890.0,
                      "symbol": "KO",
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

        # Cache-Daten-Typ für den Schlüssel
        data_type = f"reinvested_profit_{frequency}"

        # Versuche, Daten aus dem Cache zu laden
        if use_cache:
            cached_data = self._load_cached_data(symbol, data_type)
            if cached_data is not None:
                return cached_data

        try:
            # Nettoeinkommen abrufen
            financials = self.get_stock_financials(symbol, frequency)
            if isinstance(financials, dict) and "error" in financials:
                return financials

            if not isinstance(financials, pd.DataFrame) or financials.empty:
                raise ValueError(f"Keine Finanzdaten für {symbol} ({frequency}) gefunden.")

            # Nettoeinkommen ermitteln
            net_income = None
            for label in ["Net Income Common Stockholders","Net Income", "Net Income Applicable To Common Shares"]:
                if label in financials.index:
                    net_income = financials.loc[label].iloc[0]
                    if pd.notna(net_income):
                        break
            if net_income is None or pd.isna(net_income):
                return {
                    "error": f"Kein Nettoeinkommen für {symbol} ({frequency}) gefunden. Verfügbare Labels: {list(financials.index)}",
                    "symbol": symbol
                }

            # Ausgezahlte Dividenden abrufen
            dividends_result = self.get_paid_dividends(symbol, frequency)
            if "error" in dividends_result:
                return dividends_result
            paid_dividends = dividends_result["paid_dividends"]

            # Wenn paid_dividends nan ist, setze es auf 0.0
            if pd.isna(paid_dividends):
                paid_dividends = 0.0

            # Reinvestierten Gewinn berechnen
            reinvested_profit = net_income - abs(paid_dividends)
            reinvested_profit = round(reinvested_profit, 2)
            # Datum extrahieren
            latest_date = financials.columns[0]
            if not isinstance(latest_date, pd.Timestamp):
                latest_date = pd.to_datetime(latest_date)

            # Ergebnis-Dictionary erstellen
            result = {
                "reinvested_profit": float(reinvested_profit),
                "symbol": symbol,
                "frequency": frequency,
                "date": latest_date.strftime("%Y-%m-%d")
            }

            # Ergebnis im Cache speichern, wenn Cache aktiviert
            if use_cache:
                self._cache_data(result, symbol, data_type)

            return result

        except Exception as e:
            return {"error": f"Fehler beim Berechnen des reinvestierten Gewinns für {symbol} ({frequency}): {str(e)}",
                    "symbol": symbol}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_inflation_data(self, use_cache=True, start_date=None, end_date=None):
        if start_date:
            try:
                datetime.strptime(start_date, '%Y-%m-%d')
            except ValueError:
                return {"error": "start_date muss im Format 'YYYY-MM-DD' sein."}
        if end_date:
            try:
                datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                return {"error": "end_date muss im Format 'YYYY-MM-DD' sein."}
        if start_date and end_date and datetime.strptime(end_date, '%Y-%m-%d') <= datetime.strptime(start_date, '%Y-%m-%d'):
            return {"error": "end_date muss nach start_date liegen."}
        if use_cache:
            cache_key = f"cpi_inflation_data_{start_date}_{end_date}" if start_date and end_date else "cpi_inflation_data"
            cached_data = self._load_cached_data(cache_key, "inflation_data")
            if cached_data is not None and "error" not in cached_data:
                return cached_data
        try:
            api_key = '87050e1670abe158b4dbaebdc8910d49'
            series_id = 'CPIAUCSL'
            url = f'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json'
            if start_date and end_date:
                url += f'&observation_start={start_date}&observation_end={end_date}'
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()['observations']
            if len(data) < 2:
                raise ValueError("Nicht genügend CPI-Daten für die Berechnung der Inflationsrate.")
            cpi_data = [{'date': entry['date'], 'value': float(entry['value'])} for entry in data]
            if use_cache:
                cache_key = f"cpi_inflation_data_{start_date}_{end_date}" if start_date and end_date else "cpi_inflation_data"
                self._cache_data(cpi_data, cache_key, "inflation_data")
            return cpi_data
        except Exception as e:
            return {"error": f"Fehler beim Abrufen der Inflationsdaten: {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_gdp_data_grpwth(self, use_cache=True):
        if use_cache:
            cached_data = self._load_cached_data("gdp_data", "gdp_value")
            if cached_data is not None and "error" not in cached_data:
                return cached_data
        try:
            api_key = '87050e1670abe158b4dbaebdc8910d49'
            series_id = 'GDPC1'
            url = f'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json'
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()['observations']
            if len(data) < 2:
                raise ValueError("Nicht genügend BIP-Daten verfügbar.")
            latest_gdp = float(data[-1]['value'])
            latest_date = data[-1]['date']
            # Vorjahres-Daten finden (angenommen, Daten sind chronologisch sortiert)
            one_year_ago = datetime.strptime(latest_date, '%Y-%m-%d') - relativedelta(years=1)
            for entry in reversed(data):
                if datetime.strptime(entry['date'], '%Y-%m-%d') <= one_year_ago:
                    previous_gdp = float(entry['value'])
                    previous_date = entry['date']
                    break
            else:
                raise ValueError("Kein Vorjahres-BIP-Daten gefunden.")
            # BIP-Wachstum berechnen und auf 2 Dezimalstellen runden
            gdp_growth = round(((latest_gdp - previous_gdp) / previous_gdp) * 100, 2)
            data = {
                'gdp_value': latest_gdp,
                'date': latest_date,
                'previous_gdp_value': previous_gdp,
                'previous_date': previous_date,
                'gdp_growth': gdp_growth
            }
            if use_cache:
                self._cache_data(data, "gdp_data", "gdp_value")
            return data
        except Exception as e:
            return {"error": f"Fehler beim Abrufen der BIP-Daten: {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_company_profits(self, symbol, use_cache=True, frequency="annual"):
        if use_cache:
            cached_data = self._load_cached_data(symbol, f"company_profits_{frequency}")
            if cached_data is not None and "error" not in cached_data:
                return cached_data
        try:
            financials = self.get_stock_financials(symbol, frequency=frequency)
            if 'Net Income' not in financials.index:
                raise ValueError(f"Keine quartalsweisen Gewinndaten für {symbol} verfügbar.")
            net_incomes = financials.loc['Net Income']
            if len(net_incomes) < 2:
                raise ValueError(f"Nicht genügend Gewinndaten für {symbol} zur Berechnung des Wachstums.")
            latest_net_income = float(net_incomes.iloc[0])
            latest_date = net_incomes.index[0].strftime('%Y-%m-%d')
            previous_net_income = float(net_incomes.iloc[1])
            previous_date = net_incomes.index[1].strftime('%Y-%m-%d')
            latest_dt = datetime.strptime(latest_date, '%Y-%m-%d')
            previous_dt = datetime.strptime(previous_date, '%Y-%m-%d')
            days_diff = (latest_dt - previous_dt).days

            if frequency == "quarterly":
                if not (85 <= days_diff <= 95):
                    raise ValueError(
                        f"Die Gewinndaten für {symbol} decken keinen quartalsweisen Zeitraum ab: {previous_date} bis {latest_date} ({days_diff} Tage).")
            elif frequency == "annual":
                if not (350 <= days_diff <= 380):
                    raise ValueError(
                        f"Die Gewinndaten für {symbol} decken keinen jährlichen Zeitraum ab: {previous_date} bis {latest_date} ({days_diff} Tage).")

            if abs(latest_net_income) >= 1_000_000_000:
                latest_net_income_display = latest_net_income / 1_000_000_000
                previous_net_income_display = previous_net_income / 1_000_000_000
                unit = "Milliarden USD"
            else:
                latest_net_income_display = latest_net_income / 1_000_000
                previous_net_income_display = previous_net_income / 1_000_000
                unit = "Millionen USD"
            latest_net_income_display = round(latest_net_income_display, 2)
            previous_net_income_display = round(previous_net_income_display, 2)
            data = {
                'latest_net_income': float(latest_net_income),
                'latest_date': latest_date,
                'previous_net_income': float(previous_net_income),
                'previous_date': previous_date,
                'net_income_display': latest_net_income_display,
                'previous_net_income_display': previous_net_income_display,
                'unit': unit,
                'frequency': frequency
            }
            if use_cache:
                self._cache_data(data, symbol, f"company_profits_{frequency}")
            return data
        except Exception as e:
            return {"error": f"Fehler beim Abrufen der Gewinndaten für {symbol}: {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_enterprise_value(self, symbol: str, use_cache = True, frequency: str = "annual") -> dict:
        """
        Berechnet den Enterprise Value (EV) eines Unternehmens.
        Priorisiert die eigene Berechnung; bei NaN-Werten für net_debt greift es auf yfinance zurück.
        Formel: EV = Marktkapitalisierung + Gesamtverbindlichkeiten - Liquide Mittel

        Args:
            symbol (str): Aktiensymbol (z. B. 'AAPL').
            frequency (str): 'annual' für jährliche Daten (TTM), 'quarterly' für quartalsweise Daten.

        Returns:
            dict: Enthält den Enterprise Value, Symbol und Frequenz, oder Fehlerdetails mit Symbol.
        """
        try:
            if use_cache:
                cached_data = self._load_cached_data(symbol, f"enterprise_value{frequency}")
                if cached_data is not None and "error" not in cached_data:
                    return cached_data

            # Prüfen, ob Frequenz gültig ist
            if frequency not in ["annual", "quarterly"]:
                return {"error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.",
                        "symbol": symbol}

            # Daten abrufen
            current_price = self.get_current_price_per_share(symbol)
            shares = self.get_shares_outstanding(symbol)
            net_debt_data = self.get_net_debt_data(symbol, frequency=frequency)

            # Fehlerprüfung für Daten
            if isinstance(current_price, dict) and "error" in current_price:
                error_msg = current_price["error"]
                if "HTTP Error 404" in error_msg:
                    error_msg = f"Ungültiges Symbol: {symbol}. {error_msg}"
                return {"error": error_msg, "symbol": symbol}
            if isinstance(shares, dict) and "error" in shares:
                return {"error": shares["error"], "symbol": symbol}
            if isinstance(net_debt_data, dict) and "error" in net_debt_data:
                return {"error": net_debt_data["error"], "symbol": symbol}

            # Werte extrahieren
            total_debt = net_debt_data["total_debt"]
            cash = net_debt_data["cash"]
            net_debt = net_debt_data["net_debt"]

            # Prüfen auf ungültige Werte
            if shares <= 0:
                return {"error": f"Ungültige Anzahl ausstehender Aktien für {symbol}: {shares}.", "symbol": symbol}
            if current_price <= 0:
                return {"error": f"Ungültiger Aktienkurs für {symbol}: {current_price}.", "symbol": symbol}
            if total_debt is None and cash is None:
                # Verwende net_debt direkt, wenn total_debt und cash nicht verfügbar sind
                if net_debt is None or net_debt != net_debt:  # Prüft auf NaN
                    # Fallback auf yfinance für NaN-Werte
                    try:
                        ticker = yf.Ticker(symbol)
                        enterprise_value = ticker.info.get('enterpriseValue')
                        if enterprise_value is not None and isinstance(enterprise_value,
                                                                       (int, float)) and enterprise_value > 0:
                            return {
                                "enterprise_value": float(enterprise_value),
                                "symbol": symbol,
                                "frequency": frequency
                            }
                        else:
                            return {
                                "error": f"Kein gültiger Enterprise Value von yfinance für {symbol} ({frequency}) verfügbar.",
                                "symbol": symbol}
                    except Exception as yf_error:
                        return {
                            "error": f"Keine gültigen Nettoschulden-Daten für {symbol} ({frequency}) verfügbar, und yfinance fehlgeschlagen: {str(yf_error)}",
                            "symbol": symbol}
            else:
                # Verwende total_debt und cash, wenn verfügbar
                if total_debt is None:
                    return {"error": f"Keine Schulden-Daten für {symbol} ({frequency}) verfügbar.", "symbol": symbol}
                if cash is None:
                    return {"error": f"Keine Liquiditäts-Daten für {symbol} ({frequency}) verfügbar.", "symbol": symbol}
                net_debt = total_debt - cash

            # Marktkapitalisierung berechnen
            market_cap = current_price * shares

            # Enterprise Value berechnen
            enterprise_value = market_cap + net_debt

            # Ergebnis zurückgeben
            data = {
                "enterprise_value": float(enterprise_value),
                "symbol": symbol,
                "frequency": frequency
            }
            if use_cache:
                self._cache_data(data, symbol, f"enterprise_value{frequency}")

            return data

        except Exception as e:
            return {"error": f"Fehler beim Berechnen des Enterprise Value für {symbol}: {str(e)}", "symbol": symbol}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_minority_interest(self, symbol: str, frequency: str = "annual", use_cache: bool = True) -> dict:
        """
        Liefert Minderheitenanteile (Non-controlling Interests) aus der Bilanz.
        Rückgabe: {"minority_interest": float, "symbol": ..., "frequency": ..., "date": "..."}
        Fallback: 0.0 mit 'warning', wenn kein passendes Label gefunden wird.
        """
        if frequency not in ["annual", "quarterly"]:
            return {"error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.", "symbol": symbol}

        cache_key = f"minority_interest_{frequency}"
        if use_cache:
            cached = self._load_cached_data(symbol, cache_key)
            if cached is not None and "error" not in cached:
                return cached

        try:
            bs = self.get_balance_sheet(symbol, frequency=frequency)
            if isinstance(bs, dict) and "error" in bs:
                return bs
            if not isinstance(bs, pd.DataFrame) or bs.empty:
                return {"error": f"Keine Bilanzdaten für {symbol} ({frequency}) gefunden.", "symbol": symbol}

            # Mögliche Label-Varianten bei Yahoo Finance
            candidates = [
                "Minority Interest",  # häufig
                "Noncontrolling Interests",  # alternative Bezeichnung
                "Non Controlling Interest",  # Variante
                "Minority Interest Liabilities",  # selten
                "Noncontrolling Interest"  # Singular
            ]

            value = None
            for label in candidates:
                if label in bs.index:
                    value = bs.loc[label].iloc[0]
                    if pd.notna(value):
                        break

            latest_date = bs.columns[0]
            if not isinstance(latest_date, pd.Timestamp):
                latest_date = pd.to_datetime(latest_date)

            if value is None or pd.isna(value):
                result = {
                    "minority_interest": 0.0,
                    "symbol": symbol,
                    "frequency": frequency,
                    "date": latest_date.strftime("%Y-%m-%d"),
                    "warning": f"Kein Minderheitenanteil-Label gefunden; als 0.0 angenommen."
                }
            else:
                result = {
                    "minority_interest": float(value),
                    "symbol": symbol,
                    "frequency": frequency,
                    "date": latest_date.strftime("%Y-%m-%d")
                }

            if use_cache:
                self._cache_data(result, symbol, cache_key)
            return result

        except Exception as e:
            return {"error": f"Fehler beim Abrufen der Minderheitenanteile für {symbol} ({frequency}): {str(e)}",
                    "symbol": symbol}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_preferred_stock(self, symbol: str, frequency: str = "annual", use_cache: bool = True) -> dict:
        """
        Liefert Vorzugsaktien (Preferred Stock) aus der Bilanz.
        Rückgabe: {"preferred_stock": float, "symbol": ..., "frequency": ..., "date": "..."}
        Fallback: 0.0 mit 'warning', wenn kein passendes Label gefunden wird.
        """
        if frequency not in ["annual", "quarterly"]:
            return {"error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.", "symbol": symbol}

        cache_key = f"preferred_stock_{frequency}"
        if use_cache:
            cached = self._load_cached_data(symbol, cache_key)
            if cached is not None and "error" not in cached:
                return cached

        try:
            bs = self.get_balance_sheet(symbol, frequency=frequency)
            if isinstance(bs, dict) and "error" in bs:
                return bs
            if not isinstance(bs, pd.DataFrame) or bs.empty:
                return {"error": f"Keine Bilanzdaten für {symbol} ({frequency}) gefunden.", "symbol": symbol}

            # Mögliche Label-Varianten
            candidates = [
                "Preferred Stock",
                "Preferred Stock Equity",
                "Preferred Shares",
                "Preferred Stock And Other Adjustments"
            ]

            value = None
            for label in candidates:
                if label in bs.index:
                    value = bs.loc[label].iloc[0]
                    if pd.notna(value):
                        break

            latest_date = bs.columns[0]
            if not isinstance(latest_date, pd.Timestamp):
                latest_date = pd.to_datetime(latest_date)

            if value is None or pd.isna(value):
                result = {
                    "preferred_stock": 0.0,
                    "symbol": symbol,
                    "frequency": frequency,
                    "date": latest_date.strftime("%Y-%m-%d"),
                    "warning": f"Kein Vorzugsaktien-Label gefunden; als 0.0 angenommen."
                }
            else:
                result = {
                    "preferred_stock": float(value),
                    "symbol": symbol,
                    "frequency": frequency,
                    "date": latest_date.strftime("%Y-%m-%d")
                }

            if use_cache:
                self._cache_data(result, symbol, cache_key)
            return result

        except Exception as e:
            return {"error": f"Fehler beim Abrufen der Vorzugsaktien für {symbol} ({frequency}): {str(e)}",
                    "symbol": symbol}

    def _load_cached_data(self, symbol, data_type):
        filepath = os.path.join(self.cache_dir, f"{symbol}_{data_type}.json")
        cache_duration = timedelta(days=600) if "historical" in data_type else timedelta(minutes=10)
        if os.path.exists(filepath) and (
                datetime.now() - datetime.fromtimestamp(os.path.getmtime(filepath))) < cache_duration:
            with open(filepath, "r") as f:
                data = json.load(f)

                def convert_to_pandas(obj):
                    if isinstance(obj, str):
                        try:
                            return pd.read_json(StringIO(obj))
                        except ValueError:
                            return obj
                    elif isinstance(obj, dict):
                        return {k: convert_to_pandas(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_pandas(item) for item in obj]
                    return obj

                return convert_to_pandas(data)
        return None

    def _cache_data(self, data, symbol, data_type):
        self.logger.info(f"Cache Daten für {symbol}_{data_type} in {self.cache_dir}")
        os.makedirs(self.cache_dir, exist_ok=True)
        filepath = os.path.join(self.cache_dir, f"{symbol}_{data_type}.json")
        self.logger.debug(f"Schreibe Cache-Datei: {filepath}")

        def convert_pandas(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_json()
            elif isinstance(obj, pd.Series):
                return obj.to_json()
            elif isinstance(obj, dict):
                return {k: convert_pandas(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_pandas(item) for item in obj]
            return obj

        try:
            data_to_cache = convert_pandas(data)
            with open(filepath, "w") as f:
                json.dump(data_to_cache, f)
            self.logger.info(f"Cache-Datei erfolgreich geschrieben: {filepath}")
        except Exception as e:
            self.logger.error(f"Fehler beim Schreiben der Cache-Datei {filepath}: {e}")
            raise

    def _clear_cache(self):
        """Löscht alle zwischengespeicherten Daten."""
        self._cache = {}  # Annahme: _cache ist ein Dictionary, das die gecachten Daten enthält
        # Falls der Cache in einer Datei gespeichert wird, lösche die Datei hier