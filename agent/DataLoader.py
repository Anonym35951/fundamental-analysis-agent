import logging
from time import sleep
from typing import Optional, Dict
from io import StringIO
import pandas as pd
import yfinance as yf
import requests
import os
import json
from datetime import datetime, timedelta, timezone

from dateutil.relativedelta import relativedelta
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type, RetryError
from dotenv import load_dotenv
from agent.data_sources.sec_source import SecSource

load_dotenv()


def describe_exception(e: Exception) -> str:
    """Unwraps tenacity's RetryError to the actual exception that caused the
    final retry attempt to fail, so user-facing error messages show the real
    cause (e.g. a timeout or rate limit) instead of the unreadable
    "RetryError[<Future at ... raised ...>]" wrapper text."""
    if isinstance(e, RetryError):
        try:
            underlying = e.last_attempt.exception()
            if underlying is not None:
                return str(underlying)
        except Exception:
            pass
    return str(e)


CACHE_MAX_AGE = timedelta(days=620)  # etwas über der 600-Tage-Historical-TTL
CACHE_MAX_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB Obergrenze fürs Cache-Verzeichnis
CACHE_PRUNE_INTERVAL = timedelta(hours=1)  # Pruning nicht bei jeder Instanziierung neu durchlaufen

# TTL-Staffelung für _load_cached_data/_cache_data (Dateicache je Symbol):
# - "historical_*"-Keys: mehrjährige Zeitreihen, ändern sich nur am aktuellen
#   Rand -> lange TTL (unverändert, s. HISTORICAL_CACHE_TTL).
# - PRICE_SENSITIVE_CACHE_KEYWORDS: Werte, die den aktuellen Börsenkurs
#   einpreisen (Marktkapitalisierung, Enterprise Value, EV-/Preis-Multiples,
#   PEG-Ratio, Bandbreiten-Bewertung) - ändern sich während der Handelszeit
#   laufend, bleiben daher kurzlebig.
# - alles andere: SEC-/FRED-Fundamentaldaten (Bilanz, GuV, Cashflow,
#   Dividenden, Aktienzahl, Inflation, ROIC, ...) ändern sich real nur mit
#   einem neuen Quartals-/Jahres-Filing bzw. einer monatlichen CPI-
#   Veröffentlichung. Vorher lief hier fälschlich dieselbe 10-Sekunden-TTL
#   wie für Live-Preis-Werte, was bei praktisch jeder Kennzahl-Abfrage einen
#   erneuten SEC-/Alpha-Vantage-Request auslöste (siehe LAUNCH_AUDIT.md, H-3).
#   6 Stunden analog zur bestehenden SEC-Filing-Prüfung
#   (SecSource.get_latest_filing / filing_alert_worker in api/main.py).
HISTORICAL_CACHE_TTL = timedelta(days=600)
FUNDAMENTAL_CACHE_TTL = timedelta(hours=6)
LIVE_CACHE_TTL = timedelta(seconds=10)
PRICE_SENSITIVE_CACHE_KEYWORDS = (
    "market_cap",
    "enterprise_value",
    "ev_to_sales",
    "ev_to_ebit",
    "ev_to_ebitda",
    "price_to_ebit",
    "price_to_freecashflow",
    "peg_ratio",
    "tbv_bandwidth_eval",
    "ebit_bandwidth_eval",
)


class DataLoader:
    def __init__(self, user_agent="gecen.efe1308@gmail.com"):
        self.user_agent = user_agent
        self.ticker_cache = {}
        self.price_cache = {}
        # CACHE_DIR per Env konfigurierbar, damit ein Render Persistent Disk
        # (oder ein beliebiger anderer Mount) eingebunden werden kann, ohne den
        # Code anzufassen. agent/ ist bewusst von api/core/config entkoppelt.
        self.cache_dir = os.environ.get("CACHE_DIR", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.api_key = os.environ["ALPHA_VANTAGE_API_KEY"]
        self.base_url = "https://www.alphavantage.co/query?"
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.sec_source = SecSource(user_agent=self.user_agent)
        self._prune_cache()

    def _prune_cache(self):
        """Hält das Cache-Verzeichnis begrenzt: löscht abgelaufene Dateien und,
        falls die Gesamtgröße das Limit überschreitet, zusätzlich die ältesten
        Dateien. Läuft höchstens einmal pro CACHE_PRUNE_INTERVAL (Sentinel-Datei),
        damit es bei häufiger DataLoader-Instanziierung nicht bei jedem Analyse-Job
        erneut das ganze Verzeichnis durchläuft."""
        sentinel = os.path.join(self.cache_dir, ".last_prune")
        now = datetime.now()
        if os.path.exists(sentinel):
            last_prune = datetime.fromtimestamp(os.path.getmtime(sentinel))
            if now - last_prune < CACHE_PRUNE_INTERVAL:
                return

        try:
            entries = []
            for name in os.listdir(self.cache_dir):
                if name.startswith("."):
                    continue
                path = os.path.join(self.cache_dir, name)
                if not os.path.isfile(path):
                    continue
                mtime = os.path.getmtime(path)
                if now - datetime.fromtimestamp(mtime) > CACHE_MAX_AGE:
                    os.remove(path)
                    continue
                entries.append((mtime, os.path.getsize(path), path))

            total_size = sum(size for _, size, _ in entries)
            if total_size > CACHE_MAX_SIZE_BYTES:
                entries.sort(key=lambda e: e[0])  # älteste zuerst
                for mtime, size, path in entries:
                    if total_size <= CACHE_MAX_SIZE_BYTES:
                        break
                    os.remove(path)
                    total_size -= size

            with open(sentinel, "w") as f:
                f.write(now.isoformat())
        except OSError as e:
            self.logger.warning(f"Cache-Pruning fehlgeschlagen: {e}")


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
        """
        Ruft GuV-/Income-Statement-Daten über SEC ab.
        Yahoo-kompatible Rückgabe: pandas DataFrame mit Kennzahlen als Index und Perioden als Spalten.
        """
        if frequency not in ["annual", "quarterly"]:
            return {
                "error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.",
                "symbol": symbol
            }

        cache_key = f"stock_financials_{frequency}"

        if use_cache:
            cached_data = self._load_cached_data(symbol, cache_key)
            if cached_data is not None:
                if isinstance(cached_data, dict) and "error" in cached_data:
                    return cached_data
                if not isinstance(cached_data, pd.DataFrame):
                    return {
                        "error": f"Cached-Daten für {symbol} ({frequency}) sind kein DataFrame: {str(cached_data)}",
                        "symbol": symbol
                    }
                return cached_data

        try:
            financials = self.sec_source.get_stock_financials(
                symbol=symbol,
                frequency=frequency,
                use_cache=use_cache
            )

            if isinstance(financials, dict) and "error" in financials:
                return financials

            if not isinstance(financials, pd.DataFrame):
                return {
                    "error": f"SEC-Finanzdaten für {symbol} ({frequency}) sind kein DataFrame: {str(financials)}",
                    "symbol": symbol
                }

            if financials.empty:
                return {
                    "error": f"Keine SEC-Finanzdaten ({frequency}) für {symbol} gefunden.",
                    "symbol": symbol
                }

            if use_cache:
                self._cache_data(financials, symbol, cache_key)

            return financials

        except Exception as e:
            return {
                "error": f"Fehler beim Abrufen der SEC-Finanzdaten für {symbol} ({frequency}): {describe_exception(e)}",
                "symbol": symbol
            }

    def get_data_source_summary(self, symbol: str, frequency: str = "annual") -> dict:
        """Liefert die Herkunft und Aktualität der Fundamentaldaten eines Symbols
        fürs Transparenz-Versprechen (Quellen-Badge im Frontend).

        Income Statement und Bilanz kommen in dieser Engine praktisch immer aus
        SEC-Filings (get_stock_financials/get_balance_sheet haben keinen
        Yahoo-Fallback) — daher hier bewusst ein pro Symbol/Frequenz einmalig
        bestimmter Herkunftsstatus statt eine Quellenmarkierung pro einzelner
        Kennzahl, was hunderte Berechnungsfunktionen unberührt lässt und trotzdem
        eine ehrliche, akkurate Aussage liefert: "diese Fundamentaldaten stammen
        aus SEC-Filing X, Stand Y".

        Returns:
            {
                "symbol": "AAPL",
                "frequency": "annual",
                "source": "SEC",
                "as_of": "2025-09-30",   # jüngste Berichtsperiode in den Rohdaten
                "fetched_at": "2026-07-02T21:00:00+00:00",
            }
            oder bei Fehler: {"error": "...", "symbol": symbol}
        """
        fetched_at = datetime.now(timezone.utc).isoformat()

        financials = self.get_stock_financials(symbol, frequency=frequency, use_cache=True)
        if isinstance(financials, dict) and "error" in financials:
            return {
                "error": f"Keine Fundamentaldaten für {symbol} ({frequency}) verfügbar.",
                "symbol": symbol,
            }

        as_of = None
        try:
            period_dates = pd.to_datetime(financials.columns, errors="coerce").dropna()
            if len(period_dates) > 0:
                as_of = period_dates.max().strftime("%Y-%m-%d")
        except Exception:
            as_of = None

        return {
            "symbol": symbol.upper(),
            "frequency": frequency,
            "source": "SEC",
            "as_of": as_of,
            "fetched_at": fetched_at,
        }

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

    @retry(stop=stop_after_attempt(3),wait=wait_fixed(2),retry=retry_if_exception_type(Exception))
    def get_peg_ratio(self, symbol, use_cache=True):
        """
        Berechnet die PEG Ratio vollständig aus SEC-Daten.

        Formel:

            PEG = PE Ratio / Gewinnwachstum (%)

            PE = Aktienkurs / EPS

            Gewinnwachstum =
                (Aktuelles Net Income - Vorjahres Net Income)
                / |Vorjahres Net Income|
                * 100

        Returns:
            {
                "peg_ratio": 1.52,
                "symbol": "AAPL",
                "method": "sec_calculated",
                "trailing_pe": 29.4,
                "earnings_growth": 19.3,
                "eps": 6.12
            }

        oder

            {
                "error": "...",
                "symbol": "AAPL"
            }
        """

        cache_key = "peg_ratio"
        if use_cache:
            cached_data = self._load_cached_data(symbol, cache_key)
            if (
                    cached_data is not None
                    and "error" not in cached_data
            ):
                return cached_data

        try:
            current_price = self.get_current_price_per_share(symbol)

            if isinstance(current_price, dict):
                return {
                    "error": current_price.get(
                        "error",
                        f"Fehler beim Abrufen des Aktienkurses für {symbol}"
                    ),
                    "symbol": symbol
                }

            if (
                    current_price is None
                    or pd.isna(current_price)
                    or float(current_price) <= 0
            ):
                return {
                    "error": f"Ungültiger Aktienkurs für {symbol}: {current_price}",
                    "symbol": symbol
                }
            shares_result = self.get_shares_outstanding(symbol)

            if isinstance(shares_result, dict):

                if "error" in shares_result:
                    return shares_result

                shares_outstanding = shares_result.get(
                    "shares_outstanding"
                )

            else:
                shares_outstanding = shares_result

            if (
                    shares_outstanding is None
                    or pd.isna(shares_outstanding)
                    or float(shares_outstanding) <= 0
            ):
                return {
                    "error": f"Ungültige Shares Outstanding für {symbol}: {shares_outstanding}",
                    "symbol": symbol
                }

            financials = self.get_stock_financials(
                symbol=symbol,
                frequency="annual",
                use_cache=use_cache
            )

            if isinstance(financials, dict) and "error" in financials:
                return financials

            if (
                    not isinstance(financials, pd.DataFrame)
                    or financials.empty
            ):
                return {
                    "error": f"Keine Finanzdaten für {symbol} gefunden.",
                    "symbol": symbol
                }

            net_income_label = None

            for label in [
                "Net Income Common Stockholders",
                "Net Income",
                "Net Income Applicable To Common Shares"
            ]:

                if label in financials.index:
                    net_income_label = label
                    break

            if net_income_label is None:
                return {
                    "error": (
                        f"Keine Net-Income-Daten für {symbol} gefunden. "
                        f"Verfügbare Labels: {list(financials.index)}"
                    ),
                    "symbol": symbol
                }

            net_income_series = financials.loc[
                net_income_label
            ].dropna()

            if len(net_income_series) < 2:
                return {
                    "error": (
                        f"Nicht genügend Net-Income-Daten "
                        f"für PEG-Berechnung bei {symbol}."
                    ),
                    "symbol": symbol
                }

            latest_net_income = float(
                net_income_series.iloc[0]
            )

            previous_net_income = float(
                net_income_series.iloc[1]
            )

            eps = (
                    latest_net_income
                    / float(shares_outstanding)
            )

            if (
                    pd.isna(eps)
                    or eps <= 0
            ):
                return {
                    "error": f"Ungültiges EPS für {symbol}: {eps}",
                    "symbol": symbol
                }

            trailing_pe = (
                    float(current_price)
                    / float(eps)
            )

            if (
                    pd.isna(trailing_pe)
                    or trailing_pe <= 0
            ):
                return {
                    "error": f"Ungültige PE Ratio für {symbol}: {trailing_pe}",
                    "symbol": symbol
                }

            if previous_net_income == 0:
                return {
                    "error": (
                        f"Vorjahresgewinn für {symbol} ist 0. "
                        f"PEG kann nicht berechnet werden."
                    ),
                    "symbol": symbol
                }

            earnings_growth = (
                                      (
                                              latest_net_income
                                              - previous_net_income
                                      )
                                      / abs(previous_net_income)
                              ) * 100

            #
            # ==========================================================
            # Gewinnwachstum validieren
            # ==========================================================
            #

            if pd.isna(earnings_growth):
                return {
                    "error": (
                        f"Ungültiges Gewinnwachstum "
                        f"für {symbol}: {earnings_growth}"
                    ),
                    "symbol": symbol
                }

            #
            # ==========================================================
            # Negatives Wachstum → PEG nicht sinnvoll
            # ==========================================================
            #

            if earnings_growth <= 0:

                result = {
                    "peg_ratio": None,
                    "symbol": symbol.upper(),
                    "method": "sec_calculated",
                    "trailing_pe": round(float(trailing_pe), 2),
                    "earnings_growth": round(float(earnings_growth), 2),
                    "eps": round(float(eps), 4),
                    "latest_net_income": float(latest_net_income),
                    "previous_net_income": float(previous_net_income),
                    "reason": "negative_growth"
                }

                if use_cache:
                    self._cache_data(
                        result,
                        symbol,
                        cache_key
                    )

                return result

            #
            # ==========================================================
            # Normale PEG-Berechnung
            # ==========================================================
            #

            peg_ratio = (
                    float(trailing_pe)
                    / float(earnings_growth)
            )

            if pd.isna(peg_ratio):
                return {
                    "error": f"Ungültige PEG Ratio für {symbol}: {peg_ratio}",
                    "symbol": symbol
                }

            result = {
                "peg_ratio": round(float(peg_ratio), 2),
                "symbol": symbol.upper(),
                "method": "sec_calculated",
                "trailing_pe": round(float(trailing_pe), 2),
                "earnings_growth": round(float(earnings_growth), 2),
                "eps": round(float(eps), 4),
                "latest_net_income": float(latest_net_income),
                "previous_net_income": float(previous_net_income)
            }

            if use_cache:
                self._cache_data(
                    result,
                    symbol,
                    cache_key
                )

            return result

        except Exception as e:

            return {
                "error": (
                    f"Fehler beim Berechnen der "
                    f"PEG Ratio für {symbol}: {str(e)}"
                ),
                "symbol": symbol
            }

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_book_value(self, symbol):
        """
        Ruft den Buchwert je Aktie (Book Value Per Share) ab.

        Priorität:
        1. SEC Balance Sheet + Shares Outstanding
        2. Yahoo Finance Fallback

        Formel:
            Book Value Per Share =
            Stockholders Equity / Shares Outstanding
        """

        try:

            #
            # ==========================================================
            # 1. SEC PRIORITÄT
            # ==========================================================
            #

            try:

                balance_sheet = self.sec_source.get_balance_sheet(
                    symbol=symbol,
                    frequency="annual",
                    use_cache=True,
                    scope="core",
                )

                if (
                        isinstance(balance_sheet, pd.DataFrame)
                        and not balance_sheet.empty
                ):

                    equity = None

                    for label in [
                        "Stockholders Equity",
                        "Common Stock Equity",
                        "Total Stockholders Equity",
                    ]:

                        if label in balance_sheet.index:

                            value = balance_sheet.loc[label].iloc[0]

                            if pd.notna(value):
                                equity = float(value)
                                break

                    shares_outstanding = self.get_shares_outstanding(symbol)

                    if (
                            equity is not None
                            and isinstance(shares_outstanding, (int, float))
                            and shares_outstanding > 0
                    ):
                        return equity / float(shares_outstanding)

            except Exception:
                pass

            #
            # ==========================================================
            # 2. YAHOO FALLBACK
            # ==========================================================
            #

            stock = yf.Ticker(symbol)

            book_value = stock.info.get("bookValue")

            if book_value is None:
                raise ValueError(
                    f"Keine Buchwert-Daten für {symbol} gefunden."
                )

            return float(book_value)

        except Exception as e:

            return {
                "error": (
                    f"Fehler beim Abrufen des Buchwerts "
                    f"für {symbol}: {str(e)}"
                )
            }


    def is_financial_sector(self, symbol: str) -> bool:
        """Prüft, ob ein Symbol eine Bank oder ein Versicherer ist.

        Banken und Versicherer weisen Handelsbestände/kurzfristige
        Wertpapiere in derselben SEC-Bilanzposition wie 'Cash' aus, was
        Kennzahlen wie Cash/Market Cap für diese Unternehmen strukturell
        unsinnig macht. yfinance ordnet auch Zahlungsdienstleister/Broker
        (z. B. PayPal, Visa) dem groben Sektor "Financial Services" zu, ohne
        dieses Bilanzierungsproblem zu haben — daher gezielt auf die
        feingranularere `industry`-Kategorie prüfen statt auf den Sektor."""
        try:
            stock = yf.Ticker(symbol)
            industry = stock.info.get("industry") or ""
            return industry.startswith("Banks") or industry.startswith("Insurance")
        except Exception:
            return False

    def get_company_profile(self, symbol: str, use_cache: bool = True) -> dict:
        """Liefert Name/Sektor/Industrie eines Symbols (yfinance `.info`) für
        die Branchen-Klassifikation in calculate_crv_by_sector_multiples
        (agent/industry_multiples.resolve_multiples). Sektor/Industrie
        ändern sich praktisch nie - Cache-Key bewusst mit "historical_"
        präfigiert, damit _load_cached_data die lange TTL (600 Tage statt
        10 Sekunden) verwendet, siehe dortige Fallunterscheidung.

        Ein unbekanntes/ungültiges Symbol liefert von yfinance ein leeres
        oder fast leeres `.info` (kein sector/longName/regularMarketPrice) -
        in dem Fall wird bewusst NICHT gecacht, damit ein späterer, echter
        yfinance-Ausfall nicht fälschlich als "Symbol existiert nicht"
        einbrennt."""
        cache_key = "historical_company_profile"

        if use_cache:
            cached = self._load_cached_data(symbol, cache_key)
            if cached is not None and "error" not in cached:
                return cached

        try:
            info = yf.Ticker(symbol).info or {}
            has_market_data = any(
                info.get(field) is not None
                for field in ("regularMarketPrice", "longName", "shortName", "sector")
            )
            if not has_market_data:
                return {"error": f"Unbekanntes Symbol: {symbol}"}

            profile = {
                "symbol": symbol,
                "name": info.get("longName") or info.get("shortName") or symbol,
                "sector": info.get("sector"),
                "industry": info.get("industry"),
            }

            if use_cache:
                self._cache_data(profile, symbol, cache_key)

            return profile
        except Exception as e:
            return {"error": f"Fehler beim Abrufen des Unternehmensprofils für {symbol}: {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_current_price_per_share(self, symbol):
        """Gibt den aktuellen Preis einer Aktie in US-Dollar zurück.

        Fängt Exceptions bewusst NICHT selbst ab (anders als die meisten
        DataLoader-Methoden) - der @retry-Decorator braucht eine tatsächlich
        propagierende Exception, um zu greifen. Mit einem internen try/except
        plus error-dict-Rückgabe feuert @retry nie (LAUNCH_AUDIT.md P2-12).
        Live-Kurse sind der Fall, in dem transiente Yahoo-Fehler (429s,
        Timeouts - insbesondere von Cloud-/Rechenzentrums-IPs aus) am
        häufigsten auftreten, daher hier zuerst behoben. Alle Aufrufer
        (direkt und transitiv über get_current_tbv_and_price) haben ein
        eigenes umschließendes try/except und fangen die letzte Exception
        nach den drei Versuchen weiterhin als Error-Dict/generische
        API-Fehlermeldung ab - siehe LAUNCH.md."""
        stock = yf.Ticker(symbol)
        current_price = stock.info.get("regularMarketPrice")
        if current_price is None:
            hist = stock.history(period="1d")
            if hist.empty:
                raise ValueError(f"Keine aktuellen Preisdaten für {symbol} gefunden.")
            current_price = hist["Close"].iloc[-1]
        return current_price

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_balance_sheet(self, symbol, frequency="annual", use_cache=True):
        if use_cache:
            cached_data = self._load_cached_data(symbol, f"balance_sheet_{frequency}")
            if cached_data is not None and "error" not in cached_data:
                return cached_data

        try:
            balance_sheet = self.sec_source.get_balance_sheet(
                symbol=symbol,
                frequency=frequency,
                use_cache=use_cache,
            )

            if isinstance(balance_sheet, dict) and "error" in balance_sheet:
                return balance_sheet

            if balance_sheet.empty:
                raise ValueError(f"Keine SEC-Bilanzdaten ({frequency}) für {symbol} gefunden.")

            if use_cache:
                self._cache_data(balance_sheet, symbol, f"balance_sheet_{frequency}")

            return balance_sheet

        except Exception as e:
            return {
                "error": f"Fehler beim Abrufen der Bilanzdaten für {symbol} ({frequency}) über SEC: {str(e)}",
                "symbol": symbol,
            }

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2),retry=retry_if_exception_type(Exception))
    def get_market_cap(self, symbol: str, use_cache: bool = True) -> dict:
        """
        Berechnet die aktuelle Marktkapitalisierung.

        Formel:

            Market Cap =
            Current Share Price × Shares Outstanding

        Returns:
            {
                "market_cap": 1234567890000.0,
                "symbol": "AAPL",
                "date": "2026-04-17",
                "source": "Price × SEC Shares Outstanding"
            }

        oder

            {
                "error": "...",
                "symbol": "AAPL"
            }
        """

        data_type = "market_cap"

        #
        # Cache
        #

        if use_cache:

            cached_data = self._load_cached_data(
                symbol,
                data_type
            )

            if (
                    cached_data is not None
                    and "error" not in cached_data
            ):
                return cached_data

        try:

            #
            # ==========================================================
            # Aktienkurs
            # ==========================================================
            #

            current_price = self.get_current_price_per_share(
                symbol
            )

            if isinstance(current_price, dict):
                return {
                    "error": current_price.get(
                        "error",
                        f"Fehler beim Abrufen des Aktienkurses für {symbol}"
                    ),
                    "symbol": symbol
                }

            #
            # ==========================================================
            # Shares Outstanding (SEC)
            # ==========================================================
            #

            shares_result = self.get_shares_outstanding(
                symbol
            )

            if isinstance(shares_result, dict):

                if "error" in shares_result:
                    return {
                        "error": shares_result["error"],
                        "symbol": symbol
                    }

                if "shares_outstanding" not in shares_result:
                    return {
                        "error": (
                            f"Keine Shares Outstanding "
                            f"für {symbol} gefunden."
                        ),
                        "symbol": symbol
                    }

                shares_outstanding = shares_result[
                    "shares_outstanding"
                ]

            else:

                #
                # Rückwärtskompatibilität
                #

                shares_outstanding = shares_result

            #
            # ==========================================================
            # Validierung
            # ==========================================================
            #

            if (
                    shares_outstanding is None
                    or pd.isna(shares_outstanding)
                    or float(shares_outstanding) <= 0
            ):
                raise ValueError(
                    f"Ungültige Shares Outstanding "
                    f"für {symbol}: {shares_outstanding}"
                )

            if (
                    current_price is None
                    or pd.isna(current_price)
                    or float(current_price) <= 0
            ):
                raise ValueError(
                    f"Ungültiger Aktienkurs "
                    f"für {symbol}: {current_price}"
                )

            #
            # ==========================================================
            # Market Cap berechnen
            # ==========================================================
            #

            market_cap = (
                    float(current_price)
                    * float(shares_outstanding)
            )

            if (
                    pd.isna(market_cap)
                    or market_cap <= 0
            ):
                raise ValueError(
                    f"Ungültige Marktkapitalisierung "
                    f"für {symbol}: {market_cap}"
                )

            #
            # ==========================================================
            # Ergebnis
            # ==========================================================
            #

            result = {
                "market_cap": float(market_cap),
                "symbol": symbol.upper(),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "source": "Price × SEC Shares Outstanding"
            }

            #
            # Cache
            #

            if use_cache:
                self._cache_data(
                    result,
                    symbol,
                    data_type
                )

            return result

        except Exception as e:

            return {
                "error": (
                    f"Fehler beim Berechnen der "
                    f"Marktkapitalisierung für "
                    f"{symbol}: {str(e)}"
                ),
                "symbol": symbol
            }


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

            # mögliche Labels (yfinance variiert je nach Company). Die
            # kombinierten Cash+Short-Term-Investments-Labels stehen bewusst
            # vor der reinen "Cash And Cash Equivalents"-Position: bei
            # cash-reichen Unternehmen (z. B. AAPL, MSFT) liegt ein
            # Großteil der liquiden Mittel in kurzfristigen Wertpapieren,
            # die eine reine Cash-Kennzahl systematisch unterschätzen würde.
            cash_labels = [
                "Cash Cash Equivalents And Short Term Investments",
                "Cash And Cash Equivalents And Short Term Investments",
                "Cash And Short Term Investments",
                "Cash And Cash Equivalents",
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
    def get_shares_outstanding(self, symbol: str) -> dict:
        """
        Ruft die Anzahl der ausstehenden Aktien ab.

        Priorität:
        1. SEC Company Facts
        2. Yahoo Finance Fallback

        Returns:
            {
                "shares_outstanding": 15000000000,
                "symbol": "AAPL",
                "date": "2025-09-27"
            }

        oder

            {
                "error": "...",
                "symbol": "AAPL"
            }
        """

        data_type = "shares_outstanding"

        # Cache prüfen
        cached_data = self._load_cached_data(symbol, data_type)
        if cached_data is not None:
            return cached_data

        try:

            #
            # ==========================================================
            # 1. SEC PRIORITÄT
            # ==========================================================
            # Foreign Private Issuers / ADRs (z. B. BABA) melden bei SEC die
            # zugrunde liegenden Ordinary Shares, nicht die an der Börse
            # gehandelten ADS-Einheiten — bei BABA z. B. 8 Ordinary Shares
            # pro ADS. Der Aktienkurs (get_current_price_per_share) ist aber
            # immer der ADS-Preis, daher würde "Ordinary Shares Number" die
            # Market Cap um den ADS-Verhältnis-Faktor überhöhen. Für solche
            # Symbole direkt zum Yahoo-Fallback springen, dessen
            # sharesOutstanding bereits in ADS-Einheiten vorliegt.
            #

            if not self.sec_source.is_foreign_private_issuer(symbol):
                try:
                    sec_result = self.sec_source.get_balance_sheet_line_item(
                        symbol=symbol,
                        line_item="Ordinary Shares Number",
                        frequency="annual",
                        scope="core",
                    )

                    if (
                            isinstance(sec_result, dict)
                            and "error" not in sec_result
                            and sec_result.get("value") is not None
                    ):
                        result = {
                            "shares_outstanding": float(sec_result["value"]),
                            "symbol": symbol.upper(),
                            "date": sec_result.get("date"),
                            "source": "SEC"
                        }

                        self._cache_data(result, symbol, data_type)
                        return result

                except Exception:
                    pass

            #
            # ==========================================================
            # 2. YAHOO FALLBACK
            # ==========================================================
            #

            stock = yf.Ticker(symbol)

            shares = stock.info.get("sharesOutstanding")

            if shares is None:
                return {
                    "error": f"Keine Daten zu ausstehenden Aktien für {symbol} gefunden.",
                    "symbol": symbol
                }

            result = {
                "shares_outstanding": float(shares),
                "symbol": symbol.upper(),
                "date": None,
                "source": "Yahoo Finance"
            }

            self._cache_data(result, symbol, data_type)

            return result

        except Exception as e:
            return {
                "error": f"Fehler beim Abrufen der ausstehenden Aktien für {symbol}: {str(e)}",
                "symbol": symbol
            }


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_dividend_data(self, symbol, use_cache=True):
        """Ruft die Daten für die Dividenden ab."""
        if use_cache:
            cached_data = self._load_cached_data(symbol, "dividend_data")
            if cached_data is not None and "error" not in cached_data:
                return cached_data
        try:
            stock = yf.Ticker(symbol)
            if 'regularMarketPrice' not in stock.info:
                raise ValueError(f"Aktueller Marktpreis für {symbol} nicht verfügbar.")
            # Fehlendes 'dividendRate' heißt meist schlicht: das Unternehmen
            # zahlt aktuell keine Dividende (z. B. Wachstumswerte) - ein
            # gültiges, häufiges Ergebnis, kein Fehlerfall. Früher brach das
            # die komplette Dividenden-/Average-Grower-Analyse ab, obwohl
            # "0% Rendite" schlicht ein nicht erfülltes Kriterium ist
            # (LAUNCH_AUDIT.md P1-3).
            dividend_rate = stock.info.get('dividendRate') or 0
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
    def get_paid_dividends(
            self,
            symbol: str,
            frequency: str = "annual",
            use_cache: bool = True
    ) -> dict:
        """
        Ruft die ausgezahlten Dividenden ab.

        Priorität:
        1. SEC Company Facts
        2. Yahoo Finance Fallback

        Returns:
            {
                "paid_dividends": -8340000000.0,
                "symbol": "KO",
                "frequency": "annual",
                "date": "2024-12-31",
                "source": "SEC"
            }

        oder

            {
                "error": "...",
                "symbol": "KO"
            }
        """

        if frequency not in ["annual", "quarterly"]:
            return {
                "error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.",
                "symbol": symbol
            }

        data_type = f"paid_dividends_{frequency}"

        # Cache prüfen
        if use_cache:
            cached_data = self._load_cached_data(symbol, data_type)
            if cached_data is not None:
                return cached_data

        try:

            #
            # ==========================================================
            # 1. SEC PRIORITÄT
            # ==========================================================
            #

            try:

                sec_result = self.sec_source.get_cashflow_statement_line_item(
                    symbol=symbol,
                    line_item="Dividends Paid",
                    frequency=frequency,
                    scope="core",
                )

                if (
                        isinstance(sec_result, dict)
                        and "error" not in sec_result
                        and sec_result.get("value") is not None
                ):

                    result = {
                        "paid_dividends": float(sec_result["value"]),
                        "symbol": symbol.upper(),
                        "frequency": frequency,
                        "date": sec_result.get("date"),
                        "source": "SEC"
                    }

                    if use_cache:
                        self._cache_data(result, symbol, data_type)

                    return result

            except Exception:
                pass

            #
            # ==========================================================
            # 2. YAHOO FALLBACK
            # ==========================================================
            #

            stock = yf.Ticker(symbol)

            if frequency == "annual":
                cashflow = stock.cashflow
            else:
                cashflow = stock.quarterly_cashflow

            if not isinstance(cashflow, pd.DataFrame) or cashflow.empty:
                return {
                    "error": f"Keine Cash Flow-Daten für {symbol} ({frequency}) gefunden.",
                    "symbol": symbol
                }

            dividends_paid = None

            for label in [
                "Dividends Paid",
                "Common Stock Dividends Paid",
                "Preferred Stock Dividends Paid",
                "Total Dividends Paid",
                "Cash Dividends Paid",
            ]:

                if label in cashflow.index:

                    value = cashflow.loc[label].iloc[0]

                    if pd.notna(value):
                        dividends_paid = value
                        break

            #
            # Falls Yahoo nichts findet:
            # wie bisher 0 zurückgeben
            #

            if dividends_paid is None or pd.isna(dividends_paid):
                dividends_paid = 0.0

            latest_date = cashflow.columns[0]

            if not isinstance(latest_date, pd.Timestamp):
                latest_date = pd.to_datetime(latest_date)

            result = {
                "paid_dividends": float(dividends_paid),
                "symbol": symbol.upper(),
                "frequency": frequency,
                "date": latest_date.strftime("%Y-%m-%d"),
                "source": "Yahoo Finance"
            }

            if use_cache:
                self._cache_data(result, symbol, data_type)

            return result

        except Exception as e:
            return {
                "error": f"Fehler beim Abrufen der ausgezahlten Dividenden für {symbol} ({frequency}): {str(e)}",
                "symbol": symbol
            }

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_net_debt_data(self, symbol, frequency="annual"):
        """
        Ruft Daten für die Berechnung des Nettoschuldenstands ab.
        frequency: 'annual' oder 'quarterly'
        """
        try:
            balance_sheet = self.get_balance_sheet(symbol, frequency=frequency)
            if isinstance(balance_sheet, dict) and "error" in balance_sheet:
                return balance_sheet

            if balance_sheet is None or getattr(balance_sheet, "empty", True):
                return {"error": f"Keine Bilanzdaten für {symbol} ({frequency}) gefunden.", "symbol": symbol}

            date_col = balance_sheet.columns[0]
            date = date_col.strftime("%Y-%m-%d") if hasattr(date_col, "strftime") else str(date_col)

            # Cash holen
            cash = None
            for label in [
                "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments",
                "Cash",
            ]:
                if label in balance_sheet.index and pd.notna(balance_sheet.loc[label].iloc[0]):
                    cash = float(balance_sheet.loc[label].iloc[0])
                    break

            if cash is None:
                return {
                    "error": f"Keine Cash-Daten für {symbol} ({frequency}) gefunden.",
                    "symbol": symbol
                }

            # Total Debt holen oder aus Einzelpositionen berechnen
            total_debt = None

            if "Total Debt" in balance_sheet.index and pd.notna(balance_sheet.loc["Total Debt"].iloc[0]):
                total_debt = float(balance_sheet.loc["Total Debt"].iloc[0])
            else:
                debt_parts = []

                for label in [
                    "Long Term Debt",
                    "Short Term Debt",
                    "Current Debt",
                    "Long Term Debt And Capital Lease Obligation",
                    "Current Debt And Capital Lease Obligation",
                ]:
                    if label in balance_sheet.index and pd.notna(balance_sheet.loc[label].iloc[0]):
                        debt_parts.append(float(balance_sheet.loc[label].iloc[0]))

                total_debt = sum(debt_parts) if debt_parts else 0.0

            # Net Debt direkt verwenden, falls vorhanden und gültig
            if "Net Debt" in balance_sheet.index and pd.notna(balance_sheet.loc["Net Debt"].iloc[0]):
                net_debt = float(balance_sheet.loc["Net Debt"].iloc[0])
            else:
                net_debt = total_debt - cash

            return {
                "total_debt": float(total_debt),
                "cash": float(cash),
                "net_debt": float(net_debt),
                "symbol": symbol,
                "frequency": frequency,
                "date": date
            }

        except Exception as e:
            return {
                "error": f"Fehler beim Abrufen der Nettoschuldendaten für {symbol} ({frequency}): {str(e)}",
                "symbol": symbol
            }

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_ebitda_data(self, symbol, frequency="annual"):
        try:
            financials = self.get_stock_financials(symbol, frequency=frequency)
            if isinstance(financials, dict) and "error" in financials:
                return financials

            if not isinstance(financials, pd.DataFrame) or financials.empty:
                return {"error": f"Keine Finanzdaten für {symbol} ({frequency}) gefunden.", "symbol": symbol}

            latest_date = financials.columns[0]
            date = latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, "strftime") else str(latest_date)

            # 1. Direktes EBITDA
            if "EBITDA" in financials.index and pd.notna(financials.loc["EBITDA"].iloc[0]):
                ebitda = float(financials.loc["EBITDA"].iloc[0])
            else:
                # 2. Fallback: EBIT + Abschreibungen
                ebit = None
                for label in ["EBIT", "Operating Income"]:
                    if label in financials.index and pd.notna(financials.loc[label].iloc[0]):
                        ebit = float(financials.loc[label].iloc[0])
                        break

                depreciation = None
                for label in [
                    "Depreciation And Amortization",
                    "Depreciation Depletion And Amortization",
                    "Depreciation",
                    "Amortization Of Intangible Assets",
                ]:
                    if label in financials.index and pd.notna(financials.loc[label].iloc[0]):
                        depreciation = float(financials.loc[label].iloc[0])
                        break

                if ebit is None:
                    return {
                        "error": f"Keine EBIT-Daten für EBITDA-Berechnung bei {symbol} ({frequency}) gefunden.",
                        "symbol": symbol
                    }

                if depreciation is None:
                    return {
                        "error": f"Keine Abschreibungsdaten für EBITDA-Berechnung bei {symbol} ({frequency}) gefunden.",
                        "symbol": symbol
                    }

                ebitda = ebit + depreciation

            return {
                "ebitda": float(ebitda),
                "symbol": symbol,
                "frequency": frequency,
                "date": date
            }

        except Exception as e:
            return {
                "error": f"Fehler beim Abrufen der EBITDA-Daten für {symbol} ({frequency}): {str(e)}",
                "symbol": symbol
            }

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
        cache_key = f"interest_expense_all_{frequency}"

        if use_cache:
            cached_data = self._load_cached_data(symbol, cache_key)
            if cached_data is not None and "error" not in cached_data:
                return cached_data

        try:
            facts = self.sec_source.get_company_facts(symbol, use_cache=use_cache)

            if isinstance(facts, dict) and "error" in facts:
                facts["symbol"] = symbol
                return facts

            us_gaap = facts.get("facts", {}).get("us-gaap", {})

            interest_tags = [
                "InterestExpense",
                "InterestExpenseDebt",
                "InterestExpenseNonoperating",
                "FinanceLeaseInterestExpense",
                "InterestCostsIncurred",
                "InterestPaid",
                "InterestPaidNet",
                "InterestOnConvertibleDebtNetOfTax",
                "InterestIncomeExpenseNet",
                "InterestIncomeExpenseNonoperatingNet",
                "IncomeTaxExaminationInterestExpense",
                "UnrecognizedTaxBenefitsIncomeTaxPenaltiesAndInterestExpense",
                "UnrecognizedTaxBenefitsInterestOnIncomeTaxesExpense",
                "FinanceLeaseInterestPaymentOnLiability",
            ]

            items = []

            for tag in interest_tags:
                fact = us_gaap.get(tag)

                if not fact:
                    continue

                label = fact.get("label") or tag
                description = fact.get("description") or ""
                units = fact.get("units", {})

                for unit, values in units.items():
                    series = self.sec_source._fact_values_to_series(
                        values=values,
                        frequency=frequency,
                    )

                    if series is None or series.dropna().empty:
                        continue

                    clean_series = series.dropna()
                    latest_date = clean_series.index[0]
                    latest_value = clean_series.iloc[0]

                    date_str = (
                        latest_date.strftime("%Y-%m-%d")
                        if isinstance(latest_date, pd.Timestamp)
                        else str(latest_date)
                    )

                    items.append(
                        {
                            "tag": tag,
                            "label": label,
                            "description": description,
                            "unit": unit,
                            "date": date_str,
                            "value": float(latest_value),
                            "abs_value": abs(float(latest_value)),
                            "series": {
                                (
                                    date.strftime("%Y-%m-%d")
                                    if isinstance(date, pd.Timestamp)
                                    else str(date)
                                ): float(value)
                                for date, value in clean_series.items()
                            },
                        }
                    )

            if not items:
                return {
                    "error": f"Keine Interest-Expense-Tags für {symbol} ({frequency}) gefunden.",
                    "symbol": symbol,
                }

            items = sorted(items, key=lambda x: x["date"], reverse=True)

            data = {
                "symbol": symbol,
                "frequency": frequency,
                "interest_expense_items": items,
                "count": len(items),
                "latest_item": items[0],
            }

            if use_cache:
                self._cache_data(data, symbol, cache_key)

            return data

        except Exception as e:
            return {
                "error": f"Fehler beim Abrufen der Interest-Expense-Daten für {symbol} ({frequency}): {str(e)}",
                "symbol": symbol,
            }

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
            return {
                "error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.",
                "symbol": symbol
            }

        # Cache-Daten-Typ für den Schlüssel
        data_type = f"free_cashflow_{frequency}"

        # Versuche, Daten aus dem Cache zu laden
        cached_data = self._load_cached_data(symbol, data_type)
        if cached_data is not None:
            return cached_data

        try:
            # SEC-Cashflow-Daten abrufen
            cashflow_data = self.sec_source.get_cashflow_statement(
                symbol=symbol,
                frequency=frequency,
                use_cache=True,
                scope="core"
            )

            if isinstance(cashflow_data, dict) and "error" in cashflow_data:
                return cashflow_data

            if cashflow_data.empty:
                return {
                    "error": f"Keine Cashflow-Daten für {symbol} ({frequency}) gefunden.",
                    "symbol": symbol
                }

            # Standard-Datum (neueste Spalte)
            latest_date = cashflow_data.columns[0].strftime("%Y-%m-%d")

            #
            # Schritt 1:
            # Reported Free Cash Flow bevorzugen
            #

            if "Free Cash Flow" in cashflow_data.index:

                fcf_series = cashflow_data.loc["Free Cash Flow"].dropna()

                if not fcf_series.empty:
                    free_cashflow = fcf_series.iloc[0]

                    result = {
                        "free_cashflow": float(free_cashflow),
                        "symbol": symbol,
                        "frequency": frequency,
                        "date": fcf_series.index[0].strftime("%Y-%m-%d")
                    }

                    self._cache_data(result, symbol, data_type)

                    return result

            #
            # Schritt 2:
            # Fallback auf Berechnung
            #

            operating_cashflow = None

            for label in [
                "Operating Cash Flow",
                "Cash Flow From Continuing Operating Activities",
            ]:
                if label in cashflow_data.index:

                    series = cashflow_data.loc[label].dropna()

                    if not series.empty:
                        operating_cashflow = series.iloc[0]
                        break

            if operating_cashflow is None or pd.isna(operating_cashflow):
                return {
                    "error": (
                        f"Kein operativer Cashflow für {symbol} ({frequency}) gefunden. "
                        f"Verfügbare Labels: {list(cashflow_data.index)}"
                    ),
                    "symbol": symbol
                }

            #
            # CapEx ermitteln
            #

            capex = None

            for label in [
                "Capital Expenditure",
                "Purchase Of PPE",
                "Capital Expenditure Reported",
            ]:
                if label in cashflow_data.index:

                    series = cashflow_data.loc[label].dropna()

                    if not series.empty:
                        capex = series.iloc[0]
                        break

            if capex is None or pd.isna(capex):
                return {
                    "error": (
                        f"Keine Capital Expenditures für {symbol} ({frequency}) gefunden. "
                        f"Verfügbare Labels: {list(cashflow_data.index)}"
                    ),
                    "symbol": symbol
                }

            #
            # SEC verwendet bereits Yahoo-kompatible Vorzeichen:
            # CapEx negativ => FCF = OCF + CapEx
            #

            free_cashflow = operating_cashflow + capex

            if pd.isna(free_cashflow):
                return {
                    "error": (
                        f"Ungültiger berechneter Free Cashflow "
                        f"für {symbol} ({frequency}): {free_cashflow}."
                    ),
                    "symbol": symbol
                }

            result = {
                "free_cashflow": float(free_cashflow),
                "symbol": symbol,
                "frequency": frequency,
                "date": latest_date
            }

            self._cache_data(result, symbol, data_type)

            return result

        except Exception as e:
            return {
                "error": (
                    f"Fehler beim Abrufen des Free Cashflows "
                    f"für {symbol} ({frequency}): {str(e)}"
                ),
                "symbol": symbol
            }

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
            return {
                "error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.",
                "symbol": symbol
            }

        # Cache-Daten-Typ für den Schlüssel
        data_type = f"operating_cashflow_{frequency}"

        # Versuche, Daten aus dem Cache zu laden
        cached_data = self._load_cached_data(symbol, data_type)
        if cached_data is not None:
            return cached_data

        try:
            # SEC-Cashflow-Daten abrufen
            cashflow_data = self.sec_source.get_cashflow_statement(
                symbol=symbol,
                frequency=frequency,
                use_cache=True,
                scope="core"
            )

            if isinstance(cashflow_data, dict) and "error" in cashflow_data:
                return cashflow_data

            if cashflow_data.empty:
                return {
                    "error": f"Keine Cashflow-Daten für {symbol} ({frequency}) gefunden.",
                    "symbol": symbol
                }

            # Datum extrahieren
            latest_date = cashflow_data.columns[0].strftime("%Y-%m-%d")

            # Operativen Cashflow ermitteln
            operating_cashflow = None

            for label in [
                "Operating Cash Flow",
                "Cash Flow From Continuing Operating Activities",
            ]:
                if label in cashflow_data.index:
                    operating_cashflow = cashflow_data.loc[label].iloc[0]

                    if not pd.isna(operating_cashflow):
                        break

            if operating_cashflow is None or pd.isna(operating_cashflow):
                return {
                    "error": (
                        f"Kein operativer Cashflow für {symbol} ({frequency}) gefunden. "
                        f"Verfügbare Labels: {list(cashflow_data.index)}"
                    ),
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
            return {
                "error": (
                    f"Fehler beim Abrufen des operativen Cashflows "
                    f"für {symbol} ({frequency}): {str(e)}"
                ),
                "symbol": symbol
            }

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
            return {
                "error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.",
                "symbol": symbol
            }

        data_type = f"revenue_{frequency}"

        cached_data = self._load_cached_data(symbol, data_type)
        if cached_data is not None and "error" not in cached_data:
            return cached_data

        try:
            financial_data = self.get_stock_financials(
                symbol=symbol,
                frequency=frequency,
                use_cache=True
            )

            if isinstance(financial_data, dict) and "error" in financial_data:
                return financial_data

            if not isinstance(financial_data, pd.DataFrame) or financial_data.empty:
                return {
                    "error": f"Keine Finanzdaten für {symbol} ({frequency}) gefunden.",
                    "symbol": symbol
                }

            latest_date = financial_data.columns[0]
            if isinstance(latest_date, pd.Timestamp):
                latest_date = latest_date.strftime("%Y-%m-%d")
            else:
                latest_date = str(latest_date)

            revenue = None

            for label in [
                "Revenue",
                "Total Revenue",
                "Sales Revenue Net"
            ]:
                if label in financial_data.index:
                    revenue = financial_data.loc[label].iloc[0]
                    if not pd.isna(revenue):
                        break

            if revenue is None or pd.isna(revenue):
                return {
                    "error": (
                        f"Kein Umsatz für {symbol} ({frequency}) gefunden. "
                        f"Verfügbare Labels: {list(financial_data.index)}"
                    ),
                    "symbol": symbol
                }

            result = {
                "revenue": float(revenue),
                "symbol": symbol,
                "frequency": frequency,
                "date": latest_date
            }

            self._cache_data(result, symbol, data_type)
            return result

        except Exception as e:
            return {
                "error": f"Fehler beim Abrufen des Umsatzes für {symbol} ({frequency}): {str(e)}",
                "symbol": symbol
            }

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
            return {
                "error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.",
                "symbol": symbol
            }

        data_type = f"inventory_{frequency}"

        cached_data = self._load_cached_data(symbol, data_type)
        if cached_data is not None:
            return cached_data

        try:
            balance_data = self.get_balance_sheet(
                symbol=symbol,
                frequency=frequency,
            )

            if isinstance(balance_data, dict) and "error" in balance_data:
                return balance_data

            if not isinstance(balance_data, pd.DataFrame) or balance_data.empty:
                return {
                    "error": f"Keine Bilanzdaten für {symbol} ({frequency}) gefunden.",
                    "symbol": symbol
                }

            latest_date = balance_data.columns[0]
            if isinstance(latest_date, pd.Timestamp):
                latest_date = latest_date.strftime("%Y-%m-%d")
            else:
                latest_date = str(latest_date)

            inventory = None

            for label in [
                "Inventory",
                "Inventories",
                "Total Inventories",
            ]:
                if label in balance_data.index:
                    inventory = balance_data.loc[label].iloc[0]
                    if pd.notna(inventory):
                        break

            if inventory is None or pd.isna(inventory):
                return {
                    "error": f"Keine Vorräte für {symbol} ({frequency}) gefunden. Verfügbare Labels: {list(balance_data.index)}",
                    "symbol": symbol
                }

            result = {
                "inventory": float(inventory),
                "symbol": symbol,
                "frequency": frequency,
                "date": latest_date
            }

            self._cache_data(result, symbol, data_type)

            return result

        except Exception as e:
            return {
                "error": f"Fehler beim Abrufen der Vorräte für {symbol} ({frequency}): {str(e)}",
                "symbol": symbol
            }

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

        if start_date and end_date and datetime.strptime(end_date, '%Y-%m-%d') <= datetime.strptime(start_date,
                                                                                                    '%Y-%m-%d'):
            return {"error": "end_date muss nach start_date liegen."}

        cache_key = f"cpi_inflation_data_{start_date}_{end_date}" if start_date and end_date else "cpi_inflation_data"

        if use_cache:
            cached_data = self._load_cached_data(cache_key, "inflation_data")
            if cached_data is not None and "error" not in cached_data:
                return cached_data

        try:
            api_key = os.environ["FRED_API_KEY"]
            series_id = 'CPIAUCSL'
            url = f'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json'

            if start_date and end_date:
                url += f'&observation_start={start_date}&observation_end={end_date}'

            response = requests.get(url)
            response.raise_for_status()

            observations = response.json()['observations']

            cpi_data = []
            for entry in observations:
                raw_value = entry.get("value")

                if raw_value in [None, "", "."]:
                    continue

                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue

                cpi_data.append({
                    'date': entry['date'],
                    'value': value
                })

            if len(cpi_data) < 2:
                return {"error": "Nicht genügend gültige CPI-Daten für die Berechnung der Inflationsrate."}

            if use_cache:
                self._cache_data(cpi_data, cache_key, "inflation_data")

            return cpi_data

        except Exception as e:
            # Nicht str(e) an den Client zurückgeben: requests hängt bei
            # HTTPError/raise_for_status die volle Request-URL inkl.
            # api_key-Query-Parameter in die Exception-Message — das würde
            # den FRED_API_KEY an den Browser leaken. Voller Fehler nur ins
            # Server-Log, Client bekommt eine sichere, aber weiterhin
            # quellenbenannte Meldung.
            self.logger.error(f"Fehler beim Abrufen der Inflationsdaten (FRED): {describe_exception(e)}")
            return {"error": "Inflationsdaten von der FRED-API aktuell nicht verfügbar. Bitte versuche es später erneut."}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_gdp_data_grpwth(self, use_cache=True):
        if use_cache:
            cached_data = self._load_cached_data("gdp_data", "gdp_value")
            if cached_data is not None and "error" not in cached_data:
                return cached_data
        try:
            api_key = os.environ["FRED_API_KEY"]
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
            # Siehe Kommentar in get_inflation_data zum FRED_API_KEY-Leak-Risiko
            # über requests-Exception-Messages — str(e) bewusst nicht an den
            # Client zurückgeben.
            self.logger.error(f"Fehler beim Abrufen der BIP-Daten (FRED): {describe_exception(e)}")
            return {"error": "BIP-Daten von der FRED-API aktuell nicht verfügbar. Bitte versuche es später erneut."}

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
    def get_enterprise_value(self, symbol: str, use_cache=True, frequency: str = "annual") -> dict:
        """
        Berechnet den Enterprise Value (EV).

        Formel:
            EV = Market Cap + Net Debt
        """

        if frequency not in ["annual", "quarterly"]:
            return {
                "error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.",
                "symbol": symbol
            }

        cache_key = f"enterprise_value{frequency}"

        if use_cache:
            cached_data = self._load_cached_data(symbol, cache_key)
            if cached_data is not None and "error" not in cached_data:
                return cached_data

        try:
            market_cap_data = self.get_market_cap(
                symbol=symbol,
                use_cache=use_cache
            )

            if isinstance(market_cap_data, dict) and "error" in market_cap_data:
                return {
                    "error": market_cap_data["error"],
                    "symbol": symbol
                }

            market_cap = market_cap_data.get("market_cap")

            if market_cap is None or pd.isna(market_cap) or market_cap <= 0:
                return {
                    "error": f"Ungültige Marktkapitalisierung für {symbol}: {market_cap}",
                    "symbol": symbol
                }

            net_debt_data = self.get_net_debt_data(
                symbol=symbol,
                frequency=frequency
            )

            if isinstance(net_debt_data, dict) and "error" in net_debt_data:
                return {
                    "error": net_debt_data["error"],
                    "symbol": symbol
                }

            net_debt = net_debt_data.get("net_debt")

            if net_debt is None or pd.isna(net_debt):
                total_debt = net_debt_data.get("total_debt")
                cash = net_debt_data.get("cash")

                if total_debt is None or pd.isna(total_debt):
                    return {
                        "error": f"Keine gültigen Schulden-Daten für {symbol} ({frequency}) verfügbar.",
                        "symbol": symbol
                    }

                if cash is None or pd.isna(cash):
                    return {
                        "error": f"Keine gültigen Liquiditäts-Daten für {symbol} ({frequency}) verfügbar.",
                        "symbol": symbol
                    }

                net_debt = float(total_debt) - float(cash)

            enterprise_value = float(market_cap) + float(net_debt)

            if pd.isna(enterprise_value) or enterprise_value <= 0:
                return {
                    "error": f"Ungültiger Enterprise Value für {symbol} ({frequency}): {enterprise_value}",
                    "symbol": symbol
                }

            result = {
                "enterprise_value": float(enterprise_value),
                "market_cap": float(market_cap),
                "net_debt": float(net_debt),
                "symbol": symbol.upper(),
                "frequency": frequency,
                "date": net_debt_data.get("date"),
                "source": "Market Cap + SEC Net Debt"
            }

            if use_cache:
                self._cache_data(result, symbol, cache_key)

            return result

        except Exception as e:
            return {
                "error": f"Fehler beim Berechnen des Enterprise Value für {symbol}: {str(e)}",
                "symbol": symbol
            }

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

    def _cache_duration_for(self, data_type: str) -> timedelta:
        lowered = data_type.lower()
        if "historical" in lowered:
            return HISTORICAL_CACHE_TTL
        if any(keyword in lowered for keyword in PRICE_SENSITIVE_CACHE_KEYWORDS):
            return LIVE_CACHE_TTL
        return FUNDAMENTAL_CACHE_TTL

    def _load_cached_data(self, symbol, data_type):
        filepath = os.path.join(self.cache_dir, f"{symbol}_{data_type}.json")
        cache_duration = self._cache_duration_for(data_type)
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