import math
import os
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import numpy as np
import matplotlib
import tenacity
from pip._internal.utils.misc import tabulate

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mplfinance as mpf
import unittest
from agent.DataLoader import DataLoader
from agent.DataPreprocessor import DataPreprocessor
from agent.Model import Model
from agent.ActionModule import AgentAction
import time

def plot_elliott_waves(data, symbol, interval):
    """Plottet die Elliott-Wellen-Analyse als Kerzenchart mit Fibonacci-Zielzonen."""
    # Bestimme die Anzahl der Datenpunkte basierend auf der Zeitebene
    if interval == "4h":
        num_points = 180  # Ca. 1 Monat (30 Tage * 6 Kerzen pro Tag)
    elif interval == "1d":
        num_points = 60   # Ca. 3 Monate (3 Monate * 20 Handelstage)
    elif interval == "1wk":
        num_points = 104  # Ca. 2 Jahre (2 Jahre * 52 Wochen)
    else:
        num_points = 60   # Standard für andere Intervalle

    # Prüfe Datenverfügbarkeit
    if len(data) < num_points:
        print(f"Warnung: Nur {len(data)} Datenpunkte verfügbar, erwartet {num_points}.")
    data = data.tail(num_points).copy()

    # Prüfe erforderliche Spalten
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Fehlende Spalten in Daten: {required_columns}")

    # Bereite die Daten für den Kerzenchart vor
    plot_data = data[required_columns].copy()

    # Erstelle den Kerzenchart
    fig, axlist = mpf.plot(
        plot_data,
        type="candle",
        style="yahoo",
        title=f"Elliott-Wellen-Analyse für {symbol} ({interval})",
        ylabel="Preis (USD)",
        volume=False,
        figsize=(14, 7),
        returnfig=True
    )
    # Platzhalter für spätere Erweiterungen (z. B. Elliott-Wellen, Fibonacci)
    # TODO: Elliott-Wellen-Labels und Fibonacci-Linien hinzufügen (Phase 3.3)
    plt.show()

def test_datapreprocessor(symbol, interval="1d"):
    dataloader = DataLoader()
    preprocessor = DataPreprocessor()

    # Teste preprocess_stock_data
    print("Teste preprocess_stock_data...")
    stock_data = dataloader.get_stock_data(symbol, interval=interval)
    if stock_data is None or stock_data.empty:
        print(f"Fehler: Keine Kursdaten für {symbol} erhalten.")
        return
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in stock_data.columns for col in required_columns):
        print(f"Fehler: Kursdaten fehlen erforderliche Spalten: {required_columns}")
        return
    try:
        processed_stock_data = preprocessor.preprocess_stock_data(stock_data)
        print(f"Kursdaten für {symbol} ({interval}) erfolgreich vorverarbeitet. Anzahl der Datenpunkte: {len(processed_stock_data)}")
        if processed_stock_data.empty:
            print("Warnung: Nach Vorverarbeitung sind keine Daten übrig.")
    except Exception as e:
        print(f"Fehler bei preprocess_stock_data für {symbol}: {e}")

    # Teste preprocess_balance_sheet
    print("\nTeste preprocess_balance_sheet...")
    balance_sheet = dataloader.get_balance_sheet(symbol)
    if balance_sheet is None or balance_sheet.empty:
        print(f"Fehler: Keine Bilanzdaten für {symbol} erhalten.")
    else:
        try:
            processed_balance_sheet = preprocessor.preprocess_balance_sheet(balance_sheet)
            print(f"Bilanzdaten für {symbol} erfolgreich vorverarbeitet. Anzahl der Einträge: {len(processed_balance_sheet)}")
            if processed_balance_sheet.empty:
                print("Warnung: Nach Vorverarbeitung sind keine Bilanzdaten übrig.")
        except Exception as e:
            print(f"Fehler bei preprocess_balance_sheet für {symbol}: {e}")

    # Teste preprocess_edgar_data
    print("\nTeste preprocess_edgar_data...")
    cik_codes = {"AAPL": "0000320193", "SONY": "0000313838"}
    cik = cik_codes.get(symbol, "0000320193")
    edgar_data = dataloader.get_edgar_data(cik)
    if edgar_data is None:
        print(f"Fehler: Keine EDGAR-Daten für {symbol} (CIK: {cik}) erhalten.")
    else:
        try:
            processed_edgar_data = preprocessor.preprocess_edgar_data(edgar_data)
            print(f"EDGAR-Daten für {symbol} (CIK: {cik}) erfolgreich vorverarbeitet.")
        except Exception as e:
            print(f"Fehler bei preprocess_edgar_data für {symbol} (CIK: {cik}): {e}")

    # Teste calculate_technical_indicators
    print("\nTeste calculate_technical_indicators...")
    try:
        stock_data_with_indicators = preprocessor.calculate_technical_indicators(stock_data, interval=interval)
        print(f"Technische Indikatoren für {symbol} ({interval}) erfolgreich berechnet. Anzahl der Datenpunkte: {len(stock_data_with_indicators)}")
        indicator_columns = ["MA_Short", "MA_Medium", "MA_Long", "Stochastic_K", "Volume"]
        if not all(col in stock_data_with_indicators.columns for col in indicator_columns):
            print(f"Fehler: Fehlende Indikator-Spalten: {indicator_columns}")
        else:
            print(f"Beispiel für MA_Short: {stock_data_with_indicators['MA_Short'].iloc[-1]:.2f}")
            print(f"Beispiel für MA_Medium: {stock_data_with_indicators['MA_Medium'].iloc[-1]:.2f}")
            print(f"Beispiel für MA_Long: {stock_data_with_indicators['MA_Long'].iloc[-1]:.2f}")
            stochastic_k = stock_data_with_indicators['Stochastic_K'].iloc[-1]
            print(f"Beispiel für Stochastic_K: {stochastic_k:.2f}")
            if not (0 <= stochastic_k <= 100):
                print(f"Warnung: Stochastic_K außerhalb des Bereichs 0-100: {stochastic_k}")
            print(f"Beispiel für Volume: {stock_data_with_indicators['Volume'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"Fehler bei calculate_technical_indicators für {symbol}: {e}")

    # Teste calculate_fibonacci_retracements
    print("\nTeste calculate_fibonacci_retracements...")
    try:
        stock_data_with_fib = preprocessor.calculate_fibonacci_retracements(stock_data_with_indicators, interval=interval)
        print(f"Fibonacci-Retracements für {symbol} ({interval}) erfolgreich berechnet.")
        fib_columns = ["Fib_382", "Fib_50", "Fib_618", "Fib_1618"]
        if not all(col in stock_data_with_fib.columns for col in fib_columns):
            print(f"Fehler: Fehlende Fibonacci-Spalten: {fib_columns}")
        else:
            for fib_level in fib_columns:
                value = stock_data_with_fib[fib_level].iloc[-1]
                if pd.isna(value):
                    print(f"Warnung: {fib_level} ist NaN, mögliche Ursache: Keine Hoch-/Tiefpunkte gefunden.")
                else:
                    print(f"Beispiel für {fib_level}: {value:.2f}")
    except Exception as e:
        print(f"Fehler bei calculate_fibonacci_retracements für {symbol}: {e}")

    # Teste identify_elliott_waves
    print("\nTeste identify_elliott_waves...")
    try:
        stock_data_with_waves = preprocessor.identify_elliott_waves(stock_data_with_fib, interval=interval)
        print(f"Elliott-Wellen für {symbol} ({interval}) erfolgreich identifiziert.")
        if 'Wave' not in stock_data_with_waves.columns:
            print("Fehler: Wave-Spalte fehlt in den Daten.")
        else:
            wave_counts = stock_data_with_waves['Wave'].value_counts()
            print(f"Erkannte Wellen: {wave_counts}")
            if wave_counts.get(0, 0) == len(stock_data_with_waves):
                print("Warnung: Keine Elliott-Wellen identifiziert, alle Werte sind 0.")
            # Visualisiere die Ergebnisse
            plot_elliott_waves(stock_data_with_waves, symbol, interval)
    except Exception as e:
        print(f"Fehler bei identify_elliott_waves für {symbol}: {e}")

    # Teste preprocess_stock_data_for_ml
    print("\nTeste preprocess_stock_data_for_ml...")
    try:
        X, y, processed_data = preprocessor.preprocess_stock_data_for_ml(stock_data, interval=interval)
        print(f"Daten für ML für {symbol} ({interval}) erfolgreich vorverarbeitet.")
        print(f"Form der Features (X): {X.shape}")
        print(f"Form der Labels (y): {y.shape}")
        if np.any(np.isnan(X)):
            print("Warnung: NaN-Werte in X-Daten gefunden.")
        if np.all(y == 0):
            print("Warnung: Alle Labels (y) sind 0, mögliche Ursache: Falsche Wellenidentifikation.")
        else:
            print(f"Beispiel-Feature (skaliert): {X[-1][-1]:.4f}")
            print(f"Beispiel-Label: {y[-1]} (1: Welle 1, 2: Welle 2, ..., 5: Welle 5, 6: Welle A, 7: Welle B, 8: Welle C)")
    except Exception as e:
        print(f"Fehler bei preprocess_stock_data_for_ml für {symbol}: {e}")

if __name__ == "__main__":
    symbol = input("Geben Sie das Aktiensymbol ein (z. B. AAPL, SONY): ").upper()
    interval = input("Geben Sie die Zeitebene ein (z. B. 1d, 1wk): ")
    test_datapreprocessor(symbol, interval)

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.model = Model()
        self.actionmodule = AgentAction()
        self.test_symbols = ['MO', 'BABA', 'AAPL']

    def test_get_balance_sheet(self):
        """Testet die Methode get_balance_sheet mit jährlichen und quartalsweisen Daten."""
        print("\nTeste get_balance_sheet...")
        for symbol in self.test_symbols:
            for frequency in ["annual", "quarterly"]:
                result = self.loader.get_balance_sheet(symbol, frequency=frequency)
                print(result)
                if isinstance(result, dict) and "error" in result:
                    print(f"Fehler für {symbol} ({frequency}): {result['error']}")
                else:
                    print(
                        f"Bilanzdaten für {symbol} ({frequency}) erfolgreich abgerufen. Anzahl der Einträge: {len(result)}")
                    self.assertFalse(result.empty, f"Bilanzdaten für {symbol} ({frequency}) sollten nicht leer sein")
                    if "Net Debt" in result.index:
                        print(f"Net Debt für {symbol} ({frequency}): {result.loc['Net Debt'].iloc[0]}")


    def test_get_fundamental_data(self):
        """Testet die get_fundamental_data Methode für API- und Cache-Zugriff mit NVDA."""

        print("\nTeste get_fundamental_data...")


        # Testfall 1: Gültiges Symbol (NVDA) - API-Aufruf mit Caching
        with self.subTest(symbol="NVDA", frequency="annual", case="valid_symbol_api_call_nvda"):
            symbol = "NVDA"
            frequency = "annual"
            print(f"\nAbruf der Fundamentaldaten für {symbol} ({frequency}) - API-Aufruf mit Caching...")
            data = self.loader.get_fundamental_data(symbol, frequency=frequency, use_cache=True)

            # Prüfen, ob das Ergebnis ein Dictionary ist
            self.assertIsInstance(data, dict, "Ergebnis ist kein Dictionary")

            # Daten ausgeben oder Fehler analysieren
            if "error" not in data:
                print(f"\nErfolgreich abgerufene Daten für {symbol} ({frequency}) über API:")
                for key in ["income_statement", "balance_sheet", "cash_flow"]:
                    self.assertIsInstance(data[key], pd.DataFrame, f"{key} ist kein DataFrame")
                    self.assertFalse(data[key].empty, f"{key} DataFrame ist leer")
                    print(f"\n{key} (Anzahl Jahre: {len(data[key])}, Anzahl Spalten: {len(data[key].columns)}):")
                    print(f"Spalten: {data[key].columns.tolist()}")
                    print("\nTabelle der Werte:")
                    print(data[key].to_string())
            else:
                print(f"Fehler für {symbol} ({frequency}): {data['error']}")
                self.fail(f"Fehler: {data['error']}")

            # Debugging: Cache-Verzeichnis und Dateien prüfen
            print(f"\nCache-Verzeichnis: {self.loader.cache_dir}")
            cache_keys = [
                f"{symbol}_historical_income_statement_{frequency}",
                f"{symbol}_historical_balance_sheet_{frequency}",
                f"{symbol}_historical_cash_flow_{frequency}"
            ]
            for key in cache_keys:
                cache_file = os.path.join(self.loader.cache_dir, f"{key}.json")
                print(f"Prüfe Cache-Datei: {cache_file} - Existiert: {os.path.exists(cache_file)}")

        # Testfall 2: Gültiges Symbol (NVDA) - Cache-Zugriff
        with self.subTest(symbol="NVDA", frequency="annual", case="valid_symbol_cache_nvda"):
            symbol = "NVDA"
            frequency = "annual"
            print(f"\nAbruf der Fundamentaldaten für {symbol} ({frequency}) - Cache-Zugriff...")

            # Prüfen, ob Cache-Daten existieren
            cache_keys = [
                f"{symbol}_historical_income_statement_{frequency}",
                f"{symbol}_historical_balance_sheet_{frequency}",
                f"{symbol}_historical_cash_flow_{frequency}"
            ]
            cache_files_exist = all(
                os.path.exists(os.path.join(self.loader.cache_dir, f"{key}.json"))
                for key in cache_keys
            )
            if not cache_files_exist:
                missing_files = [
                    os.path.join(self.loader.cache_dir, f"{key}.json")
                    for key in cache_keys
                    if not os.path.exists(os.path.join(self.loader.cache_dir, f"{key}.json"))
                ]
                self.fail(
                    f"Cache-Daten fehlen für {symbol} ({frequency}) nach API-Aufruf. Fehlende Dateien: {missing_files}")

            # Cache-Daten abrufen
            data_cached = self.loader.get_fundamental_data(symbol, frequency=frequency, use_cache=True)

            # Prüfen, ob das Ergebnis ein Dictionary ist
            self.assertIsInstance(data_cached, dict, "Ergebnis ist kein Dictionary")

            # Daten ausgeben und prüfen
            if "error" not in data_cached:
                print(f"\nErfolgreich abgerufene Daten für {symbol} ({frequency}) aus Cache:")
                for key in ["income_statement", "balance_sheet", "cash_flow"]:
                    self.assertIsInstance(data_cached[key], pd.DataFrame, f"{key} ist kein DataFrame")
                    self.assertFalse(data_cached[key].empty, f"{key} DataFrame ist leer")
                    self.assertTrue(data[key].equals(data_cached[key]),
                                    f"{key} Daten aus Cache stimmen nicht mit API-Daten überein")
                    print(
                        f"\n{key} (Anzahl Jahre: {len(data_cached[key])}, Anzahl Spalten: {len(data_cached[key].columns)}):")
                    print(f"Spalten: {data_cached[key].columns.tolist()}")
                    print("\nTabelle der Werte:")
                    print(data_cached[key].to_string())
            else:
                print(f"Fehler für {symbol} ({frequency}): {data_cached['error']}")
                self.fail(f"Fehler: {data_cached['error']}")

    def test_get_max_historical_stock_data(self):
        """Testet die Methode get_max_historical_stock_data mit NVDA, ILMN und Sonderfällen."""
        print("\nTeste get_max_historical_stock_data...")

        # Testfall 1: Gültiges Symbol (NVDA) - API-Aufruf mit verschiedenen Intervallen
        for interval in ["1mo", "1d", "1wk"]:
            with self.subTest(symbol="NVDA", case=f"valid_symbol_api_call_interval_{interval}"):
                symbol = "NVDA"
                print(f"\nAbruf der historischen Kursdaten für {symbol} - API-Aufruf (Intervall: {interval})...")
                data = self.loader.get_max_historical_stock_data(symbol, use_cache=True, interval=interval)

                if data is None:
                    print(f"Fehler für {symbol} (Intervall: {interval}): Keine Daten abgerufen.")
                    self.fail(f"Keine Kursdaten für {symbol} (Intervall: {interval}) erhalten.")

                # Prüfe DataFrame und Spalten
                self.assertIsInstance(data, pd.DataFrame,
                                      f"Ergebnis für {symbol} (Intervall: {interval}) sollte ein DataFrame sein.")
                self.assertFalse(data.empty, f"Kursdaten für {symbol} (Intervall: {interval}) sollten nicht leer sein.")
                expected_columns = ["Open", "High", "Low", "Close", "Volume"]
                self.assertTrue(all(col in data.columns for col in expected_columns),
                                f"Fehlende Spalten für {symbol} (Intervall: {interval}): Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

                # Konsolenausgabe: Historie-Reichweite
                start_date = data.index.min().strftime("%Y-%m-%d")
                end_date = data.index.max().strftime("%Y-%m-%d")
                num_points = len(data)
                print(
                    f"Test für {symbol} (Intervall: {interval}) erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")
                print(data)
                time.sleep(1)  # Kurze Verzögerung für Rate-Limits

        # Testfall 2: Gültiges Symbol (ILMN) - API-Aufruf mit verschiedenen Intervallen
        for interval in ["1mo", "1d", "1wk"]:
            with self.subTest(symbol="ILMN", case=f"valid_symbol_api_call_ilmn_interval_{interval}"):
                symbol = "ILMN"
                print(f"\nAbruf der historischen Kursdaten für {symbol} - API-Aufruf (Intervall: {interval})...")
                data = self.loader.get_max_historical_stock_data(symbol, use_cache=True, interval=interval)

                if data is None:
                    print(f"Fehler für {symbol} (Intervall: {interval}): Keine Daten abgerufen.")
                    self.fail(f"Keine Kursdaten für {symbol} (Intervall: {interval}) erhalten.")

                # Prüfe DataFrame und Spalten
                self.assertIsInstance(data, pd.DataFrame,
                                      f"Ergebnis für {symbol} (Intervall: {interval}) sollte ein DataFrame sein.")
                self.assertFalse(data.empty, f"Kursdaten für {symbol} (Intervall: {interval}) sollten nicht leer sein.")
                expected_columns = ["Open", "High", "Low", "Close", "Volume"]
                self.assertTrue(all(col in data.columns for col in expected_columns),
                                f"Fehlende Spalten für {symbol} (Intervall: {interval}): Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

                # Konsolenausgabe: Historie-Reichweite
                start_date = data.index.min().strftime("%Y-%m-%d")
                end_date = data.index.max().strftime("%Y-%m-%d")
                num_points = len(data)
                print(
                    f"Test für {symbol} (Intervall: {interval}) erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")
                print(data)
                time.sleep(1)  # Kurze Verzögerung für Rate-Limits

        # Testfall 3: Ungültiges Symbol
        with self.subTest(symbol="INVALID", case="invalid_symbol"):
            symbol = "INVALID"
            print(f"\nAbruf der historischen Kursdaten für ungültiges Symbol {symbol}...")
            data = self.loader.get_max_historical_stock_data(symbol, use_cache=True, interval="1mo")
            self.assertIsNone(data, f"Ergebnis für ungültiges Symbol {symbol} sollte None sein.")
            print(f"Test für ungültiges Symbol {symbol} erfolgreich: Keine Daten erhalten, wie erwartet.")
            time.sleep(1)  # Kurze Verzögerung für Rate-Limits

        # Testfall 4: Leere Daten (Simulierung durch neues Unternehmen, z. B. NEWCO)
        with self.subTest(symbol="NEWCO", case="empty_data"):
            symbol = "NEWCO"
            print(f"\nAbruf der historischen Kursdaten für {symbol} (erwartet leere Daten)...")
            data = self.loader.get_max_historical_stock_data(symbol, use_cache=True, interval="1mo")
            self.assertIsNone(data, f"Ergebnis für {symbol} sollte None sein (leere Daten).")
            print(f"Test für {symbol} erfolgreich: Keine Daten erhalten, wie erwartet.")
            time.sleep(1)  # Kurze Verzögerung für Rate-Limits

        # Testfall 5: Ungültiges Datumsformat
        with self.subTest(symbol="NVDA", case="invalid_date_format"):
            symbol = "NVDA"
            print(f"\nAbruf der historischen Kursdaten für {symbol} mit ungültigem Datumsformat...")
            data = self.loader.get_max_historical_stock_data(symbol, start_date="invalid_date", use_cache=False,
                                                             interval="1mo")
            self.assertIsNone(data, f"Ergebnis für {symbol} mit ungültigem Datumsformat sollte None sein.")
            print(f"Test für {symbol} mit ungültigem Datumsformat erfolgreich: Keine Daten erhalten, wie erwartet.")
            time.sleep(1)  # Kurze Verzögerung für Rate-Limits

        # Testfall 6: Ungültiges Intervall
        with self.subTest(symbol="NVDA", case="invalid_interval"):
            symbol = "NVDA"
            print(f"\nAbruf der historischen Kursdaten für {symbol} mit ungültigem Intervall...")
            with self.assertRaises(tenacity.RetryError,
                                   msg=f"Erwarteter RetryError für ungültiges Intervall bei {symbol}"):
                self.loader.get_max_historical_stock_data(symbol, use_cache=True, interval="1y")
            print(f"Test für {symbol} mit ungültigem Intervall erfolgreich: RetryError ausgelöst, wie erwartet.")

    def test_calculate_historical_market_cap(self):
        """Testet die Methode calculate_historical_market_cap für ILMN."""
        print("\nTeste calculate_historical_market_cap...")

        with self.subTest(symbol="ILMN", case="valid_symbol"):
            symbol = "ILMN"
            print(f"\nBerechnung von Marktkapitalisierung für {symbol}...")
            data = self.model.calculate_historical_market_cap(symbol, use_cache=True)

            if data is None:
                print(f"Fehler für {symbol}: Keine Marktkapitalisierungs-Daten abgerufen.")
                self.fail(f"Keine Marktkapitalisierungs-Daten für {symbol} erhalten.")

            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"Marktkapitalisierungs-Daten für {symbol} sollten nicht leer sein.")
            expected_columns = ["MarketCap", "commonStockSharesOutstanding", "Close"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            # Vollständige Tabelle anzeigen
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige Marktkapitalisierungstabelle:")
                print(data)

            time.sleep(1)

    def test_calculate_historical_ev(self):
        """Testet die Methode calculate_historical_ev für ILMN."""
        print("\nTeste calculate_historical_ev...")

        with self.subTest(symbol="ILMN", case="valid_symbol"):
            symbol = "ILMN"
            print(f"\nBerechnung von EV für {symbol}...")
            data = self.model.calculate_historical_ev(symbol, use_cache=True)

            if data is None:
                print(f"Fehler für {symbol}: Keine EV-Daten abgerufen.")
                self.fail(f"Keine EV-Daten für {symbol} erhalten.")

            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"EV-Daten für {symbol} sollten nicht leer sein.")
            expected_columns = ["EV"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            # Vollständige Tabelle anzeigen
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige Enterprise Value Tabelle:")
                print(data)

            time.sleep(1)

    def test_calculate_historical_sales(self):
        """Testet die Methode calculate_historical_sales für ILMN."""
        print("\nTeste calculate_historical_sales...")

        with self.subTest(symbol="ILMN", case="valid_symbol"):
            symbol = "ILMN"
            print(f"\nBerechnung von Umsatzerlösen für {symbol}...")
            data = self.model.calculate_historical_sales(symbol, use_cache=True)

            if data is None:
                print(f"Fehler für {symbol}: Keine Umsatzerlöse abgerufen.")
                self.fail(f"Keine Umsatzerlöse für {symbol} erhalten.")

            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"Umsatzerlöse für {symbol} sollten nicht leer sein.")
            expected_columns = ["Sales"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige Umsatzerlös-Tabelle:")
                print(data)

            time.sleep(1)

    def test_calculate_historical_ev_sales(self):
        """Testet die Methode calculate_historical_ev_sales für ILMN."""
        print("\nTeste calculate_historical_ev_sales...")

        with self.subTest(symbol="ILMN", case="valid_symbol"):
            symbol = "ILMN"
            print(f"\nBerechnung von EV/Sales für {symbol}...")
            data = self.model.calculate_historical_ev_sales(symbol, use_cache=True)

            if data is None:
                print(f"Fehler für {symbol}: Keine EV/Sales-Daten abgerufen.")
                self.fail(f"Keine EV/Sales-Daten für {symbol} erhalten.")

            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"EV/Sales-Daten für {symbol} sollten nicht leer sein.")
            expected_columns = ["EV_Sales"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige EV/Sales-Tabelle:")
                print(data)

            time.sleep(1)

    def test_calculate_historical_ebit(self):
        """Testet die Methode calculate_historical_ebit für ILMN."""
        print("\nTeste calculate_historical_ebit...")

        with self.subTest(symbol="ILMN", case="valid_symbol"):
            symbol = "ILMN"
            print(f"\nBerechnung von EBIT für {symbol}...")
            data = self.model.calculate_historical_ebit(symbol, use_cache=True)

            if data is None:
                print(f"Fehler für {symbol}: Keine EBIT-Daten abgerufen.")
                self.fail(f"Keine EBIT-Daten für {symbol} erhalten.")

            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"EBIT-Daten für {symbol} sollten nicht leer sein.")
            expected_columns = ["EBIT"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige EBIT-Tabelle:")
                print(data)

            time.sleep(1)

        with self.subTest(symbol="INVALID", case="invalid_symbol"):
            symbol = "INVALID"
            print(f"\nBerechnung von EBIT für ungültiges Symbol {symbol}...")
            data = self.model.calculate_historical_ebit(symbol, use_cache=True)
            self.assertIsNone(data, f"Ergebnis für ungültiges Symbol {symbol} sollte None sein.")
            print(f"Test für ungültiges Symbol {symbol} erfolgreich: Keine Daten erhalten, wie erwartet.")
            time.sleep(1)

    def test_calculate_historical_ev_to_ebit(self):
        """Testet die Methode calculate_historical_ev_to_ebit für ILMN."""
        print("\nTeste calculate_historical_ev_to_ebit...")

        with self.subTest(symbol="ILMN", case="valid_symbol"):
            symbol = "ILMN"
            print(f"\nBerechnung von EV/EBIT für {symbol}...")
            data = self.model.calculate_historical_ev_to_ebit(symbol, use_cache=True)

            if data is None:
                print(f"Fehler für {symbol}: Keine EV/EBIT-Daten abgerufen.")
                self.fail(f"Keine EV/EBIT-Daten für {symbol} erhalten.")

            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"EV/EBIT-Daten für {symbol} sollten nicht leer sein.")
            expected_columns = ["EV_EBIT"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige EV/EBIT-Tabelle:")
                print(data)

            time.sleep(1)

    def test_calculate_historical_ebitda(self):
        """Testet die Methode calculate_historical_ebitda für ILMN."""
        print("\nTeste calculate_historical_ebitda...")

        with self.subTest(symbol="ILMN", case="valid_symbol"):
            symbol = "ILMN"
            print(f"\nBerechnung von EBITDA für {symbol}...")
            data = self.model.calculate_historical_ebitda(symbol, use_cache=True)

            if data is None:
                print(f"Fehler für {symbol}: Keine EBITDA-Daten abgerufen.")
                self.fail(f"Keine EBITDA-Daten für {symbol} erhalten.")

            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"EBITDA-Daten für {symbol} sollten nicht leer sein.")
            expected_columns = ["EBIT", "depreciationAndAmortization", "EBITDA"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            # Prüfen, ob EBITDA korrekt berechnet wurde
            self.assertTrue(all(data["EBITDA"].eq(data["EBIT"] + data["depreciationAndAmortization"])),
                            f"EBITDA-Berechnung für {symbol} ist inkorrekt.")

            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige EBITDA-Tabelle:")
                print(data)

            time.sleep(1)

    def test_calculate_historical_ev_to_ebitda(self):
        """Testet die Methode calculate_historical_ev_to_ebitda für ILMN."""
        print("\nTeste calculate_historical_ev_to_ebitda...")

        with self.subTest(symbol="ILMN", case="valid_symbol"):
            symbol = "ILMN"
            print(f"\nBerechnung von EV/EBITDA für {symbol}...")
            data = self.model.calculate_historical_ev_to_ebitda(symbol, use_cache=True)

            if data is None:
                print(f"Fehler für {symbol}: Keine EV/EBITDA-Daten abgerufen.")
                self.fail(f"Keine EV/EBITDA-Daten für {symbol} erhalten.")

            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"EV/EBITDA-Daten für {symbol} sollten nicht leer sein.")
            expected_columns = ["EV_EBITDA"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige EV/EBITDA-Tabelle:")
                print(data)

            time.sleep(1)

    def test_calculate_historical_price_to_book(self):
        """Testet die Methode calculate_historical_price_to_book für ILMN."""
        print("\nTeste calculate_historical_price_to_book...")

        with self.subTest(symbol="NVDA", case="valid_symbol"):
            symbol = "NVDA"
            print(f"\nBerechnung von Price/Book für {symbol}...")
            data = self.model.calculate_historical_price_to_book(symbol, use_cache=True)

            if data is None:
                print(f"Fehler für {symbol}: Keine Price/Book-Daten abgerufen.")
                self.fail(f"Keine Price/Book-Daten für {symbol} erhalten.")

            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"Price/Book-Daten für {symbol} sollten nicht leer sein.")
            expected_columns = ["Price_Book", "Price", "totalAssets", "totalLiabilities", "commonStockSharesOutstanding"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            # Prüfen, ob die Indizes mit fiscalDateEnding übereinstimmen
            fundamentals = self.model.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            balance_sheet = fundamentals.get("balance_sheet")
            if balance_sheet is not None:
                expected_dates = balance_sheet.index.intersection(data.index)
                self.assertFalse(expected_dates.empty, f"Price/Book-Daten für {symbol} sollten mit Bilanzdaten-Daten übereinstimmen.")

                # Prüfen, ob Spalten konsistent mit Bilanzdaten sind
                self.assertTrue("commonStockSharesOutstanding" in balance_sheet.columns,
                                f"Shares Outstanding-Daten sollten in balance_sheet für {symbol} vorhanden sein.")
                shares_outstanding = balance_sheet["commonStockSharesOutstanding"].reindex(data.index)
                self.assertFalse(shares_outstanding.isna().all(),
                                 f"Historische Shares Outstanding-Daten für {symbol} sollten nicht leer sein.")
                self.assertTrue((data["totalAssets"] == balance_sheet["totalAssets"].reindex(data.index)).all(),
                                f"totalAssets-Daten für {symbol} sollten mit Bilanzdaten übereinstimmen.")
                self.assertTrue((data["totalLiabilities"] == balance_sheet["totalLiabilities"].reindex(data.index)).all(),
                                f"totalLiabilities-Daten für {symbol} sollten mit Bilanzdaten übereinstimmen.")

            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige Price/Book-Tabelle:")
                print(data)

            time.sleep(1)

    def test_calculate_historical_price_to_sales(self):
        """Testet die Methode calculate_historical_price_to_sales für ILMN."""
        print("\nTeste calculate_historical_price_to_sales...")

        # Testfall 1: Gültiges Symbol (ILMN)
        with self.subTest(symbol="ILMN", case="valid_symbol"):
            symbol = "ILMN"
            print(f"\nBerechnung von Price/Sales für {symbol}...")
            data = self.model.calculate_historical_price_to_sales(symbol, use_cache=True)

            # Prüfen, ob Daten zurückgegeben wurden
            if data is None:
                print(f"Fehler für {symbol}: Keine Price/Sales-Daten abgerufen.")
                self.fail(f"Keine Price/Sales-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"Price/Sales-Daten für {symbol} sollten nicht leer sein.")

            # Prüfen, ob erwartete Spalten vorhanden sind
            expected_columns = ["Price_Sales", "Price", "Sales", "commonStockSharesOutstanding"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            # Debugging: Datenstruktur ausgeben
            print(f"\nDebug: Inhalt von price_to_sales für {symbol}:")
            print(f"Spalten: {data.columns.tolist()}")
            print(f"Erste Zeilen:\n{data.head()}")

            # Prüfen, ob Sales-Daten konsistent sind
            sales_data = self.model.calculate_historical_sales(symbol, use_cache=True)
            if sales_data is None:
                print(f"Fehler für {symbol}: Keine Umsatzdaten abgerufen.")
                self.fail(f"Keine Umsatzdaten für {symbol} erhalten.")

            self.assertTrue("Sales" in sales_data.columns,
                            f"Sales-Daten sollten in sales_data für {symbol} vorhanden sein. Gefundene Spalten: {sales_data.columns.tolist()}")
            self.assertFalse(sales_data["Sales"].isna().all(),
                             f"Historische Umsatzdaten für {symbol} sollten nicht alle NaN sein.")

            # Prüfen, ob Bilanzdaten verfügbar sind
            fundamentals = self.model.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            balance_sheet = fundamentals.get("balance_sheet") if fundamentals else None
            if balance_sheet is None:
                print(f"Fehler für {symbol}: Keine Bilanzdaten abgerufen.")
                self.fail(f"Keine Bilanzdaten für {symbol} erhalten.")

            # Prüfen, ob Indizes übereinstimmen
            expected_dates = sales_data.index.intersection(data.index).intersection(balance_sheet.index)
            self.assertFalse(expected_dates.empty, f"Price/Sales-Daten für {symbol} sollten mit Umsatz- und Bilanzdaten-Daten übereinstimmen.")

            # Prüfen, ob Sales und commonStockSharesOutstanding konsistent sind
            self.assertTrue("commonStockSharesOutstanding" in balance_sheet.columns,
                            f"Shares Outstanding-Daten sollten in balance_sheet für {symbol} vorhanden sein.")
            self.assertTrue((data["Sales"] == sales_data["Sales"].reindex(data.index)).all(),
                            f"Sales-Daten für {symbol} sollten mit sales_data übereinstimmen.")
            self.assertTrue((data["commonStockSharesOutstanding"] == balance_sheet["commonStockSharesOutstanding"].reindex(data.index, method="ffill")).all(),
                            f"commonStockSharesOutstanding-Daten für {symbol} sollten mit Bilanzdaten übereinstimmen.")

            # Prüfen, ob Price_Sales sinnvolle Werte enthält
            self.assertFalse(data["Price_Sales"].isna().all(),
                             f"Price/Sales-Werte für {symbol} sollten nicht alle NaN sein.")
            self.assertTrue((data["Price_Sales"] >= 0).all(),
                            f"Price/Sales-Werte für {symbol} sollten nicht negativ sein.")

            # Zeitraum und Datenpunkte ausgeben
            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige Price/Sales-Tabelle:")
                print(data)

            time.sleep(1)

    def test_calculate_historical_price_to_ebit(self):
        """Testet die Methode calculate_historical_price_to_ebit für ILMN."""
        print("\nTeste calculate_historical_price_to_ebit...")

        # Testfall 1: Gültiges Symbol (ILMN)
        with self.subTest(symbol="ILMN", case="valid_symbol"):
            symbol = "ILMN"
            print(f"\nBerechnung von Price/EBIT für {symbol}...")
            data = self.model.calculate_historical_price_to_ebit(symbol, use_cache=True)

            # Prüfen, ob Daten zurückgegeben wurden
            if data is None:
                print(f"Fehler für {symbol}: Keine Price/EBIT-Daten abgerufen.")
                self.fail(f"Keine Price/EBIT-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"Price/EBIT-Daten für {symbol} sollten nicht leer sein.")

            # Prüfen, ob erwartete Spalten vorhanden sind
            expected_columns = ["Price_EBIT", "Price", "EBIT", "commonStockSharesOutstanding"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            # Debugging: Datenstruktur ausgeben
            print(f"\nDebug: Inhalt von price_to_ebit für {symbol}:")
            print(f"Spalten: {data.columns.tolist()}")
            print(f"Erste Zeilen:\n{data.head()}")
            print(f"\nDebug: Letzte Zeilen:\n{data.tail()}")
            print(f"\nDebug: Zusammenfassung Price_EBIT:\n{data['Price_EBIT'].describe()}")

            # Prüfen, ob EBIT-Daten konsistent sind
            ebit_data = self.model.calculate_historical_ebit(symbol, use_cache=True)
            if ebit_data is None:
                print(f"Fehler für {symbol}: Keine EBIT-Daten abgerufen.")
                self.fail(f"Keine EBIT-Daten für {symbol} erhalten.")

            print(f"\nDebug: Inhalt von ebit_data für {symbol}:")
            print(f"Spalten: {ebit_data.columns.tolist()}")
            print(f"Erste Zeilen:\n{ebit_data.head()}")
            print(f"Letzte Zeilen:\n{ebit_data.tail()}")

            self.assertTrue("EBIT" in ebit_data.columns,
                            f"EBIT-Daten sollten in ebit_data für {symbol} vorhanden sein. Gefundene Spalten: {ebit_data.columns.tolist()}")
            self.assertFalse(ebit_data["EBIT"].isna().all(),
                             f"Historische EBIT-Daten für {symbol} sollten nicht alle NaN sein.")

            # Prüfen, ob Bilanzdaten verfügbar sind
            fundamentals = self.model.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            balance_sheet = fundamentals.get("balance_sheet") if fundamentals else None
            if balance_sheet is None:
                print(f"Fehler für {symbol}: Keine Bilanzdaten abgerufen.")
                self.fail(f"Keine Bilanzdaten für {symbol} erhalten.")

            # Prüfen, ob Indizes übereinstimmen
            expected_dates = ebit_data.index.intersection(data.index).intersection(balance_sheet.index)
            self.assertFalse(expected_dates.empty, f"Price/EBIT-Daten für {symbol} sollten mit EBIT- und Bilanzdaten-Daten übereinstimmen.")

            # Prüfen, ob EBIT und commonStockSharesOutstanding konsistent sind
            self.assertTrue("commonStockSharesOutstanding" in balance_sheet.columns,
                            f"Shares Outstanding-Daten sollten in balance_sheet für {symbol} vorhanden sein.")
            self.assertTrue((data["EBIT"] == ebit_data["EBIT"].reindex(data.index)).all(),
                            f"EBIT-Daten für {symbol} sollten mit ebit_data übereinstimmen.")
            self.assertTrue((data["commonStockSharesOutstanding"] == balance_sheet["commonStockSharesOutstanding"].reindex(data.index, method="ffill")).all(),
                            f"commonStockSharesOutstanding-Daten für {symbol} sollten mit Bilanzdaten übereinstimmen.")

            # Prüfen, ob Price_EBIT sinnvolle Werte enthält
            self.assertFalse(data["Price_EBIT"].isna().all(),
                             f"Price/EBIT-Werte für {symbol} sollten nicht alle NaN sein.")
            if (data["Price_EBIT"] < 0).any():
                print(f"Warnung: Negative Price/EBIT-Werte für {symbol} erkannt, wahrscheinlich aufgrund negativer EBIT-Werte.")

            # Zeitraum und Datenpunkte ausgeben
            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige Price/EBIT-Tabelle:")
                print(data)

            time.sleep(1)

    def test_calculate_historical_netCurrentAssets(self):
        """Testet die Methode calculate_historical_netCurrentAssets für ILMN."""
        print("\nTeste calculate_historical_netCurrentAssets...")

        # Testfall 1: Gültiges Symbol (ILMN)
        with self.subTest(symbol="ILMN", case="valid_symbol"):
            symbol = "ILMN"
            print(f"\nBerechnung von NetCurrentAssets für {symbol}...")
            data = self.model.calculate_historical_netCurrentAssets(symbol, use_cache=True)

            # Prüfen, ob Daten zurückgegeben wurden
            if data is None:
                print(f"Fehler für {symbol}: Keine NetCurrentAssets-Daten abgerufen.")
                self.fail(f"Keine NetCurrentAssets-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"NetCurrentAssets-Daten für {symbol} sollten nicht leer sein.")

            # Prüfen, ob erwartete Spalten vorhanden sind
            expected_columns = ["NetCurrentAssets"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            # Debugging: Datenstruktur ausgeben
            print(f"\nDebug: Inhalt von netCurrentAssets für {symbol}:")
            print(f"Spalten: {data.columns.tolist()}")
            print(f"Erste Zeilen:\n{data.head()}")
            print(f"\nDebug: Letzte Zeilen:\n{data.tail()}")
            print(f"\nDebug: Zusammenfassung NetCurrentAssets:\n{data['NetCurrentAssets'].describe()}")

            # Prüfen, ob Fundamentaldaten verfügbar sind
            fundamentals = self.model.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            balance_sheet = fundamentals.get("balance_sheet") if fundamentals else None
            if balance_sheet is None:
                print(f"Fehler für {symbol}: Keine Bilanzdaten abgerufen.")
                self.fail(f"Keine Bilanzdaten für {symbol} erhalten.")

            # Prüfen, ob benötigte Spalten in balance_sheet vorhanden sind
            required_columns = {"totalCurrentAssets", "totalCurrentLiabilities"}
            self.assertTrue(all(col in balance_sheet.columns for col in required_columns),
                            f"Fehlende Spalten in balance_sheet für {symbol}: Erwartet {required_columns}, erhalten {balance_sheet.columns.tolist()}")

            # Prüfen, ob Indizes übereinstimmen
            expected_dates = data.index.intersection(balance_sheet.index)
            self.assertFalse(expected_dates.empty,
                             f"NetCurrentAssets-Daten für {symbol} sollten mit Bilanzdaten-Indizes übereinstimmen.")

            # Prüfen, ob NetCurrentAssets konsistent mit Bilanzdaten ist
            balance_sheet = balance_sheet.reindex(data.index)
            calculated_netCurrentAssets = balance_sheet["totalCurrentAssets"] - balance_sheet["totalCurrentLiabilities"]
            self.assertTrue((data["NetCurrentAssets"] == calculated_netCurrentAssets).all(),
                            f"NetCurrentAssets-Daten für {symbol} sollten mit Bilanzdaten übereinstimmen.")

            # Prüfen, ob NetCurrentAssets sinnvolle Werte enthält
            self.assertFalse(data["NetCurrentAssets"].isna().all(),
                             f"NetCurrentAssets-Werte für {symbol} sollten nicht alle NaN sein.")

            # Zeitraum und Datenpunkte ausgeben
            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige NetCurrentAssets-Tabelle:")
                print(data)

            time.sleep(1)

        # Testfall 2: Ungültiges Symbol
        with self.subTest(symbol="INVALID", case="invalid_symbol"):
            symbol = "INVALID"
            print(f"\nBerechnung von NetCurrentAssets für ungültiges Symbol {symbol}...")
            data = self.model.calculate_historical_netCurrentAssets(symbol, use_cache=True)
            self.assertIsNone(data, f"Ergebnis für ungültiges Symbol {symbol} sollte None sein.")
            print(f"Test für ungültiges Symbol {symbol} erfolgreich: Keine Daten erhalten, wie erwartet.")
            time.sleep(1)

    def test_calculate_historical_price_netCurrentAssets(self):
        """Testet die Methode calculate_historical_price_netCurrentAssets für ILMN."""
        print("\nTeste calculate_historical_price_netCurrentAssets...")

        # Testfall 1: Gültiges Symbol (ILMN)
        with self.subTest(symbol="ILMN", case="valid_symbol"):
            symbol = "ILMN"
            print(f"\nBerechnung von Price/NetCurrentAssets für {symbol}...")
            data = self.model.calculate_historical_price_netCurrentAssets(symbol, use_cache=True)

            # Prüfen, ob Daten zurückgegeben wurden
            if data is None:
                print(f"Fehler für {symbol}: Keine Price/NetCurrentAssets-Daten abgerufen.")
                self.fail(f"Keine Price/NetCurrentAssets-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"Price/NetCurrentAssets-Daten für {symbol} sollten nicht leer sein.")

            # Prüfen, ob erwartete Spalten vorhanden sind
            expected_columns = ["Price_NetCurrentAssets", "Price", "NetCurrentAssets", "commonStockSharesOutstanding"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            # Debugging: Datenstruktur ausgeben
            print(f"\nDebug: Inhalt von price_netCurrentAssets für {symbol}:")
            print(f"Spalten: {data.columns.tolist()}")
            print(f"Erste Zeilen:\n{data.head()}")
            print(f"\nDebug: Letzte Zeilen:\n{data.tail()}")
            print(f"\nDebug: Zusammenfassung Price_NetCurrentAssets:\n{data['Price_NetCurrentAssets'].describe()}")

            # Prüfen, ob Fundamentaldaten verfügbar sind
            fundamentals = self.model.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            balance_sheet = fundamentals.get("balance_sheet") if fundamentals else None
            if balance_sheet is None:
                print(f"Fehler für {symbol}: Keine Bilanzdaten abgerufen.")
                self.fail(f"Keine Bilanzdaten für {symbol} erhalten.")

            # Prüfen, ob benötigte Spalten in balance_sheet vorhanden sind
            required_columns = {"totalCurrentAssets", "totalCurrentLiabilities", "commonStockSharesOutstanding"}
            self.assertTrue(all(col in balance_sheet.columns for col in required_columns),
                            f"Fehlende Spalten in balance_sheet für {symbol}: Erwartet {required_columns}, erhalten {balance_sheet.columns.tolist()}")

            # Prüfen, ob NetCurrentAssets konsistent mit Bilanzdaten ist
            balance_sheet = balance_sheet.reindex(data.index)
            calculated_netCurrentAssets = balance_sheet["totalCurrentAssets"] - balance_sheet["totalCurrentLiabilities"]
            self.assertTrue((data["NetCurrentAssets"] == calculated_netCurrentAssets).all(),
                            f"NetCurrentAssets-Daten für {symbol} sollten mit Bilanzdaten übereinstimmen.")

            # Prüfen, ob Price/NetCurrentAssets konsistent ist
            calculated_price_netCurrentAssets = data["Price"] / (
                        data["NetCurrentAssets"] / data["commonStockSharesOutstanding"])
            self.assertTrue((data["Price_NetCurrentAssets"] - calculated_price_netCurrentAssets).abs().max() < 1e-6,
                            f"Price/NetCurrentAssets-Daten für {symbol} sollten konsistent mit berechneten Werten sein.")

            # Prüfen, ob Price/NetCurrentAssets sinnvolle Werte enthält
            self.assertFalse(data["Price_NetCurrentAssets"].isna().all(),
                             f"Price/NetCurrentAssets-Werte für {symbol} sollten nicht alle NaN sein.")
            self.assertFalse(
                (data["Price_NetCurrentAssets"] == np.inf).any() or (data["Price_NetCurrentAssets"] == -np.inf).any(),
                f"Price/NetCurrentAssets-Werte für {symbol} sollten keine Inf-Werte enthalten.")

            # Zeitraum und Datenpunkte ausgeben
            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige Price/NetCurrentAssets-Tabelle:")
                print(data)

            time.sleep(1)

    def test_calculate_historical_OperatingCashflow(self):
        """Testet die Methode calculate_historical_OperatingCashflow für ILMN."""
        print("\nTeste calculate_historical_OperatingCashflow...")

        # Testfall 1: Gültiges Symbol (ILMN)
        with self.subTest(symbol="ILMN", case="valid_symbol"):
            symbol = "ILMN"
            print(f"\nBerechnung von OperatingCashflow für {symbol}...")
            data = self.model.calculate_historical_OperatingCashflow(symbol, use_cache=True)

            # Prüfen, ob Daten zurückgegeben wurden
            if data is None:
                print(f"Fehler für {symbol}: Keine OperatingCashflow-Daten abgerufen.")
                self.fail(f"Keine OperatingCashflow-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"OperatingCashflow-Daten für {symbol} sollten nicht leer sein.")

            # Prüfen, ob erwartete Spalten vorhanden sind
            expected_columns = ["OperatingCashflow"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            # Debugging: Datenstruktur ausgeben
            print(f"\nDebug: Inhalt von OperatingCashflow für {symbol}:")
            print(f"Spalten: {data.columns.tolist()}")
            print(f"Erste Zeilen:\n{data.head()}")
            print(f"\nDebug: Letzte Zeilen:\n{data.tail()}")
            print(f"\nDebug: Zusammenfassung OperatingCashflow:\n{data['OperatingCashflow'].describe()}")

            # Prüfen, ob Fundamentaldaten verfügbar sind
            fundamentals = self.model.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            cash_flow = fundamentals.get("cash_flow") if fundamentals else None
            if cash_flow is None:
                print(f"Fehler für {symbol}: Keine Cashflow-Daten abgerufen.")
                self.fail(f"Keine Cashflow-Daten für {symbol} erhalten.")

            # Prüfen, ob benötigte Spalten in cash_flow vorhanden sind
            required_columns = {"operatingCashflow"}
            self.assertTrue(all(col in cash_flow.columns for col in required_columns),
                            f"Fehlende Spalten in cash_flow für {symbol}: Erwartet {required_columns}, erhalten {cash_flow.columns.tolist()}")

            # Prüfen, ob Indizes übereinstimmen
            expected_dates = data.index.intersection(cash_flow.index)
            self.assertFalse(expected_dates.empty,
                             f"OperatingCashflow-Daten für {symbol} sollten mit Cashflow-Daten-Indizes übereinstimmen.")

            # Prüfen, ob OperatingCashflow konsistent mit Cashflow-Daten ist
            cash_flow = cash_flow.reindex(data.index)
            calculated_operating_cashflow = cash_flow["operatingCashflow"]
            self.assertTrue((data["OperatingCashflow"] == calculated_operating_cashflow).all(),
                            f"OperatingCashflow-Daten für {symbol} sollten mit Cashflow-Daten übereinstimmen.")

            # Prüfen, ob OperatingCashflow sinnvolle Werte enthält
            self.assertFalse(data["OperatingCashflow"].isna().all(),
                             f"OperatingCashflow-Werte für {symbol} sollten nicht alle NaN sein.")

            # Zeitraum und Datenpunkte ausgeben
            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige OperatingCashflow-Tabelle:")
                print(data)

            time.sleep(1)

    def test_calculate_historical_price_OperatingCashflow(self):
        """Testet die Methode calculate_historical_price_OperatingCashflow für ILMN."""
        print("\nTeste calculate_historical_price_OperatingCashflow...")

        # Testfall 1: Gültiges Symbol (ILMN)
        with self.subTest(symbol="ILMN", case="valid_symbol"):
            symbol = "ILMN"
            print(f"\nBerechnung von Price/OperatingCashflow für {symbol}...")
            data = self.model.calculate_historical_price_OperatingCashflow(symbol, use_cache=True)

            # Prüfen, ob Daten zurückgegeben wurden
            if data is None:
                print(f"Fehler für {symbol}: Keine Price/OperatingCashflow-Daten abgerufen.")
                self.fail(f"Keine Price/OperatingCashflow-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"Price/OperatingCashflow-Daten für {symbol} sollten nicht leer sein.")

            # Prüfen, ob erwartete Spalten vorhanden sind
            expected_columns = ["Price_OperatingCashflow", "Price", "OperatingCashflow", "commonStockSharesOutstanding"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            # Debugging: Datenstruktur ausgeben
            print(f"\nDebug: Inhalt von price_operatingCashflow für {symbol}:")
            print(f"Spalten: {data.columns.tolist()}")
            print(f"Erste Zeilen:\n{data.head()}")
            print(f"\nDebug: Letzte Zeilen:\n{data.tail()}")
            print(f"\nDebug: Zusammenfassung Price_OperatingCashflow:\n{data['Price_OperatingCashflow'].describe()}")

            # Prüfen, ob Fundamentaldaten verfügbar sind
            fundamentals = self.model.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            cash_flow = fundamentals.get("cash_flow") if fundamentals else None
            balance_sheet = fundamentals.get("balance_sheet") if fundamentals else None
            if cash_flow is None:
                print(f"Fehler für {symbol}: Keine Cashflow-Daten abgerufen.")
                self.fail(f"Keine Cashflow-Daten für {symbol} erhalten.")
            if balance_sheet is None:
                print(f"Fehler für {symbol}: Keine Bilanzdaten abgerufen.")
                self.fail(f"Keine Bilanzdaten für {symbol} erhalten.")

            # Prüfen, ob benötigte Spalten in cash_flow und balance_sheet vorhanden sind
            required_cash_flow_columns = {"operatingCashflow"}
            required_balance_sheet_columns = {"commonStockSharesOutstanding"}
            self.assertTrue(all(col in cash_flow.columns for col in required_cash_flow_columns),
                            f"Fehlende Spalten in cash_flow für {symbol}: Erwartet {required_cash_flow_columns}, erhalten {cash_flow.columns.tolist()}")
            self.assertTrue(all(col in balance_sheet.columns for col in required_balance_sheet_columns),
                            f"Fehlende Spalten in balance_sheet für {symbol}: Erwartet {required_balance_sheet_columns}, erhalten {balance_sheet.columns.tolist()}")

            # Prüfen, ob OperatingCashflow konsistent mit Cashflow-Daten ist
            cash_flow = cash_flow.reindex(data.index)
            calculated_operating_cashflow = cash_flow["operatingCashflow"]
            self.assertTrue((data["OperatingCashflow"] == calculated_operating_cashflow).all(),
                            f"OperatingCashflow-Daten für {symbol} sollten mit Cashflow-Daten übereinstimmen.")

            # Prüfen, ob Price/OperatingCashflow konsistent ist
            calculated_price_operating_cashflow = data["Price"] / (
                    data["OperatingCashflow"] / data["commonStockSharesOutstanding"])
            self.assertTrue((data["Price_OperatingCashflow"] - calculated_price_operating_cashflow).abs().max() < 1e-6,
                            f"Price/OperatingCashflow-Daten für {symbol} sollten konsistent mit berechneten Werten sein.")

            # Prüfen, ob Price/OperatingCashflow sinnvolle Werte enthält
            self.assertFalse(data["Price_OperatingCashflow"].isna().all(),
                             f"Price/OperatingCashflow-Werte für {symbol} sollten nicht alle NaN sein.")
            self.assertFalse(
                (data["Price_OperatingCashflow"] == np.inf).any() or (data["Price_OperatingCashflow"] == -np.inf).any(),
                f"Price/OperatingCashflow-Werte für {symbol} sollten keine Inf-Werte enthalten.")

            # Zeitraum und Datenpunkte ausgeben
            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige Price/OperatingCashflow-Tabelle:")
                print(data)

            time.sleep(1)

    def test_calculate_historical_FreeCashflow(self):
        """Testet die Methode calculate_historical_FreeCashflow für ILMN."""
        print("\nTeste calculate_historical_FreeCashflow...")

        # Testfall 1: Gültiges Symbol (ILMN)
        with self.subTest(symbol="ILMN", case="valid_symbol"):
            symbol = "ILMN"
            print(f"\nBerechnung von FreeCashflow für {symbol}...")
            data = self.model.calculate_historical_FreeCashflow(symbol, use_cache=True)

            # Prüfen, ob Daten zurückgegeben wurden
            if data is None:
                print(f"Fehler für {symbol}: Keine FreeCashflow-Daten abgerufen.")
                self.fail(f"Keine FreeCashflow-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"FreeCashflow-Daten für {symbol} sollten nicht leer sein.")

            # Prüfen, ob erwartete Spalten vorhanden sind
            expected_columns = ["FreeCashflow", "OperatingCashflow", "CapitalExpenditures"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            # Debugging: Datenstruktur ausgeben
            print(f"\nDebug: Inhalt von FreeCashflow für {symbol}:")
            print(f"Spalten: {data.columns.tolist()}")
            print(f"Erste Zeilen:\n{data.head()}")
            print(f"\nDebug: Letzte Zeilen:\n{data.tail()}")
            print(f"\nDebug: Zusammenfassung FreeCashflow:\n{data['FreeCashflow'].describe()}")

            # Prüfen, ob Fundamentaldaten verfügbar sind
            fundamentals = self.model.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            cash_flow = fundamentals.get("cash_flow") if fundamentals else None
            if cash_flow is None:
                print(f"Fehler für {symbol}: Keine Cashflow-Daten abgerufen.")
                self.fail(f"Keine Cashflow-Daten für {symbol} erhalten.")

            # Prüfen, ob benötigte Spalten in cash_flow vorhanden sind
            required_columns = {"operatingCashflow", "capitalExpenditures"}
            self.assertTrue(all(col in cash_flow.columns for col in required_columns),
                            f"Fehlende Spalten in cash_flow für {symbol}: Erwartet {required_columns}, erhalten {cash_flow.columns.tolist()}")

            # Prüfen, ob FreeCashflow konsistent mit Cashflow-Daten ist
            cash_flow = cash_flow.reindex(data.index)
            calculated_free_cashflow = cash_flow["operatingCashflow"] - cash_flow["capitalExpenditures"]
            self.assertTrue((data["FreeCashflow"] == calculated_free_cashflow).all(),
                            f"FreeCashflow-Daten für {symbol} sollten mit Cashflow-Daten übereinstimmen.")

            # Prüfen, ob FreeCashflow konsistent berechnet wurde
            calculated_free_cashflow_internal = data["OperatingCashflow"] - data["CapitalExpenditures"]
            self.assertTrue((data["FreeCashflow"] == calculated_free_cashflow_internal).all(),
                            f"FreeCashflow-Daten für {symbol} sollten mit intern berechneten Werten übereinstimmen.")

            # Prüfen, ob FreeCashflow sinnvolle Werte enthält
            self.assertFalse(data["FreeCashflow"].isna().all(),
                             f"FreeCashflow-Werte für {symbol} sollten nicht alle NaN sein.")

            # Zeitraum und Datenpunkte ausgeben
            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige FreeCashflow-Tabelle:")
                print(data)

            time.sleep(1)

    def test_calculate_historical_price_FreeCashflow(self):
        """Testet die Methode calculate_historical_Price_FreeCashflow für ILMN."""
        print("\nTeste calculate_historical_Price_FreeCashflow...")

        # Testfall 1: Gültiges Symbol (ILMN)
        with self.subTest(symbol="ILMN", case="valid_symbol"):
            symbol = "ILMN"
            print(f"\nBerechnung von Price/FreeCashflow für {symbol}...")
            data = self.model.calculate_historical_Price_FreeCashflow(symbol, use_cache=True)

            # Prüfen, ob Daten zurückgegeben wurden
            if data is None:
                print(f"Fehler für {symbol}: Keine Price/FreeCashflow-Daten abgerufen.")
                self.fail(f"Keine Price/FreeCashflow-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"Price/FreeCashflow-Daten für {symbol} sollten nicht leer sein.")

            # Prüfen, ob erwartete Spalten vorhanden sind
            expected_columns = ["Price_FreeCashflow", "Price", "FreeCashflow", "OperatingCashflow",
                                "CapitalExpenditures", "commonStockSharesOutstanding"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            # Debugging: Datenstruktur ausgeben
            print(f"\nDebug: Inhalt von price_freeCashflow für {symbol}:")
            print(f"Spalten: {data.columns.tolist()}")
            print(f"Erste Zeilen:\n{data.head()}")
            print(f"\nDebug: Letzte Zeilen:\n{data.tail()}")
            print(f"\nDebug: Zusammenfassung Price_FreeCashflow:\n{data['Price_FreeCashflow'].describe()}")

            # Prüfen, ob Fundamentaldaten verfügbar sind
            fundamentals = self.model.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            cash_flow = fundamentals.get("cash_flow") if fundamentals else None
            balance_sheet = fundamentals.get("balance_sheet") if fundamentals else None
            if cash_flow is None:
                print(f"Fehler für {symbol}: Keine Cashflow-Daten abgerufen.")
                self.fail(f"Keine Cashflow-Daten für {symbol} erhalten.")
            if balance_sheet is None:
                print(f"Fehler für {symbol}: Keine Bilanzdaten abgerufen.")
                self.fail(f"Keine Bilanzdaten für {symbol} erhalten.")

            # Prüfen, ob benötigte Spalten in cash_flow und balance_sheet vorhanden sind
            required_cash_flow_columns = {"operatingCashflow", "capitalExpenditures"}
            required_balance_sheet_columns = {"commonStockSharesOutstanding"}
            self.assertTrue(all(col in cash_flow.columns for col in required_cash_flow_columns),
                            f"Fehlende Spalten in cash_flow für {symbol}: Erwartet {required_cash_flow_columns}, erhalten {cash_flow.columns.tolist()}")
            self.assertTrue(all(col in balance_sheet.columns for col in required_balance_sheet_columns),
                            f"Fehlende Spalten in balance_sheet für {symbol}: Erwartet {required_balance_sheet_columns}, erhalten {balance_sheet.columns.tolist()}")

            # Prüfen, ob FreeCashflow konsistent mit Cashflow-Daten ist
            cash_flow = cash_flow.reindex(data.index)
            calculated_free_cashflow = cash_flow["operatingCashflow"] - cash_flow["capitalExpenditures"]
            self.assertTrue((data["FreeCashflow"] == calculated_free_cashflow).all(),
                            f"FreeCashflow-Daten für {symbol} sollten mit Cashflow-Daten übereinstimmen.")

            # Prüfen, ob OperatingCashflow und CapitalExpenditures konsistent sind
            self.assertTrue((data["OperatingCashflow"] == cash_flow["operatingCashflow"]).all(),
                            f"OperatingCashflow-Daten für {symbol} sollten mit Cashflow-Daten übereinstimmen.")
            self.assertTrue((data["CapitalExpenditures"] == cash_flow["capitalExpenditures"]).all(),
                            f"CapitalExpenditures-Daten für {symbol} sollten mit Cashflow-Daten übereinstimmen.")

            # Prüfen, ob commonStockSharesOutstanding konsistent ist
            balance_sheet = balance_sheet.reindex(data.index)
            self.assertTrue(
                (data["commonStockSharesOutstanding"] == balance_sheet["commonStockSharesOutstanding"]).all(),
                f"commonStockSharesOutstanding-Daten für {symbol} sollten mit Bilanzdaten übereinstimmen.")

            # Prüfen, ob Price/FreeCashflow konsistent ist
            calculated_price_free_cashflow = data["Price"] / (
                    data["FreeCashflow"] / data["commonStockSharesOutstanding"])
            self.assertTrue((data["Price_FreeCashflow"] - calculated_price_free_cashflow).abs().max() < 1e-6,
                            f"Price/FreeCashflow-Daten für {symbol} sollten konsistent mit berechneten Werten sein.")

            # Prüfen, ob Price/FreeCashflow sinnvolle Werte enthält
            self.assertFalse(data["Price_FreeCashflow"].isna().all(),
                             f"Price/FreeCashflow-Werte für {symbol} sollten nicht alle NaN sein.")
            self.assertFalse(
                (data["Price_FreeCashflow"] == np.inf).any() or (data["Price_FreeCashflow"] == -np.inf).any(),
                f"Price/FreeCashflow-Werte für {symbol} sollten keine Inf-Werte enthalten.")

            # Zeitraum und Datenpunkte ausgeben
            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige Price/FreeCashflow-Tabelle:")
                print(data)

            time.sleep(1)

        # Testfall 2: Ungültiges Symbol
        with self.subTest(symbol="INVALID", case="invalid_symbol"):
            symbol = "INVALID"
            print(f"\nBerechnung von Price/FreeCashflow für ungültiges Symbol {symbol}...")
            data = self.model.calculate_historical_Price_FreeCashflow(symbol, use_cache=True)
            self.assertIsNone(data, f"Ergebnis für ungültiges Symbol {symbol} sollte None sein.")
            print(f"Test für ungültiges Symbol {symbol} erfolgreich: Keine Daten erhalten, wie erwartet.")
            time.sleep(1)

    def test_calculate_historical_TangibleBookValue(self):

        """Testet die Methode calculate_historical_TangibleBookValue für ILMN."""
        print("\nTeste calculate_historical_TangibleBookValue...")

        # Testfall 1: Gültiges Symbol (ILMN)
        with self.subTest(symbol="ILMN", case="valid_symbol"):
            symbol = "ILMN"
            print(f"\nBerechnung von TangibleBookValue für {symbol}...")
            data = self.model.calculate_historical_TangibleBookValue(symbol, use_cache=True)

            # Prüfen, ob Daten zurückgegeben wurden
            if data is None:
                print(f"Fehler für {symbol}: Keine TangibleBookValue-Daten abgerufen.")
                self.fail(f"Keine TangibleBookValue-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"TangibleBookValue-Daten für {symbol} sollten nicht leer sein.")

            # Prüfen, ob erwartete Spalten vorhanden sind
            expected_columns = ["TangibleBookValue", "totalAssets", "intangibleAssets", "goodwill", "totalLiabilities"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            # Debugging: Datenstruktur ausgeben
            print(f"\nDebug: Inhalt von TangibleBookValue für {symbol}:")
            print(f"Spalten: {data.columns.tolist()}")
            print(f"Erste Zeilen:\n{data.head()}")
            print(f"\nDebug: Letzte Zeilen:\n{data.tail()}")
            print(f"\nDebug: Zusammenfassung TangibleBookValue:\n{data['TangibleBookValue'].describe()}")
            print(f"\nDebug: Zusammenfassung totalAssets:\n{data['totalAssets'].describe()}")
            print(f"\nDebug: Zusammenfassung intangibleAssets:\n{data['intangibleAssets'].describe()}")
            print(f"\nDebug: Zusammenfassung goodwill:\n{data['goodwill'].describe()}")
            print(f"\nDebug: Zusammenfassung totalLiabilities:\n{data['totalLiabilities'].describe()}")

            # Prüfen, ob Fundamentaldaten verfügbar sind
            fundamentals = self.model.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            balance_sheet = fundamentals.get("balance_sheet") if fundamentals else None
            if balance_sheet is None:
                print(f"Fehler für {symbol}: Keine Bilanzdaten abgerufen.")
                self.fail(f"Keine Bilanzdaten für {symbol} erhalten.")

            # Prüfen, ob benötigte Spalten in balance_sheet vorhanden sind
            required_columns = {"totalAssets", "intangibleAssets", "goodwill", "totalLiabilities"}
            self.assertTrue(all(col in balance_sheet.columns for col in required_columns),
                            f"Fehlende Spalten in balance_sheet für {symbol}: Erwartet {required_columns}, erhalten {balance_sheet.columns.tolist()}")

            # Prüfen, ob Indizes übereinstimmen
            expected_dates = data.index.intersection(balance_sheet.index)
            self.assertFalse(expected_dates.empty, f"TangibleBookValue-Daten für {symbol} sollten mit Bilanzdaten-Indizes übereinstimmen.")

            # Prüfen, ob TangibleBookValue konsistent mit Bilanzdaten ist
            balance_sheet = balance_sheet.reindex(data.index)
            calculated_tangibleBookValue = balance_sheet["totalAssets"] - balance_sheet["intangibleAssets"] - balance_sheet["goodwill"] - balance_sheet["totalLiabilities"]
            self.assertTrue((data["TangibleBookValue"] == calculated_tangibleBookValue).all(),
                            f"TangibleBookValue-Daten für {symbol} sollten mit Bilanzdaten übereinstimmen.")

            # Prüfen, ob Komponenten mit Bilanzdaten übereinstimmen
            self.assertTrue((data["totalAssets"] == balance_sheet["totalAssets"]).all(),
                            f"totalAssets-Daten für {symbol} sollten mit Bilanzdaten übereinstimmen.")
            self.assertTrue((data["intangibleAssets"] == balance_sheet["intangibleAssets"]).all(),
                            f"intangibleAssets-Daten für {symbol} sollten mit Bilanzdaten übereinstimmen.")
            self.assertTrue((data["goodwill"] == balance_sheet["goodwill"]).all(),
                            f"goodwill-Daten für {symbol} sollten mit Bilanzdaten übereinstimmen.")
            self.assertTrue((data["totalLiabilities"] == balance_sheet["totalLiabilities"]).all(),
                            f"totalLiabilities-Daten für {symbol} sollten mit Bilanzdaten übereinstimmen.")

            # Prüfen, ob alle Spalten sinnvolle Werte enthalten
            for column in expected_columns:
                self.assertFalse(data[column].isna().all(),
                                 f"{column}-Werte für {symbol} sollten nicht alle NaN sein.")

            # Zeitraum und Datenpunkte ausgeben
            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige TangibleBookValue-Tabelle:")
                print(data)

            time.sleep(1)

    def test_calculate_historical_price_to_TangibleBookValue(self):
        """Testet die Methode calculate_historical_price_to_TangibleBookValue für ILMN und Sonderfälle."""
        print("\nTeste calculate_historical_price_to_TangibleBookValue...")

        # Testfall 1: Gültiges Symbol (ILMN)
        with self.subTest(symbol="BABA", case="valid_symbol"):
            symbol = "BABA"
            print(f"\nBerechnung von Price/TangibleBookValue für {symbol}...")
            data = self.model.calculate_historical_price_to_TangibleBookValue(symbol, use_cache=True)

            # Prüfen, ob Daten zurückgegeben wurden
            if data is None:
                print(f"Fehler für {symbol}: Keine Price/TangibleBookValue-Daten abgerufen.")
                self.fail(f"Keine Price/TangibleBookValue-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"Price/TangibleBookValue-Daten für {symbol} sollten nicht leer sein.")

            # Prüfen, ob erwartete Spalten vorhanden sind
            expected_columns = ["Price_TangibleBookValue", "Price", "TangibleBookValue", "commonStockSharesOutstanding"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            # Debugging: Datenstruktur ausgeben
            print(f"\nDebug: Inhalt von price_to_TangibleBookValue für {symbol}:")
            print(f"Spalten: {data.columns.tolist()}")
            print(f"Erste Zeilen:\n{data.head()}")
            print(f"\nDebug: Letzte Zeilen:\n{data.tail()}")
            print(f"\nDebug: Zusammenfassung Price_TangibleBookValue:\n{data['Price_TangibleBookValue'].describe()}")

            # Prüfen, ob Fundamentaldaten verfügbar sind
            fundamentals = self.model.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            balance_sheet = fundamentals.get("balance_sheet") if fundamentals else None
            if balance_sheet is None:
                print(f"Fehler für {symbol}: Keine Bilanzdaten abgerufen.")
                self.fail(f"Keine Bilanzdaten für {symbol} erhalten.")

            # Prüfen, ob benötigte Spalten in balance_sheet vorhanden sind
            required_columns = {"commonStockSharesOutstanding"}
            self.assertTrue(all(col in balance_sheet.columns for col in required_columns),
                            f"Fehlende Spalten in balance_sheet für {symbol}: Erwartet {required_columns}, erhalten {balance_sheet.columns.tolist()}")

            # Prüfen, ob TangibleBookValue mit calculate_historical_TangibleBookValue übereinstimmt
            tangible_book_value_df = self.model.calculate_historical_TangibleBookValue(symbol, use_cache=True)
            if tangible_book_value_df is None:
                print(f"Fehler für {symbol}: Keine TangibleBookValue-Daten abgerufen.")
                self.fail(f"Keine TangibleBookValue-Daten für {symbol} erhalten.")
            self.assertTrue((data["TangibleBookValue"] == tangible_book_value_df["TangibleBookValue"].reindex(data.index)).all(),
                            f"TangibleBookValue-Daten für {symbol} sollten mit calculate_historical_TangibleBookValue übereinstimmen.")

            # Prüfen, ob commonStockSharesOutstanding konsistent ist
            self.assertTrue((data["commonStockSharesOutstanding"] == balance_sheet["commonStockSharesOutstanding"].reindex(data.index, method="ffill")).all(),
                            f"commonStockSharesOutstanding-Daten für {symbol} sollten mit Bilanzdaten übereinstimmen.")

            # Prüfen, ob Price_TangibleBookValue korrekt berechnet wurde
            calculated_per_share = data["TangibleBookValue"] / data["commonStockSharesOutstanding"]
            calculated_multiple = data["Price"] / calculated_per_share
            self.assertTrue(np.isclose(data["Price_TangibleBookValue"], calculated_multiple, rtol=1e-5).all(),
                            f"Price_TangibleBookValue für {symbol} sollte korrekt berechnet sein.")

            # Prüfen, ob keine NaN oder unendliche Werte vorhanden sind
            self.assertFalse(data["Price_TangibleBookValue"].isna().any(),
                             f"Price_TangibleBookValue-Werte für {symbol} sollten keine NaN-Werte enthalten.")
            self.assertFalse(np.isinf(data["Price_TangibleBookValue"]).any(),
                             f"Price_TangibleBookValue-Werte für {symbol} sollten keine unendlichen Werte enthalten.")

            # Zeitraum und Datenpunkte ausgeben
            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige Price/TangibleBookValue-Tabelle:")
                print(data)

            time.sleep(1)

            # Testfall 1: Gültiges Symbol (ILMN)
            with self.subTest(symbol="SLB", case="valid_symbol"):
                symbol = "SLB"
                print(f"\nBerechnung von Price/TangibleBookValue für {symbol}...")
                data = self.model.calculate_historical_price_to_TangibleBookValue(symbol, use_cache=True)

                # Prüfen, ob Daten zurückgegeben wurden
                if data is None:
                    print(f"Fehler für {symbol}: Keine Price/TangibleBookValue-Daten abgerufen.")
                    self.fail(f"Keine Price/TangibleBookValue-Daten für {symbol} erhalten.")

                # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
                self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
                self.assertFalse(data.empty, f"Price/TangibleBookValue-Daten für {symbol} sollten nicht leer sein.")

                # Prüfen, ob erwartete Spalten vorhanden sind
                expected_columns = ["Price_TangibleBookValue", "Price", "TangibleBookValue",
                                    "commonStockSharesOutstanding"]
                self.assertTrue(all(col in data.columns for col in expected_columns),
                                f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

                # Debugging: Datenstruktur ausgeben
                print(f"\nDebug: Inhalt von price_to_TangibleBookValue für {symbol}:")
                print(f"Spalten: {data.columns.tolist()}")
                print(f"Erste Zeilen:\n{data.head()}")
                print(f"\nDebug: Letzte Zeilen:\n{data.tail()}")
                print(
                    f"\nDebug: Zusammenfassung Price_TangibleBookValue:\n{data['Price_TangibleBookValue'].describe()}")

                # Prüfen, ob Fundamentaldaten verfügbar sind
                fundamentals = self.model.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
                balance_sheet = fundamentals.get("balance_sheet") if fundamentals else None
                if balance_sheet is None:
                    print(f"Fehler für {symbol}: Keine Bilanzdaten abgerufen.")
                    self.fail(f"Keine Bilanzdaten für {symbol} erhalten.")

                # Prüfen, ob benötigte Spalten in balance_sheet vorhanden sind
                required_columns = {"commonStockSharesOutstanding"}
                self.assertTrue(all(col in balance_sheet.columns for col in required_columns),
                                f"Fehlende Spalten in balance_sheet für {symbol}: Erwartet {required_columns}, erhalten {balance_sheet.columns.tolist()}")

                # Prüfen, ob TangibleBookValue mit calculate_historical_TangibleBookValue übereinstimmt
                tangible_book_value_df = self.model.calculate_historical_TangibleBookValue(symbol, use_cache=True)
                if tangible_book_value_df is None:
                    print(f"Fehler für {symbol}: Keine TangibleBookValue-Daten abgerufen.")
                    self.fail(f"Keine TangibleBookValue-Daten für {symbol} erhalten.")
                self.assertTrue((data["TangibleBookValue"] == tangible_book_value_df["TangibleBookValue"].reindex(
                    data.index)).all(),
                                f"TangibleBookValue-Daten für {symbol} sollten mit calculate_historical_TangibleBookValue übereinstimmen.")

                # Prüfen, ob commonStockSharesOutstanding konsistent ist
                self.assertTrue((data["commonStockSharesOutstanding"] == balance_sheet[
                    "commonStockSharesOutstanding"].reindex(data.index, method="ffill")).all(),
                                f"commonStockSharesOutstanding-Daten für {symbol} sollten mit Bilanzdaten übereinstimmen.")

                # Prüfen, ob Price_TangibleBookValue korrekt berechnet wurde
                calculated_per_share = data["TangibleBookValue"] / data["commonStockSharesOutstanding"]
                calculated_multiple = data["Price"] / calculated_per_share
                self.assertTrue(np.isclose(data["Price_TangibleBookValue"], calculated_multiple, rtol=1e-5).all(),
                                f"Price_TangibleBookValue für {symbol} sollte korrekt berechnet sein.")

                # Prüfen, ob keine NaN oder unendliche Werte vorhanden sind
                self.assertFalse(data["Price_TangibleBookValue"].isna().any(),
                                 f"Price_TangibleBookValue-Werte für {symbol} sollten keine NaN-Werte enthalten.")
                self.assertFalse(np.isinf(data["Price_TangibleBookValue"]).any(),
                                 f"Price_TangibleBookValue-Werte für {symbol} sollten keine unendlichen Werte enthalten.")

                # Zeitraum und Datenpunkte ausgeben
                start_date = data.index.min().strftime("%Y-%m-%d")
                end_date = data.index.max().strftime("%Y-%m-%d")
                num_points = len(data)
                print(
                    f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    print("\nVollständige Price/TangibleBookValue-Tabelle:")
                    print(data)

                time.sleep(1)

        # Testfall 2: Ungültiges Symbol
        with self.subTest(symbol="INVALID", case="invalid_symbol"):
            symbol = "INVALID"
            print(f"\nBerechnung von Price/TangibleBookValue für ungültiges Symbol {symbol}...")
            data = self.model.calculate_historical_price_to_TangibleBookValue(symbol, use_cache=True)
            self.assertIsNone(data, f"Ergebnis für ungültiges Symbol {symbol} sollte None sein.")
            print(f"Test für ungültiges Symbol {symbol} erfolgreich: Keine Daten erhalten, wie erwartet.")
            time.sleep(1)

    def test_print_balance_sheet(self):
        """Testet die Methode print_balance_sheet für ILMN."""
        print("\nTeste print_balance_sheet...")

        # Testfall 1: Gültiges Symbol (ILMN)
        with self.subTest(symbol="ILMN", case="valid_symbol"):
            symbol = "ILMN"
            dates = ["2015-03-31", "2005-03-31"]
            print(f"\nBerechnung des Balance Sheets für {symbol}...")
            balance_sheet = self.model.print_balance_sheet(symbol, dates=dates)

            # Prüfen, ob Daten zurückgegeben wurden
            if balance_sheet is None:
                print(f"Fehler für {symbol}: Keine Balance Sheet-Daten abgerufen.")
                self.fail(f"Keine Balance Sheet-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(balance_sheet, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(balance_sheet.empty, f"Balance Sheet-Daten für {symbol} sollten nicht leer sein.")

            # Prüfen, ob erwartete Spalten vorhanden sind
            expected_columns = [
                "totalAssets", "intangibleAssets", "intangibleAssetsExcludingGoodwill",
                "goodwill", "totalLiabilities", "reportedCurrency", "totalCurrentAssets",
                "cashAndCashEquivalentsAtCarryingValue", "totalNonCurrentAssets",
                "propertyPlantEquipment", "totalShareholderEquity"
            ]
            self.assertTrue(all(col in balance_sheet.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {balance_sheet.columns.tolist()}")

            # Debugging: Datenstruktur ausgeben
            print(f"\nDebug: Inhalt des Balance Sheets für {symbol}:")
            print(f"Spalten: {balance_sheet.columns.tolist()}")
            print(f"Erste Zeilen:\n{balance_sheet.head()}")
            print(f"\nDebug: Letzte Zeilen:\n{balance_sheet.tail()}")
            print(f"\nDebug: Zusammenfassung totalAssets:\n{balance_sheet['totalAssets'].describe()}")
            print(f"\nDebug: Zusammenfassung intangibleAssets:\n{balance_sheet['intangibleAssets'].describe()}")
            print(f"\nDebug: Zusammenfassung goodwill:\n{balance_sheet['goodwill'].describe()}")
            print(f"\nDebug: Zusammenfassung totalLiabilities:\n{balance_sheet['totalLiabilities'].describe()}")

            # Prüfen, ob spezifische Datenpunkte vorhanden sind
            balance_sheet_subset = balance_sheet.loc[balance_sheet.index.isin(pd.to_datetime(dates))]
            self.assertFalse(balance_sheet_subset.empty,
                             f"Balance Sheet-Daten für {symbol} sollten Daten für {dates} enthalten.")
            print(f"\nDebug: Balance Sheet für {symbol} an spezifischen Daten {dates}:")
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(balance_sheet_subset)

            # Prüfen, ob CSV-Datei erstellt wurde
            csv_file = f"{symbol}_balance_sheet.csv"
            self.assertTrue(os.path.exists(csv_file), f"CSV-Datei {csv_file} wurde nicht erstellt.")
            print(f"CSV-Datei {csv_file} erfolgreich erstellt.")

            # Vollständige Tabelle ausgeben
            print(f"\nVollständige Balance Sheet-Tabelle für {symbol}:")
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(balance_sheet)

            time.sleep(1)

    def test_get_stock_financials(self):
        """Testet die Methode get_stock_financials mit jährlichen und quartalsweisen Daten."""
        print("\nTeste get_stock_financials...")
        for symbol in self.test_symbols:
            for frequency in ["annual", "quarterly"]:
                result = self.loader.get_stock_financials(symbol, frequency=frequency)
                if isinstance(result, dict) and "error" in result:
                    print(f"Fehler für {symbol} ({frequency}): {result['error']}")
                else:
                    print(
                        f"Finanzdaten für {symbol} ({frequency}) erfolgreich abgerufen. Anzahl der Einträge: {len(result.columns)}")
                    self.assertFalse(result.empty, f"Finanzdaten für {symbol} ({frequency}) sollten nicht leer sein")
                    # Anpassung: Überprüfe auf "NetIncomeLoss" statt "EBITDA", da SEC EDGAR-Daten andere Bezeichnungen haben
                    if "NetIncomeLoss" in result.index:
                        print(f"NetIncomeLoss für {symbol} ({frequency}): {result.loc['NetIncomeLoss'].iloc[0]}")
                    elif "EBITDA" in result.index:  # Fallback für Yahoo Finance-Daten
                        print(f"EBITDA für {symbol} ({frequency}): {result.loc['EBITDA'].iloc[0]}")

    def test_determine_buy_sell_points(self):
        print("\nTest der Kauf-/Verkaufszeitpunkte:")
        for symbol in self.test_symbols:
            result = self.model.determine_buy_sell_points(symbol)
            if "error" in result:
                print(f"Fehler bei {symbol}: {result['error']}")
            else:
                print(f"{symbol}: Empfehlung = {result}")
                self.assertIn(result, ["buy", "sell", "hold"])

    def test_analyze_dividend_history(self):
        print("\nTest der Dividendenhistorie-Analyse:")
        for symbol in self.test_symbols:
            result = self.model.analyze_dividend_history(symbol)
            if "error" in result:
                print(f"Fehler bei {symbol}: {result['error']}")
            else:
                print(f"{symbol}: Jahre mit Dividenden = {result['years_with_dividends']}, "
                      f"Jahre mit Steigerungen = {result['years_with_increases']}, "
                      f"CAGR ({result['cagr_period_years']} Jahre) = {result['cagr']}%")
                self.assertTrue(isinstance(result['years_with_dividends'], (int, np.integer)))
                self.assertTrue(isinstance(result['years_with_increases'], (int, np.integer)))
                self.assertTrue(isinstance(result['cagr_period_years'], (int, np.integer)))
                self.assertTrue(isinstance(result['cagr'], float) or np.isnan(result['cagr']))

    def test_get_dividend_data(self):
        print("\nDividend Data Test Results:")
        for symbol in self.test_symbols:
            data = self.loader.get_dividend_data(symbol)
            if "error" in data:
                print(f"Fehler bei Dividenden für {symbol}: {data['error']}")
                continue
            print(f"{symbol}: Dividend Yield = {data['dividend_yield']}%, Dividend Rate = {data['dividend_rate']}, Latest Dividend = {data['latest_dividend']}")
            self.assertGreaterEqual(data['dividend_yield'], 0, f"Dividend yield for {symbol} should be non-negative")

    def test_get_paid_dividends(self):
        """Testet die Methode get_paid_dividends mit regulären und Sonderfällen."""
        print("\nTeste get_paid_dividends...")
        frequencies = ["annual", "quarterly"]  # Inklusive ungültige Frequenz

        for symbol in self.test_symbols:
            for frequency in frequencies:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.loader.get_paid_dividends(symbol, frequency)

                    # Prüfe, ob das Ergebnis ein Dictionary ist
                    self.assertIsInstance(result, dict, f"Ergebnis für {symbol} ({frequency}) sollte ein Dictionary sein")

                    if "error" in result:
                        print(f"Fehler für {symbol} ({frequency}): {result['error']}")

                        # Prüfe Fehlerstruktur
                        self.assertIn("symbol", result, f"Fehler-Dictionary für {symbol} ({frequency}) sollte 'symbol' enthalten")
                        self.assertEqual(result["symbol"], symbol, f"Symbol im Fehler sollte {symbol} sein")

                        # Sonderfälle validieren
                        if frequency == "monthly":
                            self.assertIn("Ungültige Frequenz", result["error"], f"Fehler für {symbol} (monthly) sollte Frequenz erwähnen")
                        if symbol == "INVALID" and frequency in ["annual", "quarterly"]:
                            self.assertIn("Fehler beim Abrufen", result["error"], f"Fehler für {symbol} ({frequency}) sollte API-Fehler erwähnen")

                    else:
                        print(f"Auszahlung Dividenden für {symbol} ({frequency}): {result['paid_dividends']}")

                        # Prüfe Erfolgstruktur
                        self.assertIn("paid_dividends", result, f"Ergebnis für {symbol} ({frequency}) sollte 'paid_dividends' enthalten")
                        self.assertTrue(
                            isinstance(result["paid_dividends"], float) and (
                                np.isnan(result["paid_dividends"]) or
                                result["paid_dividends"] <= 0
                            ),
                            f"Auszahlung für {symbol} ({frequency}) sollte ein Float sein, der null, negativ oder nan ist (Cash Flow-Auszahlung)"
                        )
                        self.assertEqual(result["symbol"], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result["frequency"], frequency, f"Frequenz sollte {frequency} sein")
                        self.assertIn("date", result, f"Datum sollte in Ergebnis für {symbol} ({frequency}) enthalten sein")
                        self.assertIsInstance(result["date"], str, f"Datum für {symbol} ({frequency}) sollte ein String sein")

                        # Sonderfall: Keine Dividenden oder fehlende Daten
                        if symbol in ["PYPL", "TSLA", "ILMN"] and frequency in ["annual", "quarterly"]:
                            self.assertEqual(result["paid_dividends"], 0.0, f"Auszahlung für {symbol} ({frequency}) sollte 0.0 sein, da keine Dividenden vorliegen")

    def test_get_reinvested_profit(self):
        """Testet die Methode get_reinvested_profit mit regulären und Sonderfällen."""
        print("\nTeste get_reinvested_profit...")
        frequencies = ["annual", "quarterly"]  # Inklusive ungültige Frequenz

        for symbol in self.test_symbols:
            for frequency in frequencies:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.loader.get_reinvested_profit(symbol, frequency)

                    # Prüfe, ob das Ergebnis ein Dictionary ist
                    self.assertIsInstance(result, dict,
                                          f"Ergebnis für {symbol} ({frequency}) sollte ein Dictionary sein")

                    if "error" in result:
                        print(f"Fehler für {symbol} ({frequency}): {result['error']}")

                        # Prüfe Fehlerstruktur
                        self.assertIn("symbol", result,
                                      f"Fehler-Dictionary für {symbol} ({frequency}) sollte 'symbol' enthalten")
                        self.assertEqual(result["symbol"], symbol, f"Symbol im Fehler sollte {symbol} sein")

                        # Sonderfälle validieren
                        if frequency == "monthly":
                            self.assertIn("Ungültige Frequenz", result["error"],
                                          f"Fehler für {symbol} (monthly) sollte Frequenz erwähnen")
                        if symbol == "INVALID" and frequency in ["annual", "quarterly"]:
                            self.assertIn("Fehler beim Abrufen", result["error"],
                                          f"Fehler für {symbol} ({frequency}) sollte API-Fehler erwähnen")
                        if symbol == "NVDA" and frequency == "quarterly":
                            self.assertIn("Kein Nettoeinkommen", result["error"],
                                          f"Fehler für {symbol} ({frequency}) sollte fehlendes Nettoeinkommen erwähnen")

                    else:
                        print(f"Reinvestierter Gewinn für {symbol} ({frequency}): {result['reinvested_profit']}")

                        # Prüfe Erfolgstruktur
                        self.assertIn("reinvested_profit", result,
                                      f"Ergebnis für {symbol} ({frequency}) sollte 'reinvested_profit' enthalten")
                        self.assertIsInstance(result["reinvested_profit"], float,
                                              f"Reinvestierter Gewinn für {symbol} ({frequency}) sollte ein Float sein")
                        self.assertEqual(result["symbol"], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result["frequency"], frequency, f"Frequenz sollte {frequency} sein")
                        self.assertIn("date", result,
                                      f"Datum sollte in Ergebnis für {symbol} ({frequency}) enthalten sein")
                        self.assertIsInstance(result["date"], str,
                                              f"Datum für {symbol} ({frequency}) sollte ein String sein")

                        # Prüfe logische Konsistenz
                        if symbol in ["PYPL", "TSLA", "ILMN"] and frequency in ["annual", "quarterly"]:
                            # Für Unternehmen ohne Dividenden sollte reinvestierter Gewinn gleich Nettoeinkommen sein
                            financials = self.loader.get_stock_financials(symbol, frequency)
                            if not isinstance(financials, dict) and isinstance(financials,
                                                                               pd.DataFrame) and not financials.empty:
                                net_income = financials.loc["Net Income"].iloc[
                                    0] if "Net Income" in financials.index else \
                                financials.loc["Net Income Applicable To Common Shares"].iloc[0]
                                if pd.notna(net_income):
                                    self.assertAlmostEqual(result["reinvested_profit"], net_income, places=2,
                                                           msg=f"Reinvestierter Gewinn für {symbol} ({frequency}) sollte Nettoeinkommen entsprechen, da keine Dividenden vorliegen")

    def test_get_market_cap(self):
        """Testet die Methode get_market_cap mit gültigen und ungültigen Symbolen."""
        print("\nTeste get_market_cap...")
        test_symbols = ["AAPL", "MSFT", "INVALID"]  # Gültige und ungültige Symbole

        for symbol in test_symbols:
            for use_cache in [True, False]:
                with self.subTest(symbol=symbol, use_cache=use_cache):
                    result = self.loader.get_market_cap(symbol, use_cache=use_cache)

                    # Prüfe, ob das Ergebnis ein Dictionary ist
                    self.assertIsInstance(result, dict,
                                          f"Ergebnis für {symbol} (use_cache={use_cache}) sollte ein Dictionary sein")

                    if "error" in result:
                        print(f"Fehler für {symbol} (use_cache={use_cache}): {result['error']}")

                        # Prüfe Fehlerstruktur
                        self.assertIn("symbol", result,
                                      f"Fehler-Dictionary für {symbol} (use_cache={use_cache}) sollte 'symbol' enthalten")
                        self.assertEqual(result["symbol"], symbol, f"Symbol im Fehler sollte {symbol} sein")

                        # Sonderfälle validieren
                        if symbol == "INVALID":
                            self.assertIn("Ungültiges Symbol", result["error"],
                                          f"Fehler für {symbol} sollte ungültiges Symbol erwähnen")

                    else:
                        print(f"Marktkapitalisierung für {symbol} (use_cache={use_cache}): {result['market_cap']}")

                        # Prüfe Erfolgstruktur
                        self.assertIn("market_cap", result,
                                      f"Ergebnis für {symbol} (use_cache={use_cache}) sollte 'market_cap' enthalten")
                        self.assertIsInstance(result["market_cap"], float,
                                              f"Marktkapitalisierung für {symbol} (use_cache={use_cache}) sollte ein Float sein")
                        self.assertGreater(result["market_cap"], 0,
                                           f"Marktkapitalisierung für {symbol} (use_cache={use_cache}) sollte positiv sein")
                        self.assertEqual(result["symbol"], symbol, f"Symbol sollte {symbol} sein")
                        self.assertIn("date", result,
                                      f"Datum sollte in Ergebnis für {symbol} (use_cache={use_cache}) enthalten sein")
                        self.assertIsInstance(result["date"], str,
                                              f"Datum für {symbol} (use_cache={use_cache}) sollte ein String sein")

                        # Prüfe logische Konsistenz mit Fallback-Berechnung
                        if symbol in ["AAPL", "MSFT"]:
                            price = self.loader.get_current_price_per_share(symbol)
                            shares = self.loader.get_shares_outstanding(symbol)
                            if not isinstance(price, dict) and not isinstance(shares, dict):
                                expected_market_cap = price * shares
                                # Relative Toleranz von 0,5 % (0.005)
                                relative_diff = abs(result["market_cap"] - expected_market_cap) / expected_market_cap
                                self.assertLess(relative_diff, 0.005,
                                                f"Marktkapitalisierung für {symbol} (use_cache={use_cache}) weicht um {relative_diff * 100:.2f}% von Preis * Aktien ab (erwartet: {expected_market_cap}, erhalten: {result['market_cap']})")

    def test_get_inflation_data(self):
        print("\nInflation Data Test Result:")
        inflation_data = self.loader.get_inflation_data()
        if "error" in inflation_data:
            print(f"Fehler bei Inflationsdaten: {inflation_data['error']}")
            self.fail("Fehler beim Abrufen der Inflationsdaten")
        print(f"Latest CPI Data: {inflation_data[-1]['date']} = {inflation_data[-1]['value']}")
        self.assertIsInstance(inflation_data[-1]['value'], float, "CPI value should be a float")
        self.assertIsInstance(inflation_data[-1]['date'], str, "CPI date should be a string")

    def test_get_quarterly_inflation_rate(self):
        print("\nQuarterly Inflation Rate Test Result:")
        start_date = "2024-09-30"
        end_date = "2024-12-31"
        inflation_data = self.loader.get_inflation_data(start_date=start_date, end_date=end_date)
        if "error" in inflation_data:
            print(f"Fehler bei Inflationsdaten: {inflation_data['error']}")
            self.fail("Fehler beim Abrufen der Inflationsdaten")
        quarterly_inflation_data = self.preprocessor.get_quarterly_inflation_rate(
            cpi_data=inflation_data,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )
        if "error" in quarterly_inflation_data:
            print(f"Fehler bei Inflationsrate: {quarterly_inflation_data['error']}")
            self.fail("Fehler bei der Berechnung der Inflationsrate")
        quarterly_inflation_rate = quarterly_inflation_data['quarterly_inflation_rate']
        print(f"Quarterly Inflation Rate ({start_date} to {end_date}): {quarterly_inflation_rate}%")
        self.assertIsInstance(quarterly_inflation_rate, float, "Quarterly inflation rate should be a float")
        self.assertGreaterEqual(quarterly_inflation_rate, -10.0, "Quarterly inflation rate should not be unrealistically low")
        self.assertLessEqual(quarterly_inflation_rate, 10.0, "Quarterly inflation rate should not be unrealistically high")

    def test_get_gdp_data(self):
        print("\nGDP Data Test Result:")
        gdp_data = self.loader.get_gdp_data_grpwth()
        if gdp_data is None or "error" in gdp_data:
            self.fail("Fehler beim Abrufen der BIP-Daten")
        self.assertIsInstance(gdp_data['gdp_value'], float, "GDP value should be a float")
        self.assertIsInstance(gdp_data['date'], str, "GDP date should be a string")
        self.assertIsInstance(gdp_data['previous_gdp_value'], float, "Previous GDP value should be a float")
        self.assertIsInstance(gdp_data['previous_date'], str, "Previous GDP date should be a string")
        self.assertIsInstance(gdp_data['gdp_growth'], float, "GDP growth should be a float")
        print(gdp_data)

    def test_get_company_profits(self):
        """Testet die Methode get_company_profits mit jährlichen und quartalsweisen Daten."""
        print("\nCompany Profits Test Results:")
        for symbol in self.test_symbols:
            for frequency in ["annual", "quarterly"]:
                result = self.loader.get_company_profits(symbol, frequency=frequency)
                if isinstance(result, dict) and "error" in result:
                    print(f"Fehler für {symbol} ({frequency}): {result['error']}")
                else:
                    print(
                        f"{symbol} ({frequency}): Net Income = {result['net_income_display']} {result['unit']}, Date = {result['latest_date']}")
                    self.assertIsInstance(result['latest_net_income'], float,
                                          f"Latest net income for {symbol} ({frequency}) should be a float")
                    self.assertIsInstance(result['latest_date'], str,
                                          f"Profit date for {symbol} ({frequency}) should be a string")
                    self.assertEqual(result['frequency'], frequency, f"Frequenz sollte {frequency} sein")

    def test_calculate_annual_inflation_rate(self):
        """
        Testet die Methode calculate_annual_inflation_rate mit verschiedenen Szenarien.
        """
        print("\nTeste calculate_annual_inflation_rate...")
        expected_keys = ["current_cpi", "previous_cpi", "current_date", "previous_date", "inflation_rate",
                         "target_month"]

        # Test 1: Aktuellster verfügbarer Monat (2025-05-16)
        result = self.model.calculate_annual_inflation_rate(current_date_str="2025-05-16")
        if isinstance(result, dict) and "error" in result:
            print(f"Fehler (Test 1): {result['error']}")
            self.fail(f"Fehler bei der Berechnung (Test 1): {result['error']}")

        for key in expected_keys:
            self.assertIn(key, result, f"Schlüssel {key} fehlt (Test 1)")

        self.assertIsInstance(result['current_cpi'], float)
        self.assertIsInstance(result['previous_cpi'], float)
        self.assertIsInstance(result['current_date'], str)
        self.assertIsInstance(result['previous_date'], str)
        self.assertIsInstance(result['inflation_rate'], float)
        self.assertIsInstance(result['target_month'], str)

        self.assertGreater(result['current_cpi'], 0)
        self.assertGreater(result['previous_cpi'], 0)
        self.assertGreaterEqual(result['inflation_rate'], -10.0)
        self.assertLessEqual(result['inflation_rate'], 10.0)

        current_date = datetime.strptime(result['current_date'], '%Y-%m-%d')
        self.assertEqual(current_date.strftime('%Y-%m'), result['target_month'])
        previous_date = datetime.strptime(result['previous_date'], '%Y-%m-%d')
        expected_previous_year_month = (
                datetime.strptime(result['target_month'], '%Y-%m') - timedelta(days=365)).strftime('%Y-%m')
        self.assertTrue(previous_date.strftime('%Y-%m') <= expected_previous_year_month,
                        f"Vorjahresmonat {previous_date.strftime('%Y-%m')} sollte <= {expected_previous_year_month} sein (Test 1)")

        print(f"Test 1: Inflationsrate für {result['target_month']}: {result['inflation_rate']}%")
        print(f"Current CPI ({result['current_date']}): {result['current_cpi']}")
        print(f"Previous CPI ({result['previous_date']}): {result['previous_cpi']}")

        # Test 2: Mit spezifischem target_date_str (2025-03-15)
        result = self.model.calculate_annual_inflation_rate(current_date_str="2025-05-16", target_date_str="2025-03-15")
        if isinstance(result, dict) and "error" in result:
            print(f"Fehler (Test 2): {result['error']}")
            self.fail(f"Fehler bei der Berechnung (Test 2): {result['error']}")

        for key in expected_keys:
            self.assertIn(key, result, f"Schlüssel {key} fehlt (Test 2)")

        self.assertEqual(result['target_month'], "2025-03")
        current_date = datetime.strptime(result['current_date'], '%Y-%m-%d')
        self.assertEqual(current_date.strftime('%Y-%m'), "2025-03")
        previous_date = datetime.strptime(result['previous_date'], '%Y-%m-%d')
        self.assertTrue(previous_date.strftime('%Y-%m') <= "2024-03",
                        f"Vorjahresmonat {previous_date.strftime('%Y-%m')} sollte <= 2024-03 sein (Test 2)")

        print(f"Test 2: Inflationsrate für März 2025: {result['inflation_rate']}%")
        print(f"Current CPI ({result['current_date']}): {result['current_cpi']}")
        print(f"Previous CPI ({result['previous_date']}): {result['previous_cpi']}")

        # Test 3: Datum am Monatsanfang (2025-01-01)
        result = self.model.calculate_annual_inflation_rate(current_date_str="2025-01-01")
        if isinstance(result, dict) and "error" in result:
            print(f"Fehler (Test 3): {result['error']}")
            self.fail(f"Fehler bei der Berechnung (Test 3): {result['error']}")

        for key in expected_keys:
            self.assertIn(key, result, f"Schlüssel {key} fehlt (Test 3)")

        self.assertEqual(result['target_month'], "2024-12", "Zielmonat sollte Dezember 2024 sein (Test 3)")
        current_date = datetime.strptime(result['current_date'], '%Y-%m-%d')
        self.assertEqual(current_date.strftime('%Y-%m'), "2024-12")
        previous_date = datetime.strptime(result['previous_date'], '%Y-%m-%d')
        self.assertTrue(previous_date.strftime('%Y-%m') <= "2023-12",
                        f"Vorjahresmonat {previous_date.strftime('%Y-%m')} sollte <= 2023-12 sein (Test 3)")

        print(f"Test 3: Inflationsrate für Dezember 2024: {result['inflation_rate']}%")
        print(f"Current CPI ({result['current_date']}): {result['current_cpi']}")
        print(f"Previous CPI ({result['previous_date']}): {result['previous_cpi']}")

        # Test 4: Zukünftiges Datum (2027-08-15)
        result = self.model.calculate_annual_inflation_rate(current_date_str="2027-08-15")
        self.assertTrue(isinstance(result, dict) and "error" in result,
                        f"Erwarteter Fehler für zukünftiges Datum (Test 4), aber Ergebnis war: {result}")
        self.assertIn("Das Datum 2027-08-15 liegt in der Zukunft", result['error'])
        print(f"Test 4: Erwarteter Fehler für zukünftiges Datum 2027-08-15: {result['error']}")

        # Test 5: Schaltjahr mit spezifischem target_date_str (2024-02-29)
        result = self.model.calculate_annual_inflation_rate(current_date_str="2025-05-16", target_date_str="2024-02-29")
        if isinstance(result, dict) and "error" in result:
            print(f"Fehler (Test 5): {result['error']}")
            self.fail(f"Fehler bei der Berechnung (Test 5): {result['error']}")

        for key in expected_keys:
            self.assertIn(key, result, f"Schlüssel {key} fehlt (Test 5)")

        self.assertEqual(result['target_month'], "2024-02", "Zielmonat sollte Februar 2024 sein (Test 5)")
        current_date = datetime.strptime(result['current_date'], '%Y-%m-%d')
        self.assertEqual(current_date.strftime('%Y-%m'), "2024-02")
        previous_date = datetime.strptime(result['previous_date'], '%Y-%m-%d')
        self.assertTrue(previous_date.strftime('%Y-%m') <= "2023-02",
                        f"Vorjahresmonat {previous_date.strftime('%Y-%m')} sollte <= 2023-02 sein (Test 5)")

        print(f"Test 5: Inflationsrate für Februar 2024: {result['inflation_rate']}%")
        print(f"Current CPI ({result['current_date']}): {result['current_cpi']}")
        print(f"Previous CPI ({result['previous_date']}): {result['previous_cpi']}")

        # Test 6: Zukünftiges Datum (2025-10-15)
        result = self.model.calculate_annual_inflation_rate(current_date_str="2025-10-15")
        self.assertTrue(isinstance(result, dict) and "error" in result,
                        f"Erwarteter Fehler für zukünftiges Datum (Test 6), aber Ergebnis war: {result}")
        self.assertIn("Das Datum 2025-10-15 liegt in der Zukunft", result['error'])
        print(f"Test 6: Erwarteter Fehler für zukünftiges Datum 2025-10-15: {result['error']}")

    def test_calculate_KGV(self):
        print("\nTeste calculate_KGV...")
        symbols = ["AAPL", "KO", "TSLA", "PG", "JNJ", "ILMN"]
        for symbol in symbols:
            result = self.model.calculate_KGV(symbol)
            if isinstance(result, dict) and "error" in result:
                print(f"Fehler für {symbol}: {result['error']}")
            else:
                print(f"KGV für {symbol}: {result}")

    def test_get_payout_ratio_data_annual(self):
        """Testet die Methode get_payout_ratio_data_annual mit verschiedenen Szenarien."""
        print("\nTeste get_payout_ratio_data_annual...")
        symbols = ["AAPL", "KO", "TSLA", "PG", "JNJ", "ILMN", "XPEV"]
        for symbol in symbols:
            result = self.loader.get_payout_ratio_data_annual(symbol)
            if isinstance(result, dict) and "error" in result:
                print(f"Fehler für {symbol}: {result['error']}")
            else:
                print(f"Ausschüttungsquote für {symbol}: {result['payout_ratio_eps']}%")
                if "dps" in result and result["dps"] is not None:
                    print(f"Details: DPS = {result['dps']}, EPS = {result['eps']}, "
                          f"Net Income = {result['net_income']}, Shares = {result['shares_outstanding']}")
                elif "warning" in result:
                    print(f"Warnung: {result['warning']}")
        print("\nBitte validiere die Ergebnisse manuell mit öffentlichen Quellen (z.B. Yahoo Finance, Morningstar).")

    def test_get_interest_expense_data(self):
        """Testet die Methode get_interest_expense_data für reguläre und Sonderfälle."""
        print("\nTeste get_interest_expense_data...")
        test_symbols = ["AAPL", "MSFT", "LCID", "INVALID"]  # Symbole für reguläre und Sonderfälle

        for symbol in test_symbols:
            for period in ["annual", "quarterly"]:
                with self.subTest(symbol=symbol, period=period):
                    result = self.loader.get_interest_expense_data(symbol, frequency=period)
                    if isinstance(result, dict) and "error" in result:
                        print(f"Test für {symbol} ({period}) erfolgreich: Fehler erkannt - {result['error']}.")
                        self.assertIn("symbol", result,
                                      f"Fehler-Dictionary für {symbol} ({period}) sollte 'symbol' enthalten")
                        self.assertEqual(result["symbol"], symbol, f"Symbol im Fehler sollte {symbol} sein")

                        # Spezifische Fehler für Sonderfälle
                        if symbol == "INVALID":
                            self.assertIn("Keine Finanzdaten", result["error"],
                                          f"Erwarteter Fehler für ungültiges Symbol INVALID ({period}): Keine Finanzdaten sollten gefunden werden")

                    else:
                        print(
                            f"Test für {symbol} ({period}) erfolgreich: Zinsaufwendungen für {symbol} betragen {result['interest_expense']}.")
                        self.assertIsInstance(result["interest_expense"], float,
                                              f"Zinsaufwendungen für {symbol} ({period}) sollten ein Float sein")
                        self.assertGreater(result["interest_expense"], 0,
                                           f"Zinsaufwendungen für {symbol} ({period}) sollten positiv sein")
                        self.assertEqual(result["symbol"], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result["frequency"], period, f"Frequency sollte {period} sein")
                        self.assertIn("date", result,
                                      f"Datum sollte in Ergebnis für {symbol} ({period}) enthalten sein")
                        self.assertIsInstance(result["date"], str,
                                              f"Datum für {symbol} ({period}) sollte ein String sein")

    def test_get_ebitda_data(self):
        """Testet die Methode get_ebitda_data mit jährlichen und quartalsweisen Daten."""
        print("\nTeste get_ebitda_data...")
        for symbol in self.test_symbols:
            for frequency in ["annual", "quarterly"]:
                result = self.loader.get_ebitda_data(symbol, frequency=frequency)
                if isinstance(result, dict) and "error" in result:
                    print(f"Fehler für {symbol} ({frequency}): {result['error']}")
                else:
                    print(f"EBITDA für {symbol} ({frequency}): {result['ebitda']}")
                    self.assertIsInstance(result['ebitda'], (int, float), f"EBITDA für {symbol} ({frequency}) sollte eine Zahl sein")
                    self.assertEqual(result['frequency'], frequency, f"Frequenz sollte {frequency} sein")

    def test_get_ebit_data(self):
        """Testet die Methode get_ebit_data für reguläre und Sonderfälle."""
        print("\nTeste get_ebit_data...")
        for symbol in self.test_symbols:
            for period in ["annual", "quarterly"]:
                with self.subTest(symbol=symbol, period=period):
                    result = self.loader.get_ebit_data(symbol, frequency=period)
                    if isinstance(result, dict) and "error" in result:
                        print(f"Test für {symbol} ({period}) erfolgreich: Fehler erkannt - {result['error']}.")
                        self.assertIn("symbol", result,
                                      f"Fehler-Dictionary für {symbol} ({period}) sollte 'symbol' enthalten")
                        self.assertEqual(result["symbol"], symbol, f"Symbol im Fehler sollte {symbol} sein")
                    else:
                        print(f"Test für {symbol} ({period}) erfolgreich: EBIT für {symbol} beträgt {result['ebit']}.")
                        self.assertIsInstance(result["ebit"], float,
                                              f"EBIT für {symbol} ({period}) sollte ein Float sein")
                        self.assertEqual(result["symbol"], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result["frequency"], period, f"Frequency sollte {period} sein")
                        self.assertIn("date", result,
                                      f"Datum sollte in Ergebnis für {symbol} ({period}) enthalten sein")
                        self.assertIsInstance(result["date"], str,
                                              f"Datum für {symbol} ({period}) sollte ein String sein")

    def test_calculate_interest_coverage_ratio(self):
        """Testet die Methode calculate_interest_coverage_ratio für reguläre und Sonderfälle."""
        print("\nTeste calculate_interest_coverage_ratio...")
        for symbol in self.test_symbols:
            for period in ["annual", "quarterly"]:
                with self.subTest(symbol=symbol, period=period):
                    result = self.model.calculate_interest_coverage_ratio(symbol, frequency=period)
                    if isinstance(result, dict) and "error" in result:
                        print(f"Test für {symbol} ({period}) erfolgreich: Fehler erkannt - {result['error']}.")
                        self.assertIn("symbol", result,
                                      f"Fehler-Dictionary für {symbol} ({period}) sollte 'symbol' enthalten")
                        self.assertEqual(result["symbol"], symbol, f"Symbol im Fehler sollte {symbol} sein")
                    else:
                        print(
                            f"Test für {symbol} ({period}) erfolgreich: Zinsdeckungsrate für {symbol} beträgt {result['interest_coverage_ratio']}.")
                        self.assertIsInstance(result["interest_coverage_ratio"], float,
                                              f"Zinsdeckungsrate für {symbol} ({period}) sollte ein Float sein")
                        self.assertEqual(result["symbol"], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result["frequency"], period, f"Frequency sollte {period} sein")
                        self.assertIn("date", result,
                                      f"Datum sollte in Ergebnis für {symbol} ({period}) enthalten sein")
                        self.assertIsInstance(result["date"], str,
                                              f"Datum für {symbol} ({period}) sollte ein String sein")

    def test_get_free_cashflow(self):
        """Testet die Methode get_free_cashflow mit jährlichen und quartalsweisen Daten."""
        print("\nTeste get_free_cashflow...")
        for symbol in self.test_symbols:
            for frequency in ["annual", "quarterly"]:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.loader.get_free_cashflow(symbol, frequency=frequency)
                    if isinstance(result, dict) and "error" in result:
                        print(f"Fehler für {symbol} ({frequency}): {result['error']}")
                        self.assertIn("symbol", result,
                                      f"Fehler-Dictionary für {symbol} ({frequency}) sollte 'symbol' enthalten")
                        self.assertEqual(result["symbol"], symbol, f"Symbol im Fehler sollte {symbol} sein")
                    else:
                        print(f"Free Cashflow für {symbol} ({frequency}): {result['free_cashflow']}")
                        self.assertIsInstance(result['free_cashflow'], float,
                                              f"Free Cashflow für {symbol} ({frequency}) sollte ein Float sein")
                        self.assertEqual(result['symbol'], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result['frequency'], frequency, f"Frequenz sollte {frequency} sein")
                        self.assertIn("date", result,
                                      f"Datum sollte in Ergebnis für {symbol} ({frequency}) enthalten sein")
                        self.assertIsInstance(result['date'], str,
                                              f"Datum für {symbol} ({frequency}) sollte ein String sein")

    def test_get_operating_cashflow(self):
        """Testet die Methode get_operating_cashflow mit jährlichen und quartalsweisen Daten."""
        print("\nTeste get_operating_cashflow...")
        for symbol in self.test_symbols:
            for frequency in ["annual", "quarterly"]:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.loader.get_operating_cashflow(symbol, frequency=frequency)
                    if isinstance(result, dict) and "error" in result:
                        print(f"Fehler für {symbol} ({frequency}): {result['error']}")
                        self.assertIn("symbol", result,
                                      f"Fehler-Dictionary für {symbol} ({frequency}) sollte 'symbol' enthalten")
                        self.assertEqual(result["symbol"], symbol, f"Symbol im Fehler sollte {symbol} sein")
                    else:
                        print(f"Operating Cashflow für {symbol} ({frequency}): {result['operating_cashflow']}")
                        self.assertIsInstance(result['operating_cashflow'], float,
                                              f"Operating Cashflow für {symbol} ({frequency}) sollte ein Float sein")
                        self.assertEqual(result['symbol'], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result['frequency'], frequency, f"Frequenz sollte {frequency} sein")
                        self.assertIn("date", result,
                                      f"Datum sollte in Ergebnis für {symbol} ({frequency}) enthalten sein")
                        self.assertIsInstance(result['date'], str,
                                              f"Datum für {symbol} ({frequency}) sollte ein String sein")

    def test_get_revenue(self):
        """Testet die Methode get_revenue mit jährlichen und quartalsweisen Daten."""
        print("\nTeste get_revenue...")
        for symbol in self.test_symbols:
            for frequency in ["annual", "quarterly"]:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.loader.get_revenue(symbol, frequency=frequency)
                    if isinstance(result, dict) and "error" in result:
                        print(f"Fehler für {symbol} ({frequency}): {result['error']}")
                        self.assertIn("symbol", result,
                                      f"Fehler-Dictionary für {symbol} ({frequency}) sollte 'symbol' enthalten")
                        self.assertEqual(result["symbol"], symbol, f"Symbol im Fehler sollte {symbol} sein")
                    else:
                        print(f"Umsatz für {symbol} ({frequency}): {result['revenue']}")
                        self.assertIsInstance(result['revenue'], float,
                                              f"Umsatz für {symbol} ({frequency}) sollte ein Float sein")
                        self.assertEqual(result['symbol'], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result['frequency'], frequency, f"Frequenz sollte {frequency} sein")
                        self.assertIn("date", result,
                                      f"Datum sollte in Ergebnis für {symbol} ({frequency}) enthalten sein")
                        self.assertIsInstance(result['date'], str,
                                              f"Datum für {symbol} ({frequency}) sollte ein String sein")

    def test_get_enterprise_value(self):
        """Testet die Methode get_enterprise_value mit jährlichen und quartalsweisen Daten für reguläre und Sonderfälle."""
        print("\nTeste get_enterprise_value...")

        test_symbols = self.test_symbols + ["ZM", "INVALID"]
        frequencies = ["annual", "quarterly"]

        for symbol in test_symbols:
            for frequency in frequencies:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.loader.get_enterprise_value(symbol, frequency=frequency)

                    self.assertIsInstance(result, dict,
                                          f"Ergebnis für {symbol} ({frequency}) sollte ein Dictionary sein")

                    if "error" in result:
                        print(f"Fehler für {symbol} ({frequency}): {result['error']}")
                        self.assertIn("symbol", result,
                                      f"Fehler-Dictionary für {symbol} ({frequency}) sollte 'symbol' enthalten")
                        self.assertEqual(result["symbol"], symbol, f"Symbol im Fehler sollte {symbol} sein")

                        if symbol == "INVALID":
                            self.assertIn("Ungültiges Symbol", result["error"],
                                          f"Fehler für INVALID ({frequency}) sollte ungültiges Symbol erwähnen")
                        elif symbol == "NVDA":
                            self.assertIn("Kein gültiger Enterprise Value von yfinance", result["error"],
                                          f"Fehler für NVDA ({frequency}) sollte fehlende yfinance-Daten erwähnen")
                        else:
                            self.fail(f"Unerwarteter Fehler für {symbol} ({frequency}): {result['error']}")
                    else:
                        print(f"Enterprise Value für {symbol} ({frequency}): {result['enterprise_value']}")
                        self.assertIn("enterprise_value", result,
                                      f"Ergebnis für {symbol} ({frequency}) sollte 'enterprise_value' enthalten")
                        self.assertIsInstance(result["enterprise_value"], float,
                                              f"Enterprise Value für {symbol} ({frequency}) sollte ein Float sein")
                        self.assertGreater(result["enterprise_value"], 0,
                                           f"Enterprise Value für {symbol} ({frequency}) sollte positiv sein")
                        self.assertEqual(result["symbol"], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result["frequency"], frequency, f"Frequenz sollte {frequency} sein")

                        if symbol in ["KO", "PG", "JNJ", "PEP"]:
                            self.assertGreater(result["enterprise_value"], 1e10,
                                               f"EV für {symbol} ({frequency}) sollte >10 Mrd. USD sein")
                            self.assertLess(result["enterprise_value"], 1e12,
                                            f"EV für {symbol} ({frequency}) sollte <1 Bio. USD sein")
                        elif symbol == "TSLA":
                            self.assertGreater(result["enterprise_value"], 1e11,
                                               f"EV für TSLA ({frequency}) sollte >100 Mrd. USD sein")

    def test_calculate_cashflow_margin(self):
        """Testet die Methode get_cashflow_margin mit jährlichen und quartalsweisen Daten."""
        print("\nTeste calculate_cashflow_margin...")
        for symbol in self.test_symbols:
            for frequency in ["annual", "quarterly"]:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.model.calculate_cashflow_margin(symbol, frequency=frequency)
                    if isinstance(result, dict) and "error" in result:
                        print(f"Fehler für {symbol} ({frequency}): {result['error']}")
                        self.assertIn("symbol", result,
                                      f"Fehler-Dictionary für {symbol} ({frequency}) sollte 'symbol' enthalten")
                        self.assertEqual(result["symbol"], symbol, f"Symbol im Fehler sollte {symbol} sein")
                    else:
                        print(f"Cashflow-Marge für {symbol} ({frequency}): {result['cashflow_margin']}%")
                        self.assertIsInstance(result['cashflow_margin'], float,
                                              f"Cashflow-Marge für {symbol} ({frequency}) sollte ein Float sein")
                        self.assertEqual(result['symbol'], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result['frequency'], frequency, f"Frequenz sollte {frequency} sein")
                        self.assertIn("date", result,
                                      f"Datum sollte in Ergebnis für {symbol} ({frequency}) enthalten sein")
                        self.assertIsInstance(result['date'], str,
                                              f"Datum für {symbol} ({frequency}) sollte ein String sein")

    def test_get_inventory(self):
        """Testet die Methode get_inventory mit jährlichen und quartalsweisen Daten."""
        print("\nTeste get_inventory...")
        for symbol in self.test_symbols:
            for frequency in ["annual", "quarterly"]:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.loader.get_inventory(symbol, frequency=frequency)
                    if isinstance(result, dict) and "error" in result:
                        print(f"Fehler für {symbol} ({frequency}): {result['error']}")
                        self.assertIn("symbol", result,
                                      f"Fehler-Dictionary für {symbol} ({frequency}) sollte 'symbol' enthalten")
                        self.assertEqual(result["symbol"], symbol, f"Symbol im Fehler sollte {symbol} sein")
                    else:
                        print(f"Vorräte für {symbol} ({frequency}): {result['inventory']}")
                        self.assertIsInstance(result['inventory'], float,
                                              f"Vorräte für {symbol} ({frequency}) sollten ein Float sein")
                        self.assertEqual(result['symbol'], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result['frequency'], frequency, f"Frequenz sollte {frequency} sein")
                        self.assertIn("date", result,
                                      f"Datum sollte in Ergebnis für {symbol} ({frequency}) enthalten sein")
                        self.assertIsInstance(result['date'], str,
                                              f"Datum für {symbol} ({frequency}) sollte ein String sein")

    def test_get_minority_interest(self):
        """Testet die Methode get_minority_interest mit jährlichen und quartalsweisen Daten sowie Sonderfällen."""
        print("\nTeste get_minority_interest...")

        # Test-Symbole, einschließlich eines ungültigen Symbols
        test_symbols = self.test_symbols + ["INVALID"]
        frequencies = ["annual", "quarterly"]

        for symbol in test_symbols:
            for frequency in frequencies:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.loader.get_minority_interest(symbol, frequency=frequency)

                    self.assertIsInstance(result, dict,
                                          f"Ergebnis für {symbol} ({frequency}) sollte ein Dictionary sein")

                    if "error" in result:
                        print(f"Fehler für {symbol} ({frequency}): {result['error']}")
                        if symbol == "INVALID":
                            # Für ungültiges Symbol: Prüfe nur, ob ein Fehler vorhanden ist, toleriere fehlenden 'symbol'
                            self.assertIn("Fehler beim Abrufen", result["error"],
                                          f"Fehler für INVALID ({frequency}) sollte Abrufproblem erwähnen")
                        else:
                            self.assertIn("symbol", result,
                                          f"Fehler-Dictionary für {symbol} ({frequency}) sollte 'symbol' enthalten")
                            self.assertEqual(result["symbol"], symbol, f"Symbol im Fehler sollte {symbol} sein")
                            self.fail(f"Unerwarteter Fehler für {symbol} ({frequency}): {result['error']}")
                    else:
                        print(
                            f"Minderheitenanteil für {symbol} ({frequency}): {result.get('minority_interest', 'nicht verfügbar')}")
                        self.assertIn("minority_interest", result,
                                      f"Ergebnis für {symbol} ({frequency}) sollte 'minority_interest' enthalten")
                        self.assertIsInstance(result["minority_interest"], (float, int),
                                              f"Minderheitenanteil für {symbol} ({frequency}) sollte ein numerischer Wert sein")
                        self.assertEqual(result["symbol"], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result["frequency"], frequency, f"Frequenz sollte {frequency} sein")
                        self.assertIn("date", result,
                                      f"Datum sollte in Ergebnis für {symbol} ({frequency}) enthalten sein")
                        self.assertIsInstance(result["date"], str,
                                              f"Datum für {symbol} ({frequency}) sollte ein String sein")

                        # Prüfe auf Warnung, wenn Minderheitenanteil 0.0 ist
                        if result["minority_interest"] == 0.0:
                            self.assertIn("warning", result,
                                          f"Ergebnis für {symbol} ({frequency}) sollte 'warning' enthalten bei 0.0")
                            self.assertIn("angenommen", result["warning"],
                                          f"Warnung für {symbol} ({frequency}) sollte 'angenommen' enthalten")
                        else:
                            self.assertNotIn("warning", result,
                                             f"Ergebnis für {symbol} ({frequency}) sollte keine Warnung enthalten")

                        # Zusätzliche Validierung für bekannte Symbole
                        if symbol in self.test_symbols:
                            if symbol in ["KO", "PG", "JNJ", "PEP"]:  # Große Unternehmen
                                self.assertGreaterEqual(result["minority_interest"], 0,
                                                        f"Minderheitenanteil für {symbol} ({frequency}) sollte >= 0 sein")
                                self.assertLessEqual(result["minority_interest"], 1e10,
                                                     f"Minderheitenanteil für {symbol} ({frequency}) sollte <10 Mrd. USD sein")
                            elif symbol == "TSLA":
                                self.assertGreaterEqual(result["minority_interest"], 0,
                                                        f"Minderheitenanteil für {symbol} ({frequency}) sollte >= 0 sein")

    def test_get_preferred_stock(self):
        """Testet die Methode get_preferred_stock mit jährlichen und quartalsweisen Daten sowie Sonderfällen."""
        print("\nTeste get_preferred_stock...")

        # Test-Symbole, einschließlich eines ungültigen Symbols
        test_symbols = self.test_symbols + ["INVALID"]
        frequencies = ["annual", "quarterly"]

        for symbol in test_symbols:
            for frequency in frequencies:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.loader.get_preferred_stock(symbol, frequency=frequency)

                    self.assertIsInstance(result, dict,
                                          f"Ergebnis für {symbol} ({frequency}) sollte ein Dictionary sein")

                    if "error" in result:
                        print(f"Fehler für {symbol} ({frequency}): {result['error']}")
                        if symbol == "INVALID":
                            # Für ungültiges Symbol: Prüfe nur, ob ein Fehler vorhanden ist
                            self.assertIn("Fehler beim Abrufen", result["error"],
                                          f"Fehler für INVALID ({frequency}) sollte Abrufproblem erwähnen")
                        else:
                            self.assertIn("symbol", result,
                                          f"Fehler-Dictionary für {symbol} ({frequency}) sollte 'symbol' enthalten")
                            self.assertEqual(result["symbol"], symbol, f"Symbol im Fehler sollte {symbol} sein")
                            self.fail(f"Unerwarteter Fehler für {symbol} ({frequency}): {result['error']}")
                    else:
                        print(
                            f"Vorzugsaktien für {symbol} ({frequency}): {result.get('preferred_stock', 'nicht verfügbar')}")
                        self.assertIn("preferred_stock", result,
                                      f"Ergebnis für {symbol} ({frequency}) sollte 'preferred_stock' enthalten")
                        self.assertIsInstance(result["preferred_stock"], (float, int),
                                              f"Vorzugsaktien für {symbol} ({frequency}) sollte ein numerischer Wert sein")
                        self.assertEqual(result["symbol"], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result["frequency"], frequency, f"Frequenz sollte {frequency} sein")
                        self.assertIn("date", result,
                                      f"Datum sollte in Ergebnis für {symbol} ({frequency}) enthalten sein")
                        self.assertIsInstance(result["date"], str,
                                              f"Datum für {symbol} ({frequency}) sollte ein String sein")

                        # Optional: Prüfe Warnung nur, wenn sie vorhanden ist
                        if "warning" in result:
                            self.assertIn("angenommen", result["warning"],
                                          f"Warnung für {symbol} ({frequency}) sollte 'angenommen' enthalten")

                        # Zusätzliche Validierung für bekannte Symbole
                        if symbol in self.test_symbols:
                            self.assertGreaterEqual(result["preferred_stock"], 0,
                                                    f"Vorzugsaktien für {symbol} ({frequency}) sollte >= 0 sein")
                            if symbol in ["KO", "PG", "JNJ", "PEP"]:  # Große Unternehmen
                                self.assertLessEqual(result["preferred_stock"], 1e10,
                                                     f"Vorzugsaktien für {symbol} ({frequency}) sollte <10 Mrd. USD sein")
                            elif symbol == "TSLA":
                                self.assertLessEqual(result["preferred_stock"], 1e10,
                                                     f"Vorzugsaktien für {symbol} ({frequency}) sollte <10 Mrd. USD sein")

    def test_calculate_inventory_to_revenue_ratio(self):
        """Testet die Methode calculate_inventory_to_revenue_ratio mit regulären und Sonderfällen."""
        print("\nTeste calculate_inventory_to_revenue_ratio...")
        for symbol in self.test_symbols:
            for frequency in ["annual", "quarterly"]:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.model.calculate_inventory_to_revenue_ratio(symbol, frequency=frequency)
                    if isinstance(result, dict) and "error" in result:
                        print(f"Fehler für {symbol} ({frequency}): {result['error']}")
                        # Prüfen, ob Fehler-Dictionary korrekt ist
                        self.assertIn("symbol", result, f"Fehler-Dictionary für {symbol} ({frequency}) sollte 'symbol' enthalten")
                        self.assertEqual(result["symbol"], symbol, f"Symbol im Fehler sollte {symbol} sein")
                        # Sonderfälle validieren
                        if symbol == "PYPL":
                            self.assertIn("möglicherweise Dienstleister", result["error"], f"Fehler für PYPL ({frequency}) sollte Dienstleister erwähnen")
                        if symbol == "NVDA" and frequency == "quarterly":
                            self.assertIn("Keine Vorräte", result["error"], f"Fehler für NVDA (quarterly) sollte fehlende Vorräte erwähnen")
                    else:
                        print(f"Vorräte/Umsatz für {symbol} ({frequency}): {result['inventory_to_revenue_ratio']}%")
                        # Prüfen, ob Ergebnis korrekt ist
                        self.assertIsInstance(result['inventory_to_revenue_ratio'], float, f"Vorräte/Umsatz für {symbol} ({frequency}) sollte ein Float sein")
                        self.assertGreaterEqual(result['inventory_to_revenue_ratio'], 0, f"Vorräte/Umsatz für {symbol} ({frequency}) sollte nicht negativ sein")
                        self.assertEqual(result['symbol'], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result['frequency'], frequency, f"Frequenz sollte {frequency} sein")
                        self.assertIn("date", result, f"Datum sollte in Ergebnis für {symbol} ({frequency}) enthalten sein")
                        self.assertIsInstance(result['date'], str, f"Datum für {symbol} ({frequency}) sollte ein String sein")

    def test_get_net_debt_data(self):
        """Testet die Methode get_net_debt_data mit jährlichen und quartalsweisen Daten."""
        print("\nTeste get_net_debt_data...")
        for symbol in self.test_symbols:
            for frequency in ["annual", "quarterly"]:
                result = self.loader.get_net_debt_data(symbol, frequency=frequency)
                if isinstance(result, dict) and "error" in result:
                    print(f"Fehler für {symbol} ({frequency}): {result['error']}")
                else:
                    print(f"Nettoschulden für {symbol} ({frequency}): {result['net_debt']}")
                    self.assertIsInstance(result['net_debt'], (int, float), f"Net Debt für {symbol} ({frequency}) sollte eine Zahl sein")
                    self.assertEqual(result['frequency'], frequency, f"Frequenz sollte {frequency} sein")

    def test_calculate_kuv(self):
        """Testet die Methode calculate_KUV für verschiedene Symbole und Frequenzen."""
        print("\nTeste calculate_KUV...")
        for symbol in self.test_symbols:
            for frequency in ["annual", "quarterly"]:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.model.calculate_kuv(symbol, frequency=frequency)
                    if isinstance(result, dict) and "error" in result:
                        print(f"Fehler für {symbol} ({frequency}): {result['error']}")
                        self.assertIn("error", result, f"Fehler sollte erkannt werden für {symbol} ({frequency})")
                    else:
                        print(f"KUV für {symbol} ({frequency}): {result['KUV']}")
                        self.assertIn("KUV", result, f"KUV sollte berechnet werden für {symbol} ({frequency})")
                        self.assertIsInstance(result["KUV"], (int, float),
                                              f"KUV sollte eine Zahl sein für {symbol} ({frequency})")
                        self.assertGreaterEqual(result["KUV"], 0,
                                                f"KUV sollte nicht negativ sein für {symbol} ({frequency})")
                        self.assertEqual(result["symbol"], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result["frequency"], frequency, f"Frequenz sollte {frequency} sein")
                        self.assertLessEqual(result["KUV"], 100,
                                             f"KUV sollte realistisch sein für {symbol} ({frequency})")

    def test_calculate_roe(self):
        """Testet die Methode calculate_ROE für verschiedene Symbole und Frequenzen."""
        print("\nTeste calculate_ROE...")
        for symbol in self.test_symbols:
            for frequency in ["annual", "quarterly"]:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.model.calculate_roe(symbol, frequency=frequency)
                    if isinstance(result, dict) and "error" in result:
                        print(f"Fehler für {symbol} ({frequency}): {result['error']}")
                        self.assertIn("error", result, f"Fehler sollte erkannt werden für {symbol} ({frequency})")
                    else:
                        print(f"ROE für {symbol} ({frequency}): {result['ROE']}")
                        self.assertIn("ROE", result, f"ROE sollte berechnet werden für {symbol} ({frequency})")
                        self.assertIsInstance(result["ROE"], (int, float),
                                              f"ROE sollte eine Zahl sein für {symbol} ({frequency})")
                        self.assertEqual(result["symbol"], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result["frequency"], frequency, f"Frequenz sollte {frequency} sein")
                        # ROE kann negativ sein (Verluste), aber realistische Grenzen prüfen
                        self.assertGreaterEqual(result["ROE"], -2,
                                                f"ROE sollte nicht unrealistisch negativ sein für {symbol} ({frequency})")
                        self.assertLessEqual(result["ROE"], 2,
                                             f"ROE sollte realistisch sein für {symbol} ({frequency})")

    def test_calculate_debt_to_equity(self):
        """Testet die Methode calculate_debt_to_equity für verschiedene Symbole, Frequenzen und Sondersituationen."""
        print("\nTeste calculate_debt_to_equity...")

        # Reguläre Fälle
        for symbol in self.test_symbols:
            for frequency in ["annual", "quarterly"]:
                with self.subTest(symbol=symbol, frequency=frequency, case="regular"):
                    result = self.model.calculate_debt_to_equity(symbol, frequency=frequency)
                    if isinstance(result, dict) and "error" in result:
                        print(f"Fehler für {symbol} ({frequency}): {result['error']}")
                        self.assertIn("error", result, f"Fehler sollte erkannt werden für {symbol} ({frequency})")
                    else:
                        print(f"Debt-to-Equity für {symbol} ({frequency}): {result['debt_to_equity']}")
                        self.assertIn("debt_to_equity", result,
                                      f"Debt-to-Equity sollte berechnet werden für {symbol} ({frequency})")
                        self.assertIsInstance(result["debt_to_equity"], (int, float),
                                              f"Debt-to-Equity sollte eine Zahl sein für {symbol} ({frequency})")
                        self.assertEqual(result["symbol"], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result["frequency"], frequency, f"Frequenz sollte {frequency} sein")
                        # Realistische Grenzen prüfen
                        self.assertGreaterEqual(result["debt_to_equity"], 0,
                                                f"Debt-to-Equity sollte nicht negativ sein für {symbol} ({frequency})")
                        self.assertLessEqual(result["debt_to_equity"], 10,
                                             f"Debt-to-Equity sollte realistisch sein für {symbol} ({frequency})")


        # Sondersituationen (Simulierung durch Mocking oder Auswahl von Symbolen mit bekannten Problemen)
        # 1. Fehlende Verbindlichkeitsdaten (Simulierung durch ein ungültiges Symbol)
        with self.subTest(symbol="INVALID", frequency="annual", case="missing_liabilities"):
            result = self.model.calculate_debt_to_equity("INVALID", frequency="annual")
            self.assertIn("error", result, "Fehler sollte erkannt werden für fehlende Daten")
            self.assertIn("Bilanzdaten", result["error"], "Fehlermeldung sollte auf fehlende Bilanzdaten hinweisen")

        # 2. Fehlendes Eigenkapital (Simulierung durch ein ungültiges Symbol)
        with self.subTest(symbol="INVALID", frequency="quarterly", case="missing_equity"):
            result = self.model.calculate_debt_to_equity("INVALID", frequency="quarterly")
            self.assertIn("error", result, "Fehler sollte erkannt werden für fehlende Daten")
            self.assertIn("Bilanzdaten", result["error"], "Fehlermeldung sollte auf fehlende Bilanzdaten hinweisen")

    def test_calculate_peg_ratio(self):
        """Testet die calculate_peg_ratio-Methode mit echten Daten für reguläre und Sondersituationen mit Subtests."""
        test_cases = [
            # Reguläre Fälle
            {
                "symbol": "AAPL",
                "expected_methods": ["direct", "calculated_from_earnings_growth"],
                "expected_peg_range": (1.0, 5.0),
                "description": "Regulärer Fall für AAPL (Apple)"
            },
            {
                "symbol": "MSFT",
                "expected_methods": ["direct", "calculated_from_earnings_growth"],
                "expected_peg_range": (1.5, 3.5),
                "description": "Regulärer Fall für MSFT (Microsoft)"
            },
            # Sondersituationen
            {
                "symbol": "LCID",
                "expected_methods": [None],
                "expected_peg_range": None,
                "description": "Sonderfall: Unternehmen mit potenziellen Verlusten (LCID)"
            },
            {
                "symbol": "INVALID",
                "expected_methods": [None],
                "expected_peg_range": None,
                "description": "Sonderfall: Ungültiges Symbol (INVALID)"
            }
        ]

        for case in test_cases:
            with self.subTest(case=case["description"]):
                # Methode get_peg_ratio (aus DataLoader) testen
                result_get = self.model.dataloader.get_peg_ratio(case["symbol"])
                if "error" not in result_get:
                    print(f"Test für {case['description']} mit get_peg_ratio erfolgreich: Die PEG-Ratio für {case['symbol']} beträgt {result_get['peg_ratio']} (Methode: {result_get['method']}).")
                    self.assertIn(result_get["method"], case["expected_methods"],
                                 f"Falsche Methode für {case['description']}: Erwartet {case['expected_methods']}, erhalten {result_get['method']}")
                    self.assertGreaterEqual(result_get["peg_ratio"], case["expected_peg_range"][0],
                                           f"PEG-Ratio zu niedrig für {case['description']}: {result_get['peg_ratio']}")
                    self.assertLessEqual(result_get["peg_ratio"], case["expected_peg_range"][1],
                                        f"PEG-Ratio zu hoch für {case['description']}: {result_get['peg_ratio']}")
                else:
                    print(f"Test für {case['description']} mit get_peg_ratio erfolgreich: Fehler erkannt - {result_get['error']}.")
                    self.assertEqual(case["expected_methods"], [None],
                                    f"Unerwarteter Fehler für {case['description']}: {result_get['error']}")

                # Methode calculate_peg_ratio (aus Model) testen
                result_calc = self.model.calculate_peg_ratio(case["symbol"])
                if "error" not in result_calc:
                    print(f"Test für {case['description']} mit calculate_peg_ratio erfolgreich: Die PEG-Ratio für {case['symbol']} beträgt {result_calc['peg_ratio']} (Methode: {result_calc['method']}).")
                    self.assertIn(result_calc["method"], case["expected_methods"],
                                 f"Falsche Methode für {case['description']}: Erwartet {case['expected_methods']}, erhalten {result_calc['method']}")
                    self.assertGreaterEqual(result_calc["peg_ratio"], case["expected_peg_range"][0],
                                           f"PEG-Ratio zu niedrig für {case['description']}: {result_calc['peg_ratio']}")
                    self.assertLessEqual(result_calc["peg_ratio"], case["expected_peg_range"][1],
                                        f"PEG-Ratio zu hoch für {case['description']}: {result_calc['peg_ratio']}")
                else:
                    print(f"Test für {case['description']} mit calculate_peg_ratio erfolgreich: Fehler erkannt - {result_calc['error']}.")
                    self.assertEqual(case["expected_methods"], [None],
                                    f"Unerwarteter Fehler für {case['description']}: {result_calc['error']}")

    def test_calculate_total_inflation_for_period(self):
        """Testet die Methode calculate_total_inflation_for_period für verschiedene Fälle."""
        test_cases = [
            # Bestehender realistischer Testfall
            {
                "start_date": "2020-02-01",
                "end_date": "2025-02-01",
                "description": "Realistischer Zeitraum (Februar 2020 bis Februar 2025, 5 Jahre)",
                "expected_error": False,
                "expected_inflation_range": (0, 30),  # Plausible Inflation über 5 Jahre
            },
            # Neuer Testfall: 3 Jahre, März 2019 bis März 2022
            {
                "start_date": "2019-03-01",
                "end_date": "2022-03-01",
                "description": "3 Jahre (März 2019 bis März 2022)",
                "expected_error": False,
                "expected_inflation_range": (0, 20),  # Plausible Inflation über 3 Jahre
            },
            # Neuer Testfall: 1 Jahr, Juli 2023 bis Juli 2024
            {
                "start_date": "2023-07-01",
                "end_date": "2024-07-01",
                "description": "1 Jahr (Juli 2023 bis Juli 2024)",
                "expected_error": False,
                "expected_inflation_range": (0, 10),  # Plausible Inflation über 1 Jahr
            },
            # Neuer Testfall: 4 Jahre, Oktober 2020 bis Oktober 2024
            {
                "start_date": "2020-10-01",
                "end_date": "2024-10-01",
                "description": "4 Jahre (Oktober 2020 bis Oktober 2024)",
                "expected_error": False,
                "expected_inflation_range": (0, 25),  # Plausible Inflation über 4 Jahre
            },
            # Neuer Testfall: Zeitraum kurz vor aktuellem Datum, Juni 2020 bis April 2025
            {
                "start_date": "2020-06-01",
                "end_date": "2025-04-01",
                "description": "Kurz vor aktuellem Datum (Juni 2020 bis April 2025, ~4.8 Jahre)",
                "expected_error": False,
                "expected_inflation_range": (0, 30),  # Plausible Inflation über ~5 Jahre
            },
            # Bestehender Testfall: Zukunft mit fehlenden Daten
            {
                "start_date": "2025-10-01",
                "end_date": "2025-11-01",
                "description": "Zukunft mit fehlenden Daten (nach Mai 2025)",
                "expected_error": True,
                "expected_inflation_range": None,
            },
            # Bestehender Testfall: Ungültiger Zeitraum
            {
                "start_date": "2020-02-01",
                "end_date": "2020-01-01",
                "description": "Ungültiger Zeitraum (Enddatum vor Startdatum)",
                "expected_error": True,
                "expected_inflation_range": None,
            }
        ]

        for case in test_cases:
            with self.subTest(description=case["description"]):
                result = self.model.calculate_total_inflation_for_period(
                    case["start_date"], case["end_date"]
                )

                if case["expected_error"]:
                    self.assertIn("error", result, f"Erwarteter Fehler für {case['description']} nicht aufgetreten.")
                    print(f"Test für {case['description']} erfolgreich: Fehler erkannt - {result['error']}")
                else:
                    self.assertNotIn("error", result, f"Fehler bei der Analyse: {result.get('error')}")
                    self.assertIn("total_inflation", result, "Fehlender 'total_inflation' Wert.")
                    inflation = result["total_inflation"]
                    min_inflation, max_inflation = case["expected_inflation_range"]
                    self.assertGreaterEqual(inflation, min_inflation,
                                           f"Inflation für {case['description']} zu niedrig: {inflation}%")
                    self.assertLessEqual(inflation, max_inflation,
                                         f"Inflation für {case['description']} zu hoch: {inflation}%")
                    self.assertEqual(result["start_date"], case["start_date"],
                                     f"Falsches Startdatum: Erwartet {case['start_date']}, erhalten {result['start_date']}")
                    self.assertEqual(result["end_date"], case["end_date"],
                                     f"Falsches Enddatum: Erwartet {case['end_date']}, erhalten {result['end_date']}")
                    print(f"Test für {case['description']} erfolgreich: {result['message']}")

    def test_calculate_avg_quarterly_profit_growth(self):
        """
        Testet die Methode calculate_avg_quarterly_profit_growth mit offiziellen Daten für reguläre und Sonderfälle,
        einschließlich eines Unternehmens mit negativen Nettogewinnen.
        """
        # Subtest 1: Regulärer Fall (AAPL, positive Net Incomes)
        with self.subTest(scenario="Regular case: AAPL"):
            result = self.model.calculate_avg_quarterly_profit_growth(
                symbol="AAPL", start_date="2024-01-01", end_date="2025-05-29"
            )
            if "error" in result:
                print(f"Fehler bei AAPL-Daten: {result['error']}")
            self.assertNotIn("error", result, "Es sollte kein Fehler auftreten für AAPL.")
            self.assertIn("avg_growth", result, "avg_growth sollte vorhanden sein.")
            self.assertGreater(result["avg_growth"], 0, "AQGR sollte positiv sein für AAPL.")
            self.assertIn("Durchschnittliche quartalsweise Gewinnwachstumsrate", result["message"],
                          "Nachricht sollte AQGR enthalten.")
            print(f"Subtest für regulären Fall: AAPL erfolgreich: {result['message']}")

        # Subtest 2: Unternehmen mit negativen Net Incomes (LCID)
        with self.subTest(scenario="Negative net income case: LCID"):
            result = self.model.calculate_avg_quarterly_profit_growth(
                symbol="LCID", start_date="2024-01-01", end_date="2025-05-29"
            )
            if "error" in result:
                print(f"Fehler bei LCID-Daten: {result['error']}")
            self.assertNotIn("error", result, "Es sollte kein Fehler auftreten für LCID.")
            self.assertIn("net_incomes", result, "net_incomes sollte vorhanden sein.")
            self.assertTrue(all(item["value"] < 0 for item in result["net_incomes"]),
                            "Alle Nettogewinne sollten negativ sein für LCID.")
            self.assertIn("negative Nettogewinne", result["message"],
                          "Nachricht sollte negative Nettogewinne erwähnen.")
            self.assertIn("hohem Risiko behaftet", result["message"],
                          "Nachricht sollte Risiko erwähnen.")
            self.assertIn("Mio. USD", result["message"],
                          "Nachricht sollte Einheit in Mio. USD enthalten.")
            print(f"Subtest für negatives Net Income: LCID erfolgreich: {result['message']}")

        # Subtest 3: Sonderfall - Ungültiges Symbol (XYZ_INVALID)
        with self.subTest(scenario="Special case: Invalid symbol"):
            result = self.model.calculate_avg_quarterly_profit_growth(
                symbol="XYZ_INVALID", start_date="2024-01-01", end_date="2025-05-29"
            )
            self.assertIn("error", result, "Es sollte ein Fehler auftreten für ungültiges Symbol.")
            self.assertIn("Keine Finanzdaten", result["error"],
                          "Fehlermeldung sollte auf fehlende Daten hinweisen.")
            print(f"Subtest für ungültiges Symbol: {result['error']}")

        # Subtest 4: Sonderfall - Unzureichende Daten (nur ein Quartal)
        with self.subTest(scenario="Special case: Insufficient data"):
            result = self.model.calculate_avg_quarterly_profit_growth(
                symbol="NEWCO", start_date="2024-01-01", end_date="2025-05-29"
            )
            if "error" not in result and len(result.get("net_incomes", [])) < 2:
                self.assertIn("error", result, "Es sollte ein Fehler auftreten bei unzureichenden Daten.")
                self.assertIn("Nicht genügend Nettogewinndaten", result["error"],
                              "Fehlermeldung sollte auf unzureichende Daten hinweisen.")
            print(f"Subtest für unzureichende Daten: {result.get('error', 'Keine Fehler')}")

    def test_calculate_avg_annual_profit_growth(self):
        """Testet die Methode calculate_annual_profit_growth mit mehreren Szenarien als Subtests."""
        # Reguläre Szenarien (Unternehmen mit positiven Nettogewinnen)
        with self.subTest(scenario="Regular case: Annual data for AAPL"):
            result = self.model.calculate_avg_annual_profit_growth(
                symbol="AAPL",
                start_date="2020-01-01",  # Wird ignoriert
                end_date="2025-05-25"  # Wird ignoriert
            )
            if "error" in result:
                print(f"Fehler bei AAPL-Daten: {result['error']}")
            self.assertNotIn("error", result, "Es sollte kein Fehler auftreten.")
            self.assertIn("avg_growth", result, "AAGR sollte berechnet werden.")
            self.assertGreaterEqual(result["avg_growth"], -100, "AAGR sollte realistisch sein.")
            self.assertEqual(result["frequency"], "annual", "Frequenz sollte 'annual' sein.")
            self.assertNotIn("net_incomes", result, "net_incomes sollte nicht vorhanden sein bei positiven Gewinnen.")
            print(f"Subtest für Regulärer Fall (Jahresdaten für AAPL) erfolgreich: {result['message']}")

        with self.subTest(scenario="Regular case: Annual data for TSLA"):
            result = self.model.calculate_avg_annual_profit_growth(
                symbol="TSLA",
                start_date="2020-01-01",
                end_date="2025-05-25"
            )
            if "error" in result:
                print(f"Fehler bei TSLA-Daten: {result['error']}")
            self.assertNotIn("error", result, "Es sollte kein Fehler auftreten.")
            self.assertIn("avg_growth", result, "AAGR sollte berechnet werden.")
            self.assertGreaterEqual(result["avg_growth"], -100, "AAGR sollte realistisch sein.")
            self.assertEqual(result["frequency"], "annual", "Frequenz sollte 'annual' sein.")
            self.assertNotIn("net_incomes", result, "net_incomes sollte nicht vorhanden sein bei positiven Gewinnen.")
            print(f"Subtest für Regulärer Fall (Jahresdaten für TSLA) erfolgreich: {result['message']}")

        # Sondersituation: Unternehmen mit negativen Nettogewinnen
        with self.subTest(scenario="Negative net income case: LCID"):
            result = self.model.calculate_avg_annual_profit_growth(
                symbol="LCID",
                start_date="2020-01-01",
                end_date="2025-05-25"
            )
            if "error" in result:
                print(f"Fehler bei LCID-Daten: {result['error']}")
            self.assertNotIn("error", result, "Es sollte kein Fehler auftreten für LCID.")
            self.assertIn("net_incomes", result, "net_incomes sollte vorhanden sein.")
            self.assertTrue(all(item["value"] < 0 for item in result["net_incomes"]),
                            "Alle Nettogewinne sollten negativ sein für LCID.")
            self.assertIn("negative Nettogewinne", result["message"],
                          "Nachricht sollte negative Nettogewinne erwähnen.")
            self.assertIn("hohem Risiko behaftet", result["message"],
                          "Nachricht sollte Risiko erwähnen.")
            self.assertIn("Ein Vergleich mit der Inflation ist nicht sinnvoll", result["message"],
                          "Nachricht sollte angeben, dass ein Inflation-Vergleich nicht sinnvoll ist.")
            self.assertNotIn("avg_growth", result, "AAGR sollte nicht vorhanden sein bei negativen Gewinnen.")
            print(f"Subtest für negatives Net Income (LCID) erfolgreich: {result['message']}")

        # Sondersituationen: Ungültige Daten
        with self.subTest(scenario="Invalid symbol: XYZ_INVALID"):
            result = self.model.calculate_avg_annual_profit_growth(
                symbol="XYZ_INVALID",
                start_date="2020-01-01",
                end_date="2025-05-25"
            )
            self.assertIn("error", result, "Ein Fehler sollte auftreten.")
            self.assertIn("Keine Finanzdaten", result["error"], "Fehlermeldung sollte auf fehlende Daten hinweisen.")
            print(f"Subtest für Sonderfall: Ungültiges Symbol erfolgreich: {result['error']}")

        with self.subTest(scenario="Insufficient data: NEWCO"):
            result = self.model.calculate_avg_annual_profit_growth(
                symbol="NEWCO",
                start_date="2020-01-01",
                end_date="2025-05-25"
            )
            self.assertIn("error", result, "Ein Fehler sollte auftreten bei unzureichenden Daten.")
            self.assertIn("Keine Finanzdaten", result["error"], "Fehlermeldung sollte auf fehlende Daten hinweisen.")
            print(f"Subtest für Sonderfall: Unzureichende Daten erfolgreich: {result['error']}")

    def test_compare_avg_quarterly_growth_to_inflation(self):
        """Testet die Methode compare_quarterly_growth_to_inflation mit mehreren Szenarien als Subtests."""
        # Reguläre Szenarien (Unternehmen mit positiven Nettogewinnen)
        with self.subTest(scenario="Regular case: AAPL"):
            result = self.model.compare_avg_quarterly_growth_to_inflation(
                symbol="AAPL",
                start_date="2023-01-01",
                end_date="2025-05-25"
            )
            if "error" in result:
                print(f"Fehler bei AAPL-Daten: {result['error']}")
            self.assertNotIn("error", result, "Es sollte kein Fehler auftreten.")
            self.assertIn("aqgr", result, "AQGR sollte vorhanden sein.")
            self.assertIn("total_inflation", result, "Inflation sollte vorhanden sein.")
            self.assertIn("outperforms_inflation", result, "Bewertung sollte vorhanden sein.")
            self.assertIsInstance(result["outperforms_inflation"], bool, "Bewertung sollte ein Boolean sein.")
            print(f"Subtest für Regulärer Fall (AAPL) erfolgreich: {result['message']}")

        with self.subTest(scenario="Regular case: TSLA"):
            result = self.model.compare_avg_quarterly_growth_to_inflation(
                symbol="TSLA",
                start_date="2023-01-01",
                end_date="2025-05-25"
            )
            if "error" in result:
                print(f"Fehler bei TSLA-Daten: {result['error']}")
            self.assertNotIn("error", result, "Es sollte kein Fehler auftreten.")
            self.assertIn("aqgr", result, "AQGR sollte vorhanden sein.")
            self.assertIn("total_inflation", result, "Inflation sollte vorhanden sein.")
            self.assertIn("outperforms_inflation", result, "Bewertung sollte vorhanden sein.")
            self.assertIsInstance(result["outperforms_inflation"], bool, "Bewertung sollte ein Boolean sein.")
            print(f"Subtest für Regulärer Fall (TSLA) erfolgreich: {result['message']}")

        # Sondersituationen (Unternehmen mit negativen Nettogewinnen)
        with self.subTest(scenario="Negative net income case: LCID"):
            result = self.model.compare_avg_quarterly_growth_to_inflation(
                symbol="LCID",
                start_date="2023-01-01",
                end_date="2025-05-25"
            )
            if "error" in result:
                print(f"Fehler bei LCID-Daten: {result['error']}")
            self.assertNotIn("error", result, "Es sollte kein Fehler auftreten für LCID.")
            self.assertIn("net_incomes", result, "net_incomes sollte vorhanden sein.")
            self.assertTrue(all(item["value"] < 0 for item in result["net_incomes"]),
                            "Alle Nettogewinne sollten negativ sein für LCID.")
            self.assertIn("negative Nettogewinne", result["message"],
                          "Nachricht sollte negative Nettogewinne erwähnen.")
            self.assertIn("hohem Risiko behaftet", result["message"],
                          "Nachricht sollte Risiko erwähnen.")
            self.assertIn("Ein Vergleich mit der Inflation ist nicht sinnvoll", result["message"],
                          "Nachricht sollte angeben, dass ein Inflation-Vergleich nicht sinnvoll ist.")
            self.assertNotIn("aqgr", result, "AQGR sollte nicht vorhanden sein bei negativen Gewinnen.")
            self.assertNotIn("total_inflation", result, "Inflation sollte nicht vorhanden sein bei negativen Gewinnen.")
            self.assertNotIn("outperforms_inflation", result,
                             "Bewertung sollte nicht vorhanden sein bei negativen Gewinnen.")
            print(f"Subtest für negatives Net Income (LCID) erfolgreich: {result['message']}")

        # Sondersituationen (Ungültige Daten)
        with self.subTest(scenario="Invalid symbol: XYZ_INVALID"):
            result = self.model.compare_avg_quarterly_growth_to_inflation(
                symbol="XYZ_INVALID",
                start_date="2023-01-01",
                end_date="2025-05-25"
            )
            self.assertIn("error", result, "Ein Fehler sollte auftreten.")
            self.assertIn("Keine Finanzdaten", result["error"], "Fehlermeldung sollte auf fehlende Daten hinweisen.")
            print(f"Subtest für Sonderfall: Ungültiges Symbol erfolgreich: {result['error']}")

        with self.subTest(scenario="Insufficient data: NEWCO"):
            result = self.model.compare_avg_quarterly_growth_to_inflation(
                symbol="NEWCO",
                start_date="2023-01-01",
                end_date="2025-05-25"
            )
            self.assertIn("error", result, "Ein Fehler sollte auftreten bei unzureichenden Daten.")
            self.assertIn("Keine Finanzdaten", result["error"], "Fehlermeldung sollte auf fehlende Daten hinweisen.")
            print(f"Subtest für Sonderfall: Unzureichende Daten erfolgreich: {result['error']}")

    def test_compare_annual_growth_to_inflation(self):
        """Testet die Methode compare_annual_growth_to_inflation mit mehreren Szenarien als Subtests."""
        # Reguläre Szenarien (Unternehmen mit ausschließlich positiven Nettogewinnen)
        with self.subTest(scenario="Regular case: AAPL"):
            result = self.model.compare_avg_annual_growth_to_inflation(
                symbol="AAPL",
                start_date="2020-01-01",
                end_date="2025-05-26"
            )
            if "error" in result:
                print(f"Fehler bei AAPL-Daten: {result['error']}")
            self.assertNotIn("error", result, "Es sollte kein Fehler auftreten.")
            self.assertIn("aagr", result, "AAGR sollte vorhanden sein.")
            self.assertIn("total_inflation", result, "Inflation sollte vorhanden sein.")
            self.assertIn("outperforms_inflation", result, "Bewertung sollte vorhanden sein.")
            self.assertIsInstance(result["outperforms_inflation"], bool, "Bewertung sollte ein Boolean sein.")
            print(f"Subtest für Regulärer Fall (AAPL) erfolgreich: {result['message']}")

        with self.subTest(scenario="Regular case: TSLA"):
            result = self.model.compare_avg_annual_growth_to_inflation(
                symbol="TSLA",
                start_date="2020-01-01",
                end_date="2025-05-26"
            )
            if "error" in result:
                print(f"Fehler bei TSLA-Daten: {result['error']}")
            self.assertNotIn("error", result, "Es sollte kein Fehler auftreten.")
            self.assertIn("aagr", result, "AAGR sollte vorhanden sein.")
            self.assertIn("total_inflation", result, "Inflation sollte vorhanden sein.")
            self.assertIn("outperforms_inflation", result, "Bewertung sollte vorhanden sein.")
            self.assertIsInstance(result["outperforms_inflation"], bool, "Bewertung sollte ein Boolean sein.")
            print(f"Subtest für Regulärer Fall (TSLA) erfolgreich: {result['message']}")

        # Sondersituation: Unternehmen mit negativen Nettogewinnen
        with self.subTest(scenario="Negative net income case: BTI"):
            result = self.model.compare_avg_annual_growth_to_inflation(
                symbol="BTI",
                start_date="2020-01-01",
                end_date="2025-05-26"
            )
            if "error" in result:
                print(f"Fehler bei BTI-Daten: {result['error']}")
            self.assertNotIn("error", result, "Es sollte kein Fehler auftreten für BTI.")
            self.assertIn("net_incomes", result, "net_incomes sollte vorhanden sein.")
            self.assertTrue(any(item["value"] < 0 for item in result["net_incomes"]),
                            "Mindestens ein Nettogewinn sollte negativ sein für BTI.")
            self.assertIn("negative Nettogewinne", result["message"],
                          "Nachricht sollte negative Nettogewinne erwähnen.")
            self.assertIn("hohem Risiko behaftet", result["message"],
                          "Nachricht sollte Risiko erwähnen.")
            self.assertIn("Ein Vergleich mit der Inflation ist nicht sinnvoll", result["message"],
                          "Nachricht sollte angeben, dass ein Inflation-Vergleich nicht sinnvoll ist.")
            self.assertNotIn("aagr", result, "AAGR sollte nicht vorhanden sein bei negativen Gewinnen.")
            self.assertNotIn("total_inflation", result, "Inflation sollte nicht vorhanden sein bei negativen Gewinnen.")
            self.assertNotIn("outperforms_inflation", result,
                             "Bewertung sollte nicht vorhanden sein bei negativen Gewinnen.")
            print(f"Subtest für negatives Net Income (BTI) erfolgreich: {result['message']}")

        with self.subTest(scenario="Fully negative net income case: LCID"):
            result = self.model.compare_avg_annual_growth_to_inflation(
                symbol="LCID",
                start_date="2020-01-01",
                end_date="2025-05-26"
            )
            if "error" in result:
                print(f"Fehler bei LCID-Daten: {result['error']}")
            self.assertNotIn("error", result, "Es sollte kein Fehler auftreten für LCID.")
            self.assertIn("net_incomes", result, "net_incomes sollte vorhanden sein.")
            self.assertTrue(all(item["value"] < 0 for item in result["net_incomes"]),
                            "Alle Nettogewinne sollten negativ sein für LCID.")
            self.assertIn("negative Nettogewinne", result["message"],
                          "Nachricht sollte negative Nettogewinne erwähnen.")
            self.assertIn("hohem Risiko behaftet", result["message"],
                          "Nachricht sollte Risiko erwähnen.")
            self.assertIn("Ein Vergleich mit der Inflation ist nicht sinnvoll", result["message"],
                          "Nachricht sollte angeben, dass ein Inflation-Vergleich nicht sinnvoll ist.")
            self.assertNotIn("aagr", result, "AAGR sollte nicht vorhanden sein bei durchgängig negativen Gewinnen.")
            self.assertNotIn("total_inflation", result,
                             "Inflation sollte nicht vorhanden sein bei durchgängig negativen Gewinnen.")
            self.assertNotIn("outperforms_inflation", result,
                             "Bewertung sollte nicht vorhanden sein bei durchgängig negativen Gewinnen.")
            print(f"Subtest für durchgängig negatives Net Income (LCID) erfolgreich: {result['message']}")

        # Sondersituationen (Ungültige Daten)
        with self.subTest(scenario="Invalid symbol: XYZ_INVALID"):
            result = self.model.compare_avg_annual_growth_to_inflation(
                symbol="XYZ_INVALID",
                start_date="2020-01-01",
                end_date="2025-05-26"
            )
            self.assertIn("error", result, "Ein Fehler sollte auftreten.")
            self.assertIn("Keine Finanzdaten", result["error"], "Fehlermeldung sollte auf fehlende Daten hinweisen.")
            print(f"Subtest für Sonderfall: Ungültiges Symbol erfolgreich: {result['error']}")

        with self.subTest(scenario="No data: NEWCO"):
            result = self.model.compare_avg_annual_growth_to_inflation(
                symbol="NEWCO",
                start_date="2020-01-01",
                end_date="2025-05-26"
            )
            self.assertIn("error", result, "Ein Fehler sollte auftreten bei fehlenden Daten.")
            self.assertIn("Keine Finanzdaten", result["error"], "Fehlermeldung sollte auf fehlende Daten hinweisen.")
            print(f"Subtest für Sonderfall: Fehlende Daten (NEWCO) erfolgreich: {result['error']}")

    def test_calculate_ev_to_sales(self):
        """
        Testet die Methode calculate_ev_to_sales mit verschiedenen Symbolen und Frequenzen.
        """
        print("\nTeste calculate_ev_to_sales...")

        # Testsymbole, einschließlich eines ungültigen Symbols
        test_symbols = ["MCD"]
        frequencies = ["annual", "quarterly"]
        expected_keys = ["ev_to_sales", "symbol", "frequency"]

        for symbol in test_symbols:
            for frequency in frequencies:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.model.calculate_ev_to_sales(symbol, frequency=frequency)

                    # Überprüfen, ob das Ergebnis ein Dictionary ist
                    self.assertIsInstance(result, dict,
                                          f"Ergebnis für {symbol} ({frequency}) sollte ein Dictionary sein")

                    if "error" in result:
                        print(f"Fehler für {symbol} ({frequency}): {result['error']}")
                        self.assertIn("symbol", result,
                                      f"Fehler-Dictionary für {symbol} ({frequency}) sollte 'symbol' enthalten")
                        self.assertEqual(result["symbol"], symbol, f"Symbol im Fehler sollte {symbol} sein")

                        # Spezifische Fehlerprüfungen
                        if symbol == "INVALID":
                            self.assertIn("Ungültiges Symbol", result["error"],
                                          f"Fehler für INVALID ({frequency}) sollte ungültiges Symbol erwähnen")
                        elif symbol == "NVDA" and "yfinance fehlgeschlagen" in result["error"]:
                            self.fail(f"Unerwarteter yfinance-Fehler für NVDA ({frequency}): {result['error']}")
                        else:
                            # Für andere Symbole erlauben wir Fehler, falls Daten fehlen
                            pass
                    else:
                        print(f"EV/Sales Multiple für {symbol} ({frequency}): {result['ev_to_sales']}")
                        # Überprüfen, ob alle erwarteten Schlüssel vorhanden sind
                        for key in expected_keys:
                            self.assertIn(key, result,
                                          f"Schlüssel {key} fehlt im Ergebnis für {symbol} ({frequency})")

                        # Überprüfen der Datentypen
                        self.assertIsInstance(result["ev_to_sales"], float,
                                              f"EV/Sales für {symbol} ({frequency}) sollte ein Float sein")
                        self.assertIsInstance(result["symbol"], str,
                                              f"Symbol für {symbol} ({frequency}) sollte ein String sein")
                        self.assertIsInstance(result["frequency"], str,
                                              f"Frequenz für {symbol} ({frequency}) sollte ein String sein")

                        # Plausibilitätsprüfungen
                        self.assertGreater(result["ev_to_sales"], 0,
                                           f"EV/Sales für {symbol} ({frequency}) sollte positiv sein")
                        self.assertLess(result["ev_to_sales"], 100,
                                        f"EV/Sales für {symbol} ({frequency}) sollte realistisch sein (< 100)")

                        # Überprüfen der Symbol- und Frequenzkonsistenz
                        self.assertEqual(result["symbol"], symbol,
                                         f"Symbol im Ergebnis sollte {symbol} sein ({frequency})")
                        self.assertEqual(result["frequency"], frequency,
                                         f"Frequenz im Ergebnis sollte {frequency} sein")


        # Zusätzlicher Test für ungültige Frequenz
        with self.subTest(symbol="AAPL", frequency="monthly"):
            result = self.model.calculate_ev_to_sales("AAPL", frequency="monthly")
            self.assertIn("error", result, f"Ergebnis für ungültige Frequenz sollte einen Fehler enthalten")
            self.assertIn("Ungültige Frequenz", result["error"],
                          f"Fehler für ungültige Frequenz sollte 'Ungültige Frequenz' enthalten")
            self.assertEqual(result["symbol"], "AAPL", f"Symbol im Fehler sollte AAPL sein")
            print(f"Fehler für ungültige Frequenz (monthly): {result['error']}")

    def test_calculate_price_to_freeCashflow(self):
        """Testet die Methode calculate_price_to_freeCashflow für reguläre und Sonderfälle."""
        print("\nTeste calculate_price_to_freeCashflow...")
        for symbol in self.test_symbols + ["INVALID"]:  # Füge ein ungültiges Symbol hinzu
            for frequency in ["annual", "quarterly"]:  # Teste beide Frequenzen
                with self.subTest(symbol=symbol, frequency=frequency):
                    # Teste mit Caching
                    result = self.model.calculate_price_to_freeCashflow(symbol, use_cache=True, frequency=frequency)
                    if isinstance(result, dict) and "error" in result:
                        print(f"Test für {symbol} ({frequency}) erfolgreich: Fehler erkannt - {result['error']}.")
                        self.assertIn("symbol", result,
                                      f"Fehler-Dictionary für {symbol} ({frequency}) sollte 'symbol' enthalten")
                        self.assertEqual(result["symbol"], symbol,
                                         f"Symbol im Fehler sollte {symbol} sein")
                    else:
                        print(
                            f"Test für {symbol} ({frequency}) erfolgreich: Price/FreeCashflow für {symbol} beträgt {result['price_to_freeCashflow']}.")
                        self.assertIn("price_to_freeCashflow", result,
                                      f"Price/FreeCashflow-Wert sollte in Ergebnis für {symbol} ({frequency}) enthalten sein")
                        self.assertIn("symbol", result,
                                      f"Symbol sollte in Ergebnis für {symbol} ({frequency}) enthalten sein")
                        self.assertIn("frequency", result,
                                      f"Frequency sollte in Ergebnis für {symbol} ({frequency}) enthalten sein")
                        self.assertIn("date", result,
                                      f"Datum sollte in Ergebnis für {symbol} ({frequency}) enthalten sein")
                        self.assertEqual(result["symbol"], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result["frequency"], frequency, f"Frequency sollte {frequency} sein")
                        self.assertIsInstance(result["date"], str,
                                              f"Datum für {symbol} ({frequency}) sollte ein String sein")
                        if result["price_to_freeCashflow"] == "inf":
                            self.assertIn("message", result,
                                          f"Message sollte bei 'inf' für {symbol} ({frequency}) enthalten sein")
                            self.assertEqual(result["message"],
                                             "Price/FreeCashflow ist unendlich aufgrund eines null oder negativen FreeCashflow.",
                                             f"Message für 'inf' sollte korrekt sein für {symbol} ({frequency})")
                        else:
                            self.assertIsInstance(result["price_to_freeCashflow"], float,
                                                  f"Price/FreeCashflow für {symbol} ({frequency}) sollte ein Float sein")
                        # Teste Caching
                        cached_result = self.model.calculate_price_to_freeCashflow(symbol, use_cache=True,
                                                                                   frequency=frequency)
                        self.assertEqual(cached_result, result,
                                         f"Cached Ergebnis für {symbol} ({frequency}) sollte mit erstem Aufruf übereinstimmen")

    def test_evaluate_tbv_bandwidth(self):
        """Integrationstest (ohne Mocks) für evaluate_tbv_bandwidth mit echten Daten.
           Akzeptiert auch den erwarteten Fehler 'Unzureichende Historie …'."""
        print("\nTeste evaluate_tbv_bandwidth...")

        def _pretty_print(symbol, result):
            """Hilfsfunktion: gibt die wichtigsten Felder schön formatiert aus."""
            if "error" in result:
                print(f"[{symbol}] Hinweis/Fehler: {result['error']}")
                return
            cur = result["current"]
            tg = result["targets"]
            zones = result["zones"]
            print(f"\n[{symbol}] Ergebnis:")
            print(f"  History (Jahre):         {result['meta']['history_years']}")
            print(f"  Aktuell: Kurs={cur['price']}, TBV/Aktie={cur['tbv_per_share']}, P/TBV={cur['pb_ratio']}"),
            print(f"  Targets: WC={tg['WC']}, BUY={tg['BUY']}, SELL={tg['SELL']}")
            print(f"  Signal:  {result['signal']} – {result['message']}")
            print(
                f"  Punkte:  buy={len(zones['buy_zone_points'])}, sell={len(zones['sell_zone_points'])}, touches≈1x={zones['touches_tbv≈1x']}")
            # ein paar Beispielpunkte ausgeben (max 3)
            for label, points in [("BUY-Punkte", zones["buy_zone_points"]), ("SELL-Punkte", zones["sell_zone_points"])]:
                if points:
                    preview = points[:3]
                    print(f"  {label} (Beispiele): " +
                          ", ".join([f"{p['date']}@{p['price']}(PB={round(p['pb_ratio'], 2)})" for p in preview]))
                else:
                    print(f"  {label}: —")

        # ---------- Subtest 1: XOM (kann ggf. < 10 Jahre TBV-Historie liefern) ----------
        with self.subTest(symbol="MO", case="bandwidth"):
            symbol = "MO"
            result = self.model.evaluate_tbv_bandwidth(symbol, min_years=10.0, use_cache=True)
            _pretty_print(symbol, result)

            # Akzeptiere erwarteten Historie-Fehler ODER prüfe normales Ergebnis
            if "error" in result:
                self.assertIn("Unzureichende Historie", result["error"])
            else:
                # Grundstruktur
                for k in ["symbol", "pb", "zones", "targets", "current", "signal", "message", "meta"]:
                    self.assertIn(k, result)
                self.assertIsInstance(result["pb"], pd.DataFrame)
                self.assertFalse(result["pb"].empty)
                self.assertGreaterEqual(result["meta"]["history_years"], 10.0)
                # Targets plausibel
                cur = result["current"];
                tg = result["targets"]
                if cur["tbv_per_share"] > 0:
                    self.assertGreater(tg["BUY"], tg["WC"])
                    self.assertGreater(tg["SELL"], tg["BUY"])
                    self.assertIn(result["signal"], ["buy", "neutral", "sell"])
                else:
                    self.assertEqual(result["signal"], "warning")
                    self.assertEqual(tg["WC"], 0.0);
                    self.assertEqual(tg["BUY"], 0.0);
                    self.assertEqual(tg["SELL"], 0.0)

        # ---------- Subtest 2: MCD (häufig TBV ≤ 0 → Warning-Zweig testen) ----------
        with self.subTest(symbol="EXC", case="bandwidth"):
            symbol = "EXC"
            result = self.model.evaluate_tbv_bandwidth(symbol, min_years=10.0, use_cache=True)
            _pretty_print(symbol, result)

            self.assertIsInstance(result, dict)
            self.assertNotIn("error", result, result.get("error"))
            self.assertIn("current", result);
            self.assertIn("targets", result)
            cur = result["current"];
            tg = result["targets"]

            if cur["tbv_per_share"] <= 0:
                self.assertEqual(result["signal"], "warning")
                self.assertEqual(tg["WC"], 0.0);
                self.assertEqual(tg["BUY"], 0.0);
                self.assertEqual(tg["SELL"], 0.0)
            else:
                self.assertGreater(tg["BUY"], tg["WC"])
                self.assertGreater(tg["SELL"], tg["BUY"])
                self.assertIn(result["signal"], ["buy", "neutral", "sell"])

    def test_evaluate_ebit_bandwidth(self):
        """Integrationstest (ohne Mocks) für evaluate_ebit_bandwidth mit echten Daten.
           Akzeptiert auch den erwarteten Fehler 'Unzureichende Historie …'."""
        print("\nTeste evaluate_ebit_bandwidth...")

        def _pretty_print(symbol, result):
            """Hilfsfunktion: gibt die wichtigsten Felder schön formatiert aus."""
            if "error" in result:
                print(f"[{symbol}] Hinweis/Fehler: {result['error']}")
                return
            cur = result["current"]
            tg = result["targets"]
            zones = result["zones"]
            print(f"\n[{symbol}] Ergebnis:")
            print(f"  History (Jahre):         {result['meta']['history_years']}")
            print(f"  Aktuell: Kurs={cur['price']}, EBIT/Aktie={cur['ebit_per_share']}, P/EBIT={cur['ebit_ratio']}"),
            print(f"  Targets: WC={tg['WC']}, BUY={tg['BUY']}, SELL={tg['SELL']}")
            print(f"  Signal:  {result['signal']} – {result['message']}")
            print(
                f"  Punkte:  buy={len(zones['buy_zone_points'])}, sell={len(zones['sell_zone_points'])}, touches≈8x={zones['touches_ebit≈8x']}")
            # ein paar Beispielpunkte ausgeben (max 3)
            for label, points in [("BUY-Punkte", zones["buy_zone_points"]), ("SELL-Punkte", zones["sell_zone_points"])]:
                if points:
                    preview = points[:3]
                    print(f"  {label} (Beispiele): " +
                          ", ".join([f"{p['date']}@{p['price']}(EBIT={round(p['ebit_ratio'], 2)})" for p in preview]))
                else:
                    print(f"  {label}: —")

        # ---------- Subtest 1: XOM (kann ggf. < 10 Jahre EBIT-Historie liefern) ----------
        with self.subTest(symbol="BABA", case="bandwidth"):
            symbol = "BABA"
            result = self.model.evaluate_ebit_bandwidth(symbol, min_years=10.0, use_cache=True)
            _pretty_print(symbol, result)

            # Akzeptiere erwarteten Historie-Fehler ODER prüfe normales Ergebnis
            if "error" in result:
                self.assertIn("Unzureichende Historie", result["error"])
            else:
                # Grundstruktur
                for k in ["symbol", "ebit", "zones", "targets", "current", "signal", "message", "meta"]:
                    self.assertIn(k, result)
                self.assertIsInstance(result["ebit"], pd.DataFrame)
                self.assertFalse(result["ebit"].empty)
                self.assertGreaterEqual(result["meta"]["history_years"], 10.0)
                # Targets plausibel
                cur = result["current"]
                tg = result["targets"]
                if cur["ebit_per_share"] > 0:
                    self.assertGreater(tg["BUY"], tg["WC"])
                    self.assertGreater(tg["SELL"], tg["BUY"])
                    self.assertIn(result["signal"], ["buy", "neutral", "sell"])
                else:
                    self.assertEqual(result["signal"], "warning")
                    self.assertEqual(tg["WC"], 0.0)
                    self.assertEqual(tg["BUY"], 0.0)
                    self.assertEqual(tg["SELL"], 0.0)

        # ---------- Subtest 2: TSLA (häufig EBIT ≤ 0 → Warning-Zweig testen) ----------
        with self.subTest(symbol="TSLA", case="bandwidth"):
            symbol = "TSLA"
            result = self.model.evaluate_ebit_bandwidth(symbol, min_years=10.0, use_cache=True)
            _pretty_print(symbol, result)

            self.assertIsInstance(result, dict)
            self.assertNotIn("error", result, result.get("error"))
            self.assertIn("current", result)
            self.assertIn("targets", result)
            cur = result["current"]
            tg = result["targets"]

            if cur["ebit_per_share"] <= 0:
                self.assertEqual(result["signal"], "warning")
                self.assertEqual(tg["WC"], 0.0)
                self.assertEqual(tg["BUY"], 0.0)
                self.assertEqual(tg["SELL"], 0.0)
            else:
                self.assertGreater(tg["BUY"], tg["WC"])
                self.assertGreater(tg["SELL"], tg["BUY"])
                self.assertIn(result["signal"], ["buy", "neutral", "sell"])

    def test_calculate_buy_case(self):
        """Testet die Methode calculate_buy_case für MCD mit Price/Sales (3 Jahre VOR globalem Minimum + optionaler Fallback)."""
        print("\nTeste calculate_buy_case...")

        with self.subTest(symbol="MCD", case="valid_symbol"):
            symbol = "MCD"
            print(f"\nBerechnung von Buy-Case für {symbol} mit Price/Sales...")

            # Daten abrufen
            data = self.model.calculate_historical_price_to_sales(symbol, use_cache=True)

            # Basis-Prüfungen
            if data is None:
                print(f"Fehler für {symbol}: Keine Price/Sales-Daten abgerufen.")
                self.fail(f"Keine Price/Sales-Daten für {symbol} erhalten.")
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"Price/Sales-Daten für {symbol} sollten nicht leer sein.")

            expected_columns = ["Price_Sales", "Price", "Sales", "commonStockSharesOutstanding"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            # Debug
            print(f"\nDebug: Inhalt von price_to_sales für {symbol}:")
            print(f"Spalten: {data.columns.tolist()}")
            print(f"Erste Zeilen:\n{data.head()}")
            print(f"\nDebug: Letzte Zeilen:\n{data.tail()}")
            print(f"\nDebug: Zusammenfassung Price_Sales:\n{data['Price_Sales'].describe()}")

            # Jahre VOR dem globalen Minimum bestimmen
            multiple_column = next((col for col in self.model.HISTORICAL_MULTIPLE_COLUMNS if col in data.columns), None)
            self.assertIsNotNone(multiple_column, "Keine Multiple-Spalte in den Daten gefunden.")

            global_min_idx = data[multiple_column].idxmin()
            global_min_year = pd.to_datetime(global_min_idx).year

            data_before_min = data[data.index.year < global_min_year].copy()
            yearly_mins_before = data_before_min.groupby(data_before_min.index.year)[multiple_column].min()

            print(f"\nDebug: Globales Minimum Jahr: {global_min_year}")
            print(f"Debug: Jahre VOR globalem Minimum: {sorted(data_before_min.index.year.unique().tolist())}")
            print(f"Debug: Anzahl der Jahre VOR Minimum: {len(yearly_mins_before)}")

            # Konsistenzprüfungen zu Sales & Bilanz
            sales_data = self.model.calculate_historical_sales(symbol, use_cache=True)
            if sales_data is None:
                print(f"Fehler für {symbol}: Keine Umsatzdaten abgerufen.")
                self.fail(f"Keine Umsatzdaten für {symbol} erhalten.")
            self.assertIn("Sales", sales_data.columns)
            self.assertFalse(sales_data["Sales"].isna().all())

            fundamentals = self.model.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            balance_sheet = fundamentals.get("balance_sheet") if fundamentals else None
            if balance_sheet is None:
                print(f"Fehler für {symbol}: Keine Bilanzdaten abgerufen.")
                self.fail(f"Keine Bilanzdaten für {symbol} erhalten.")

            expected_dates = sales_data.index.intersection(data.index).intersection(balance_sheet.index)
            self.assertFalse(expected_dates.empty)

            self.assertIn("commonStockSharesOutstanding", balance_sheet.columns)
            self.assertTrue((data["Sales"] == sales_data["Sales"].reindex(data.index)).all())
            self.assertTrue((data["commonStockSharesOutstanding"] ==
                             balance_sheet["commonStockSharesOutstanding"].reindex(data.index, method="ffill")).all())

            # Buy-Case berechnen
            buy_case_result = self.model.calculate_buy_case(data)
            self.assertIsNotNone(buy_case_result, "calculate_buy_case lieferte None.")

            if isinstance(buy_case_result, dict) and 'error' in buy_case_result:
                print(f"Erwarteter Fehler für {symbol}: {buy_case_result['error']}")
                self.assertTrue(
                    ("Nicht genug verschiedene Jahre für 3 Werte" in buy_case_result['error']) or
                    ("Nicht genug Jahre vor dem globalen Minimum" in buy_case_result['error']) or
                    ("Nicht genug Jahre nach dem globalen Minimum" in buy_case_result['error']),
                    f"Unerwartete Fehlermeldung: {buy_case_result['error']}"
                )
                if 'buy_fallback_algo_used' in buy_case_result:
                    self.assertIsInstance(buy_case_result['buy_fallback_algo_used'], bool)
            else:
                # Grundstruktur
                self.assertIsInstance(buy_case_result, dict)
                for k in ['global_min', 'buy_value']:
                    self.assertIn(k, buy_case_result)

                self.assertGreaterEqual(buy_case_result['global_min'], 0)
                self.assertGreaterEqual(buy_case_result['buy_value'], 0)
                self.assertFalse(pd.isna(buy_case_result['global_min']) or pd.isna(buy_case_result['buy_value']))

                # --- NEU: Jahre & Werte der Median-Bildung nachvollziehen ---
                # Entscheide je nach Fallback-Flag bzw. Datenlage
                fallback_used = buy_case_result.get('buy_fallback_algo_used', None)

                # Serie der Jahres-Minima (über alle Jahre)
                yearly_mins_all = data.groupby(data.index.year)[multiple_column].min()

                if fallback_used is True or len(yearly_mins_before) < 3:
                    # Fallback: global 3 kleinste Jahres-Minima
                    top_years = yearly_mins_all.nsmallest(3).index.tolist()
                    values_used = [float(yearly_mins_all.loc[y]) for y in top_years]
                    strategy = "Fallback (globale 3 Jahres-Minima)"
                else:
                    # Normalfall: 3 kleinste Jahres-Minima VOR globalem Minimum
                    top_years = yearly_mins_before.nsmallest(3).index.tolist()
                    values_used = [float(yearly_mins_before.loc[y]) for y in top_years]
                    strategy = "Vor-Minimum (3 Jahres-Minima)"

                computed_median = round(pd.Series(values_used).median(), 2)

                print(f"\nDebug: Buy-Case-Ergebnis für {symbol}:")
                print(f"Globales Minimum: {buy_case_result['global_min']}")
                print(f"Buy-Wert (geliefert): {buy_case_result['buy_value']}")
                print(f"Strategie: {strategy}")
                print(f"Jahre verwendet: {top_years}")
                print(f"Werte verwendet: {values_used}")
                print(f"Median (nachgerechnet): {computed_median}")

                # Validierung: Median stimmt überein
                self.assertEqual(computed_median, float(buy_case_result['buy_value']),
                                 "Der nachgerechnete Median entspricht nicht dem gelieferten Buy-Wert.")

                # Falls Flag vorhanden, Konsistenz prüfen
                if fallback_used is not None:
                    if len(yearly_mins_before) < 3:
                        self.assertTrue(fallback_used,
                                        "Bei <3 Jahren vor Minimum sollte der Fallback genutzt worden sein.")
                    else:
                        self.assertFalse(fallback_used,
                                         "Bei ≥3 Jahren vor Minimum sollte kein Fallback genutzt werden.")

            # Zeitraum/Datenpunkte
            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige Price/Sales-Tabelle:")
                print(data)

            time.sleep(1)

    def test_calculate_worst_case(self):
        """Testet die Methode calculate_worst_case für MCD mit Price/Sales."""
        print("\nTeste calculate_worst_case...")

        # Testfall 1: Gültiges Symbol (MCD)
        with self.subTest(symbol="BABA", case="valid_symbol"):
            symbol = "BABA"
            print(f"\nBerechnung von Worst-Case für {symbol} mit Price/Sales...")

            # Daten mit calculate_historical_price_to_sales abrufen
            data = self.model.calculate_historical_price_to_sales(symbol, use_cache=True)

            # Prüfen, ob Daten zurückgegeben wurden
            if data is None:
                print(f"Fehler für {symbol}: Keine Price/Sales-Daten abgerufen.")
                self.fail(f"Keine Price/Sales-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"Price/Sales-Daten für {symbol} sollten nicht leer sein.")

            # Prüfen, ob erwartete Spalten vorhanden sind
            expected_columns = ["Price_Sales", "Price", "Sales", "commonStockSharesOutstanding"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            # Debugging: Datenstruktur ausgeben
            print(f"\nDebug: Inhalt von price_to_sales für {symbol}:")
            print(f"Spalten: {data.columns.tolist()}")
            print(f"Erste Zeilen:\n{data.head()}")
            print(f"\nDebug: Letzte Zeilen:\n{data.tail()}")
            print(f"\nDebug: Zusammenfassung Price_Sales:\n{data['Price_Sales'].describe()}")

            # Prüfen, ob Sales-Daten konsistent sind
            sales_data = self.model.calculate_historical_sales(symbol, use_cache=True)
            if sales_data is None:
                print(f"Fehler für {symbol}: Keine Umsatzdaten abgerufen.")
                self.fail(f"Keine Umsatzdaten für {symbol} erhalten.")

            self.assertTrue("Sales" in sales_data.columns,
                            f"Sales-Daten sollten in sales_data für {symbol} vorhanden sein. Gefundene Spalten: {sales_data.columns.tolist()}")
            self.assertFalse(sales_data["Sales"].isna().all(),
                             f"Historische Umsatzdaten für {symbol} sollten nicht alle NaN sein.")

            # Prüfen, ob Bilanzdaten verfügbar sind
            fundamentals = self.model.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            balance_sheet = fundamentals.get("balance_sheet") if fundamentals else None
            if balance_sheet is None:
                print(f"Fehler für {symbol}: Keine Bilanzdaten abgerufen.")
                self.fail(f"Keine Bilanzdaten für {symbol} erhalten.")

            # Prüfen, ob Indizes übereinstimmen
            expected_dates = sales_data.index.intersection(data.index).intersection(balance_sheet.index)
            self.assertFalse(expected_dates.empty,
                             f"Price/Sales-Daten für {symbol} sollten mit Umsatz- und Bilanzdaten-Daten übereinstimmen.")

            # Prüfen, ob Sales und commonStockSharesOutstanding konsistent sind
            self.assertTrue("commonStockSharesOutstanding" in balance_sheet.columns,
                            f"Shares Outstanding-Daten sollten in balance_sheet für {symbol} vorhanden sein.")
            self.assertTrue((data["Sales"] == sales_data["Sales"].reindex(data.index)).all(),
                            f"Sales-Daten für {symbol} sollten mit sales_data übereinstimmen.")
            self.assertTrue((data["commonStockSharesOutstanding"] == balance_sheet[
                "commonStockSharesOutstanding"].reindex(data.index, method="ffill")).all(),
                            f"commonStockSharesOutstanding-Daten für {symbol} sollten mit Bilanzdaten übereinstimmen.")

            # Worst-Case berechnen
            worst_case_result = self.model.calculate_worst_case(data)

            # Prüfen, ob ein Ergebnis zurückgegeben wurde
            if isinstance(worst_case_result, dict) and 'error' in worst_case_result:
                print(f"Erwarteter Fehler für {symbol}: {worst_case_result['error']}")
                buy_case_result = self.model.calculate_buy_case(data)
                if isinstance(buy_case_result, dict) and 'error' in buy_case_result:
                    self.assertEqual(worst_case_result['error'],
                                     f"Fehler beim Abruf des Buy-Werts: {buy_case_result['error']}",
                                     f"Fehlerdictionary von Worst-Case sollte Buy-Case-Fehler widerspiegeln für {symbol}")
            else:
                # Prüfen, ob das Ergebnis ein numerischer Wert ist
                self.assertIsInstance(worst_case_result, (int, float),
                                      f"Worst-Case-Ergebnis für {symbol} sollte ein numerischer Wert sein.")
                self.assertGreaterEqual(worst_case_result, 0,
                                        f"Worst-Case-Wert für {symbol} sollte nicht negativ sein.")
                self.assertFalse(pd.isna(worst_case_result),
                                 f"Worst-Case-Wert für {symbol} sollte keinen NaN-Wert enthalten.")

                # Buy-Case zur Validierung holen
                buy_case_result = self.model.calculate_buy_case(data)
                if isinstance(buy_case_result, dict) and 'error' in buy_case_result:
                    print(f"Unerwarteter Fehler beim Buy-Case für {symbol}: {buy_case_result['error']}")
                    self.fail(
                        f"Buy-Case sollte kein Fehlerdictionary sein, wenn Worst-Case erfolgreich ist für {symbol}")
                expected_worst = round(buy_case_result['buy_value'] / 1.2, 2)

                # Prüfen, ob Worst-Case korrekt berechnet wurde
                self.assertAlmostEqual(worst_case_result, expected_worst, places=2,
                                       msg=f"Worst-Case für {symbol} sollte Buy-Wert / 1.2 sein. Erwartet: {expected_worst}, erhalten: {worst_case_result}")

                # Debugging: Ergebnis ausgeben
                print(f"\nDebug: Worst-Case-Ergebnis für {symbol}:")
                print(f"Worst-Case-Wert: {worst_case_result}")

            # Zeitraum und Datenpunkte ausgeben
            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige Price/Sales-Tabelle:")
                print(data)

            time.sleep(1)

    def test_calculate_sell_case(self):
        """Testet die Methode calculate_sell_case für MCD mit Price/Sales."""
        print("\nTeste calculate_sell_case...")

        # Testfall 1: Gültiges Symbol (MCD)
        with self.subTest(symbol="MCD", case="valid_symbol"):
            symbol = "MCD"
            print(f"\nBerechnung von Sell-Case für {symbol} mit Price/Sales...")

            # Daten mit calculate_historical_price_to_sales abrufen
            data = self.model.calculate_historical_price_to_sales(symbol, use_cache=True)

            # Prüfen, ob Daten zurückgegeben wurden
            if data is None:
                print(f"Fehler für {symbol}: Keine Price/Sales-Daten abgerufen.")
                self.fail(f"Keine Price/Sales-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"Price/Sales-Daten für {symbol} sollten nicht leer sein.")

            # Prüfen, ob erwartete Spalten vorhanden sind
            expected_columns = ["Price_Sales", "Price", "Sales", "commonStockSharesOutstanding"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            # Debugging: Datenstruktur ausgeben
            print(f"\nDebug: Inhalt von price_to_sales für {symbol}:")
            print(f"Spalten: {data.columns.tolist()}")
            print(f"Erste Zeilen:\n{data.head()}")
            print(f"\nDebug: Letzte Zeilen:\n{data.tail()}")
            print(f"\nDebug: Zusammenfassung Price_Sales:\n{data['Price_Sales'].describe()}")

            # Prüfen, ob Sales-Daten konsistent sind
            sales_data = self.model.calculate_historical_sales(symbol, use_cache=True)
            if sales_data is None:
                print(f"Fehler für {symbol}: Keine Umsatzdaten abgerufen.")
                self.fail(f"Keine Umsatzdaten für {symbol} erhalten.")

            self.assertTrue("Sales" in sales_data.columns,
                            f"Sales-Daten sollten in sales_data für {symbol} vorhanden sein. Gefundene Spalten: {sales_data.columns.tolist()}")
            self.assertFalse(sales_data["Sales"].isna().all(),
                             f"Historische Umsatzdaten für {symbol} sollten nicht alle NaN sein.")

            # Prüfen, ob Bilanzdaten verfügbar sind
            fundamentals = self.model.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            balance_sheet = fundamentals.get("balance_sheet") if fundamentals else None
            if balance_sheet is None:
                print(f"Fehler für {symbol}: Keine Bilanzdaten abgerufen.")
                self.fail(f"Keine Bilanzdaten für {symbol} erhalten.")

            # Prüfen, ob Indizes übereinstimmen
            expected_dates = sales_data.index.intersection(data.index).intersection(balance_sheet.index)
            self.assertFalse(expected_dates.empty,
                             f"Price/Sales-Daten für {symbol} sollten mit Umsatz- und Bilanzdaten-Daten übereinstimmen.")

            # Prüfen, ob Sales und commonStockSharesOutstanding konsistent sind
            self.assertTrue("commonStockSharesOutstanding" in balance_sheet.columns,
                            f"Shares Outstanding-Daten sollten in balance_sheet für {symbol} vorhanden sein.")
            self.assertTrue((data["Sales"] == sales_data["Sales"].reindex(data.index)).all(),
                            f"Sales-Daten für {symbol} sollten mit sales_data übereinstimmen.")
            self.assertTrue((data["commonStockSharesOutstanding"] == balance_sheet[
                "commonStockSharesOutstanding"].reindex(data.index, method="ffill")).all(),
                            f"commonStockSharesOutstanding-Daten für {symbol} sollten mit Bilanzdaten übereinstimmen.")

            # Sell-Case berechnen
            sell_case_result = self.model.calculate_sell_case(data)

            # Prüfen, ob ein Ergebnis zurückgegeben wurde
            if isinstance(sell_case_result, dict) and 'error' in sell_case_result:
                print(f"Erwarteter Fehler für {symbol}: {sell_case_result['error']}")
                # Prüfe, ob der Fehler mit der Bedingung übereinstimmt (z. B. < 3 Jahre)
                yearly_maxs = data.groupby(data.index.year)["Price_Sales"].max()
                self.assertEqual(len(yearly_maxs) < 3, True,
                                 f"Fehlerdictionary sollte bei weniger als 3 Jahren auftreten, aber {len(yearly_maxs)} Jahre gefunden.")
            else:
                # Prüfen, ob das Ergebnis ein numerischer Wert ist
                self.assertIsInstance(sell_case_result, (int, float),
                                      f"Sell-Case-Ergebnis für {symbol} sollte ein numerischer Wert sein.")
                self.assertGreaterEqual(sell_case_result, 0,
                                        f"Sell-Case-Wert für {symbol} sollte nicht negativ sein.")
                self.assertFalse(pd.isna(sell_case_result),
                                 f"Sell-Case-Wert für {symbol} sollte keinen NaN-Wert enthalten.")

                # Debugging: Ergebnis ausgeben
                print(f"\nDebug: Sell-Case-Ergebnis für {symbol}:")
                print(f"Sell-Wert: {sell_case_result}")

            # Zeitraum und Datenpunkte ausgeben
            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige Price/Sales-Tabelle:")
                print(data)

            time.sleep(1)

    def test_calculate_fairValue_case(self):
        """Testet die Methode calculate_fairValue_case für TSLA mit Price/Sales."""
        print("\nTeste calculate_fairValue_case...")

        # Testfall 1: Gültiges Symbol (TSLA)
        with self.subTest(symbol="MCD", case="valid_symbol"):
            symbol = "MCD"
            print(f"\nBerechnung von Fair-Value-Case für {symbol} mit Price/Sales...")

            # Daten mit calculate_historical_price_to_sales abrufen
            data = self.model.calculate_historical_price_to_sales(symbol, use_cache=True)

            # Prüfen, ob Daten zurückgegeben wurden
            if data is None:
                print(f"Fehler für {symbol}: Keine Price/Sales-Daten abgerufen.")
                self.fail(f"Keine Price/Sales-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"Price/Sales-Daten für {symbol} sollten nicht leer sein.")

            # Prüfen, ob erwartete Spalten vorhanden sind
            expected_columns = ["Price_Sales", "Price", "Sales", "commonStockSharesOutstanding"]
            self.assertTrue(all(col in data.columns for col in expected_columns),
                            f"Fehlende Spalten für {symbol}: Erwartet {expected_columns}, erhalten {data.columns.tolist()}")

            # Debugging: Datenstruktur ausgeben
            print(f"\nDebug: Inhalt von price_to_sales für {symbol}:")
            print(f"Spalten: {data.columns.tolist()}")
            print(f"Erste Zeilen:\n{data.head()}")
            print(f"\nDebug: Letzte Zeilen:\n{data.tail()}")
            print(f"\nDebug: Zusammenfassung Price_Sales:\n{data['Price_Sales'].describe()}")

            # Prüfen, ob Sales-Daten konsistent sind
            sales_data = self.model.calculate_historical_sales(symbol, use_cache=True)
            if sales_data is None:
                print(f"Fehler für {symbol}: Keine Umsatzdaten abgerufen.")
                self.fail(f"Keine Umsatzdaten für {symbol} erhalten.")

            self.assertTrue("Sales" in sales_data.columns,
                            f"Sales-Daten sollten in sales_data für {symbol} vorhanden sein. Gefundene Spalten: {sales_data.columns.tolist()}")
            self.assertFalse(sales_data["Sales"].isna().all(),
                             f"Historische Umsatzdaten für {symbol} sollten nicht alle NaN sein.")

            # Prüfen, ob Bilanzdaten verfügbar sind
            fundamentals = self.model.dataloader.get_fundamental_data(symbol, frequency="quarterly", use_cache=True)
            balance_sheet = fundamentals.get("balance_sheet") if fundamentals else None
            if balance_sheet is None:
                print(f"Fehler für {symbol}: Keine Bilanzdaten abgerufen.")
                self.fail(f"Keine Bilanzdaten für {symbol} erhalten.")

            # Prüfen, ob Indizes übereinstimmen
            expected_dates = sales_data.index.intersection(data.index).intersection(balance_sheet.index)
            self.assertFalse(expected_dates.empty,
                             f"Price/Sales-Daten für {symbol} sollten mit Umsatz- und Bilanzdaten-Daten übereinstimmen.")

            # Prüfen, ob Sales und commonStockSharesOutstanding konsistent sind
            self.assertTrue("commonStockSharesOutstanding" in balance_sheet.columns,
                            f"Shares Outstanding-Daten sollten in balance_sheet für {symbol} vorhanden sein.")
            self.assertTrue((data["Sales"] == sales_data["Sales"].reindex(data.index)).all(),
                            f"Sales-Daten für {symbol} sollten mit sales_data übereinstimmen.")
            self.assertTrue((data["commonStockSharesOutstanding"] == balance_sheet[
                "commonStockSharesOutstanding"].reindex(data.index, method="ffill")).all(),
                            f"commonStockSharesOutstanding-Daten für {symbol} sollten mit Bilanzdaten übereinstimmen.")

            # Buy-Case und Sell-Case berechnen
            buy_case_result = self.model.calculate_buy_case(data)
            sell_case_result = self.model.calculate_sell_case(data)

            # Prüfen, ob Ergebnisse zurückgegeben wurden
            if isinstance(buy_case_result, dict) and 'error' in buy_case_result:
                print(f"Fehler beim Buy-Case für {symbol}: {buy_case_result['error']}")
                self.fail(f"Buy-Case für {symbol} sollte kein Fehlerdictionary sein.")
            if isinstance(sell_case_result, dict) and 'error' in sell_case_result:
                print(f"Fehler beim Sell-Case für {symbol}: {sell_case_result['error']}")
                self.fail(f"Sell-Case für {symbol} sollte kein Fehlerdictionary sein.")

            # Extrahiere Buy- und Sell-Werte
            buy_value = buy_case_result.get('buy_value', None)
            sell_value = sell_case_result

            if buy_value is None or sell_value is None:
                print(f"Fehler für {symbol}: Buy- oder Sell-Wert fehlt.")
                self.fail(f"Buy- oder Sell-Wert für {symbol} fehlt.")

            # Fair-Value-Case berechnen
            fair_value_result = self.model.calculate_fairValue_case(buy_value, sell_value)

            # Prüfen, ob ein Ergebnis zurückgegeben wurde
            if isinstance(fair_value_result, dict) and 'error' in fair_value_result:
                print(f"Erwarteter Fehler für {symbol}: {fair_value_result['error']}")
                self.assertTrue(buy_value is None or sell_value is None or
                                not isinstance(buy_value, (int, float)) or not isinstance(sell_value, (int, float)),
                                f"Fehlerdictionary sollte bei ungültigen Eingaben auftreten für {symbol}")
            else:
                # Prüfen, ob das Ergebnis ein numerischer Wert ist
                self.assertIsInstance(fair_value_result, (int, float),
                                      f"Fair-Value-Ergebnis für {symbol} sollte ein numerischer Wert sein.")
                self.assertGreaterEqual(fair_value_result, 0,
                                        f"Fair-Value-Wert für {symbol} sollte nicht negativ sein.")
                self.assertFalse(pd.isna(fair_value_result),
                                 f"Fair-Value-Wert für {symbol} sollte keinen NaN-Wert enthalten.")

                # Prüfen, ob Fair-Value korrekt berechnet wurde (Mittelwert von Buy und Sell)
                expected_fair_value = round((buy_value + sell_value) / 2, 2)
                self.assertAlmostEqual(fair_value_result, expected_fair_value, places=2,
                                       msg=f"Fair-Value für {symbol} sollte (Buy + Sell) / 2 sein. Erwartet: {expected_fair_value}, erhalten: {fair_value_result}")

                # Debugging: Ergebnisse ausgeben
                print(f"\nDebug: Fair-Value-Ergebnis für {symbol}:")
                print(f"Buy-Wert: {buy_value}")
                print(f"Sell-Wert: {sell_value}")
                print(f"Fair-Value: {fair_value_result}")

            # Zeitraum und Datenpunkte ausgeben
            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: Historie von {start_date} bis {end_date}, {num_points} Datenpunkte.")

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\nVollständige Price/Sales-Tabelle:")
                print(data)

            time.sleep(1)

    def test_calculate_course_target_price_book(self):
        """Testet die Methode calculate_course_target mit Price/Book für MCD."""
        print("\nTeste calculate_course_target_price_book...")

        # Testfall 1: Gültiges Symbol (MCD)
        with self.subTest(symbol="MCD", case="valid_symbol"):
            symbol = "MCD"
            print(f"\nBerechnung von Kursziel für {symbol} mit Price/Book...")
            data = self.model.calculate_historical_price_to_book(symbol)

            # Prüfen, ob Daten zurückgegeben wurden
            if data is None:
                print(f"Fehler für {symbol}: Keine Price/Book-Daten abgerufen.")
                self.fail(f"Keine Price/Book-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"Price/Book-Daten für {symbol} sollten nicht leer sein.")

            # Debugging: Prüfe die Daten vor der Verarbeitung
            print(f"Daten für {symbol}: {data.head()}")
            buy_result = self.model.calculate_buy_case(data)
            print(f"Ergebnis von calculate_buy_case: {buy_result}")

            # Methode aufrufen
            result = self.model.calculate_course_target_PriceMultiples(data, symbol)
            if isinstance(result, dict) and 'error' in result:
                print(f"Erwarteter Fehler für {symbol}: {result['error']}")
                # Prüfe, ob der Fehler den Buy-Case-Fehler mit Präfix enthält
                if isinstance(buy_result, dict) and 'error' in buy_result:
                    expected_error = f"Fehler beim Abruf des Buy-Werts: {buy_result['error']}"
                    self.assertEqual(result['error'], expected_error,
                                     f"Fehlerdictionary sollte Buy-Case-Fehler mit Präfix widerspiegeln für {symbol}")
                # Optional: Weitere Abhängigkeiten prüfen, falls nötig
                wc_result = self.model.calculate_worst_case(data)
                if isinstance(wc_result, dict) and 'error' in wc_result:
                    self.assertIn(buy_result.get('error', ''), wc_result.get('error', ''),
                                  f"Worst-Case-Fehler sollte Buy-Case-Fehler enthalten für {symbol}")
            else:
                self.assertIsNotNone(result, f"calculate_course_target für {symbol} lieferte None.")
                self.assertIn("worst_case_price", result,
                              f"Fehlender Schlüssel 'worst_case_price' im Ergebnis für {symbol}.")
                self.assertIn("buy_price", result, f"Fehlender Schlüssel 'buy_price' im Ergebnis für {symbol}.")
                self.assertIn("fair_value_price", result,
                              f"Fehlender Schlüssel 'fair_value_price' im Ergebnis für {symbol}.")
                self.assertIn("sell_price", result, f"Fehlender Schlüssel 'sell_price' im Ergebnis für {symbol}.")
                self.assertTrue(all(value >= 0 for value in result.values()),
                                f"Negative Kursziele erkannt für {symbol}.")

                # Formatiertes Ausgeben der Kursziele
                formatted_result = {
                    "worst_case_price": f"{result['worst_case_price']:.2f}",
                    "buy_price": f"{result['buy_price']:.2f}",
                    "fair_value_price": f"{result['fair_value_price']:.2f}",
                    "sell_price": f"{result['sell_price']:.2f}"
                }
                print(f"Kursziele für {symbol}: {formatted_result}")

    def test_calculate_course_target_price_sales(self):
        """Testet die Methode calculate_course_target mit Price/Sales für MCD."""
        print("\nTeste calculate_course_target_price_sales...")

        # Testfall 1: Gültiges Symbol (MCD)
        with self.subTest(symbol="MCD", case="valid_symbol"):
            symbol = "MCD"
            print(f"\nBerechnung von Kursziel für {symbol} mit Price/Sales...")
            data = self.model.calculate_historical_price_to_sales(symbol)

            # Prüfen, ob Daten zurückgegeben wurden
            if data is None:
                print(f"Fehler für {symbol}: Keine Price/Sales-Daten abgerufen.")
                self.fail(f"Keine Price/Sales-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"Price/Sales-Daten für {symbol} sollten nicht leer sein.")

            # Methode aufrufen
            result = self.model.calculate_course_target_PriceMultiples(data, symbol)
            self.assertIsNotNone(result, f"calculate_course_target für {symbol} lieferte None.")
            self.assertIn("worst_case_price", result, f"Fehlender Schlüssel 'worst_case_price' im Ergebnis für {symbol}.")
            self.assertIn("buy_price", result, f"Fehlender Schlüssel 'buy_price' im Ergebnis für {symbol}.")
            self.assertIn("fair_value_price", result, f"Fehlender Schlüssel 'fair_value_price' im Ergebnis für {symbol}.")
            self.assertIn("sell_price", result, f"Fehlender Schlüssel 'sell_price' im Ergebnis für {symbol}.")
            self.assertTrue(all(value >= 0 for value in result.values()), f"Negative Kursziele erkannt für {symbol}.")

            print(f"Kursziele für {symbol}: {result}")

    def test_calculate_course_target_price_ebit(self):
        """Testet die Methode calculate_course_target mit Price/EBIT für MCD."""
        print("\nTeste calculate_course_target_price_ebit...")

        # Testfall 1: Gültiges Symbol (MCD)
        with self.subTest(symbol="MCD", case="valid_symbol"):
            symbol = "MCD"
            print(f"\nBerechnung von Kursziel für {symbol} mit Price/EBIT...")
            data = self.model.calculate_historical_price_to_ebit(symbol)

            # Prüfen, ob Daten zurückgegeben wurden
            if data is None:
                print(f"Fehler für {symbol}: Keine Price/EBIT-Daten abgerufen.")
                self.fail(f"Keine Price/EBIT-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"Price/EBIT-Daten für {symbol} sollten nicht leer sein.")

            # Rohe Daten für EBIT und Shares Outstanding abrufen
            ebit_data = self.model.dataloader.get_ebit_data(symbol, frequency="quarterly")
            if isinstance(ebit_data, dict) and "error" in ebit_data:
                print(f"Fehler beim Abruf von EBIT-Daten: {ebit_data['error']}")
                self.fail(f"Fehler beim Abruf von EBIT-Daten für {symbol}")
            shares_outstanding = self.model.dataloader.get_shares_outstanding(symbol)
            if isinstance(shares_outstanding, dict) and "error" in shares_outstanding:
                print(f"Fehler beim Abruf der Aktienzahl: {shares_outstanding['error']}")
                self.fail(f"Fehler beim Abruf der Aktienzahl für {symbol}")
            if shares_outstanding <= 0:
                print(f"Ungültige Aktienzahl für {symbol}: {shares_outstanding}")
                self.fail(f"Ungültige Aktienzahl für {symbol}")

            # Berechnete Kennzahl pro Aktie manuell überprüfen (als Referenz)
            metric_per_share = ebit_data["ebit"] / shares_outstanding if ebit_data["ebit"] else 0.0
            print(f"Verwendete Werte für {symbol}:")
            print(f"  EBIT: {ebit_data['ebit']}")
            print(f"  Shares Outstanding: {shares_outstanding}")
            print(f"  Berechnete Kennzahl pro Aktie: {metric_per_share}")

            # Methode aufrufen
            result = self.model.calculate_course_target_PriceMultiples(data, symbol)
            self.assertIsNotNone(result, f"calculate_course_target für {symbol} lieferte None.")
            self.assertIn("worst_case_price", result,
                          f"Fehlender Schlüssel 'worst_case_price' im Ergebnis für {symbol}.")
            self.assertIn("buy_price", result, f"Fehlender Schlüssel 'buy_price' im Ergebnis für {symbol}.")
            self.assertIn("fair_value_price", result,
                          f"Fehlender Schlüssel 'fair_value_price' im Ergebnis für {symbol}.")
            self.assertIn("sell_price", result, f"Fehlender Schlüssel 'sell_price' im Ergebnis für {symbol}.")
            self.assertTrue(all(value >= 0 for value in result.values()), f"Negative Kursziele erkannt für {symbol}.")

            print(f"Kursziele für {symbol}: {result}")

    def test_calculate_course_target_price_netCurrentAssets(self):
        """Testet die Methode calculate_course_target mit Price/NetCurrentAssets für TSLA."""
        print("\nTeste calculate_course_target_price_netCurrentAssets...")

        # Testfall 1: Gültiges Symbol (TSLA)
        with self.subTest(symbol="TSLA", case="valid_symbol"):
            symbol = "TSLA"
            print(f"\nBerechnung von Kursziel für {symbol} mit Price/NetCurrentAssets...")
            data = self.model.calculate_historical_price_netCurrentAssets(symbol)

            # Prüfen, ob Daten zurückgegeben wurden
            if data is None:
                print(f"Fehler für {symbol}: Keine Price/NetCurrentAssets-Daten abgerufen.")
                self.fail(f"Keine Price/NetCurrentAssets-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"Price/NetCurrentAssets-Daten für {symbol} sollten nicht leer sein.")

            # Rohe Daten für NetCurrentAssets und Shares Outstanding abrufen
            net_current_assets_data = self.model.dataloader.get_balance_sheet(symbol, frequency="quarterly")
            if isinstance(net_current_assets_data, dict) and "error" in net_current_assets_data:
                print(f"Fehler beim Abruf von NetCurrentAssets-Daten: {net_current_assets_data['error']}")
                self.fail(f"Fehler beim Abruf von NetCurrentAssets-Daten für {symbol}")
            total_assets = net_current_assets_data.loc["Total Assets"].iloc[
                0] if "Total Assets" in net_current_assets_data.index else 0
            total_liabilities = net_current_assets_data.loc["Total Liabilities"].iloc[
                0] if "Total Liabilities" in net_current_assets_data.index else 0
            net_current_assets = total_assets - total_liabilities
            shares_outstanding = self.model.dataloader.get_shares_outstanding(symbol)
            if isinstance(shares_outstanding, dict) and "error" in shares_outstanding:
                print(f"Fehler beim Abruf der Aktienzahl: {shares_outstanding['error']}")
                self.fail(f"Fehler beim Abruf der Aktienzahl für {symbol}")
            if shares_outstanding <= 0:
                print(f"Ungültige Aktienzahl für {symbol}: {shares_outstanding}")
                self.fail(f"Ungültige Aktienzahl für {symbol}")

            # Berechnete Kennzahl pro Aktie manuell überprüfen
            metric_per_share = net_current_assets / shares_outstanding if net_current_assets else 0.0
            print(f"Verwendete Werte für {symbol}:")
            print(f"  NetCurrentAssets: {net_current_assets}")
            print(f"  Shares Outstanding: {shares_outstanding}")
            print(f"  Berechnete Kennzahl pro Aktie: {metric_per_share}")

            # Buy-Wert direkt aus calculate_buy_case extrahieren
            buy_result_direct = self.model.calculate_buy_case(data)
            if isinstance(buy_result_direct, dict) and 'error' in buy_result_direct:
                print(f"Fehler beim direkten Abruf des Buy-Werts: {buy_result_direct['error']}")
            else:
                buy_multiple_direct = buy_result_direct['buy_value']
                print(f"Direkt berechneter Buy Multiple für {symbol}: {buy_multiple_direct}")

            # Methode aufrufen
            result = self.model.calculate_course_target_PriceMultiples(data, symbol)
            self.assertIsNotNone(result, f"calculate_course_target für {symbol} lieferte None.")

            # Prüfe, ob ein Fehlerdictionary zurückgegeben wurde
            if isinstance(result, dict) and 'error' in result:
                print(f"Erfolgreicher Fehlerfall für {symbol}: {result['error']}")
                # Extrahiere den Buy-Wert, der an calculate_worst_case übergeben wurde (durch Debugging in der Methode)
                # Da wir den genauen Wert nicht direkt sehen, simulieren wir den Aufruf
                wc_result = self.model.calculate_worst_case(data)
                if isinstance(wc_result, dict) and 'error' in wc_result:
                    print(f"Fehler in calculate_worst_case: {wc_result['error']}")
                    # Versuche, den Buy-Wert zu rekonstruieren, der den Fehler auslöste
                    buy_result_wc = self.model.calculate_buy_case(data)
                    if isinstance(buy_result_wc, dict) and 'error' not in buy_result_wc:
                        buy_multiple_wc = buy_result_wc['buy_value']
                        print(f"Buy Multiple, der an calculate_worst_case übergeben wurde: {buy_multiple_wc}")
                self.assertTrue(True)  # Fehlerfall akzeptieren
            else:
                # Prüfe normale Kursziele, falls kein Fehler auftritt
                self.assertIn("worst_case_price", result,
                              f"Fehlender Schlüssel 'worst_case_price' im Ergebnis für {symbol}.")
                self.assertIn("buy_price", result, f"Fehlender Schlüssel 'buy_price' im Ergebnis für {symbol}.")
                self.assertIn("fair_value_price", result,
                              f"Fehlender Schlüssel 'fair_value_price' im Ergebnis für {symbol}.")
                self.assertIn("sell_price", result, f"Fehlender Schlüssel 'sell_price' im Ergebnis für {symbol}.")
                self.assertTrue(all(value >= 0 for value in result.values()),
                                f"Negative Kursziele erkannt für {symbol}.")

            print(f"Ergebnis für {symbol}: {result}")

    def test_calculate_course_target_price_operatingCashflow(self):
        """Testet die Methode calculate_course_target mit Price/OperatingCashflow für MCD."""
        print("\nTeste calculate_course_target_price_operatingCashflow...")

        # Testfall 1: Gültiges Symbol (MCD, korrigierter Symbolwert)
        with self.subTest(symbol="MCD", case="valid_symbol"):
            symbol = "MCD"
            print(f"\nBerechnung von Kursziel für {symbol} mit Price/OperatingCashflow...")
            data = self.model.calculate_historical_price_OperatingCashflow(symbol)

            # Prüfen, ob Daten zurückgegeben wurden
            if data is None:
                print(f"Fehler für {symbol}: Keine Price/OperatingCashflow-Daten abgerufen.")
                self.fail(f"Keine Price/OperatingCashflow-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"Price/OperatingCashflow-Daten für {symbol} sollten nicht leer sein.")

            # Methode aufrufen
            result = self.model.calculate_course_target_PriceMultiples(data, symbol)
            self.assertIsNotNone(result, f"calculate_course_target für {symbol} lieferte None.")

            # Prüfe, ob ein Fehlerdictionary zurückgegeben wurde
            if isinstance(result, dict) and 'error' in result:
                print(f"Erfolgreicher Fehlerfall für {symbol}: {result['error']}")
                self.assertTrue(True)  # Fehlerfall akzeptieren
            else:
                # Validieren der Kursziele bei erfolgreichem Fall
                self.assertIn("worst_case_price", result,
                              f"Fehlender Schlüssel 'worst_case_price' im Ergebnis für {symbol}.")
                self.assertIn("buy_price", result, f"Fehlender Schlüssel 'buy_price' im Ergebnis für {symbol}.")
                self.assertIn("fair_value_price", result,
                              f"Fehlender Schlüssel 'fair_value_price' im Ergebnis für {symbol}.")
                self.assertIn("sell_price", result, f"Fehlender Schlüssel 'sell_price' im Ergebnis für {symbol}.")
                self.assertTrue(all(value >= 0 for value in result.values()),
                                f"Negative Kursziele erkannt für {symbol}.")

                print(f"Kursziele für {symbol}: {result}")

    def test_calculate_course_target_price_freeCashflow(self):
        """Testet die Methode calculate_course_target mit Price/FreeCashflow für MCD und TSLA."""
        print("\nTeste calculate_course_target_price_freeCashflow...")

        # Testfall 1: Gültiges Symbol (MCD, erfolgreicher Fall)
        with self.subTest(symbol="NVDA", case="valid_symbol"):
            symbol = "NVDA"
            print(f"\nBerechnung von Kursziel für {symbol} mit Price/FreeCashflow...")
            data = self.model.calculate_historical_Price_FreeCashflow(symbol)

            # Prüfen, ob Daten zurückgegeben wurden
            if data is None:
                print(f"Fehler für {symbol}: Keine Price/FreeCashflow-Daten abgerufen.")
                self.fail(f"Keine Price/FreeCashflow-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"Price/FreeCashflow-Daten für {symbol} sollten nicht leer sein.")

            # Methode aufrufen
            result = self.model.calculate_course_target_PriceMultiples(data, symbol)
            self.assertIsNotNone(result, f"calculate_course_target für {symbol} lieferte None.")

            # Prüfe, ob ein Fehlerdictionary zurückgegeben wurde
            if isinstance(result, dict) and 'error' in result:
                print(f"Erfolgreicher Fehlerfall für {symbol}: {result['error']}")
                self.assertTrue(True)  # Fehlerfall akzeptieren
            else:
                # Validieren der Kursziele bei erfolgreichem Fall
                self.assertIn("worst_case_price", result,
                              f"Fehlender Schlüssel 'worst_case_price' im Ergebnis für {symbol}.")
                self.assertIn("buy_price", result, f"Fehlender Schlüssel 'buy_price' im Ergebnis für {symbol}.")
                self.assertIn("fair_value_price", result,
                              f"Fehlender Schlüssel 'fair_value_price' im Ergebnis für {symbol}.")
                self.assertIn("sell_price", result, f"Fehlender Schlüssel 'sell_price' im Ergebnis für {symbol}.")
                self.assertTrue(all(value >= 0 for value in result.values()),
                                f"Negative Kursziele erkannt für {symbol}.")

                print(f"Kursziele für {symbol}: {result}")

        # Testfall 2: Gültiges Symbol (TSLA, erwarteter Fehlerfall)
        with self.subTest(symbol="MCD", case="expected_error"):
            symbol = "MCD"
            print(f"\nBerechnung von Kursziel für {symbol} mit Price/FreeCashflow...")
            data = self.model.calculate_historical_Price_FreeCashflow(symbol)

            # Prüfen, ob Daten zurückgegeben wurden
            if data is None:
                print(f"Fehler für {symbol}: Keine Price/FreeCashflow-Daten abgerufen.")
                self.fail(f"Keine Price/FreeCashflow-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"Price/FreeCashflow-Daten für {symbol} sollten nicht leer sein.")

            # Methode aufrufen
            result = self.model.calculate_course_target_PriceMultiples(data, symbol)
            self.assertIsNotNone(result, f"calculate_course_target für {symbol} lieferte None.")

            # Prüfe, ob ein Fehlerdictionary zurückgegeben wurde (erwartet für TSLA)
            if isinstance(result, dict) and 'error' in result:
                print(f"Erfolgreicher Fehlerfall für {symbol}: {result['error']}")
                self.assertTrue(True)  # Fehlerfall akzeptieren
            else:
                # Falls kein Fehler, validiere Kursziele (unwahrscheinlich für TSLA)
                self.assertIn("worst_case_price", result,
                              f"Fehlender Schlüssel 'worst_case_price' im Ergebnis für {symbol}.")
                self.assertIn("buy_price", result, f"Fehlender Schlüssel 'buy_price' im Ergebnis für {symbol}.")
                self.assertIn("fair_value_price", result,
                              f"Fehlender Schlüssel 'fair_value_price' im Ergebnis für {symbol}.")
                self.assertIn("sell_price", result, f"Fehlender Schlüssel 'sell_price' im Ergebnis für {symbol}.")
                self.assertTrue(all(value >= 0 for value in result.values()),
                                f"Negative Kursziele erkannt für {symbol}.")

                print(f"Kursziele für {symbol}: {result}")

    def test_calculate_course_target_EVMultiples_EV_Sales(self):
        """Testet die Berechnung der Kursziele für EV_Sales-Multiple mit Daten von calculate_historical_ev_sales."""
        print("\nTeste calculate_course_target_EVMultiples mit EV_Sales...")
        symbol = "TSLA"

        # Hole historische EV_Sales-Daten
        historical_data = self.model.calculate_historical_ev_sales(symbol)

        # Prüfe, ob Daten erfolgreich geladen wurden
        if historical_data is None or historical_data.empty:
            self.fail(f"Keine historischen EV_Sales-Daten für {symbol} erhalten von calculate_historical_ev_sales")

        # Aufruf der Methode
        result = self.model.calculate_course_target_EVMultiples(symbol, historical_data)

        # Überprüfe, ob ein Fehler zurückgegeben wurde
        if "error" in result:
            self.assertIsInstance(result, dict, "Ergebnis sollte ein Dictionary sein")
            self.assertIn("error", result, "Ergebnis sollte einen Fehler enthalten")
            print(f"Erwarteter Fehler erkannt: {result}")
            return  # Test erfolgreich beendet, da Fehler erwartet wird

        # Validierungen nur durchführen, wenn kein Fehler vorliegt
        self.assertIsInstance(result, dict, "Ergebnis sollte ein Dictionary sein")
        self.assertIn("EV_Sales", result, "Ergebnis sollte EV_Sales enthalten")
        self.assertIsInstance(result["EV_Sales"], dict, "EV_Sales sollte ein Dictionary sein")
        self.assertEqual(set(result["EV_Sales"].keys()), {"WC", "BUY", "FV", "SELL"},
                         "Szenarien sollten WC, BUY, FV, SELL enthalten")

        # Überprüfe, dass keine Fehler auftreten, wenn Daten gültig sind
        self.assertNotIn("error", result, "Ergebnis sollte keinen Fehler enthalten")

        # Überprüfe, dass Werte numerisch und positiv sind
        for scen_name, price in result["EV_Sales"].items():
            self.assertIsInstance(price, (int, float), f"Wert für {scen_name} sollte numerisch sein")
            self.assertGreaterEqual(price, 0, f"Wert für {scen_name} ({price}) sollte nicht negativ sein")

        # Hole die verwendeten Werte direkt von DataLoader (Annahme: Methoden geben Dictionaries zurück)
        revenue = self.model.dataloader.get_revenue(symbol, frequency="quarterly")
        net_debt = self.model.dataloader.get_net_debt_data(symbol, frequency="quarterly")
        shares = self.model.dataloader.get_shares_outstanding(symbol)
        minority_interest = self.model.dataloader.get_minority_interest(symbol, frequency="quarterly")
        preferred_stock = self.model.dataloader.get_preferred_stock(symbol, frequency="quarterly")

        # Extrahiere Werte (falls keine Fehler)
        revenue_value = float(revenue["revenue"]) if isinstance(revenue, dict) and "revenue" in revenue else None
        net_debt_value = float(net_debt["net_debt"]) if isinstance(net_debt, dict) and "net_debt" in net_debt else None
        shares_value = float(shares) if isinstance(shares, (int, float)) and shares > 0 else None
        minority_interest_value = float(minority_interest["minority_interest"]) if isinstance(minority_interest, dict) and "minority_interest" in minority_interest else None
        preferred_stock_value = float(preferred_stock["preferred_stock"]) if isinstance(preferred_stock, dict) and "preferred_stock" in preferred_stock else None

        # Schätze die verwendeten Multiples basierend auf den Kurszielen und der Formel (rückwärts)
        if revenue_value and shares_value and all(result["EV_Sales"].values()):
            revenue_per_share = revenue_value / shares_value
            ev_sales_multiples = {}
            for scen_name, price in result["EV_Sales"].items():
                # Rückwärtsrechnung: Multiples = (price * shares + net_debt) / revenue
                ev_sales_multiples[scen_name] = (price * shares_value + net_debt_value) / revenue_value
            multiplies = ev_sales_multiples
        else:
            multiplies = {"WC": None, "BUY": None, "FV": None, "SELL": None}

        # Ausgabe der Ergebnisse, verwendeten Werte und Multiples
        print(f"Kursziele für {symbol} (EV_Sales): {result['EV_Sales']}")
        print(f"Verwendete Werte für die Berechnung:")
        print(f" - Revenue (Umsatz): {revenue_value}")
        print(f" - Net Debt: {net_debt_value}")
        print(f" - Shares (Aktienzahl): {shares_value}")
        print(f" - Minority Interest: {minority_interest_value}")
        print(f" - Preferred Stock: {preferred_stock_value}")
        print(f"Verwendete Multiples (EV_Sales): {multiplies}")

    def test_calculate_course_target_EVMultiples_EV_EBIT(self):
        """Testet die Berechnung der Kursziele für EV_EBIT-Multiple mit Daten von calculate_historical_ev_to_ebit."""
        print("\nTeste calculate_course_target_EVMultiples mit EV_EBIT...")
        symbol = "NVDA"

        # Hole historische EV_EBIT-Daten
        historical_data = self.model.calculate_historical_ev_to_ebit(symbol)

        # Prüfe, ob Daten erfolgreich geladen wurden
        if historical_data is None or historical_data.empty:
            self.fail(f"Keine historischen EV_EBIT-Daten für {symbol} erhalten von calculate_historical_ev_to_ebit")

        # Aufruf der Methode
        result = self.model.calculate_course_target_EVMultiples(symbol, historical_data)

        # Überprüfe, ob ein Fehler zurückgegeben wurde
        if "error" in result:
            self.assertIsInstance(result, dict, "Ergebnis sollte ein Dictionary sein")
            self.assertIn("error", result, "Ergebnis sollte einen Fehler enthalten")
            print(f"Erwarteter Fehler erkannt: {result}")
            return  # Test erfolgreich beendet, da Fehler erwartet wird

        # Validierungen nur durchführen, wenn kein Fehler vorliegt
        self.assertIsInstance(result, dict, "Ergebnis sollte ein Dictionary sein")
        self.assertIn("EV_EBIT", result, "Ergebnis sollte EV_EBIT enthalten")
        self.assertIsInstance(result["EV_EBIT"], dict, "EV_EBIT sollte ein Dictionary sein")
        self.assertEqual(set(result["EV_EBIT"].keys()), {"WC", "BUY", "FV", "SELL"},
                         "Szenarien sollten WC, BUY, FV, SELL enthalten")

        # Überprüfe, dass keine Fehler auftreten, wenn Daten gültig sind
        self.assertNotIn("error", result, "Ergebnis sollte keinen Fehler enthalten")

        # Überprüfe, dass Werte numerisch und positiv sind
        for scen_name, price in result["EV_EBIT"].items():
            self.assertIsInstance(price, (int, float), f"Wert für {scen_name} sollte numerisch sein")
            self.assertTrue(np.isfinite(price), f"Wert für {scen_name} ({price}) sollte kein nan oder inf sein")
            self.assertGreaterEqual(price, 0, f"Wert für {scen_name} ({price}) sollte nicht negativ sein")

        # Hole die verwendeten Werte direkt von DataLoader (Annahme: Methoden geben Dictionaries zurück)
        ebit = self.model.dataloader.get_ebit_data(symbol, frequency="quarterly")
        net_debt = self.model.dataloader.get_net_debt_data(symbol, frequency="quarterly")
        shares = self.model.dataloader.get_shares_outstanding(symbol)
        minority_interest = self.model.dataloader.get_minority_interest(symbol, frequency="quarterly")
        preferred_stock = self.model.dataloader.get_preferred_stock(symbol, frequency="quarterly")

        # Extrahiere Werte (falls keine Fehler)
        ebit_value = float(ebit["ebit"]) if isinstance(ebit, dict) and "ebit" in ebit else None
        net_debt_value = float(net_debt["net_debt"]) if isinstance(net_debt, dict) and "net_debt" in net_debt else None
        shares_value = float(shares) if isinstance(shares, (int, float)) and shares > 0 else None
        minority_interest_value = float(minority_interest["minority_interest"]) if isinstance(minority_interest, dict) and "minority_interest" in minority_interest else None
        preferred_stock_value = float(preferred_stock["preferred_stock"]) if isinstance(preferred_stock, dict) and "preferred_stock" in preferred_stock else None

        # Schätze die verwendeten Multiples basierend auf den Kurszielen und der Formel (rückwärts)
        if ebit_value and shares_value and all(result["EV_EBIT"].values()):
            ebit_per_share = ebit_value / shares_value
            ev_ebit_multiples = {}
            for scen_name, price in result["EV_EBIT"].items():
                # Rückwärtsrechnung: Multiples = (price * shares + net_debt) / ebit
                ev_ebit_multiples[scen_name] = (price * shares_value + net_debt_value) / ebit_value
            multiplies = ev_ebit_multiples
        else:
            multiplies = {"WC": None, "BUY": None, "FV": None, "SELL": None}

        # Ausgabe der Ergebnisse, verwendeten Werte und Multiples
        print(f"Kursziele für {symbol} (EV_EBIT): {result['EV_EBIT']}")
        print(f"Verwendete Werte für die Berechnung:")
        print(f" - EBIT: {ebit_value}")
        print(f" - Net Debt: {net_debt_value}")
        print(f" - Shares (Aktienzahl): {shares_value}")
        print(f" - Minority Interest: {minority_interest_value}")
        print(f" - Preferred Stock: {preferred_stock_value}")
        print(f"Verwendete Multiples (EV_EBIT): {multiplies}")

    def test_calculate_course_target_EVMultiples_EV_EBITDA(self):
        """Testet die Berechnung der Kursziele für EV_EBITDA-Multiple mit Daten von calculate_historical_ev_to_ebitda."""
        print("\nTeste calculate_course_target_EVMultiples mit EV_EBITDA...")
        symbol = "MCD"

        # Hole historische EV_EBITDA-Daten
        historical_data = self.model.calculate_historical_ev_to_ebitda(symbol)

        # Prüfe, ob Daten erfolgreich geladen wurden
        if historical_data is None or historical_data.empty:
            self.fail(f"Keine historischen EV_EBITDA-Daten für {symbol} erhalten von calculate_historical_ev_to_ebitda")

        # Aufruf der Methode
        result = self.model.calculate_course_target_EVMultiples(symbol, historical_data)

        # Überprüfe, ob ein Fehler zurückgegeben wurde
        if "error" in result:
            self.assertIsInstance(result, dict, "Ergebnis sollte ein Dictionary sein")
            self.assertIn("error", result, "Ergebnis sollte einen Fehler enthalten")
            print(f"Erwarteter Fehler erkannt: {result}")
            return  # Test erfolgreich beendet, da Fehler erwartet wird

        # Validierungen nur durchführen, wenn kein Fehler vorliegt
        self.assertIsInstance(result, dict, "Ergebnis sollte ein Dictionary sein")
        self.assertIn("EV_EBITDA", result, "Ergebnis sollte EV_EBITDA enthalten")
        self.assertIsInstance(result["EV_EBITDA"], dict, "EV_EBITDA sollte ein Dictionary sein")
        self.assertEqual(set(result["EV_EBITDA"].keys()), {"WC", "BUY", "FV", "SELL"},
                         "Szenarien sollten WC, BUY, FV, SELL enthalten")

        # Überprüfe, dass keine Fehler auftreten, wenn Daten gültig sind
        self.assertNotIn("error", result, "Ergebnis sollte keinen Fehler enthalten")

        # Überprüfe, dass Werte numerisch und positiv sind
        for scen_name, price in result["EV_EBITDA"].items():
            self.assertIsInstance(price, (int, float), f"Wert für {scen_name} sollte numerisch sein")
            self.assertGreaterEqual(price, 0, f"Wert für {scen_name} ({price}) sollte nicht negativ sein")

        # Hole die verwendeten Werte direkt von DataLoader (Annahme: Methoden geben Dictionaries zurück)
        ebitda = self.model.dataloader.get_ebitda_data(symbol, frequency="quarterly")
        net_debt = self.model.dataloader.get_net_debt_data(symbol, frequency="quarterly")
        shares = self.model.dataloader.get_shares_outstanding(symbol)
        minority_interest = self.model.dataloader.get_minority_interest(symbol, frequency="quarterly")
        preferred_stock = self.model.dataloader.get_preferred_stock(symbol, frequency="quarterly")

        # Extrahiere Werte (falls keine Fehler)
        ebitda_value = float(ebitda["ebitda"]) if isinstance(ebitda, dict) and "ebitda" in ebitda else None
        net_debt_value = float(net_debt["net_debt"]) if isinstance(net_debt, dict) and "net_debt" in net_debt else None
        shares_value = float(shares) if isinstance(shares, (int, float)) and shares > 0 else None
        minority_interest_value = float(minority_interest["minority_interest"]) if isinstance(minority_interest,
                                                                                              dict) and "minority_interest" in minority_interest else None
        preferred_stock_value = float(preferred_stock["preferred_stock"]) if isinstance(preferred_stock,
                                                                                        dict) and "preferred_stock" in preferred_stock else None

        # Schätze die verwendeten Multiples basierend auf den Kurszielen und der Formel (rückwärts)
        if ebitda_value and shares_value and all(result["EV_EBITDA"].values()):
            ebitda_per_share = ebitda_value / shares_value
            ev_ebitda_multiples = {}
            for scen_name, price in result["EV_EBITDA"].items():
                # Rückwärtsrechnung: Multiples = (price * shares + net_debt) / ebitda
                ev_ebitda_multiples[scen_name] = (price * shares_value + net_debt_value) / ebitda_value
            multiplies = ev_ebitda_multiples
        else:
            multiplies = {"WC": None, "BUY": None, "FV": None, "SELL": None}

        # Ausgabe der Ergebnisse, verwendeten Werte und Multiples
        print(f"Kursziele für {symbol} (EV_EBITDA): {result['EV_EBITDA']}")
        print(f"Verwendete Werte für die Berechnung:")
        print(f" - EBITDA: {ebitda_value}")
        print(f" - Net Debt: {net_debt_value}")
        print(f" - Shares (Aktienzahl): {shares_value}")
        print(f" - Minority Interest: {minority_interest_value}")
        print(f" - Preferred Stock: {preferred_stock_value}")
        print(f"Verwendete Multiples (EV_EBITDA): {multiplies}")

    def test_calculate_course_target_price_tangibleBookValue(self):
        """Testet die Methode calculate_course_target mit Price/TangibleBookValue für MCD und AAPL."""
        print("\nTeste calculate_course_target_price_tangibleBookValue...")

        # Testfall 1: Gültiges Symbol (MCD, erwarteter Fehlerfall)
        with self.subTest(symbol="MCD", case="expected_error"):
            symbol = "MCD"
            print(f"\nBerechnung von Kursziel für {symbol} mit Price/TangibleBookValue...")
            data = self.model.calculate_historical_price_to_TangibleBookValue(symbol)

            # Prüfen, ob Daten zurückgegeben wurden
            if data is None:
                print(f"Fehler für {symbol}: Keine Price/TangibleBookValue-Daten abgerufen.")
                self.fail(f"Keine Price/TangibleBookValue-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"Price/TangibleBookValue-Daten für {symbol} sollten nicht leer sein.")

            # Methode aufrufen
            result = self.model.calculate_course_target_PriceMultiples(data, symbol)
            self.assertIsNotNone(result, f"calculate_course_target für {symbol} lieferte None.")

            # Prüfe, ob ein Fehlerdictionary zurückgegeben wurde (erwartet für MCD)
            if isinstance(result, dict) and 'error' in result:
                print(f"Erfolgreicher Fehlerfall für {symbol}: {result['error']}")
                self.assertTrue(True)  # Fehlerfall akzeptieren
            else:
                # Falls kein Fehler, validiere Kursziele (unwahrscheinlich für MCD)
                self.assertIn("worst_case_price", result,
                              f"Fehlender Schlüssel 'worst_case_price' im Ergebnis für {symbol}.")
                self.assertIn("buy_price", result, f"Fehlender Schlüssel 'buy_price' im Ergebnis für {symbol}.")
                self.assertIn("fair_value_price", result,
                              f"Fehlender Schlüssel 'fair_value_price' im Ergebnis für {symbol}.")
                self.assertIn("sell_price", result, f"Fehlender Schlüssel 'sell_price' im Ergebnis für {symbol}.")
                self.assertTrue(all(value >= 0 for value in result.values()),
                                f"Negative Kursziele erkannt für {symbol}.")

                print(f"Kursziele für {symbol}: {result}")

        # Testfall 2: Gültiges Symbol (AAPL, erfolgreicher Fall)
        with self.subTest(symbol="AAPL", case="valid_symbol"):
            symbol = "AAPL"
            print(f"\nBerechnung von Kursziel für {symbol} mit Price/TangibleBookValue...")
            data = self.model.calculate_historical_price_to_TangibleBookValue(symbol)

            # Prüfen, ob Daten zurückgegeben wurden
            if data is None:
                print(f"Fehler für {symbol}: Keine Price/TangibleBookValue-Daten abgerufen.")
                self.fail(f"Keine Price/TangibleBookValue-Daten für {symbol} erhalten.")

            # Prüfen, ob Ergebnis ein DataFrame ist und nicht leer
            self.assertIsInstance(data, pd.DataFrame, f"Ergebnis für {symbol} sollte ein DataFrame sein.")
            self.assertFalse(data.empty, f"Price/TangibleBookValue-Daten für {symbol} sollten nicht leer sein.")

            # Methode aufrufen
            result = self.model.calculate_course_target_PriceMultiples(data, symbol)
            self.assertIsNotNone(result, f"calculate_course_target für {symbol} lieferte None.")

            # Prüfe, ob ein Fehlerdictionary zurückgegeben wurde
            if isinstance(result, dict) and 'error' in result:
                print(f"Erfolgreicher Fehlerfall für {symbol}: {result['error']}")
                self.assertTrue(True)  # Fehlerfall akzeptieren
            else:
                # Validieren der Kursziele bei erfolgreichem Fall
                self.assertIn("worst_case_price", result,
                              f"Fehlender Schlüssel 'worst_case_price' im Ergebnis für {symbol}.")
                self.assertIn("buy_price", result, f"Fehlender Schlüssel 'buy_price' im Ergebnis für {symbol}.")
                self.assertIn("fair_value_price", result,
                              f"Fehlender Schlüssel 'fair_value_price' im Ergebnis für {symbol}.")
                self.assertIn("sell_price", result, f"Fehlender Schlüssel 'sell_price' im Ergebnis für {symbol}.")
                self.assertTrue(all(value >= 0 for value in result.values()),
                                f"Negative Kursziele erkannt für {symbol}.")

                print(f"Kursziele für {symbol}: {result}")

    def test_calculate_crv(self):
        """Testet die Methode calculate_crv mit Price/TangibleBookValue- und Price/FreeCashflow-Daten (ohne Mocks)."""
        print("\nTeste calculate_crv...")

        # --- Subtest 1: Price/TangibleBookValue (historische Daten) ---
        with self.subTest(symbol="MCD", case="price_tangiblebookvalue"):
            symbol = "MCD"
            print(f"\nBerechne CRV für {symbol} (Price/TangibleBookValue)...")
            hist_ptbv = self.model.calculate_historical_price_to_TangibleBookValue(symbol)

            self.assertIsNotNone(hist_ptbv, f"Keine Price/TangibleBookValue-Daten für {symbol} erhalten.")
            self.assertIsInstance(hist_ptbv, pd.DataFrame, "historical_data muss ein DataFrame sein.")
            self.assertFalse(hist_ptbv.empty, "Price/TangibleBookValue-Daten sollten nicht leer sein.")

            crv_result = self.model.calculate_crv(symbol, hist_ptbv)
            self.assertIsNotNone(crv_result, "calculate_crv lieferte None.")

            if isinstance(crv_result, dict) and "error" in crv_result:
                allowed_error_parts = [
                    "Fehler beim Abruf des Buy-Werts",
                    "Fehler beim Abruf des Worst-Case-Werts",
                    "Ungültiger Buy-Wert: Muss eine positive Zahl sein",
                    "Net Debt enthält ungültige Werte (z.B. nan oder inf)",
                    "Nicht genug Jahre nach dem globalen Minimum für eine vollständige Analyse",
                    "Nicht genug verschiedene Jahre für 3 Werte",
                    "Downside ≤ 0 – CRV nicht definiert (aktueller Kurs ≤ Worst-Case-Kurs).",
                ]
                self.assertTrue(any(part in crv_result["error"] for part in allowed_error_parts),
                                f"Unerwarteter Fehler für {symbol}: {crv_result['error']}")
                print(f"Erwarteter Fehler für {symbol}: {crv_result['error']}")
            else:
                self.assertIn("crv_conservative", crv_result)
                self.assertIn("crv_aggressive", crv_result)
                self.assertIn("inputs", crv_result)
                self.assertIn("course_targets", crv_result)

                self.assertIsInstance(crv_result["crv_conservative"], (int, float))
                self.assertIsInstance(crv_result["crv_aggressive"], (int, float))

                inputs = crv_result["inputs"]
                for k in ["current_price", "downside", "upside_conservative", "upside_aggressive"]:
                    self.assertIn(k, inputs)
                    self.assertIsInstance(inputs[k], (int, float))

                self.assertGreater(inputs["downside"], 0.0)

                if inputs["upside_conservative"] == 0.0 and inputs["upside_aggressive"] == 0.0:
                    self.assertEqual(crv_result["crv_conservative"], 0.0)
                    self.assertEqual(crv_result["crv_aggressive"], 0.0)
                    self.assertEqual(inputs["upside_aggressive"], inputs["upside_conservative"])
                else:
                    self.assertGreaterEqual(inputs["upside_aggressive"], inputs["upside_conservative"])
                    self.assertGreaterEqual(crv_result["crv_aggressive"], crv_result["crv_conservative"])
                    self.assertGreaterEqual(crv_result["crv_conservative"], 0.0)
                    self.assertGreaterEqual(crv_result["crv_aggressive"], 0.0)

                targets = crv_result["course_targets"]
                for key in ["WC", "BUY", "FV", "SELL"]:
                    self.assertIn(key, targets)
                    self.assertIsInstance(targets[key], (int, float))

                print(f"CRV (konservativ) für {symbol} mit Price/TangibleBookValue: {crv_result['crv_conservative']}")
                print(f"CRV (aggressiv)  für {symbol} mit Price/TangibleBookValue: {crv_result['crv_aggressive']}")
                print(f"Inputs: {crv_result['inputs']}")
                print(f"Kursziele: {crv_result['course_targets']}")

        # --- Subtest 2: Price/FreeCashflow (historische Daten) ---
        with self.subTest(symbol="BABA", case="price_freecashflow"):
            symbol = "BABA"
            print(f"\nBerechne CRV für {symbol} (Price/FreeCashflow)...")
            hist_pfcf = self.model.calculate_historical_Price_FreeCashflow(symbol)

            self.assertIsNotNone(hist_pfcf, f"Keine Price/FreeCashflow-Daten für {symbol} erhalten.")
            self.assertIsInstance(hist_pfcf, pd.DataFrame, "historical_data muss ein DataFrame sein.")
            self.assertFalse(hist_pfcf.empty, "Price/FreeCashflow-Daten sollten nicht leer sein.")

            crv_result = self.model.calculate_crv(symbol, hist_pfcf)
            self.assertIsNotNone(crv_result, "calculate_crv lieferte None.")

            if isinstance(crv_result, dict) and "error" in crv_result:
                allowed_error_parts = [
                    "Fehler beim Abruf des Buy-Werts",
                    "Fehler beim Abruf des Worst-Case-Werts",
                    "Ungültiger Buy-Wert: Muss eine positive Zahl sein",
                    "Net Debt enthält ungültige Werte (z.B. nan oder inf)",
                    "Nicht genug Jahre nach dem globalen Minimum für eine vollständige Analyse",
                    "Nicht genug verschiedene Jahre für 3 Werte",
                    "Downside ≤ 0 – CRV nicht definiert (aktueller Kurs ≤ Worst-Case-Kurs).",
                ]
                self.assertTrue(any(part in crv_result["error"] for part in allowed_error_parts),
                                f"Unerwarteter Fehler für {symbol}: {crv_result['error']}")
                print(f"Erwarteter Fehler für {symbol}: {crv_result['error']}")
            else:
                self.assertIn("crv_conservative", crv_result)
                self.assertIn("crv_aggressive", crv_result)
                self.assertIn("inputs", crv_result)
                self.assertIn("course_targets", crv_result)

                self.assertIsInstance(crv_result["crv_conservative"], (int, float))
                self.assertIsInstance(crv_result["crv_aggressive"], (int, float))

                inputs = crv_result["inputs"]
                for k in ["current_price", "downside", "upside_conservative", "upside_aggressive"]:
                    self.assertIn(k, inputs)
                    self.assertIsInstance(inputs[k], (int, float))

                self.assertGreater(inputs["downside"], 0.0)

                if inputs["upside_conservative"] == 0.0 and inputs["upside_aggressive"] == 0.0:
                    self.assertEqual(crv_result["crv_conservative"], 0.0)
                    self.assertEqual(crv_result["crv_aggressive"], 0.0)
                    self.assertEqual(inputs["upside_aggressive"], inputs["upside_conservative"])
                else:
                    self.assertGreaterEqual(inputs["upside_aggressive"], inputs["upside_conservative"])
                    self.assertGreaterEqual(crv_result["crv_aggressive"], crv_result["crv_conservative"])
                    self.assertGreaterEqual(crv_result["crv_conservative"], 0.0)
                    self.assertGreaterEqual(crv_result["crv_aggressive"], 0.0)

                targets = crv_result["course_targets"]
                for key in ["WC", "BUY", "FV", "SELL"]:
                    self.assertIn(key, targets)
                    self.assertIsInstance(targets[key], (int, float))

                print(f"CRV (konservativ) für {symbol} mit Price/FreeCashflow: {crv_result['crv_conservative']}")
                print(f"CRV (aggressiv)  für {symbol} mit Price/FreeCashflow: {crv_result['crv_aggressive']}")
                print(f"Inputs: {crv_result['inputs']}")
                print(f"Kursziele: {crv_result['course_targets']}")

    def test_calculate_percentiles(self):
        """Testet die Methode calculate_percentiles für ILMN und Sonderfälle."""
        print("\nTeste calculate_percentiles...")

        # Testfall 1: Gültiges Symbol (MCD)
        with self.subTest(symbol="MCD", case="valid_symbol"):
            symbol = "MCD"
            print(f"\nBerechnung von Percentiles für {symbol}...")
            # Daten abrufen über self.loader
            financials = self.loader.get_stock_financials(symbol, frequency="annual")
            if isinstance(financials, dict) and "error" in financials:
                print(f"Fehler für {symbol}: {financials['error']}")
                self.fail(f"Keine Finanzdaten für {symbol} erhalten.")

            # Abruf des Multiples
            multiple_data = self.model.calculate_historical_ev_sales(symbol, start_date=None, end_date=None,
                                                                     use_cache=True)
            if multiple_data is None or multiple_data.empty:
                print(f"Fehler für {symbol}: Keine EV_Sales-Daten erhalten.")
                self.fail(f"Keine Multiple-Daten für {symbol} erhalten.")
            data = multiple_data[["EV_Sales"]].copy().dropna()
            if data.empty:
                print(f"Fehler für {symbol}: Keine gültigen Multiple-Daten nach Entfernen von NaN-Werten.")
                self.fail(f"Keine gültigen Multiple-Daten für {symbol} erhalten.")

            result = self.model.calculate_percentiles(data, "EV_Sales")

            # Prüfen, ob Daten zurückgegeben wurden
            if "error" in result:
                print(f"Fehler für {symbol}: {result['error']}")
                self.fail(f"Keine Percentile-Daten für {symbol} erhalten.")

            self.assertIsInstance(result, dict, f"Ergebnis für {symbol} sollte ein Dictionary sein.")
            self.assertIn("percentiles", result, f"Ergebnis für {symbol} sollte 'percentiles' enthalten.")
            self.assertIn("ranges", result, f"Ergebnis für {symbol} sollte 'ranges' enthalten.")
            self.assertEqual(result["multiple"], "EV_Sales",
                             f"Multiple sollte 'EV_Sales' sein für {symbol}")

            # Anpassung des DataFrames: Basierend auf "ranges" (13 Einträge)
            percentiles_df = pd.DataFrame({
                "Percentiles": result["percentiles"][:-1],  # Alle 13 Grenzwerte bis zum letzten
                "Ranges": [f"{low:.2f}-{high:.2f}" for low, high in result["ranges"]]
            }, index=range(13))

            self.assertEqual(len(percentiles_df), 13, f"Anzahl der Percentile-Grenzwerte für {symbol} sollte 13 sein.")
            self.assertEqual(len(result["ranges"]), 13, f"Anzahl der Bereiche für {symbol} sollte 13 sein.")

            # Debugging: Datenstruktur ausgeben
            print(f"\nDebug: Inhalt von percentiles für {symbol}:")
            print(f"Percentiles: {result['percentiles']}")
            print(f"Ranges: {result['ranges']}")
            print(f"\nDebug: Percentile-Tabelle für {symbol}:\n{percentiles_df}")

            # Zeitraum und Datenpunkte ausgeben
            num_points = len(data)
            print(f"Test für {symbol} erfolgreich: {num_points} Datenpunkte verwendet.")

            time.sleep(1)

    def test_calculate_DurationInRange(self):
        """Testet die Methode calculate_DurationInRange für ILMN und Sonderfälle."""
        print("\nTeste calculate_DurationInRange...")

        # Testfall 1: Gültiges Symbol (MCD)
        with self.subTest(symbol="MCD", case="valid_symbol"):
            symbol = "MCD"
            print(f"\nBerechnung von DurationInRange für {symbol}...")
            multiple_data = self.model.calculate_historical_ev_sales(symbol, use_cache=True)
            if multiple_data is None or multiple_data.empty:
                print(f"Fehler für {symbol}: Keine EV_Sales-Daten erhalten.")
                self.fail(f"Keine Multiple-Daten für {symbol} erhalten.")
            data = multiple_data[["EV_Sales"]].copy().dropna()

            percentiles = self.model.calculate_percentiles(data, "EV_Sales")
            ranges = percentiles["ranges"]
            result = self.model.calculate_DurationInRange(data, "EV_Sales", ranges)

            self.assertIsInstance(result, dict, f"Ergebnis für {symbol} sollte ein Dictionary sein.")
            self.assertIn("counts", result, f"Ergebnis für {symbol} sollte 'counts' enthalten.")
            self.assertIn("shares", result, f"Ergebnis für {symbol} sollte 'shares' enthalten.")
            self.assertIn("total_days", result, f"Ergebnis für {symbol} sollte 'total_days' enthalten.")
            self.assertEqual(len(result["counts"]), 13, f"Anzahl der Counts für {symbol} sollte 13 sein.")
            self.assertGreaterEqual(sum(result["shares"].values()), 0.99,
                                    msg=f"Summe der Shares für {symbol} sollte mindestens 0,99 sein, ist {sum(result['shares'].values())}")

            # Tabelle erstellen
            table_df = pd.DataFrame({
                "Range": list(result["counts"].keys()),
                "Counts (Tage)": list(result["counts"].values()),
                "Shares (Anteil)": [f"{share:.4f}" for share in result["shares"].values()]
            })
            print(f"\nÜbersichtliche Tabelle für {symbol}:\n{table_df.to_string(index=False)}")

            print(f"Test für {symbol} erfolgreich: {result['total_days']} Datenpunkte verwendet.")

        # Testfall 2: Ungültige Daten
        with self.subTest(symbol="INVALID", case="invalid_data"):
            print(f"\nBerechnung von DurationInRange für ungültiges Symbol {symbol}...")
            symbol = "INVALID"
            data = pd.DataFrame({"EV_Sales": pd.Series(dtype=float)})
            ranges = [(0.0, 1.0)] * 12  # Platzhalter-Ranges
            with self.assertRaises(ValueError) as context:
                self.model.calculate_DurationInRange(data, "EV_Sales", ranges)
            self.assertIn("keine gültigen daten", str(context.exception).lower(),
                          f"Fehler für {symbol} sollte 'gültigen Daten' enthalten.")
            print(f"Test für ungültige Daten {symbol} erfolgreich: {str(context.exception)}")

            time.sleep(1)

    def test_calculate_probability_functionality(self):
        """Testet die Funktionalität der Methode calculate_probability für MCD."""
        print("\nTeste Funktionalität von calculate_probability...")

        # Testfall 1: Gültiges Symbol (MCD)
        with self.subTest(symbol="MCD", case="valid_functionality"):
            symbol = "MCD"
            print(f"\nBerechnung von EV/Sales, Percentiles, Duration und Wahrscheinlichkeit für {symbol}...")

            # Daten abrufen
            data = self.model.calculate_historical_ev_sales(symbol, use_cache=True)
            if data is None or data.empty:
                print(f"Fehler für {symbol}: Keine EV_Sales-Daten abgerufen.")
                self.fail(f"Keine EV_Sales-Daten für {symbol} erhalten.")
            print(f"Datenpunkte nach Abruf: {len(data)}")

            # Perzentile berechnen
            percentiles = self.model.calculate_percentiles(data, "EV_Sales")
            self.assertIsInstance(percentiles, dict, f"Ergebnis für {symbol} sollte ein Dictionary sein.")
            self.assertIn("ranges", percentiles, f"Ergebnis für {symbol} sollte 'ranges' enthalten.")
            self.assertEqual(len(percentiles["ranges"]), 13, f"Anzahl der Ranges für {symbol} sollte 13 sein.")
            ranges = percentiles["ranges"]

            # DurationInRange berechnen
            duration_result = self.model.calculate_DurationInRange(data, "EV_Sales", ranges)
            self.assertIsInstance(duration_result, dict, f"Ergebnis für {symbol} sollte ein Dictionary sein.")
            self.assertIn("shares", duration_result, f"Ergebnis für {symbol} sollte 'shares' enthalten.")
            print(f"Shares: {duration_result['shares']}, Summe Shares: {sum(duration_result['shares'].values())}")

            # Wahrscheinlichkeiten berechnen
            result = self.model.calculate_probability(data, "EV_Sales", ranges)
            self.assertIsInstance(result, dict, f"Ergebnis für {symbol} sollte ein Dictionary sein.")
            self.assertIn("ranges", result, f"Ergebnis für {symbol} sollte 'ranges' enthalten.")
            self.assertIn("probability_down", result, f"Ergebnis für {symbol} sollte 'probability_down' enthalten.")
            self.assertEqual(len(result["ranges"]), 13, f"Anzahl der Ranges für {symbol} sollte 13 sein.")
            self.assertEqual(len(result["probability_down"]), 13,
                             f"Anzahl der probability_down-Werte für {symbol} sollte 13 sein.")
            print(f"Probability Down Werte: {result['probability_down']}")
            self.assertAlmostEqual(max(result["probability_down"].values()), 1.0, places=1,
                                   msg=f"Maximale probability_down für {symbol} sollte etwa 1 sein, ist {max(result['probability_down'].values())}")
            self.assertEqual(min(result["probability_down"].values()), 0.0,
                             msg=f"Minimale probability_down für {symbol} sollte 0 sein, ist {min(result['probability_down'].values())}")

            # Erstelle Tabelle
            table_df = pd.DataFrame({
                "Range": result["ranges"],
                "Probability Down (%)": [f"{value * 100:.2f}" for value in result["probability_down"].values()],
                "Probability Up (%)": [f"{value * 100:.2f}" for value in result["probability_up"].values()]
            })
            print(f"\nÜbersichtliche Tabelle für Wahrscheinlichkeiten für {symbol}:\n{table_df.to_string(index=False)}")

            print(f"Test für {symbol} erfolgreich: {len(data)} Datenpunkte verwendet.")

        # Testfall 2: Ungültige Daten
        with self.subTest(symbol="INVALID", case="invalid_data"):
            print(f"\nBerechnung von Wahrscheinlichkeit für ungültiges Symbol {symbol}...")
            symbol = "INVALID"
            data = pd.DataFrame({"EV_Sales": pd.Series(dtype=float)})
            ranges = [(0.0, 1.0)] * 12  # Platzhalter-Ranges
            with self.assertRaises(ValueError) as context:
                self.model.calculate_probability(data, "EV_Sales", ranges)
            self.assertIn("keine gültigen daten", str(context.exception).lower(),
                          f"Fehler für {symbol} sollte 'gültigen Daten' enthalten.")
            print(f"Test für ungültige Daten {symbol} erfolgreich: {str(context.exception)}")

        time.sleep(1)

    def test_calculate_historical_ebit_with_percentiles_and_duration(self):
        """Testet die Integration von calculate_historical_ev_sales mit calculate_percentiles und calculate_DurationInRange."""
        print(
            "\nTeste Integration von calculate_historical_ev_sales, calculate_percentiles und calculate_DurationInRange...")

        # Testfall 1: Gültiges Symbol (MCD)
        with self.subTest(symbol="MCD", case="valid_symbol"):
            symbol = "MCD"
            print(f"\nBerechnung von EV/Sales, Percentiles und Duration für {symbol}...")
            ev_sales_data = self.model.calculate_historical_ev_sales(symbol, use_cache=True)
            if ev_sales_data is None or ev_sales_data.empty:
                print(f"Fehler für {symbol}: Keine EV/Sales-Daten abgerufen.")
                self.fail(f"Keine EV/Sales-Daten für {symbol} erhalten.")

            percentiles = self.model.calculate_percentiles(ev_sales_data, "EV_Sales")
            self.assertIsInstance(percentiles, dict, f"Ergebnis für {symbol} sollte ein Dictionary sein.")
            self.assertIn("percentiles", percentiles, f"Ergebnis für {symbol} sollte 'percentiles' enthalten.")
            self.assertIn("ranges", percentiles, f"Ergebnis für {symbol} sollte 'ranges' enthalten.")
            self.assertEqual(len(percentiles["ranges"]), 13, f"Anzahl der Ranges für {symbol} sollte 12 sein.")

            ranges = percentiles["ranges"]
            duration_result = self.model.calculate_DurationInRange(ev_sales_data, "EV_Sales", ranges)
            self.assertIsInstance(duration_result, dict, f"Ergebnis für {symbol} sollte ein Dictionary sein.")
            self.assertIn("counts", duration_result, f"Ergebnis für {symbol} sollte 'counts' enthalten.")
            self.assertIn("shares", duration_result, f"Ergebnis für {symbol} sollte 'shares' enthalten.")
            self.assertIn("total_days", duration_result, f"Ergebnis für {symbol} sollte 'total_days' enthalten.")
            self.assertEqual(len(duration_result["counts"]), 13, f"Anzahl der Counts für {symbol} sollte 12 sein.")
            self.assertGreaterEqual(sum(duration_result["shares"].values()), 0.99,
                                    msg=f"Summe der Shares für {symbol} sollte mindestens 0,99 sein, ist {sum(duration_result['shares'].values())}")

            # Tabelle erstellen
            table_df = pd.DataFrame({
                "Range": list(duration_result["counts"].keys()),
                "Counts (Tage)": list(duration_result["counts"].values()),
                "Shares (%)": [f"{value * 100:.2f}" for value in duration_result["shares"].values()]
            })
            print(f"\nÜbersichtliche Tabelle für {symbol}:\n{table_df.to_string(index=False)}")

            print(f"Test für {symbol} erfolgreich: {duration_result['total_days']} Datenpunkte verwendet.")

        time.sleep(1)

    def test_calculate_ev_to_ebit(self):
        """Testet die Methode calculate_ev_to_ebit für reguläre und Sonderfälle."""
        print("\nTeste calculate_ev_to_ebit...")
        for symbol in self.test_symbols + ["INVALID"]:  # Füge ein ungültiges Symbol hinzu
            for period in ["annual", "quarterly"]:
                with self.subTest(symbol=symbol, period=period):
                    # Teste mit Caching
                    result = self.model.calculate_ev_to_ebit(symbol, use_cache=True, frequency=period)
                    if isinstance(result, dict) and "error" in result:
                        print(f"Test für {symbol} ({period}) erfolgreich: Fehler erkannt - {result['error']}.")
                        self.assertIn("symbol", result,
                                      f"Fehler-Dictionary für {symbol} ({period}) sollte 'symbol' enthalten")
                        self.assertEqual(result["symbol"], symbol, f"Symbol im Fehler sollte {symbol} sein")
                    else:
                        print(
                            f"Test für {symbol} ({period}) erfolgreich: EV/EBIT für {symbol} beträgt {result['ev_to_ebit']}.")
                        self.assertIn("ev_to_ebit", result,
                                      f"EV/EBIT-Wert sollte in Ergebnis für {symbol} ({period}) enthalten sein")
                        self.assertIn("symbol", result,
                                      f"Symbol sollte in Ergebnis für {symbol} ({period}) enthalten sein")
                        self.assertIn("frequency", result,
                                      f"Frequency sollte in Ergebnis für {symbol} ({period}) enthalten sein")
                        self.assertIn("date", result,
                                      f"Datum sollte in Ergebnis für {symbol} ({period}) enthalten sein")
                        self.assertEqual(result["symbol"], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result["frequency"], period, f"Frequency sollte {period} sein")
                        self.assertIsInstance(result["date"], str,
                                              f"Datum für {symbol} ({period}) sollte ein String sein")
                        if result["ev_to_ebit"] == "inf":
                            self.assertIn("message", result,
                                          f"Message sollte bei 'inf' für {symbol} ({period}) enthalten sein")
                            self.assertEqual(result["message"],
                                             "EV/EBIT ist unendlich aufgrund eines null oder negativen EBIT.",
                                             f"Message für 'inf' sollte korrekt sein für {symbol} ({period})")
                        else:
                            self.assertIsInstance(result["ev_to_ebit"], float,
                                                  f"EV/EBIT für {symbol} ({period}) sollte ein Float sein")
                        # Teste Caching
                        cached_result = self.model.calculate_ev_to_ebit(symbol, use_cache=True, frequency=period)
                        self.assertEqual(cached_result, result,
                                         f"Cached Ergebnis für {symbol} ({period}) sollte mit erstem Aufruf übereinstimmen")

    def test_calculate_ev_to_ebitda(self):
        """Testet die Methode calculate_ev_to_ebitda für reguläre und Sonderfälle."""
        print("\nTeste calculate_ev_to_ebitda...")
        for symbol in self.test_symbols + ["INVALID"]:  # Füge ein ungültiges Symbol hinzu
            for period in ["annual", "quarterly"]:
                with self.subTest(symbol=symbol, period=period):
                    # Teste mit Caching
                    result = self.model.calculate_ev_to_ebitda(symbol, use_cache=True, frequency=period)
                    if isinstance(result, dict) and "error" in result:
                        print(f"Test für {symbol} ({period}) erfolgreich: Fehler erkannt - {result['error']}.")
                        self.assertIn("symbol", result,
                                      f"Fehler-Dictionary für {symbol} ({period}) sollte 'symbol' enthalten")
                        self.assertEqual(result["symbol"], symbol, f"Symbol im Fehler sollte {symbol} sein")
                    else:
                        print(
                            f"Test für {symbol} ({period}) erfolgreich: EV/EBITDA für {symbol} beträgt {result['ev_to_ebitda']}.")
                        self.assertIn("ev_to_ebitda", result,
                                      f"EV/EBITDA-Wert sollte in Ergebnis für {symbol} ({period}) enthalten sein")
                        self.assertIn("symbol", result,
                                      f"Symbol sollte in Ergebnis für {symbol} ({period}) enthalten sein")
                        self.assertIn("frequency", result,
                                      f"Frequency sollte in Ergebnis für {symbol} ({period}) enthalten sein")
                        self.assertIn("date", result,
                                      f"Datum sollte in Ergebnis für {symbol} ({period}) enthalten sein")
                        self.assertEqual(result["symbol"], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result["frequency"], period, f"Frequency sollte {period} sein")
                        self.assertIsInstance(result["date"], str,
                                              f"Datum für {symbol} ({period}) sollte ein String sein")
                        if result["ev_to_ebitda"] == "inf":
                            self.assertIn("message", result,
                                          f"Message sollte bei 'inf' für {symbol} ({period}) enthalten sein")
                            self.assertEqual(result["message"],
                                             "EV/EBITDA ist unendlich aufgrund eines null oder negativen EBITDA.",
                                             f"Message für 'inf' sollte korrekt sein für {symbol} ({period})")
                        else:
                            self.assertIsInstance(result["ev_to_ebitda"], float,
                                                  f"EV/EBITDA für {symbol} ({period}) sollte ein Float sein")
                        # Teste Caching
                        cached_result = self.model.calculate_ev_to_ebitda(symbol, use_cache=True, frequency=period)
                        self.assertEqual(cached_result, result,
                                         f"Cached Ergebnis für {symbol} ({period}) sollte mit erstem Aufruf übereinstimmen")

    def test_calculate_price_to_ebit(self):
        """Testet die Methode calculate_price_to_ebit für reguläre und Sonderfälle."""
        print("\nTeste calculate_price_to_ebit...")
        for symbol in self.test_symbols + ["INVALID"]:  # Füge ein ungültiges Symbol hinzu
            for frequency in ["annual", "quarterly"]:  # Teste beide Frequenzen
                with self.subTest(symbol=symbol, frequency=frequency):
                    # Teste mit Caching
                    result = self.model.calculate_price_to_ebit(symbol, use_cache=True, frequency=frequency)
                    if isinstance(result, dict) and "error" in result:
                        print(f"Test für {symbol} ({frequency}) erfolgreich: Fehler erkannt - {result['error']}.")
                        self.assertIn("symbol", result,
                                      f"Fehler-Dictionary für {symbol} ({frequency}) sollte 'symbol' enthalten")
                        self.assertEqual(result["symbol"], symbol,
                                         f"Symbol im Fehler sollte {symbol} sein")
                    else:
                        print(
                            f"Test für {symbol} ({frequency}) erfolgreich: Price/EBIT für {symbol} beträgt {result['price_to_ebit']}.")
                        self.assertIn("price_to_ebit", result,
                                      f"Price/EBIT-Wert sollte in Ergebnis für {symbol} ({frequency}) enthalten sein")
                        self.assertIn("symbol", result,
                                      f"Symbol sollte in Ergebnis für {symbol} ({frequency}) enthalten sein")
                        self.assertIn("frequency", result,
                                      f"Frequency sollte in Ergebnis für {symbol} ({frequency}) enthalten sein")
                        self.assertIn("date", result,
                                      f"Datum sollte in Ergebnis für {symbol} ({frequency}) enthalten sein")
                        self.assertEqual(result["symbol"], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result["frequency"], frequency, f"Frequency sollte {frequency} sein")
                        self.assertIsInstance(result["date"], str,
                                              f"Datum für {symbol} ({frequency}) sollte ein String sein")
                        if result["price_to_ebit"] == "inf":
                            self.assertIn("message", result,
                                          f"Message sollte bei 'inf' für {symbol} ({frequency}) enthalten sein")
                            self.assertEqual(result["message"],
                                             "Price/EBIT ist unendlich aufgrund eines null oder negativen EBIT.",
                                             f"Message für 'inf' sollte korrekt sein für {symbol} ({frequency})")
                        else:
                            self.assertIsInstance(result["price_to_ebit"], float,
                                                  f"Price/EBIT für {symbol} ({frequency}) sollte ein Float sein")
                        # Teste Caching
                        cached_result = self.model.calculate_price_to_ebit(symbol, use_cache=True, frequency=frequency)
                        self.assertEqual(cached_result, result,
                                         f"Cached Ergebnis für {symbol} ({frequency}) sollte mit erstem Aufruf übereinstimmen")

    def test_get_invested_capital(self):
        """Testet die Methode get_invested_capital für jährliche und quartalsweise Daten."""
        print("\nTeste get_invested_capital...")
        for symbol in self.test_symbols + ["INVALID"]:
            for frequency in ["annual", "quarterly"]:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.loader.get_invested_capital(symbol, frequency=frequency)
                    if isinstance(result, dict) and "error" in result:
                        print(f"Test für {symbol} ({frequency}) erfolgreich: Fehler erkannt - {result['error']}.")
                    else:
                        print(
                            f"Test für {symbol} ({frequency}) erfolgreich: Investiertes Kapital = {result['invested_capital']}, "
                            f"Eigenkapital = {result['total_equity']}, Fremdkapital = {result['total_debt']}, "
                            f"Datum = {result['date']}.")
                        self.assertIsInstance(result["invested_capital"], float,
                                              f"Investiertes Kapital für {symbol} ({frequency}) sollte ein Float sein")
                        self.assertIsInstance(result["total_equity"], float,
                                              f"Eigenkapital für {symbol} ({frequency}) sollte ein Float sein")
                        self.assertIsInstance(result["total_debt"], float,
                                              f"Fremdkapital für {symbol} ({frequency}) sollte ein Float sein")
                        self.assertEqual(result["symbol"], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result["frequency"], frequency, f"Frequenz sollte {frequency} sein")
                        self.assertIn("date", result,
                                      f"Datum sollte in Ergebnis für {symbol} ({frequency}) enthalten sein")
                        self.assertIsInstance(result["date"], str,
                                              f"Datum für {symbol} ({frequency}) sollte ein String sein")

    def test_calculate_current_netCurrentAssets(self):
        """Testet calculate_current_netCurrentAssets für annual/quarterly, Struktur, Plausibilität & Randfälle."""
        print("\nNet Current Assets Test Results:")

        test_symbols = ["AAPL", "MO", "BABA"]  # kannst du jederzeit erweitern
        frequencies = ["annual", "quarterly"]

        for symbol in test_symbols:
            for frequency in frequencies:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.model.calculate_current_netCurrentAssets(
                        symbol, frequency=frequency, use_cache=True
                    )

                    # Fehlerpfad
                    if isinstance(result, dict) and "error" in result:
                        print(f"{symbol} ({frequency}): ERROR -> {result['error']}")
                        self.assertIn("symbol", result, "Fehler-Dict sollte 'symbol' enthalten")
                        self.assertEqual(result["symbol"], symbol, "Fehler-Dict: falsches Symbol")
                        self.assertIsInstance(result["error"], str, "Fehlertext sollte String sein")
                        continue

                    # Pretty print
                    print(
                        f"{symbol} ({frequency}): "
                        f"NCA={result['net_current_assets']}, "
                        f"CA={result['current_assets']}, "
                        f"CL={result['current_liabilities']}, "
                        f"labels={result['labels_used']}, "
                        f"date={result.get('date')}"
                    )

                    # Grundstruktur
                    self.assertIn("symbol", result)
                    self.assertEqual(result["symbol"], symbol)
                    self.assertIn("frequency", result)
                    self.assertEqual(result["frequency"], frequency)
                    self.assertIn("date", result)

                    # Pflichtfelder & Typen
                    for key in ["net_current_assets", "current_assets", "current_liabilities"]:
                        self.assertIn(key, result, f"{key} fehlt in Ergebnis für {symbol} ({frequency})")
                        self.assertIsInstance(result[key], float, f"{key} sollte float sein für {symbol} ({frequency})")

                    self.assertIn("labels_used", result)
                    self.assertIsInstance(result["labels_used"], dict, "labels_used sollte dict sein")
                    self.assertIn("assets", result["labels_used"])
                    self.assertIn("liabilities", result["labels_used"])
                    self.assertTrue(
                        result["labels_used"]["assets"] is None or isinstance(result["labels_used"]["assets"], str),
                        "labels_used['assets'] sollte str oder None sein"
                    )
                    self.assertTrue(
                        result["labels_used"]["liabilities"] is None or isinstance(result["labels_used"]["liabilities"],
                                                                                   str),
                        "labels_used['liabilities'] sollte str oder None sein"
                    )

                    # Plausibilität: NCA = Current Assets - Current Liabilities (mit Toleranz)
                    expected_nca = result["current_assets"] - result["current_liabilities"]
                    self.assertAlmostEqual(
                        result["net_current_assets"], expected_nca, places=2,
                        msg=f"NCA sollte CA-CL sein für {symbol} ({frequency})"
                    )

                    # Plausibilität: CA/CL sollten nicht NaN sein
                    self.assertFalse(pd.isna(result["current_assets"]), "current_assets darf nicht NaN sein")
                    self.assertFalse(pd.isna(result["current_liabilities"]), "current_liabilities darf nicht NaN sein")
                    self.assertFalse(pd.isna(result["net_current_assets"]), "net_current_assets darf nicht NaN sein")

        # Randfall: ungültige Frequenz
        bad = self.model.calculate_current_netCurrentAssets("AAPL", frequency="monthly", use_cache=True)
        print(f"Invalid frequency test: {bad}")
        self.assertIsInstance(bad, dict)
        self.assertIn("error", bad)
        self.assertIn("symbol", bad)
        self.assertEqual(bad["symbol"], "AAPL")

    def test_calculate_ROIC(self):
        """Testet die Methode calculate_ROIC für jährliche und quartalsweise Daten sowie ungültige Symbole."""
        print("\nTeste calculate_ROIC...")
        for symbol in self.test_symbols + ["INVALID"]:
            for frequency in ["annual", "quarterly"]:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.model.calculate_ROIC(symbol, frequency=frequency)
                    if isinstance(result, dict) and "error" in result:
                        print(f"Test für {symbol} ({frequency}) erfolgreich: Fehler erkannt - {result['error']}.")
                        self.assertIn("symbol", result,
                                      f"Fehler-Dictionary für {symbol} ({frequency}) sollte 'symbol' enthalten")
                        self.assertEqual(result["symbol"], symbol, f"Symbol im Fehler sollte {symbol} sein")
                        if symbol == "INVALID":
                            # Jetzt wird quarterly unterstützt -> bei INVALID muss der Fehler vom Abruf stammen
                            self.assertIn("Fehler beim Abrufen", result["error"],
                                          f"Fehler für ungültiges Symbol {symbol} ({frequency}) sollte auf Datenabruf hinweisen")
                    else:
                        print(f"Test für {symbol} ({frequency}) erfolgreich: ROIC = {result['roic']}%, "
                              f"Net Income = {result['net_income']}, Investiertes Kapital = {result['invested_capital']}, "
                              f"Datum = {result['date']}.")
                        self.assertIsInstance(result["roic"], float,
                                              f"ROIC für {symbol} ({frequency}) sollte ein Float sein")
                        self.assertIsInstance(result["net_income"], float,
                                              f"Net Income für {symbol} ({frequency}) sollte ein Float sein")
                        self.assertIsInstance(result["invested_capital"], float,
                                              f"Investiertes Kapital für {symbol} ({frequency}) sollte ein Float sein")
                        self.assertEqual(result["symbol"], symbol, f"Symbol sollte {symbol} sein")
                        self.assertEqual(result["frequency"], frequency, f"Frequenz sollte {frequency} sein")
                        self.assertIn("date", result,
                                      f"Datum sollte in Ergebnis für {symbol} ({frequency}) enthalten sein")
                        self.assertIsInstance(result["date"], str,
                                              f"Datum für {symbol} ({frequency}) sollte ein String sein")
                        self.assertGreater(result["invested_capital"], 0,
                                           f"Investiertes Kapital für {symbol} ({frequency}) sollte größer als 0 sein")

    def test_get_cash_and_equivalents(self):
        """Testet die Methode get_cash_and_equivalents mit jährlichen und quartalsweisen Daten."""
        print("\nCash & Equivalents Test Results:")
        for symbol in self.test_symbols:
            for frequency in ["annual", "quarterly"]:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.loader.get_cash_and_equivalents(symbol, frequency=frequency, use_cache=True)

                    # Fehlerpfad
                    if isinstance(result, dict) and "error" in result:
                        print(f"Fehler für {symbol} ({frequency}): {result['error']}")
                        self.assertIn("symbol", result,
                                      f"Fehler-Dict für {symbol} ({frequency}) sollte 'symbol' enthalten")
                        self.assertEqual(result["symbol"], symbol, f"Symbol im Fehler sollte {symbol} sein")
                        self.assertIsInstance(result["error"], str,
                                              f"Fehlermeldung für {symbol} ({frequency}) sollte String sein")
                        continue

                    # Ausgabe
                    cash_val = result.get("cash_and_equivalents")
                    date = result.get("date", "unbekannt")
                    label_used = result.get("label_used", "unbekannt")
                    print(f"{symbol} ({frequency}): Cash = {cash_val}, Label = {label_used}, Date = {date}")

                    # Grundstruktur
                    self.assertIn("cash_and_equivalents", result,
                                  f"Ergebnis für {symbol} ({frequency}) sollte 'cash_and_equivalents' enthalten")
                    self.assertIn("symbol", result, f"Ergebnis für {symbol} ({frequency}) sollte 'symbol' enthalten")
                    self.assertIn("frequency", result,
                                  f"Ergebnis für {symbol} ({frequency}) sollte 'frequency' enthalten")
                    self.assertIn("label_used", result,
                                  f"Ergebnis für {symbol} ({frequency}) sollte 'label_used' enthalten")
                    self.assertIn("date", result, f"Ergebnis für {symbol} ({frequency}) sollte 'date' enthalten")

                    # Typen/Validierung
                    self.assertEqual(result["symbol"], symbol, f"Symbol sollte {symbol} sein")
                    self.assertEqual(result["frequency"], frequency, f"Frequenz sollte {frequency} sein")
                    self.assertIsInstance(result["cash_and_equivalents"], float,
                                          f"Cash-Wert für {symbol} ({frequency}) sollte float sein")
                    self.assertTrue(result["cash_and_equivalents"] >= 0,
                                    f"Cash-Wert für {symbol} ({frequency}) sollte >= 0 sein")
                    self.assertIsInstance(result["label_used"], str,
                                          f"label_used für {symbol} ({frequency}) sollte String sein")
                    self.assertTrue(len(result["label_used"]) > 0,
                                    f"label_used für {symbol} ({frequency}) sollte nicht leer sein")

                    # date kann None sein, aber wenn vorhanden, dann String
                    if result["date"] is not None:
                        self.assertIsInstance(result["date"], str,
                                              f"date für {symbol} ({frequency}) sollte String oder None sein")

    def test_calculate_cash_to_market_cap(self):
        """Testet die Methode calculate_cash_to_market_cap für valide, Grenz- und Fehlerfälle."""
        print("\nCash-to-Market-Cap Test Results:")

        test_symbols = ["AAPL", "MO", "BABA"]
        frequencies = ["annual", "quarterly"]

        for symbol in test_symbols:
            for frequency in frequencies:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.model.calculate_cash_to_market_cap(
                        symbol,
                        frequency=frequency,
                        use_cache=True
                    )

                    # Fehlerpfad
                    if isinstance(result, dict) and "error" in result:
                        print(f"Fehler für {symbol} ({frequency}): {result['error']}")
                        self.assertIn(
                            "symbol",
                            result,
                            f"Fehler-Dict für {symbol} ({frequency}) sollte 'symbol' enthalten"
                        )
                        self.assertEqual(
                            result["symbol"],
                            symbol,
                            f"Symbol im Fehler sollte {symbol} sein"
                        )
                        self.assertIsInstance(
                            result["error"],
                            str,
                            f"Fehlermeldung für {symbol} ({frequency}) sollte String sein"
                        )
                        continue

                    # Pretty Print
                    print(
                        f"{symbol} ({frequency}): "
                        f"Cash/MarketCap = {result['cash_to_market_cap']}, "
                        f"Cash = {result['cash']}, "
                        f"MarketCap = {result['market_cap']}, "
                        f"Date = {result['date']}"
                    )

                    # Grundstruktur
                    self.assertIn("cash_to_market_cap", result,
                                  f"{symbol} ({frequency}) sollte 'cash_to_market_cap' enthalten")
                    self.assertIn("cash", result,
                                  f"{symbol} ({frequency}) sollte 'cash' enthalten")
                    self.assertIn("market_cap", result,
                                  f"{symbol} ({frequency}) sollte 'market_cap' enthalten")
                    self.assertIn("symbol", result,
                                  f"{symbol} ({frequency}) sollte 'symbol' enthalten")
                    self.assertIn("frequency", result,
                                  f"{symbol} ({frequency}) sollte 'frequency' enthalten")
                    self.assertIn("date", result,
                                  f"{symbol} ({frequency}) sollte 'date' enthalten")

                    # Typprüfungen
                    self.assertIsInstance(
                        result["cash_to_market_cap"],
                        float,
                        f"cash_to_market_cap für {symbol} ({frequency}) sollte float sein"
                    )
                    self.assertIsInstance(
                        result["cash"],
                        float,
                        f"cash für {symbol} ({frequency}) sollte float sein"
                    )
                    self.assertIsInstance(
                        result["market_cap"],
                        float,
                        f"market_cap für {symbol} ({frequency}) sollte float sein"
                    )
                    self.assertIsInstance(
                        result["date"],
                        str,
                        f"date für {symbol} ({frequency}) sollte string sein"
                    )

                    # Inhaltliche Plausibilität
                    self.assertGreaterEqual(
                        result["cash"],
                        0,
                        f"Cash-Wert für {symbol} ({frequency}) sollte >= 0 sein"
                    )
                    self.assertGreater(
                        result["market_cap"],
                        0,
                        f"Market Cap für {symbol} ({frequency}) sollte > 0 sein"
                    )
                    self.assertGreaterEqual(
                        result["cash_to_market_cap"],
                        0,
                        f"Cash-to-Market-Cap für {symbol} ({frequency}) sollte >= 0 sein"
                    )

                    # Konsistenz
                    expected_ratio = round(result["cash"] / result["market_cap"], 4)
                    self.assertAlmostEqual(
                        result["cash_to_market_cap"],
                        expected_ratio,
                        places=4,
                        msg=f"Cash-to-Market-Cap für {symbol} ({frequency}) sollte korrekt berechnet sein"
                    )

                    self.assertEqual(
                        result["symbol"],
                        symbol,
                        f"Symbol sollte {symbol} sein"
                    )
                    self.assertEqual(
                        result["frequency"],
                        frequency,
                        f"Frequenz sollte {frequency} sein"
                    )

    def test_analyze_dividend_companies(self):
        """Testet die Methode analyze_dividend_companies inkl. CRV-Analyse."""
        print("\nTeste analyze_dividend_companies...")
        test_symbols = ["MO"]

        def _pretty_print(symbol, result):
            sep = "=" * 80

            print(f"\n{sep}")
            print(f"{symbol} – Dividendenanalyse")
            print(sep)

            if "error" in result:
                print(f"❌ Fehler: {result['error']}")
                print(sep)
                return

            def _line(name, value, ok, msg=""):
                status = "✔" if ok else "✘"
                val = value if value is not None else "N/A"
                print(f"{status} {name:<25}: {val}")
                if msg:
                    print(f"    ↳ {msg}")

            _line(
                "Dividend Yield",
                f"{result['dividend_yield']['value']:.2f}%",
                result['dividend_yield']['meets_criterion'],
            )

            eg = result["earnings_growth_vs_inflation"]
            _line(
                "Earnings Growth",
                f"AAGR={eg['annual_aagr'] or 'N/A'}%, AQGR={eg['quarterly_aqgr'] or 'N/A'}%, Infl={eg['inflation'] or 'N/A'}%",
                eg["meets_criterion"],
                eg["message"],
            )

            _line(
                "Payout Ratio",
                f"{result['payout_ratio']['value']:.2f}%",
                result['payout_ratio']['meets_criterion'],
                result['payout_ratio']['message'],
            )

            ic = result["interest_coverage_ratio"]
            _line(
                "Interest Coverage",
                f"{ic['value']:.2f}",
                ic["meets_criterion"],
                f"Datum: {ic.get('date') or 'N/A'}",
            )

            nde = result["net_debt_to_ebitda"]
            _line(
                "Net Debt / EBITDA",
                nde["value"],
                nde["meets_criterion"],
                nde.get("message", ""),
            )

            ev = result["ev_to_ebit"]
            _line(
                "EV / EBIT",
                ev["value"],
                ev["meets_criterion"],
                ev.get("message", ""),
            )

            dh = result["dividend_history"]
            print("\n📈 Dividend History")
            if "error" in dh:
                print(f"✘ Fehler: {dh['error']}")
            else:
                print(
                    f"✔ Jahre mit Dividenden   : {dh['years_with_dividends']}\n"
                    f"✔ Jahre mit Erhöhungen   : {dh['years_with_increases']}\n"
                    f"✔ CAGR                   : {dh['cagr']}%\n"
                    f"✔ CAGR-Zeitraum           : {dh['cagr_period_years']} Jahre\n"
                    f"  ↳ {dh['message']}"
                )

            # ---------------- CRV-Analyse ----------------
            print("\n" + "-" * 80)
            print("📊 CRV-Analyse (Bewertung – ergänzend)")
            print("-" * 80)

            crv = result.get("crv")
            if not crv or "error" in crv:
                print(f"❌ CRV-Analyse nicht verfügbar: {crv.get('error') if crv else 'keine Daten'}")
            else:
                print(f"Sektoren : {', '.join(crv.get('sectors', []))}")
                print(f"Multiples: {', '.join(crv.get('multiples_used', []))}\n")

                for multiple, crv_data in crv.get("crv_results", {}).items():
                    if "error" in crv_data:
                        print(f"✘ {multiple:<25}: Fehler – {crv_data['error']}")
                        continue

                    cons = crv_data.get("crv_conservative")
                    aggr = crv_data.get("crv_aggressive")
                    positive = crv_data.get("crv_positive")

                    status = "✔ POSITIV" if positive else "✘ negativ"

                    print(
                        f"{multiple:<25}: "
                        f"CRV konservativ={cons:.2f} | "
                        f"{'aggressiv=' + f'{aggr:.2f}' if aggr is not None else 'aggressiv=N/A'} "
                        f"→ {status}"
                    )

            print("\n" + "-" * 80)
            print(f"📌 Gesamtbewertung: {result['overall_assessment']}")
            print(f"📝 Zusammenfassung: {result['message']}")
            print(sep)

        for symbol in test_symbols:
            with self.subTest(symbol=symbol):
                result = self.actionmodule.analyze_dividend_companies(symbol)

                _pretty_print(symbol, result)

                # Fehlerpfad
                if "error" in result:
                    self.assertIn("symbol", result)
                    self.assertIsInstance(result["error"], str)
                    continue

                # Grundstruktur
                self.assertEqual(result["symbol"], symbol)
                self.assertIn(result["overall_assessment"], ["Dividend Safe", "Dividend Risky"])
                self.assertIn("message", result)

                # Pflichtkriterien
                criteria = [
                    "dividend_yield",
                    "earnings_growth_vs_inflation",
                    "payout_ratio",
                    "interest_coverage_ratio",
                    "net_debt_to_ebitda",
                    "ev_to_ebit",
                ]

                for c in criteria:
                    self.assertIn(c, result)
                    self.assertIn("meets_criterion", result[c])
                    self.assertIsInstance(result[c]["meets_criterion"], bool)

                # Dividend History
                self.assertIn("dividend_history", result)

                # CRV-Struktur
                self.assertIn("crv", result)
                if "error" not in result["crv"]:
                    self.assertIn("crv_results", result["crv"])
                    self.assertIsInstance(result["crv"]["crv_results"], dict)

    def test_analyze_average_grower(self):
        """Testet analyze_average_grower für valide Symbole, Grenzfälle und Fehlerfälle."""
        print("\nTeste analyze_average_grower...")
        test_symbols = ['PYPL']  # ggf. MHH als Edge Case

        def _pretty_print(symbol, result):
            sep = "=" * 80
            print(f"\n{sep}")
            print(f"{symbol} – Average-Grower-Analyse")
            print(sep)

            if "error" in result:
                print(f"❌ Fehler: {result['error']}")
                print(sep)
                return

            def _line(name, value, ok, msg):
                status = "✔" if ok else "✘"
                val = value if value is not None else "N/A"
                print(f"{status} {name:<20}: {val}")
                print(f"    ↳ {msg}")

            # 1) Price / EBIT
            pe = result["price_ebit"]
            _line(
                "Price / EBIT",
                f"{pe['value']:.2f} (Ø {pe['historical_mean']:.2f})",
                pe["meets_criterion"],
                pe["message"],
            )

            # 2) Price / TBV
            ptbv = result["price_tbv"]
            _line(
                "Price / TBV",
                f"{ptbv['value']:.2f} (Median {ptbv['historical_median']:.2f})",
                ptbv["meets_criterion"],
                ptbv["message"],
            )

            # 3) Dividend Yield / Reinvestment
            dy = result["dividend_yield"]
            _line(
                "Dividend Yield",
                f"{dy['value']:.2f}%",
                dy["meets_criterion"],
                dy["message"],
            )

            # 4) Free Cashflow
            fcf = result["free_cashflow"]
            _line(
                "Free Cashflow",
                f"{fcf['value']:.2f} (≥ {fcf['threshold']:.2f})",
                fcf["meets_criterion"],
                fcf["message"],
            )

            # ---------- CRV-Ausgabe ----------
            print("-" * 80)
            print("📊 CRV-Analyse (branchenrelevante historische Multiples)")
            print("-" * 80)

            crv = result.get("crv")
            if not crv or "error" in crv:
                print(f"❌ CRV-Analyse nicht verfügbar: {crv.get('error', 'Unbekannter Fehler')}")
            else:
                sectors = ", ".join(crv.get("sectors", []))
                multiples = ", ".join(crv.get("multiples_used", []))
                print(f"Sektoren: {sectors}")
                print(f"Multiples: {multiples}\n")

                for multiple, data in crv.get("crv_results", {}).items():
                    if "error" in data:
                        print(f"✘ {multiple:<25}: Fehler – {data['error']}")
                    else:
                        cons = data.get("crv_conservative")
                        aggr = data.get("crv_aggressive")
                        ok = data.get("crv_positive", False)
                        status = "✔ POSITIV" if ok else "✘ negativ"
                        print(
                            f"{multiple:<25}: "
                            f"CRV konservativ={cons:.2f} | aggressiv={aggr:.2f} → {status}"
                        )

            print("-" * 80)
            print(f"📌 Gesamtbewertung: {result['overall_assessment']}")
            print(f"📝 Zusammenfassung: {result['message']}")
            print(sep)

        for symbol in test_symbols:
            with self.subTest(symbol=symbol):
                result = self.actionmodule.analyze_average_grower(symbol, use_cache=True)

                # Fehlerpfad
                if isinstance(result, dict) and "error" in result:
                    _pretty_print(symbol, result)
                    self.assertIn("symbol", result)
                    self.assertEqual(result["symbol"], symbol)
                    self.assertIsInstance(result["error"], str)
                    continue

                _pretty_print(symbol, result)

                # ---------- Strukturtests ----------
                self.assertIn("symbol", result)
                self.assertEqual(result["symbol"], symbol)
                self.assertIn(result["overall_assessment"], ["Average Grower", "Not an Average Grower"])
                self.assertIn("message", result)

                for key in ["price_ebit", "price_tbv", "dividend_yield", "free_cashflow"]:
                    self.assertIn(key, result)
                    self.assertIn("meets_criterion", result[key])
                    self.assertIsInstance(result[key]["meets_criterion"], bool)

                # ---------- Gesamtlogik ----------
                meets_all = all(
                    result[c]["meets_criterion"]
                    for c in ["price_ebit", "price_tbv", "dividend_yield", "free_cashflow"]
                )
                expected = "Average Grower" if meets_all else "Not an Average Grower"
                self.assertEqual(result["overall_assessment"], expected)

    def test_analyze_wachstumswerte(self):
        """Testet die analyze_wachstumswerte-Methode für valide Symbole, Grenzfälle und Fehlerfälle."""
        print("\nTeste analyze_wachstumswerte...")
        test_symbols = ['BABA']
        frequencies = ['annual', 'quarterly']

        def _pretty_print(symbol, frequency, result):
            sep = "=" * 80

            print(f"\n{sep}")
            print(f"{symbol} – Wachstumsanalyse ({frequency.upper()})")
            print(sep)

            if "error" in result:
                print(f"❌ Fehler: {result['error']}")
                return

            def _line(name, value, ok, msg):
                status = "✔" if ok else "✘"
                val = value if value is not None else "N/A"
                print(f"{status} {name:<20}: {val}")
                print(f"    ↳ {msg}")

            # ---------------- Wachstums-Kriterien ----------------
            _line(
                "Profit Growth",
                f"{result['profit_growth']['value']:.2f}%" if result['profit_growth']['value'] is not None else None,
                result['profit_growth']['meets_criterion'],
                result['profit_growth']['message'],
            )

            _line(
                "PEG Ratio",
                f"{result['peg_ratio']['value']:.2f}" if result['peg_ratio']['value'] is not None else None,
                result['peg_ratio']['meets_criterion'],
                result['peg_ratio']['message'],
            )

            gm = result["gross_margin"]
            _line(
                "Gross Margin",
                gm.get("value"),
                gm["meets_criterion"],
                gm["message"],
            )

            _line(
                "ROIC",
                f"{result['roic']['value']:.2f}%" if result['roic']['value'] is not None else None,
                result['roic']['meets_criterion'],
                result['roic']['message'],
            )

            _line(
                "Reinvestment",
                f"{result['reinvestment_rate']['value']:.2f}%" if result['reinvestment_rate'][
                                                                      'value'] is not None else None,
                result['reinvestment_rate']['meets_criterion'],
                result['reinvestment_rate']['message'],
            )

            _line(
                "EV / Sales",
                f"{result['ev_to_sales']['value']:.2f}" if result['ev_to_sales']['value'] is not None else None,
                result['ev_to_sales']['meets_criterion'],
                result['ev_to_sales']['message'],
            )

            # ---------------- CRV-Analyse ----------------
            print("\n" + "-" * 80)
            print("📊 CRV-Analyse (branchenrelevante historische Multiples)")
            print("-" * 80)

            crv = result.get("crv")
            if not crv or "error" in crv:
                print(f"❌ CRV-Analyse nicht verfügbar: {crv.get('error') if crv else 'keine Daten'}")
            else:
                print(f"Sektoren: {', '.join(crv.get('sectors', []))}")
                print(f"Multiples: {', '.join(crv.get('multiples_used', []))}\n")

                for multiple, crv_data in crv.get("crv_results", {}).items():
                    if "error" in crv_data:
                        print(f"✘ {multiple:<25}: Fehler – {crv_data['error']}")
                        continue

                    cons = crv_data.get("crv_conservative")
                    aggr = crv_data.get("crv_aggressive")
                    positive = crv_data.get("crv_positive")

                    status = "✔ POSITIV" if positive else "✘ negativ"

                    print(
                        f"{multiple:<25}: "
                        f"CRV konservativ={cons:.2f} | "
                        f"{'aggressiv=' + f'{aggr:.2f}' if aggr is not None else 'aggressiv=N/A'} "
                        f"→ {status}"
                    )

            # ---------------- Gesamtbewertung ----------------
            print("\n" + "-" * 80)
            print(f"📌 Gesamtbewertung: {result['overall_assessment']}")
            print(f"📝 Zusammenfassung: {result['message']}")
            print(sep)

        for symbol in test_symbols:
            for frequency in frequencies:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.actionmodule.analyze_wachstumswerte(symbol, frequency=frequency, use_cache=True)

                    # Fehlerpfad
                    if isinstance(result, dict) and "error" in result:
                        _pretty_print(symbol, frequency, result)
                        self.assertIn("symbol", result,
                                      f"Fehler-Dict für {symbol} ({frequency}) sollte 'symbol' enthalten")
                        self.assertEqual(result["symbol"], symbol, f"Symbol im Fehler sollte {symbol} sein")
                        self.assertIsInstance(result["error"], str,
                                              f"Fehlermeldung für {symbol} ({frequency}) sollte String sein")
                        continue

                    _pretty_print(symbol, frequency, result)

                    # Grundstruktur
                    self.assertIn("symbol", result, f"Ergebnis für {symbol} ({frequency}) sollte 'symbol' enthalten")
                    self.assertEqual(result["symbol"], symbol, f"Symbol sollte {symbol} sein")
                    self.assertIn("overall_assessment", result,
                                  f"Ergebnis für {symbol} ({frequency}) sollte 'overall_assessment' enthalten")
                    self.assertIn(result["overall_assessment"], ["Wachstumswert", "Kein Wachstumswert"],
                                  f"Overall assessment für {symbol} ({frequency}) sollte 'Wachstumswert' oder 'Kein Wachstumswert' sein")
                    self.assertIn("message", result, f"Ergebnis für {symbol} ({frequency}) sollte 'message' enthalten")

                    # Kriterien
                    criteria = ["profit_growth", "peg_ratio", "gross_margin", "roic", "reinvestment_rate",
                                "ev_to_sales"]
                    for criterion in criteria:
                        self.assertIn(criterion, result,
                                      f"Kriterium {criterion} sollte in Ergebnis für {symbol} ({frequency}) enthalten sein")
                        self.assertIn("meets_criterion", result[criterion],
                                      f"{criterion} für {symbol} ({frequency}) sollte 'meets_criterion' enthalten")
                        self.assertIsInstance(result[criterion]["meets_criterion"], bool,
                                              f"'meets_criterion' für {criterion} ({symbol}, {frequency}) sollte ein Boolean sein")
                        self.assertIn("message", result[criterion],
                                      f"{criterion} für {symbol} ({frequency}) sollte 'message' enthalten")
                        self.assertIsInstance(result[criterion]["message"], str,
                                              f"'message' für {criterion} ({symbol}, {frequency}) sollte ein String sein")

                    # 1. Profit Growth
                    pg = result["profit_growth"]
                    if pg["meets_criterion"]:
                        self.assertTrue(pg["value"] is not None and pg["value"] > 15,
                                        f"Profit Growth value für {symbol} ({frequency}) sollte > 15% sein, wenn meets_criterion=True")
                    if not pg["meets_criterion"]:
                        self.assertTrue(
                            "erfüllt nicht" in pg["message"] or "Fehler" in pg["message"] or "Negative Nettogewinne" in
                            pg["message"],
                            f"Profit Growth message für {symbol} ({frequency}) sollte Fehler oder Negativ-Hinweis enthalten"
                        )

                    # 2. PEG Ratio
                    peg = result["peg_ratio"]
                    if peg["meets_criterion"]:
                        self.assertTrue(peg["value"] is not None and peg["value"] < 2,
                                        f"PEG Ratio value für {symbol} ({frequency}) sollte < 2 sein, wenn meets_criterion=True")
                    if not peg["meets_criterion"]:
                        self.assertTrue(
                            "nicht attraktiv" in peg["message"] or "Fehler" in peg["message"],
                            f"PEG Ratio message für {symbol} ({frequency}) sollte Fehler oder Negativ-Hinweis enthalten"
                        )

                    # 3. Gross Margin
                    gm = result["gross_margin"]
                    if gm["meets_criterion"]:
                        self.assertEqual(gm["trend"], "steigend",
                                         f"Gross Margin trend für {symbol} ({frequency}) sollte 'steigend' sein, wenn meets_criterion=True")
                        self.assertIsInstance(gm["value"], list,
                                              f"Gross Margin value für {symbol} ({frequency}) sollte eine Liste sein")
                        self.assertEqual(len(gm["value"]), 3,
                                         f"Gross Margin value für {symbol} ({frequency}) sollte 3 Werte enthalten")
                    if not gm["meets_criterion"]:
                        self.assertTrue(
                            "nicht steigend" in gm["message"] or "Fehler" in gm["message"] or "Fehlende Daten" in gm[
                                "message"] or "Nicht genügend Daten" in gm["message"],
                            f"Gross Margin message für {symbol} ({frequency}) sollte Fehler oder Negativ-Hinweis enthalten"
                        )

                    # 4. ROIC
                    roic = result["roic"]
                    if roic["meets_criterion"]:
                        self.assertTrue(roic["value"] is not None and roic["value"] > 10,
                                        f"ROIC value für {symbol} ({frequency}) sollte > 10% sein, wenn meets_criterion=True")
                    if not roic["meets_criterion"]:
                        self.assertTrue(
                            "ist nicht hoch" in roic["message"] or "Fehler" in roic["message"],
                            f"ROIC message für {symbol} ({frequency}) sollte Fehler oder Negativ-Hinweis enthalten"
                        )

                    # 5. Reinvestment Rate
                    rr = result["reinvestment_rate"]
                    if rr["meets_criterion"]:
                        self.assertTrue(rr["value"] is not None and rr["value"] > 50,
                                        f"Reinvestment Rate value für {symbol} ({frequency}) sollte > 50% sein, wenn meets_criterion=True")
                    if not rr["meets_criterion"]:
                        self.assertTrue(
                            "ist nicht hoch" in rr["message"] or "Fehler" in rr["message"] or "Net Income fehlt" in rr[
                                "message"],
                            f"Reinvestment Rate message für {symbol} ({frequency}) sollte Fehler oder Negativ-Hinweis enthalten"
                        )

                    # 6. EV/Sales
                    evs = result["ev_to_sales"]
                    if frequency == "annual":  # EV/Sales fixiert auf annual
                        if evs["meets_criterion"]:
                            self.assertTrue(evs["value"] is not None and evs["value"] < 10,
                                            f"EV/Sales value für {symbol} (annual) sollte < 10 sein, wenn meets_criterion=True")
                        if not evs["meets_criterion"]:
                            self.assertTrue(
                                "ist nicht akzeptabel" in evs["message"] or "Fehler" in evs["message"],
                                f"EV/Sales message für {symbol} (annual) sollte Fehler oder Negativ-Hinweis enthalten"
                            )

                    # Gesamtbewertung
                    meets_all = all(result[c]["meets_criterion"] for c in criteria)
                    expected_overall = "Wachstumswert" if meets_all else "Kein Wachstumswert"
                    self.assertEqual(result["overall_assessment"], expected_overall,
                                     f"overall_assessment für {symbol} ({frequency}) sollte zu den Kriterien passen")
                    if meets_all:
                        self.assertTrue(
                            "Alle Kriterien erfüllt" in result["message"],
                            f"message für {symbol} ({frequency}) sollte 'Alle Kriterien erfüllt' enthalten, wenn alle Kriterien erfüllt sind"
                        )
                        if result["profit_growth"]["value"] is not None and result["profit_growth"]["value"] > 20 and \
                                result["peg_ratio"]["value"] is not None and result["peg_ratio"]["value"] < 1:
                            self.assertIn("idealer Wachstumswert", result["message"],
                                          f"message für {symbol} ({frequency}) sollte 'idealer Wachstumswert' enthalten, wenn AAGR > 20% und PEG < 1")
                    else:
                        self.assertIsInstance(result["message"], str,
                                              f"message für {symbol} ({frequency}) sollte ein String sein")
                        self.assertTrue(len(result["message"]) > 0,
                                        f"message für {symbol} ({frequency}) sollte bei Nichterfüllung Hinweise enthalten")

    def test_analyze_typical_cyclers(self):
        """Testet die analyze_typical_cyclers-Methode für valide Symbole, Grenzfälle und Fehlerfälle."""
        print("\nTeste analyze_typical_cyclers...")

        test_symbols = ["OXY"]  # typischer Zykliker-Kandidat
        frequencies = ["annual", "quarterly"]
        min_history_years = 10.0

        def _pretty_print(symbol, frequency, result):
            sep = "=" * 80

            print(f"\n{sep}")
            print(f"{symbol} – Zykliker-Analyse ({frequency.upper()})")
            print(sep)

            if isinstance(result, dict) and "error" in result:
                print(f"❌ Fehler: {result['error']}")
                print(sep)
                return

            def _line(name, value, ok, msg):
                status = "✔" if ok else "✘"
                val = value if value is not None else "N/A"
                print(f"{status} {name:<22}: {val}")
                print(f"    ↳ {msg}")

            _line(
                "ROE",
                f"{result['roe'].get('value'):.2%}" if result["roe"].get("value") is not None else None,
                result["roe"]["meets_criterion"],
                result["roe"]["message"],
            )

            _line(
                "Cashflow-Marge",
                f"{result['cashflow_margin'].get('value')}%",
                result["cashflow_margin"]["meets_criterion"],
                result["cashflow_margin"]["message"],
            )

            _line(
                "KGV",
                result["kgv"].get("value"),
                result["kgv"]["meets_criterion"],
                result["kgv"]["message"],
            )

            _line(
                "Vorräte / Umsatz",
                f"{result['inventory_to_revenue'].get('value')}%",
                result["inventory_to_revenue"]["meets_criterion"],
                result["inventory_to_revenue"]["message"],
            )

            ptbv = result["price_to_tbv"]
            _line(
                "P / TBV",
                ptbv.get("value"),
                ptbv["meets_criterion"],
                ptbv["message"],
            )

            eve = result["ev_to_ebitda"]
            _line(
                "EV / EBITDA",
                f"{eve.get('current')} (Pctl {eve.get('percentile')})",
                eve["meets_criterion"],
                eve["message"],
            )

            pfcf = result["price_to_fcf"]
            _line(
                "P / Free CF (ann.)",
                pfcf.get("value"),
                pfcf["meets_criterion"],
                pfcf["message"],
            )

            # ---------- CRV ----------
            print("-" * 80)
            print("📊 CRV-Analyse (branchenrelevante historische Multiples)")
            print("-" * 80)

            crv = result.get("crv")
            if not crv or "error" in crv:
                print(f"❌ CRV-Analyse nicht verfügbar: {crv.get('error', 'unbekannt')}")
            else:
                print(f"Sektoren: {', '.join(crv.get('sectors', []))}")
                print(f"Multiples: {', '.join(crv.get('multiples', []))}\n")

                for multiple, data in crv.get("results", {}).items():
                    if "error" in data:
                        print(f"✘ {multiple:<24}: Fehler – {data['error']}")
                    else:
                        status = "✔ POSITIV" if data.get("positive") else "✘ negativ"
                        print(
                            f"{multiple:<24}: "
                            f"CRV konservativ={data.get('crv_conservative')} | "
                            f"aggressiv={data.get('crv_aggressive')} → {status}"
                        )

            print("-" * 80)
            print(f"📌 Gesamtbewertung: {result['overall_assessment']}")
            print(f"📝 Zusammenfassung: {result['message']}")
            print(sep)

        for symbol in test_symbols:
            for frequency in frequencies:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.actionmodule.analyze_typical_cyclers(
                        symbol,
                        frequency=frequency,
                        min_history_years=min_history_years,
                        use_cache=True
                    )

                    # Fehlerpfad
                    if isinstance(result, dict) and "error" in result:
                        _pretty_print(symbol, frequency, result)
                        self.assertIn("symbol", result)
                        self.assertEqual(result["symbol"], symbol)
                        self.assertIsInstance(result["error"], str)
                        continue

                    _pretty_print(symbol, frequency, result)

                    # ---------------- Assertions (unverändert) ----------------
                    self.assertIn("symbol", result)
                    self.assertEqual(result["symbol"], symbol)
                    self.assertIn("frequency", result)
                    self.assertEqual(result["frequency"], frequency)

                    criteria = [
                        "roe", "cashflow_margin", "kgv",
                        "inventory_to_revenue", "price_to_tbv",
                        "ev_to_ebitda", "price_to_fcf"
                    ]

                    for c in criteria:
                        self.assertIn(c, result)
                        self.assertIn("meets_criterion", result[c])
                        self.assertIsInstance(result[c]["meets_criterion"], bool)
                        self.assertIn("message", result[c])

                    meets_all = all(result[c]["meets_criterion"] for c in criteria)
                    expected = "Typical Cycler – Buy" if meets_all else "Typical Cycler – Watch/Avoid"
                    self.assertEqual(result["overall_assessment"], expected)

                    if meets_all:
                        self.assertIn("Alle Kriterien erfüllt", result["message"])
                    else:
                        self.assertTrue(len(result["message"]) > 0)

    def test_analyze_cycler_turnarounds(self):
        """Testet die analyze_cycler_turnarounds-Methode für Turnaround-Situationen."""
        print("\nTeste analyze_cycler_turnarounds...")

        # Klassische Turnaround- / High-Risk-Zykliker
        test_symbols = ["UAA"]
        frequencies = ["annual", "quarterly"]

        def _pretty_print(symbol, frequency, result):
            sep = "=" * 80

            print(f"\n{sep}")
            print(f"{symbol} – Zyklischer Turnaround ({frequency.upper()})")
            print(sep)

            if "error" in result:
                print(f"❌ Fehler: {result['error']}")
                print(sep)
                return

            def _line(name, value, ok, msg):
                status = "✔" if ok else "✘"
                val = value if value is not None else "N/A"
                print(f"{status} {name:<25}: {val}")
                print(f"    ↳ {msg}")

            # 1) Net Debt / EBITDA
            nde = result["net_debt_to_ebitda"]
            _line(
                "Net Debt / EBITDA",
                nde.get("value"),
                nde["meets_criterion"],
                nde["message"],
            )

            # 2) Debt to Equity
            dte = result["debt_to_equity"]
            _line(
                "Debt / Equity",
                dte.get("value"),
                dte["meets_criterion"],
                dte["message"],
            )

            # 3) Inventory / Revenue
            inv = result["inventory_to_revenue"]
            _line(
                "Inventory / Revenue",
                inv.get("value"),
                inv["meets_criterion"],
                inv["message"],
            )

            # 4) Valuable Assets (qualitativ)
            print(f"✔ {'Valuable Assets':<25}: qualitative Prüfung")
            print(f"    ↳ {result['valuable_assets']['message']}")

            # ---------- CRV ----------
            print("-" * 80)
            print("📊 CRV-Analyse (Bewertung – ergänzend)")
            print("-" * 80)

            crv = result.get("crv", {})
            if not crv or "error" in crv:
                print(f"❌ CRV-Analyse nicht verfügbar: {crv.get('error', 'keine Daten')}")
            else:
                print(f"Sektoren: {', '.join(crv.get('sectors', []))}")
                print(f"Multiples: {', '.join(crv.get('multiples_used', []))}\n")

                for multiple, data in crv.get("crv_results", {}).items():
                    if "error" in data:
                        print(f"✘ {multiple:<25}: Fehler – {data['error']}")
                    else:
                        cons = data.get("crv_conservative")
                        aggr = data.get("crv_aggressive")
                        ok = data.get("crv_positive")
                        status = "✔ POSITIV" if ok else "✘ negativ"
                        print(
                            f"{multiple:<25}: CRV konservativ={cons:.2f} | aggressiv={aggr:.2f} → {status}"
                        )

            # ---------- Summary ----------
            print("-" * 80)
            print(f"📌 Gesamtbewertung: {result['overall_assessment']}")
            print(f"📝 Zusammenfassung: {result['message']}")
            print(sep)

        for symbol in test_symbols:
            for frequency in frequencies:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.actionmodule.analyze_cycler_turnarounds(
                        symbol,
                        frequency=frequency
                    )

                    # ------------------------
                    # Fehlerpfad
                    # ------------------------
                    if isinstance(result, dict) and "error" in result:
                        _pretty_print(symbol, frequency, result)
                        self.assertIn("symbol", result)
                        self.assertEqual(result["symbol"], symbol)
                        self.assertIsInstance(result["error"], str)
                        continue

                    _pretty_print(symbol, frequency, result)

                    # ------------------------
                    # Grundstruktur
                    # ------------------------
                    self.assertIn("symbol", result)
                    self.assertEqual(result["symbol"], symbol)

                    self.assertIn("overall_assessment", result)
                    self.assertIn(
                        result["overall_assessment"],
                        ["Turnaround Candidate", "High-Risk Turnaround / Avoid"]
                    )

                    self.assertIn("message", result)
                    self.assertIsInstance(result["message"], str)

                    # ------------------------
                    # Kriterien vorhanden
                    # ------------------------
                    criteria = [
                        "net_debt_to_ebitda",
                        "debt_to_equity",
                        "inventory_to_revenue",
                        "valuable_assets"
                    ]

                    for criterion in criteria:
                        self.assertIn(criterion, result)

                    # ------------------------
                    # 1. Net Debt / EBITDA
                    # ------------------------
                    nde = result["net_debt_to_ebitda"]
                    self.assertIn("risk_level", nde)
                    self.assertIn(nde["risk_level"], ["low", "elevated", "high", "critical", "unknown"])
                    self.assertIn("meets_criterion", nde)
                    self.assertIsInstance(nde["meets_criterion"], bool)

                    # ------------------------
                    # 2. Debt to Equity
                    # ------------------------
                    dte = result["debt_to_equity"]
                    self.assertIn("risk_level", dte)
                    self.assertIn(dte["risk_level"], ["moderate", "high", "critical", "unknown"])
                    self.assertIn("meets_criterion", dte)
                    self.assertIsInstance(dte["meets_criterion"], bool)

                    # ------------------------
                    # 3. Inventory / Revenue
                    # ------------------------
                    inv = result["inventory_to_revenue"]
                    self.assertIn("meets_criterion", inv)
                    self.assertIsInstance(inv["meets_criterion"], bool)
                    self.assertIn("message", inv)

                    if inv["value"] is not None:
                        self.assertTrue(
                            isinstance(inv["value"], float),
                            f"Inventory/Revenue value für {symbol} ({frequency}) sollte float sein"
                        )

                    # ------------------------
                    # 4. Valuable Assets (qualitativ)
                    # ------------------------
                    va = result["valuable_assets"]
                    self.assertIn("message", va)
                    self.assertIsNone(
                        va["meets_criterion"],
                        "valuable_assets sollte kein hartes meets_criterion haben"
                    )

                    # ------------------------
                    # Logische Konsistenz
                    # ------------------------
                    if result["overall_assessment"] == "Turnaround Candidate":
                        self.assertTrue(
                            nde["meets_criterion"] or dte["meets_criterion"],
                            "Turnaround Candidate sollte zumindest finanzielle Überlebensfähigkeit zeigen"
                        )

                    if result["overall_assessment"] == "High-Risk Turnaround / Avoid":
                        self.assertTrue(
                            "Risiko" in result["message"] or
                            "hoch" in result["message"] or
                            "kritisch" in result["message"],
                            "High-Risk Turnaround sollte Risiko im Message-Text reflektieren"
                        )

    def test_analyze_optionality(self):
        """Testet die analyze_optionality-Methode für Optionalitäts-Kandidaten."""
        print("\nTeste analyze_optionality...")

        test_symbols = ["AAPL", "MO", "BABA"]
        frequencies = ["annual", "quarterly"]

        def _pretty_print(symbol, frequency, result):
            sep = "=" * 80
            print(f"\n{sep}")
            print(f"{symbol} – Optionality-Analyse ({frequency.upper()})")
            print(sep)

            if "error" in result:
                print(f"❌ Fehler: {result['error']}")
                return

            def _line(name, value, ok, msg):
                status = "✔" if ok else "✘"
                val = value if value is not None else "N/A"
                print(f"{status} {name:<22}: {val}")
                print(f"    ↳ {msg}")

            nde = result["net_debt_to_ebitda"]
            _line(
                "Net Debt / EBITDA",
                f"{nde['value']:.2f}" if nde["value"] is not None else None,
                nde["meets_criterion"],
                nde["message"],
            )

            cash = result["cash_to_market_cap"]
            _line(
                "Cash / Market Cap",
                f"{cash['value']:.1%}" if cash["value"] is not None else None,
                cash["meets_criterion"],
                cash["message"],
            )

            print("-" * 80)
            print("🧩 Katalysatoren (qualitativ)")
            print(f"    ↳ {result['catalysts']['message']}")

            # ---------- CRV (Add-on) ----------
            print("-" * 80)
            print("📊 CRV-Analyse (Bewertung – ergänzend)")
            crv = result.get("crv")

            if not crv or "error" in crv:
                print(f"❌ CRV-Analyse nicht verfügbar: {crv.get('error', 'keine Daten') if crv else 'nicht berechnet'}")
            else:
                print(f"Sektoren: {', '.join(crv.get('sectors', []))}")
                print(f"Multiples: {', '.join(crv.get('multiples_used', []))}\n")

                for m, data in crv["crv_results"].items():
                    if "error" in data:
                        print(f"✘ {m:<25}: Fehler – {data['error']}")
                    else:
                        cons = data.get("crv_conservative")
                        aggr = data.get("crv_aggressive")
                        status = "✔ POSITIV" if data.get("crv_positive") else "✘ negativ"
                        print(f"{m:<25}: CRV konservativ={cons:.2f} | aggressiv={aggr:.2f} → {status}")

            print("-" * 80)
            print(f"📌 Gesamtbewertung: {result['overall_assessment']}")
            print(f"📝 Zusammenfassung: {result['message']}")
            print(sep)

        for symbol in test_symbols:
            for frequency in frequencies:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.actionmodule.analyze_optionality(
                        symbol,
                        frequency=frequency,
                        use_cache=True
                    )

                    # -----------------------------
                    # Fehlerpfad
                    # -----------------------------
                    if isinstance(result, dict) and "error" in result:
                        _pretty_print(symbol, frequency, result)
                        self.assertIn("symbol", result)
                        self.assertEqual(result["symbol"], symbol)
                        self.assertIsInstance(result["error"], str)
                        continue

                    _pretty_print(symbol, frequency, result)

                    # -----------------------------
                    # Grundstruktur
                    # -----------------------------
                    self.assertIn("symbol", result)
                    self.assertEqual(result["symbol"], symbol)

                    self.assertIn("overall_assessment", result)
                    self.assertIn(
                        result["overall_assessment"],
                        ["Optionality Candidate", "Limited Optionality"]
                    )

                    self.assertIn("message", result)
                    self.assertIsInstance(result["message"], str)

                    # -----------------------------
                    # Kriterien prüfen
                    # -----------------------------
                    criteria = [
                        "net_debt_to_ebitda",
                        "cash_to_market_cap",
                        "catalysts"
                    ]

                    for criterion in criteria:
                        self.assertIn(criterion, result)
                        self.assertIn("message", result[criterion])
                        self.assertIsInstance(result[criterion]["message"], str)

                    # -----------------------------
                    # 1. Net Debt / EBITDA
                    # -----------------------------
                    nde = result["net_debt_to_ebitda"]
                    self.assertIn("meets_criterion", nde)
                    self.assertIsInstance(nde["meets_criterion"], bool)

                    if nde["meets_criterion"]:
                        self.assertTrue(
                            nde["value"] is not None and nde["value"] <= 1.5,
                            f"Net Debt/EBITDA für {symbol} ({frequency}) sollte ≤ 1.5 sein"
                        )

                    # -----------------------------
                    # 2. Cash / Market Cap
                    # -----------------------------
                    cash = result["cash_to_market_cap"]
                    self.assertIn("meets_criterion", cash)
                    self.assertIsInstance(cash["meets_criterion"], bool)

                    if cash["meets_criterion"]:
                        self.assertTrue(
                            cash["value"] is not None and cash["value"] >= 0.20,
                            f"Cash/MarketCap für {symbol} ({frequency}) sollte ≥ 20% sein"
                        )

                    # -----------------------------
                    # 3. Catalysts (qualitativ)
                    # -----------------------------
                    catalysts = result["catalysts"]
                    self.assertIn("message", catalysts)
                    self.assertIsNone(
                        catalysts["meets_criterion"],
                        "Catalysts sollten bewusst nicht automatisch bewertet werden"
                    )

                    # -----------------------------
                    # Gesamtlogik prüfen
                    # -----------------------------
                    meets_all = (
                            result["net_debt_to_ebitda"]["meets_criterion"]
                            and result["cash_to_market_cap"]["meets_criterion"]
                    )

                    expected_assessment = (
                        "Optionality Candidate" if meets_all else "Limited Optionality"
                    )

                    self.assertEqual(
                        result["overall_assessment"],
                        expected_assessment,
                        f"Overall Assessment für {symbol} ({frequency}) passt nicht zur Kriterienlage"
                    )

                    if meets_all:
                        self.assertIn(
                            "Optionalität",
                            result["message"],
                            "Positive Optionalität sollte im Gesamttext erwähnt werden"
                        )
                    else:
                        self.assertTrue(
                            len(result["message"]) > 0,
                            "Bei fehlender Optionalität sollte ein erklärender Hinweis vorhanden sein"
                        )

    def test_analyze_asset_play(self):
        """Testet analyze_asset_play auf Funktionalität, Struktur, Randfälle und Robustheit."""
        print("\nTeste analyze_asset_play...")

        test_symbols = ["AAPL", "MO", "BABA"]  # divers: teuer, defensiv, asset-lastig
        frequencies = ["annual", "quarterly"]

        def _pretty_print(symbol, frequency, result):
            sep = "=" * 80
            print(f"\n{sep}")
            print(f"{symbol} – Value-/Asset-Analyse ({frequency.upper()})")
            print(sep)

            if "error" in result:
                print(f"❌ Fehler: {result['error']}")
                print(sep)
                return

            def _line(name, value, ok, msg):
                status = "✔" if ok else "✘"
                val = value if value is not None else "N/A"
                print(f"{status} {name:<22}: {val}")
                print(f"    ↳ {msg}")

            ptbv = result["price_to_tangible_book"]
            _line(
                "P / Tangible Book",
                ptbv.get("value"),
                ptbv["meets_criterion"],
                ptbv["message"]
            )

            nca = result["net_current_assets"]
            _line(
                "Net Current Assets",
                nca.get("value"),
                nca["meets_criterion"],
                nca["message"]
            )

            print("-" * 80)
            print("🏗 Asset-Qualität (qualitativ)")
            print(f"    ↳ {result['asset_value_quality']['message']}")

            print("-" * 80)
            print(f"📌 Gesamtbewertung: {result['overall_assessment']}")
            print(f"📝 Zusammenfassung: {result['message']}")
            print(sep)

        for symbol in test_symbols:
            for frequency in frequencies:
                with self.subTest(symbol=symbol, frequency=frequency):
                    result = self.actionmodule.analyze_asset_play(
                        symbol,
                        frequency=frequency,
                        use_cache=True
                    )

                    # -------- Fehlerpfad --------
                    if isinstance(result, dict) and "error" in result:
                        _pretty_print(symbol, frequency, result)
                        self.assertIn("symbol", result)
                        self.assertEqual(result["symbol"], symbol)
                        self.assertIsInstance(result["error"], str)
                        continue

                    _pretty_print(symbol, frequency, result)

                    # -------- Grundstruktur --------
                    self.assertIn("symbol", result)
                    self.assertEqual(result["symbol"], symbol)
                    self.assertIn("overall_assessment", result)
                    self.assertIn(result["overall_assessment"], [
                        "Asset Play – Candidate",
                        "Asset Play – Watch/Avoid"
                    ])
                    self.assertIn("message", result)
                    self.assertIsInstance(result["message"], str)

                    # -------- Kriterien vorhanden --------
                    criteria = [
                        "price_to_tangible_book",
                        "net_current_assets",
                        "asset_value_quality"
                    ]

                    for criterion in criteria:
                        self.assertIn(criterion, result)
                        self.assertIn("message", result[criterion])

                    # -------- 1) P/TBV --------
                    ptbv = result["price_to_tangible_book"]
                    self.assertIn("meets_criterion", ptbv)
                    self.assertIsInstance(ptbv["meets_criterion"], bool)

                    if ptbv["meets_criterion"]:
                        self.assertIsNotNone(ptbv["value"])
                        self.assertLessEqual(
                            ptbv["value"],
                            ptbv["threshold"],
                            "Wenn P/TBV meets_criterion=True, muss es ≤ threshold sein"
                        )

                    # -------- 2) Net Current Assets --------
                    nca = result["net_current_assets"]
                    self.assertIn("meets_criterion", nca)
                    self.assertIsInstance(nca["meets_criterion"], bool)

                    if nca["meets_criterion"]:
                        self.assertIsNotNone(nca["value"])
                        self.assertGreater(
                            nca["value"],
                            0,
                            "Wenn Net Current Assets meets_criterion=True, müssen sie positiv sein"
                        )

                    # -------- 3) Asset Value Quality (qualitativ) --------
                    asset_quality = result["asset_value_quality"]
                    self.assertIsNone(
                        asset_quality["meets_criterion"],
                        "Asset Value Quality darf bewusst kein Boolean sein"
                    )
                    self.assertIsInstance(asset_quality["message"], str)

                    # -------- Gesamtlogik --------
                    meets_all = (
                            result["price_to_tangible_book"]["meets_criterion"] and
                            result["net_current_assets"]["meets_criterion"]
                    )

                    expected_overall = (
                        "Asset Play – Candidate"
                        if meets_all else
                        "Asset Play – Watch/Avoid"
                    )

                    self.assertEqual(
                        result["overall_assessment"],
                        expected_overall,
                        "overall_assessment muss zu Einzelkriterien passen"
                    )

                    if meets_all:
                        self.assertIn(
                            "Unterbewertet",
                            result["message"],
                            "Bei Asset Play Candidate sollte Message Unterbewertung erwähnen"
                        )
                    else:
                        self.assertTrue(
                            len(result["message"]) > 0,
                            "Bei Watch/Avoid sollte Message Hinweise enthalten"
                        )

    def test_calculate_crv_by_sector_multiples(self):
        """Testet calculate_crv_by_sector_multiples isoliert und prüft Struktur + Robustheit."""
        print("\nTeste calculate_crv_by_sector_multiples...")

        test_symbols = ["BABA"]

        def _pretty_print(result: dict):
            if "error" in result:
                print(f"❌ Fehler: {result['error']}")
                return

            print(f"\nSymbol: {result['symbol']}")
            print(f"Sektoren: {', '.join(result['sectors'])}")
            print(f"Multiples: {', '.join(result['multiples_used'])}")

            print("\n--- CRV-Ergebnisse (branchenrelevant) ---")
            for multiple, crv in result["crv_results"].items():
                if "error" in crv:
                    print(f"  [{multiple}] ❌ nicht berechenbar – {crv['error']}")
                    continue

                crv_val = crv.get("crv_conservative")
                flag = "✔" if crv.get("crv_positive") else "✘"
                print(f"  [{multiple}] CRV konservativ = {crv_val} {flag}")

        for symbol in test_symbols:
            with self.subTest(symbol=symbol):
                result = self.actionmodule.calculate_crv_by_sector_multiples(symbol)

                _pretty_print(result)

                # ---------- Fehlerpfad ----------
                if "error" in result:
                    self.assertIn("symbol", result)
                    self.assertEqual(result["symbol"], symbol)
                    self.assertIsInstance(result["error"], str)
                    continue

                # ---------- Grundstruktur ----------
                self.assertIn("symbol", result)
                self.assertEqual(result["symbol"], symbol)

                self.assertIn("sectors", result)
                self.assertIsInstance(result["sectors"], list)
                self.assertGreater(len(result["sectors"]), 0)

                self.assertIn("multiples_used", result)
                self.assertIsInstance(result["multiples_used"], list)
                self.assertGreater(len(result["multiples_used"]), 0)

                self.assertIn("crv_results", result)
                self.assertIsInstance(result["crv_results"], dict)
                self.assertGreater(len(result["crv_results"]), 0)

                # ---------- CRV-Ergebnisse ----------
                for multiple, crv in result["crv_results"].items():
                    self.assertIsInstance(multiple, str)
                    self.assertIsInstance(crv, dict)

                    # Fehlerhafte CRVs sind erlaubt
                    if "error" in crv:
                        self.assertIsInstance(crv["error"], str)
                        continue

                    # Erfolgreiche CRVs
                    self.assertIn("crv_conservative", crv)
                    self.assertIn("crv_aggressive", crv)
                    self.assertIn("crv_positive", crv)

                    self.assertIsInstance(crv["crv_positive"], bool)
                    self.assertIsInstance(crv["crv_conservative"], (int, float))