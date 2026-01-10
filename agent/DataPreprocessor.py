# (2.) Klassen oder Module, die Daten säubern, transformieren und Feature Engineering betreiben.

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks
from datetime import datetime, timedelta

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.printed_inflation_periods = set()

    def preprocess_stock_data(self, data):
        """Verarbeitet historische Kursdaten (z.B. Bereinigung)"""
        if data is None or not isinstance(data, pd.DataFrame):
            raise ValueError("Ungültige Kursdaten: Daten müssen ein pandas DataFrame sein und dürfen nicht None sein.")
        data = data.copy()
        data = data.dropna()
        if data.empty:
            raise ValueError("Nach der Bereinigung sind keine Kursdaten mehr vorhanden.")
        return data

    def preprocess_balance_sheet(self, balance_sheet):
        """Verarbeitet Bilanzdaten (z.B. Bereinigung)"""
        if balance_sheet is None or not isinstance(balance_sheet, pd.DataFrame):
            raise ValueError("Ungültige Bilanzdaten: Daten müssen ein pandas DataFrame sein und dürfen nicht None sein.")
        return balance_sheet.copy()

    def preprocess_edgar_data(self, edgar_data):
        """Verarbeitet EDGAR-Daten (z.B. Extraktion relevanter Kennzahlen)"""
        if edgar_data is None:
            raise ValueError("Ungültige EDGAR-Daten: Daten dürfen nicht None sein.")
        return edgar_data

    def calculate_technical_indicators(self, data, interval="1d"):
        """Berechnet technische Indikatoren für die Kursdaten, angepasst an die Zeitebene."""
        if data is None or not isinstance(data, pd.DataFrame):
            raise ValueError("Ungültige Kursdaten für technische Indikatoren: Daten müssen ein pandas DataFrame sein und dürfen nicht None sein.")
        data = data.copy()

        # Setze Fenstergrößen basierend auf der Zeitebene
        if interval == "4h":
            ma_short, ma_medium, ma_long = 5, 20, 200 # 20 Stunden, 80 Stunden, 800 Stunden
            stoch_period, vol_period = 14 * 4, 20 * 4 # 14 Tage (56 Stunden), 20 Tage (80 Stunden)
        elif interval == "1d":
            ma_short, ma_medium, ma_long = 5, 20, 200 # 5 Tage, 20 Tage, 200 Tage
            stoch_period, vol_period = 14, 20 # 14 Tage, 20 Tage
        elif interval == "1wk":
            ma_short, ma_medium, ma_long = 5, 20, 50 # 5 Wochen, 20 Wochen, 50 Wochen (ca. 1 Jahr, da 200 Wochen zu lang sind)
            stoch_period, vol_period = 14, 20 # 14 Wochen, 20 Wochen
        else:
            ma_short, ma_medium, ma_long = 5, 20, 200
            stoch_period, vol_period = 14, 20

        # Gleitende Durchschnitte (zur Trendbestimmung)
        data.loc[:, "MA_Short"] = data["Close"].rolling(window=ma_short).mean()
        data.loc[:, "MA_Medium"] = data["Close"].rolling(window=ma_medium).mean()
        data.loc[:, "MA_Long"] = data["Close"].rolling(window=ma_long).mean()

        #Stochastic Oscillator (%K, angepasst an die Zeitebene)
        low_period = data["Low"].rolling(window=stoch_period).min()
        high_period = data["High"].rolling(window=stoch_period).max()
        data.loc[:, "Stochastic_K"] = 100 * (data["Close"] - low_period) / (high_period - low_period)

        # Handelsvolumen (normalisiert)
        data["Volume"] = data["Volume"].astype(float)
        data.loc[:, "Volume"] = data["Volume"] / data["Volume"].rolling(window=vol_period).mean()

        # Entferne fehlende Werte
        data = data.dropna()
        if data.empty:
            raise ValueError("Nach der Berechnung der technischen Indikatoren sind keine Daten mehr vorhanden.")
        return data

    def calculate_fibonacci_retracements(self, data, interval="1d"):
        """Berechnet Fibonacci-Retracements"""
        data = data.copy()
        if interval == "4h":
            distance = 5
        elif interval == "1d":
            distance = 10
        elif interval == "1wk":
            distance = 3
        else:
            distance = 10
        peaks, _ = find_peaks(data["Close"], distance=distance)
        troughs, _ = find_peaks(-data["Close"], distance=distance)
        data["Fib_382"] = np.nan
        data["Fib_50"] = np.nan
        data["Fib_618"] = np.nan
        data["Fib_1618"] = np.nan
        for i in range(len(peaks) - 1):
            peak_idx = peaks[i]
            next_trough_idx = troughs[troughs > peak_idx][0] if len(troughs[troughs > peak_idx]) > 0 else len(data) - 1
            high = data["Close"].iloc[peak_idx]
            low = data["Close"].iloc[next_trough_idx]
            diff = high - low
            data.iloc[peak_idx:next_trough_idx + 1, data.columns.get_loc("Fib_382")] = high - diff * 0.382
            data.iloc[peak_idx:next_trough_idx + 1, data.columns.get_loc("Fib_50")] = high - diff * 0.5
            data.iloc[peak_idx:next_trough_idx + 1, data.columns.get_loc("Fib_618")] = high - diff * 0.618
            data.iloc[peak_idx:next_trough_idx + 1, data.columns.get_loc("Fib_1618")] = high + diff * 0.618
        for i in range(len(troughs) - 1):
            trough_idx = troughs[i]
            next_peak_idx = peaks[peaks > trough_idx][0] if len(peaks[peaks > trough_idx]) > 0 else len(data) - 1
            low = data["Close"].iloc[trough_idx]
            high = data["Close"].iloc[next_peak_idx]
            diff = high - low
            data.iloc[trough_idx:next_peak_idx + 1, data.columns.get_loc("Fib_382")] = low + diff * 0.382
            data.iloc[trough_idx:next_peak_idx + 1, data.columns.get_loc("Fib_50")] = low + diff * 0.5
            data.iloc[trough_idx:next_peak_idx + 1, data.columns.get_loc("Fib_618")] = low + diff * 0.618
            data.iloc[trough_idx:next_peak_idx + 1, data.columns.get_loc("Fib_1618")] = low + diff * 1.618
        return data

    def identify_elliott_waves(self, data, interval="1d"):
        """Identifiziert Elliott-Wellen basierend auf Fibonacci-Retracements und verfeinerten Regeln, angepasst an Zeitebene."""
        data = data.copy()
        if interval == "4h":
            distance = 5
        elif interval == "1d":
            distance = 10
        elif interval == "1wk":
            distance = 3
        else:
            distance = 10
        peaks, _ = find_peaks(data["Close"], distance=distance)
        troughs, _ = find_peaks(-data["Close"], distance=distance)
        data["Trend"] = np.where(data["Close"] > data["MA_Long"], 1, -1)
        data["Wave"] = np.nan
        points = sorted(list(peaks) + list(troughs))
        wave_count = 0
        i = 0
        while i < len(points) - 5:
            segment = data.iloc[points[i]:points[i + 5]]
            if len(segment) < 5:
                i += 1
                continue
            trend = segment["Trend"].iloc[0]
            if trend == 1:
                w1_start = segment["Close"].iloc[0]
                w1_end = segment["Close"].iloc[1]
                if w1_end <= w1_start:
                    i += 1
                    continue
                w2_end = segment["Close"].iloc[2]
                retracement = (w1_end - w2_end) / (w1_end - w1_start)
                if not (0.5 <= retracement <= 0.618) or w2_end < w1_start:
                    i += 1
                    continue
                w3_end = segment["Close"].iloc[3]
                if w3_end <= w2_end:
                    i += 1
                    continue
                w3_length = w3_end - w2_end
                w1_length = w1_end - w1_start
                if w3_length < w1_length:
                    i += 1
                    continue
                w4_end = segment["Close"].iloc[4]
                retracement_4 = (w3_end - w4_end) / (w3_end - w2_end)
                if not (0.382 <= retracement_4 <= 0.5) or w4_end < w1_end:
                    i += 1
                    continue
                w5_end = segment["Close"].iloc[5]
                if w5_end <= w4_end:
                    i += 1
                    continue
                w5_length = w5_end - w4_end
                if w5_length > w3_length:
                    i += 1
                    continue
                data.iloc[points[i], data.columns.get_loc("Wave")] = 1
                data.iloc[points[i + 1], data.columns.get_loc("Wave")] = 2
                data.iloc[points[i + 2], data.columns.get_loc("Wave")] = 3
                data.iloc[points[i + 3], data.columns.get_loc("Wave")] = 4
                data.iloc[points[i + 4], data.columns.get_loc("Wave")] = 5
                wave_count += 1
                i += 5
            else:
                w1_start = segment["Close"].iloc[0]
                w1_end = segment["Close"].iloc[1]
                if w1_end >= w1_start:
                    i += 1
                    continue
                w2_end = segment["Close"].iloc[2]
                retracement = (w2_end - w1_end) / (w1_start - w1_end)
                if not (0.5 <= retracement <= 0.618) or w2_end > w1_start:
                    i += 1
                    continue
                w3_end = segment["Close"].iloc[3]
                if w3_end >= w2_end:
                    i += 1
                    continue
                w3_length = w2_end - w3_end
                w1_length = w1_start - w1_end
                if w3_length < w1_length:
                    i += 1
                    continue
                w4_end = segment["Close"].iloc[4]
                retracement_4 = (w4_end - w3_end) / (w2_end - w3_end)
                if not (0.382 <= retracement_4 <= 0.5) or w4_end > w1_end:
                    i += 1
                    continue
                w5_end = segment["Close"].iloc[5]
                if w5_end >= w4_end:
                    i += 1
                    continue
                w5_length = w4_end - w5_end
                if w5_length > w3_length:
                    i += 1
                    continue
                data.iloc[points[i], data.columns.get_loc("Wave")] = 1
                data.iloc[points[i + 1], data.columns.get_loc("Wave")] = 2
                data.iloc[points[i + 2], data.columns.get_loc("Wave")] = 3
                data.iloc[points[i + 3], data.columns.get_loc("Wave")] = 4
                data.iloc[points[i + 4], data.columns.get_loc("Wave")] = 5
                wave_count += 1
                i += 5
        i = 0
        while i < len(points) - 3:
            segment = data.iloc[points[i]:points[i + 3]]
            if len(segment) < 3:
                i += 1
                continue
            trend = segment["Trend"].iloc[0]
            if trend == 1:
                wA_start = segment["Close"].iloc[0]
                wA_end = segment["Close"].iloc[1]
                if wA_end >= wA_start:
                    i += 1
                    continue
                wB_end = segment["Close"].iloc[2]
                retracement = (wB_end - wA_end) / (wA_start - wA_end)
                if not (0.5 <= retracement <= 0.618):
                    i += 1
                    continue
                wC_end = segment["Close"].iloc[3]
                if wC_end >= wB_end:
                    i += 1
                    continue
                data.iloc[points[i], data.columns.get_loc("Wave")] = 6
                data.iloc[points[i + 1], data.columns.get_loc("Wave")] = 7
                data.iloc[points[i + 2], data.columns.get_loc("Wave")] = 8
                i += 3
            else:
                wA_start = segment["Close"].iloc[0]
                wA_end = segment["Close"].iloc[1]
                if wA_end <= wA_start:
                    i += 1
                    continue
                wB_end = segment["Close"].iloc[2]
                retracement = (wA_end - wB_end) / (wA_end - wA_start)
                if not (0.5 <= retracement <= 0.618):
                    i += 1
                    continue
                wC_end = segment["Close"].iloc[3]
                if wC_end <= wB_end:
                    i += 1
                    continue
                data.iloc[points[i], data.columns.get_loc("Wave")] = 6
                data.iloc[points[i + 1], data.columns.get_loc("Wave")] = 7
                data.iloc[points[i + 2], data.columns.get_loc("Wave")] = 8
                i += 3
        data["Wave"] = data["Wave"].fillna(0)
        return data

    def preprocess_stock_data_for_ml(self, data, interval="1d"):
        """Verarbeitet historische Kursdaten für ein CNN-LSTM-Modell, angepasst an die Zeitebene"""
        if data is None or not isinstance(data, pd.DataFrame):
            raise ValueError("Ungültige Kursdaten für ML: Daten müssen ein pandas DataFrame sein und dürfen nicht None sein.")

        # Setze Timestamps basierend auf Zeitebene
        if interval == "4h":
            timesteps = 240
        elif interval == "1d":
            timesteps = 60
        elif interval == "1wk":
            timesteps = 20
        else:
            timesteps = 60

        #Bereinigung
        data = self.preprocess_stock_data(data)

        #technische indikatoren berechnen
        data = self.calculate_technical_indicators(data, interval=interval)

        #Fib-Retracements berechnen
        data = self.calculate_fibonacci_retracements(data, interval=interval)

        #elliott-wellen identifizieren
        data = self.identify_elliott_waves(data, interval=interval)

        #Features auswählen
        features = data[["Close", "MA_Short", "MA_Medium", "Stochastic_K", "Volume", "Fib_382", "Fib_50", "Fib_618", "Fib_1618"]].values
        labels = data["Wave"].values

        # Skalierung der Features
        features_scaled = self.scaler.fit_transform(features)

        # Erstelle ein CNN-LSTM
        X, y = [], []
        for i in range(len(features_scaled) - timesteps):
            X.append(features_scaled[i:i + timesteps])
            y.append(labels[i + timesteps])
        return np.array(X), np.array(y), data

    def get_quarterly_inflation_rate(self, cpi_data, start_date, end_date, use_cache=True):

        # Berechne den Zeitraum für die Inflationsdaten
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        if (end - start).days < 85: #Etwas weniger als 90 Tage, um kleine Abweichungen zu tolerieren
            raise ValueError("Zeitraum muss mindestens ein Quartal (ca. 90 Tage) umfassen.")

        # Hole Inflationsdaten für den Zeitraum plus einen zusätzichen Monat davor
        earliest_date = start - timedelta(days=31)
        cpi_data = [d for d in cpi_data if earliest_date <= datetime.strptime(d['date'], '%Y-%m-%d') <= end]
        cpi_data = sorted(cpi_data, key=lambda x: x['date'])

        # Berechne monatliche Inflationsraten
        monthly_rates = []
        for i in range(len(cpi_data) - 3, len(cpi_data)):
            if i < 1:
                raise ValueError("Nicht genügend CPI-Daten für die Berechnung der monatlichen Inflationsraten.")
            previous_cpi = cpi_data[i-1]['value']
            current_cpi = cpi_data[i]['value']
            monthly_rate = (current_cpi - previous_cpi) / previous_cpi
            monthly_rates.append(monthly_rate)

        # Berechne die kumulative Inflationsrate für das Quartal
        cumulative_rate = 1.0
        for rate in monthly_rates:
            cumulative_rate *= (1 + rate)
        quarterly_inflation_rate = (cumulative_rate - 1) * 100
        quarterly_inflation_rate = round(quarterly_inflation_rate, 2)
        return {
            'quarterly_inflation_rate': quarterly_inflation_rate,
            'monthly_rates': monthly_rates,
            'cpi_data': cpi_data
        }