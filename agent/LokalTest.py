import simfin as sf
import pandas as pd
import os
from tenacity import retry, stop_after_attempt, wait_fixed

class DataLoader:
    def __init__(self, user_agent="gecen.efe1308@gmail.com"):
        self.user_agent = user_agent
        self.ticker_cache = {}
        self.price_cache = {}
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        sf.set_api_key(api_key="dc0c0be0-5826-44d6-8315-5b542247dfe7")
        sf.set_data_dir('data/')
        self.isin_mapping = pd.read_csv('data/us-companies.csv', sep=';', usecols=['Ticker', 'ISIN'], encoding='utf-8')
        self.isin_mapping = self.isin_mapping.dropna(subset=['Ticker', 'ISIN'])
        print(f"Geladene Spalten: {self.isin_mapping.columns.tolist()}")
        self.isin_mapping.set_index('Ticker', inplace=True)

    def get_ISIN(self, ticker):
        ticker_upper = ticker.upper()
        isin = self.isin_mapping.loc[ticker_upper, 'ISIN'] if ticker_upper in self.isin_mapping.index else None
        if isin is None:
            raise ValueError(f"Kein ISIN für Ticker {ticker} gefunden.")
        return isin

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def get_historical_data(self, symbol):
        try:
            isin = self.get_ISIN(symbol)
            df_companies = sf.load_companies(market='us')
            simfin_id = df_companies[df_companies['ISIN'] == isin]['SimFinId'].iloc[0]
            if pd.isna(simfin_id):
                raise ValueError(f"Kein SimFinId für ISIN {isin} gefunden.")
            df_income = sf.load_income(variant='annual', market='us')
            df_cashflow = sf.load_cashflow(variant='annual', market='us')
            df_balance = sf.load_balance(variant='annual', market='us')
            income_data = df_income[df_income["SimFinId"] == simfin_id]
            cashflow_data = df_cashflow[df_cashflow["SimFinId"] == simfin_id]
            balance_data = df_balance[df_balance["SimFinId"] == simfin_id]
            combined_data = {
                'income': income_data,
                'cashflow': cashflow_data,
                'balance': balance_data
            }
            print(f"Historische Daten für {symbol} erfolgreich abgerufen.")
            print(f"\n--- Daten für {symbol} ---")
            for key, df in combined_data.items():
                print(f"\n{key.upper()} DATA:")
                print(f"Spalten: {df.columns.tolist()}")
                print(f"Anzahl Zeilen: {len(df)}")
                if not df.empty and 'Fiscal Year' in df.columns:
                    print(f"Zeitspanne: {df['Fiscal Year'].min()} bis {df['Fiscal Year'].max()}")
                print(df.head())
            return combined_data
        except Exception as e:
            return {"error": f"Unerwarteter Fehler beim Abrufen der Daten: {str(e)}", "symbol": symbol}

if __name__ == "__main__":
    # Installiere simfin, falls nicht vorhanden
    try:
        import simfin
    except ImportError:
        os.system('pip install simfin')
        import simfin as sf
    loader = DataLoader()
    data = loader.get_historical_data('AAPL')