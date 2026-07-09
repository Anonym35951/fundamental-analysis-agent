# compact_compare_stock_financials.py

import pandas as pd
import yfinance as yf
from data_sources.sec_source import SecSource


def compare_financials(symbol="AAPL"):
    sec = SecSource(user_agent="your_email@example.com")

    print(f"\n========== {symbol} COMPARISON ==========\n")

    # SEC
    sec_df = sec.get_stock_financials(
        symbol=symbol,
        frequency="annual",
        use_cache=False,
        scope="core"
    )

    # Yahoo
    yf_df = yf.Ticker(symbol).financials

    if isinstance(sec_df, dict) and "error" in sec_df:
        print("SEC ERROR:", sec_df["error"])
        return

    if yf_df is None or yf_df.empty:
        print("Yahoo Finance DataFrame leer.")
        return

    print("SEC Shape:", sec_df.shape)
    print("YF Shape :", yf_df.shape)

    print("\n--- SEC Columns ---")
    print(list(sec_df.columns))

    print("\n--- YF Columns ---")
    print(list(yf_df.columns))

    print("\n--- SEC Index Labels ---")
    print(list(sec_df.index))

    print("\n--- YF Index Labels ---")
    print(list(yf_df.index))

    # Überschneidung prüfen
    common_labels = set(sec_df.index).intersection(set(yf_df.index))
    only_sec = set(sec_df.index) - set(yf_df.index)
    only_yf = set(yf_df.index) - set(sec_df.index)

    print("\n--- Common Labels Count ---")
    print(len(common_labels))

    print("\n--- Only SEC Labels ---")
    print(list(sorted(only_sec))[:30])

    print("\n--- Only Yahoo Labels ---")
    print(list(sorted(only_yf))[:30])


if __name__ == "__main__":
    compare_financials("AAPL")