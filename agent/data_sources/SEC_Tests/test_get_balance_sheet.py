import unittest
import pandas as pd
import yfinance as yf

from data_sources.sec_source import SecSource


FOCUS_LABELS = [
    "Cash And Cash Equivalents",
    "Other Short Term Investments",
    "Cash Cash Equivalents And Short Term Investments",
    "Total Debt",
    "Long Term Debt",
    "Short Term Debt",
    "Current Debt",
    "Long Term Debt And Capital Lease Obligation",
    "Net Debt",
    "Invested Capital",
    "Goodwill",
    "Other Intangible Assets",
    "Stockholders Equity",
    "Total Assets",
    "Total Liabilities Net Minority Interest",
    "Tangible Book Value",
    "Net Tangible Assets",
]


def print_focused_balance_sheet_comparison(symbol, frequency, yf_bs, sec_bs):
    print(f"\n\n===== FOCUS BALANCE SHEET CHECK: {symbol} / {frequency.upper()} =====")

    if isinstance(sec_bs, dict):
        print("SEC Fehler:")
        print(sec_bs)
        return

    print("Yahoo neueste Periode:", yf_bs.columns[0])
    print("SEC neueste Periode:  ", sec_bs.columns[0])

    print("\n==============================")
    print("FOCUS LABEL PRESENCE")
    print("==============================")

    for label in FOCUS_LABELS:
        print(
            f"{label:<55} | "
            f"yf={label in yf_bs.index:<5} | "
            f"sec={label in sec_bs.index:<5}"
        )

    print("\n==============================")
    print("FOCUS VALUE COMPARISON")
    print("==============================")

    for label in FOCUS_LABELS:
        yf_exists = label in yf_bs.index
        sec_exists = label in sec_bs.index

        yf_value = yf_bs.loc[label].iloc[0] if yf_exists else None
        sec_value = sec_bs.loc[label].iloc[0] if sec_exists else None

        if pd.notna(yf_value) and pd.notna(sec_value):
            diff = sec_value - yf_value
        else:
            diff = None

        print(
            f"{label:<55} | "
            f"yf={yf_value} | "
            f"sec={sec_value} | "
            f"diff={diff}"
        )


class TestBalanceSheetFocusedComparison(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sec = SecSource(
            user_agent="deine-email@example.com"
        )

    def _run_symbol_test(self, symbol):
        for frequency in ["annual", "quarterly"]:
            print(f"\n\n########## TEST: {symbol} ({frequency}) ##########")

            ticker = yf.Ticker(symbol)

            yf_bs = (
                ticker.balance_sheet
                if frequency == "annual"
                else ticker.quarterly_balance_sheet
            )

            sec_bs = self.sec.get_balance_sheet(
                symbol=symbol,
                frequency=frequency,
                use_cache=False,
                scope="core",
            )

            self.assertIsInstance(
                yf_bs,
                pd.DataFrame,
                f"Yahoo Balance Sheet ist kein DataFrame für {symbol} ({frequency})"
            )
            self.assertFalse(
                yf_bs.empty,
                f"Yahoo Balance Sheet ist leer für {symbol} ({frequency})"
            )

            self.assertIsInstance(
                sec_bs,
                pd.DataFrame,
                f"SEC Balance Sheet ist kein DataFrame für {symbol} ({frequency}): {sec_bs}"
            )
            self.assertFalse(
                sec_bs.empty,
                f"SEC Balance Sheet ist leer für {symbol} ({frequency})"
            )

            print_focused_balance_sheet_comparison(
                symbol=symbol,
                frequency=frequency,
                yf_bs=yf_bs,
                sec_bs=sec_bs,
            )

    def test_aapl_balance_sheet(self):
        self._run_symbol_test("AAPL")

    def test_mo_balance_sheet(self):
        self._run_symbol_test("MO")

    def test_pypl_balance_sheet(self):
        self._run_symbol_test("PYPL")

    def test_ko_balance_sheet(self):
        self._run_symbol_test("KO")

    def test_msft_balance_sheet(self):
        self._run_symbol_test("MSFT")


if __name__ == "__main__":
    unittest.main()