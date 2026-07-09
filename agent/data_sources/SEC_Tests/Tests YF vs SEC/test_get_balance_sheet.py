import unittest
import pandas as pd
import yfinance as yf

from data_sources.sec_source import SecSource


FOCUS_LABELS = [
    "Cash And Cash Equivalents",
    "Other Short Term Investments",
    "Cash Cash Equivalents And Short Term Investments",

    "Accounts Receivable",
    "Inventory",

    "Current Assets",
    "Total Assets",

    "Accounts Payable",
    "Current Liabilities",

    "Long Term Debt",
    "Short Term Debt",
    "Current Debt",
    "Total Debt",

    "Total Liabilities Net Minority Interest",

    "Stockholders Equity",
    "Common Stock Equity",
    "Total Stockholders Equity",

    "Net Debt",

    "Goodwill",
    "Other Intangible Assets",

    "Tangible Book Value",
    "Net Tangible Assets",
    "Working Capital",
]


FINANCIAL_TICKERS = {
    "BAC",
    "JPM",
    "PGR",
    "BRK-B",
}


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
            f"{label:<60} | "
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
            f"{label:<60} | "
            f"yf={yf_value} | "
            f"sec={sec_value} | "
            f"diff={diff}"
        )


class TestBalanceSheetFocusedComparisonExtended(unittest.TestCase):

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

            #
            # Financials:
            # Net Debt / Working Capital / Invested Capital
            # sollen bewusst NICHT berechnet werden.
            #

            if symbol.upper() in FINANCIAL_TICKERS:
                print("\n--- FINANCIAL COMPANY CHECK ---")

                self.assertNotIn(
                    "Net Debt",
                    sec_bs.index,
                    f"{symbol}: Net Debt sollte für Financials nicht berechnet werden."
                )

                self.assertNotIn(
                    "Working Capital",
                    sec_bs.index,
                    f"{symbol}: Working Capital sollte für Financials nicht berechnet werden."
                )

                self.assertNotIn(
                    "Invested Capital",
                    sec_bs.index,
                    f"{symbol}: Invested Capital sollte für Financials nicht berechnet werden."
                )

                print("✓ Net Debt nicht vorhanden")
                print("✓ Working Capital nicht vorhanden")
                print("✓ Invested Capital nicht vorhanden")

    #
    # Tech
    #

    def test_aapl_balance_sheet(self):
        self._run_symbol_test("AAPL")

    def test_msft_balance_sheet(self):
        self._run_symbol_test("MSFT")

    def test_meta_balance_sheet(self):
        self._run_symbol_test("META")

    def test_nvda_balance_sheet(self):
        self._run_symbol_test("NVDA")

    #
    # Industrials
    #

    def test_cat_balance_sheet(self):
        self._run_symbol_test("CAT")

    def test_de_balance_sheet(self):
        self._run_symbol_test("DE")

    #
    # Consumer / Staples
    #

    def test_ko_balance_sheet(self):
        self._run_symbol_test("KO")

    def test_mo_balance_sheet(self):
        self._run_symbol_test("MO")

    #
    # Banks / Financials
    #

    def test_jpm_balance_sheet(self):
        self._run_symbol_test("JPM")

    def test_bac_balance_sheet(self):
        self._run_symbol_test("BAC")

    #
    # Insurance / Holding
    #

    def test_brk_b_balance_sheet(self):
        self._run_symbol_test("BRK-B")

    def test_pgr_balance_sheet(self):
        self._run_symbol_test("PGR")


if __name__ == "__main__":
    unittest.main()