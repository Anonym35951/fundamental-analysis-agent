import unittest
import pandas as pd
import yfinance as yf

from data_sources.sec_source import SecSource


FOCUS_LABELS = [
    "Total Revenue",
    "Revenue",
    "Cost Of Revenue",
    "Gross Profit",
    "Operating Income",
    "EBIT",
    "EBITDA",
    "Interest Expense",
    "Net Income",
    "Net Income Common Stockholders",
    "Diluted EPS",
    "Basic EPS",
    "Depreciation And Amortization",
    "Depreciation Depletion And Amortization",
]


def print_focused_income_statement_comparison(symbol, frequency, yf_is, sec_is):
    print(f"\n\n===== FOCUS INCOME STATEMENT CHECK: {symbol} / {frequency.upper()} =====")

    if isinstance(sec_is, dict):
        print("SEC Fehler:")
        print(sec_is)
        return

    print("Yahoo neueste Periode:", yf_is.columns[0])
    print("SEC neueste Periode:  ", sec_is.columns[0])

    print("\n==============================")
    print("FOCUS LABEL PRESENCE")
    print("==============================")

    for label in FOCUS_LABELS:
        print(
            f"{label:<60} | "
            f"yf={label in yf_is.index:<5} | "
            f"sec={label in sec_is.index:<5}"
        )

    print("\n==============================")
    print("FOCUS VALUE COMPARISON")
    print("==============================")

    for label in FOCUS_LABELS:
        yf_exists = label in yf_is.index
        sec_exists = label in sec_is.index

        yf_value = yf_is.loc[label].iloc[0] if yf_exists else None
        sec_value = sec_is.loc[label].iloc[0] if sec_exists else None

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


class TestIncomeStatementFocusedComparison(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sec = SecSource(
            user_agent="deine-email@example.com"
        )

    def _run_symbol_test(self, symbol):
        for frequency in ["annual", "quarterly"]:
            print(f"\n\n########## TEST: {symbol} ({frequency}) ##########")

            ticker = yf.Ticker(symbol)

            yf_is = (
                ticker.financials
                if frequency == "annual"
                else ticker.quarterly_financials
            )

            sec_is = self.sec.get_stock_financials(
                symbol=symbol,
                frequency=frequency,
                use_cache=False,
                scope="core",
            )

            self.assertIsInstance(
                yf_is,
                pd.DataFrame,
                f"Yahoo Income Statement ist kein DataFrame für {symbol} ({frequency})"
            )
            self.assertFalse(
                yf_is.empty,
                f"Yahoo Income Statement ist leer für {symbol} ({frequency})"
            )

            self.assertIsInstance(
                sec_is,
                pd.DataFrame,
                f"SEC Income Statement ist kein DataFrame für {symbol} ({frequency}): {sec_is}"
            )
            self.assertFalse(
                sec_is.empty,
                f"SEC Income Statement ist leer für {symbol} ({frequency})"
            )

            print_focused_income_statement_comparison(
                symbol=symbol,
                frequency=frequency,
                yf_is=yf_is,
                sec_is=sec_is,
            )

    def test_aapl_income_statement(self):
        self._run_symbol_test("AAPL")

    def test_msft_income_statement(self):
        self._run_symbol_test("MSFT")

    def test_meta_income_statement(self):
        self._run_symbol_test("META")

    def test_amzn_income_statement(self):
        self._run_symbol_test("AMZN")

    def test_ko_income_statement(self):
        self._run_symbol_test("KO")

    def test_mo_income_statement(self):
        self._run_symbol_test("MO")

    def test_jpm_income_statement(self):
        self._run_symbol_test("JPM")

    def test_nvda_income_statement(self):
        self._run_symbol_test("NVDA")


if __name__ == "__main__":
    unittest.main()