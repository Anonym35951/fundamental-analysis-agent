import unittest
import pandas as pd
import yfinance as yf

from data_sources.sec_source import SecSource


FOCUS_LABELS = [
    "Operating Cash Flow",
    "Cash Flow From Continuing Operating Activities",
    "Capital Expenditure",
    "Purchase Of PPE",
    "Capital Expenditure Reported",
    "Free Cash Flow",
    "Dividends Paid",
    "Repurchase Of Capital Stock",
    "Net Cash From Investing",
    "Net Cash From Financing",
    "Acquisitions",
    "Investments In Securities",
    "Debt Issuance",
    "Debt Repayment",
    "Stock Issuance",
]


def print_focused_cashflow_comparison(symbol, frequency, yf_cf, sec_cf):
    print(f"\n\n===== FOCUS CASHFLOW CHECK: {symbol} / {frequency.upper()} =====")

    if isinstance(sec_cf, dict):
        print("SEC Fehler:")
        print(sec_cf)
        return

    print("Yahoo neueste Periode:", yf_cf.columns[0])
    print("SEC neueste Periode:  ", sec_cf.columns[0])

    print("\n==============================")
    print("FOCUS LABEL PRESENCE")
    print("==============================")

    for label in FOCUS_LABELS:
        print(
            f"{label:<60} | "
            f"yf={label in yf_cf.index:<5} | "
            f"sec={label in sec_cf.index:<5}"
        )

    print("\n==============================")
    print("FOCUS VALUE COMPARISON")
    print("==============================")

    for label in FOCUS_LABELS:
        yf_exists = label in yf_cf.index
        sec_exists = label in sec_cf.index

        yf_value = yf_cf.loc[label].iloc[0] if yf_exists else None
        sec_value = sec_cf.loc[label].iloc[0] if sec_exists else None

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


class TestCashflowFocusedComparison(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sec = SecSource(
            user_agent="deine-email@example.com"
        )

    def _run_symbol_test(self, symbol):
        for frequency in ["annual", "quarterly"]:
            print(f"\n\n########## TEST: {symbol} ({frequency}) ##########")

            ticker = yf.Ticker(symbol)

            yf_cf = (
                ticker.cashflow
                if frequency == "annual"
                else ticker.quarterly_cashflow
            )

            sec_cf = self.sec.get_cashflow_statement(
                symbol=symbol,
                frequency=frequency,
                use_cache=False,
                scope="core",
            )

            self.assertIsInstance(
                yf_cf,
                pd.DataFrame,
                f"Yahoo Cashflow Statement ist kein DataFrame für {symbol} ({frequency})"
            )
            self.assertFalse(
                yf_cf.empty,
                f"Yahoo Cashflow Statement ist leer für {symbol} ({frequency})"
            )

            self.assertIsInstance(
                sec_cf,
                pd.DataFrame,
                f"SEC Cashflow Statement ist kein DataFrame für {symbol} ({frequency}): {sec_cf}"
            )
            self.assertFalse(
                sec_cf.empty,
                f"SEC Cashflow Statement ist leer für {symbol} ({frequency})"
            )

            print_focused_cashflow_comparison(
                symbol=symbol,
                frequency=frequency,
                yf_cf=yf_cf,
                sec_cf=sec_cf,
            )

    def test_aapl_cashflow_statement(self):
        self._run_symbol_test("AAPL")

    def test_mo_cashflow_statement(self):
        self._run_symbol_test("MO")

    def test_pypl_cashflow_statement(self):
        self._run_symbol_test("PYPL")

    def test_ko_cashflow_statement(self):
        self._run_symbol_test("KO")

    def test_msft_cashflow_statement(self):
        self._run_symbol_test("MSFT")

    def test_baba_cashflow_statement(self):
        self._run_symbol_test("BABA")


if __name__ == "__main__":
    unittest.main()