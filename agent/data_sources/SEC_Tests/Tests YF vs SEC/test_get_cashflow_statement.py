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


FINANCIAL_SYMBOLS = {
    "JPM",
    "BAC",
    "BRK-B",
    "PGR",
}


HIGH_VOLATILITY_CASHFLOW_SYMBOLS = {
    "NVDA",
}

VALID_QUALITY_STATES = {
    "HIGH",
    "MEDIUM",
    "LOW",
    "NONE",
    "PARTIAL_HISTORY",
    None,
}


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

    print("\n==============================")
    print("SEC ATTRIBUTES")
    print("==============================")

    attrs_to_check = [
        "capex_quality",
        "capex_latest_date",
        "capex_source_tag",
        "capex_warning",

        "fcf_quality",
        "fcf_latest_date",
        "fcf_source",
        "fcf_warning",
    ]

    for attr in attrs_to_check:
        print(f"{attr:<30} -> {sec_cf.attrs.get(attr)}")


class TestCashflowFocusedComparisonExtended(unittest.TestCase):

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
                f"Yahoo Cashflow ist kein DataFrame für {symbol} ({frequency})"
            )

            self.assertFalse(
                yf_cf.empty,
                f"Yahoo Cashflow ist leer für {symbol} ({frequency})"
            )

            self.assertIsInstance(
                sec_cf,
                pd.DataFrame,
                (
                    f"SEC Cashflow ist kein DataFrame "
                    f"für {symbol} ({frequency}): {sec_cf}"
                )
            )

            self.assertFalse(
                sec_cf.empty,
                f"SEC Cashflow ist leer für {symbol} ({frequency})"
            )

            self.assertIn(
                "Operating Cash Flow",
                sec_cf.index,
                (
                    f"Operating Cash Flow fehlt "
                    f"für {symbol} ({frequency})"
                )
            )

            if "Capital Expenditure" in sec_cf.index:

                capex_series = sec_cf.loc["Capital Expenditure"].dropna()

                if not capex_series.empty:

                    positive_capex = capex_series[capex_series > 0]

                    self.assertTrue(
                        positive_capex.empty,
                        (
                            f"CapEx enthält positive Werte "
                            f"für {symbol} ({frequency})"
                        )
                    )

            if (
                "Operating Cash Flow" in sec_cf.index
                and "Capital Expenditure" in sec_cf.index
                and "Free Cash Flow" in sec_cf.index
            ):

                ocf = sec_cf.loc["Operating Cash Flow"]
                capex = sec_cf.loc["Capital Expenditure"]
                fcf = sec_cf.loc["Free Cash Flow"]

                derived_fcf = ocf + capex

                common_cols = (
                    derived_fcf.dropna().index
                    .intersection(fcf.dropna().index)
                )

                for col in common_cols:

                    reported = fcf[col]
                    derived = derived_fcf[col]

                    if pd.isna(reported) or pd.isna(derived):
                        continue

                    tolerance = max(abs(reported) * 0.05, 1)

                    self.assertAlmostEqual(
                        reported,
                        derived,
                        delta=tolerance,
                        msg=(
                            f"FCF mismatch für "
                            f"{symbol} ({frequency}) "
                            f"am {col}"
                        )
                    )

            #
            # Quarterly YTD Erkennung
            #
            # Banken, Versicherungen und hochvolatile Cashflow-Unternehmen
            # werden ausgeschlossen.
            #

            if (
                frequency == "quarterly"
                and symbol not in FINANCIAL_SYMBOLS
                and symbol not in HIGH_VOLATILITY_CASHFLOW_SYMBOLS
                and "Operating Cash Flow" in sec_cf.index
            ):

                ocf_series = sec_cf.loc["Operating Cash Flow"].dropna()

                if len(ocf_series) >= 3:

                    median_val = ocf_series.abs().median()
                    max_val = ocf_series.abs().max()

                    if median_val > 0:

                        ratio = max_val / median_val

                        self.assertLess(
                            ratio,
                            20,
                            (
                                f"Möglicherweise unnormalisierte "
                                f"YTD Quarterly-Werte "
                                f"für {symbol}"
                            )
                        )

            capex_quality = sec_cf.attrs.get("capex_quality")

            self.assertIn(
                capex_quality,
                VALID_QUALITY_STATES,
                (
                    f"Ungültige CapEx Quality "
                    f"für {symbol} ({frequency})"
                )
            )

            if capex_quality == "PARTIAL_HISTORY":
                missing_years = sec_cf.attrs.get(
                    "capex_missing_years"
                )

                self.assertIsInstance(
                    missing_years,
                    list,
                    (
                        f"capex_missing_years fehlt "
                        f"für {symbol} ({frequency})"
                    )
                )

                self.assertGreater(
                    len(missing_years),
                    0,
                    (
                        f"PARTIAL_HISTORY ohne "
                        f"fehlende Jahre für "
                        f"{symbol} ({frequency})"
                    )
                )

            fcf_quality = sec_cf.attrs.get("fcf_quality")

            self.assertIn(
                fcf_quality,
                VALID_QUALITY_STATES,
                (
                    f"Ungültige FCF Quality "
                    f"für {symbol} ({frequency})"
                )
            )

            if capex_quality == "PARTIAL_HISTORY":
                self.assertEqual(
                    fcf_quality,
                    "PARTIAL_HISTORY",
                    (
                        f"FCF Quality sollte "
                        f"PARTIAL_HISTORY sein "
                        f"für {symbol} ({frequency})"
                    )
                )

            if "Free Cash Flow" in sec_cf.index:

                fcf_source = sec_cf.attrs.get("fcf_source")

                self.assertIsNotNone(
                    fcf_source,
                    (
                        f"FCF Source fehlt "
                        f"für {symbol} ({frequency})"
                    )
                )

            print_focused_cashflow_comparison(
                symbol=symbol,
                frequency=frequency,
                yf_cf=yf_cf,
                sec_cf=sec_cf,
            )

    def test_aapl_cashflow(self):
        self._run_symbol_test("AAPL")

    def test_msft_cashflow(self):
        self._run_symbol_test("MSFT")

    def test_meta_cashflow(self):
        self._run_symbol_test("META")

    def test_nvda_cashflow(self):
        self._run_symbol_test("NVDA")

    def test_cat_cashflow(self):
        self._run_symbol_test("CAT")

    def test_de_cashflow(self):
        self._run_symbol_test("DE")

    def test_ko_cashflow(self):
        self._run_symbol_test("KO")

    def test_mo_cashflow(self):
        self._run_symbol_test("MO")

    def test_jpm_cashflow(self):
        self._run_symbol_test("JPM")

    def test_bac_cashflow(self):
        self._run_symbol_test("BAC")

    def test_brk_b_cashflow(self):
        self._run_symbol_test("BRK-B")

    def test_pgr_cashflow(self):
        self._run_symbol_test("PGR")


if __name__ == "__main__":
    unittest.main()