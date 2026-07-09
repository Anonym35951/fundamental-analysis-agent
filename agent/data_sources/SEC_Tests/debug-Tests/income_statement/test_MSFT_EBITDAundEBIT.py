import unittest
import pandas as pd
import yfinance as yf

from data_sources.sec_source import SecSource


MSFT_EBITDA_TAGS_TO_CHECK = [
    "EarningsBeforeInterestTaxesDepreciationAndAmortization",
    "OperatingIncomeLoss",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "DepreciationDepletionAndAmortization",
    "Depreciation",
    "DepreciationAndAmortization",
    "DepreciationAmortizationAndAccretionNet",
    "AmortizationOfIntangibleAssets",
    "FiniteLivedIntangibleAssetsAmortizationExpense",
    "IncomeBeforeTaxExpenseBenefit",
    "EarningsBeforeInterestAndTaxes",
]


class TestMSFTEbitdaDebug(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sec = SecSource(
            user_agent="deine-email@example.com"
        )

    def _get_sec_series(
            self,
            us_gaap,
            tag,
            frequency,
    ):
        payload = us_gaap.get(tag)

        if not payload:
            return None

        values = payload.get(
            "units",
            {}
        ).get("USD")

        if not values:
            return None

        return self.sec._fact_values_to_series(
            values=values,
            frequency=frequency,
            statement_type="income",
        )

    def _print_yahoo_values(
            self,
            frequency,
    ):
        ticker = yf.Ticker("MSFT")

        yf_is = (
            ticker.income_stmt
            if frequency == "annual"
            else ticker.quarterly_income_stmt
        )

        print("\n==============================")
        print(f"YAHOO MSFT EBIT / EBITDA / {frequency.upper()}")
        print("==============================")

        for row in [
            "Operating Income",
            "EBIT",
            "EBITDA",
            "Interest Expense",
            "Net Income",
        ]:

            if row not in yf_is.index:
                continue

            print(f"\n{row}")

            print(
                yf_is.loc[row]
                .dropna()
                .sort_index(ascending=False)
                .head(20)
            )

    def _print_sec_tags(
            self,
            us_gaap,
            frequency,
    ):
        print("\n==============================")
        print(f"SEC MSFT EBITDA TAGS / {frequency.upper()}")
        print("==============================")

        for tag in MSFT_EBITDA_TAGS_TO_CHECK:

            payload = us_gaap.get(tag)

            if not payload:

                print(f"\nTAG: {tag}")
                print("NOT FOUND")
                continue

            series = self._get_sec_series(
                us_gaap,
                tag,
                frequency,
            )

            print(f"\nTAG: {tag}")
            print(f"LABEL: {payload.get('label')}")
            print(
                f"UNITS: "
                f"{list(payload.get('units', {}).keys())}"
            )

            if (
                    series is None
                    or series.dropna().empty
            ):
                print("Keine verwertbare Serie.")
                continue

            clean = (
                series.dropna()
                .sort_index(ascending=False)
            )

            print(clean.head(25))

    def _print_current_sec_result(
            self,
            frequency,
    ):
        sec_is = self.sec.get_stock_financials(
            symbol="MSFT",
            frequency=frequency,
            use_cache=False,
            scope="core",
        )

        print("\n==============================")
        print(
            f"CURRENT SEC RESULT / "
            f"{frequency.upper()}"
        )
        print("==============================")

        for row in [
            "Operating Income",
            "EBIT",
            "EBITDA",
            "Depreciation And Amortization",
            "Depreciation Depletion And Amortization",
            "Interest Expense",
            "Net Income",
        ]:

            if row in sec_is.index:

                print(f"\n{row}")

                print(
                    sec_is.loc[row]
                    .dropna()
                    .sort_index(ascending=False)
                    .head(25)
                )

            else:

                print(f"\n{row}: FEHLT")

    def _print_all_depreciation_tags(
            self,
            us_gaap,
    ):
        print("\n==============================")
        print("ALL DEPRECIATION TAGS")
        print("==============================")

        for tag in sorted(us_gaap.keys()):

            if "depreci" in tag.lower():
                print(tag)

    def _print_all_amortization_tags(
            self,
            us_gaap,
    ):
        print("\n==============================")
        print("ALL AMORTIZATION TAGS")
        print("==============================")

        for tag in sorted(us_gaap.keys()):

            if "amort" in tag.lower():
                print(tag)

    def _run_msft_debug(self):

        facts = self.sec.get_company_facts(
            "MSFT",
            use_cache=False,
        )

        self.assertIsInstance(
            facts,
            dict,
        )

        us_gaap = (
            facts.get("facts", {})
            .get("us-gaap", {})
        )

        self.assertTrue(us_gaap)

        self._print_all_depreciation_tags(
            us_gaap
        )

        self._print_all_amortization_tags(
            us_gaap
        )

        for frequency in [
            "annual",
            "quarterly",
        ]:

            print(
                f"\n\n########## "
                f"MSFT EBITDA DEBUG / "
                f"{frequency.upper()} "
                f"##########"
            )

            self._print_yahoo_values(
                frequency
            )

            self._print_sec_tags(
                us_gaap,
                frequency,
            )

            self._print_current_sec_result(
                frequency
            )

    def test_msft_ebitda_debug(self):
        self._run_msft_debug()


if __name__ == "__main__":
    unittest.main()