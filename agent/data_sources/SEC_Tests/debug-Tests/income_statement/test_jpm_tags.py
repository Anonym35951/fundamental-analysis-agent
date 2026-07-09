import unittest
import pandas as pd
import yfinance as yf

from data_sources.sec_source import SecSource


JPM_BANK_TAGS_TO_CHECK = [
    #
    # Revenue
    #
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",

    #
    # Banking
    #
    "InterestIncome",
    "InterestExpense",
    "NetInterestIncome",

    #
    # Pretax / Earnings
    #
    "IncomeBeforeTaxExpenseBenefit",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "OperatingIncomeLoss",
    "EarningsBeforeInterestAndTaxes",
    "EarningsBeforeInterestTaxesDepreciationAndAmortization",

    #
    # D&A
    #
    "DepreciationDepletionAndAmortization",
    "DepreciationAndAmortization",
    "Depreciation",
    "AmortizationOfIntangibleAssets",

    #
    # Net Income
    #
    "NetIncomeLoss",
    "NetIncomeLossAvailableToCommonStockholdersBasic",
]


class TestJPMBankDebug(unittest.TestCase):

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
        ticker = yf.Ticker("JPM")

        yf_is = (
            ticker.income_stmt
            if frequency == "annual"
            else ticker.quarterly_income_stmt
        )

        print("\n==============================")
        print(
            f"YAHOO JPM / "
            f"{frequency.upper()}"
        )
        print("==============================")

        for row in yf_is.index:

            print(f"\n{row}")

            print(
                yf_is.loc[row]
                .dropna()
                .sort_index(ascending=False)
                .head(10)
            )

    def _print_sec_tags(
            self,
            us_gaap,
            frequency,
    ):
        print("\n==============================")
        print(
            f"SEC JPM TAGS / "
            f"{frequency.upper()}"
        )
        print("==============================")

        for tag in JPM_BANK_TAGS_TO_CHECK:

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

            if (
                    series is None
                    or series.dropna().empty
            ):
                print("Keine verwertbare Serie.")
                continue

            print(
                series.dropna()
                .sort_index(ascending=False)
                .head(15)
            )

    def _print_all_interest_tags(
            self,
            us_gaap,
    ):
        print("\n==============================")
        print("ALL INTEREST TAGS")
        print("==============================")

        for tag in sorted(us_gaap.keys()):

            if "interest" in tag.lower():

                print(tag)

    def _print_all_income_tags(
            self,
            us_gaap,
    ):
        print("\n==============================")
        print("ALL INCOME TAGS")
        print("==============================")

        for tag in sorted(us_gaap.keys()):

            if "income" in tag.lower():

                print(tag)

    def _print_all_revenue_tags(
            self,
            us_gaap,
    ):
        print("\n==============================")
        print("ALL REVENUE TAGS")
        print("==============================")

        for tag in sorted(us_gaap.keys()):

            if (
                    "revenue" in tag.lower()
                    or "revenues" in tag.lower()
            ):
                print(tag)

    def _print_current_sec_result(
            self,
            frequency,
    ):
        sec_is = self.sec.get_stock_financials(
            symbol="JPM",
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

        for row in sec_is.index:

            print(f"\n{row}")

            print(
                sec_is.loc[row]
                .dropna()
                .sort_index(ascending=False)
                .head(10)
            )

    def _run_jpm_debug(self):

        facts = self.sec.get_company_facts(
            "JPM",
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

        self._print_all_interest_tags(
            us_gaap
        )

        self._print_all_income_tags(
            us_gaap
        )

        self._print_all_revenue_tags(
            us_gaap
        )

        for frequency in [
            "annual",
            "quarterly",
        ]:

            print(
                f"\n\n########## "
                f"JPM BANK DEBUG / "
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
                frequency,
            )

    def test_jpm_bank_debug(self):
        self._run_jpm_debug()


if __name__ == "__main__":
    unittest.main()