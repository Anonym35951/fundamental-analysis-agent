import unittest
import pandas as pd
import yfinance as yf

from data_sources.sec_source import SecSource


AMZN_INTEREST_TAGS_TO_CHECK = [
    #
    # Aktuelles Mapping
    #
    "InterestExpenseNonoperating",
    "InterestExpense",
    "InterestExpenseDebt",
    "FinanceLeaseInterestExpense",
    "InterestCostsIncurred",

    #
    # Erweiterungskandidaten
    #
    "InterestAndDebtExpense",
    "InterestExpenseAndDebtExpense",
    "InterestExpenseDebtExcludingAmortization",
    "InterestExpenseBorrowings",
    "InterestExpenseDeposits",
    "InterestExpenseOther",
    "InterestExpenseAndOther",
    "InterestExpenseRelatedParty",
    "InterestExpenseLongTermDebt",
    "InterestExpenseShortTermBorrowings",
    "InterestExpenseCapitalLease",
]


class TestAMZNInterestExpenseDebug(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sec = SecSource(
            user_agent="deine-email@example.com"
        )

    def _get_sec_series(self, us_gaap, tag, frequency):
        payload = us_gaap.get(tag)

        if not payload:
            return None

        values = payload.get("units", {}).get("USD")

        if not values:
            return None

        return self.sec._fact_values_to_series(
            values=values,
            frequency=frequency,
            statement_type="income",
        )

    def _print_yahoo_interest(self, frequency):
        ticker = yf.Ticker("AMZN")

        yf_is = (
            ticker.income_stmt
            if frequency == "annual"
            else ticker.quarterly_income_stmt
        )

        print("\n==============================")
        print(f"YAHOO AMZN INTEREST EXPENSE / {frequency.upper()}")
        print("==============================")

        if "Interest Expense" not in yf_is.index:
            print("Yahoo hat keinen Interest Expense Eintrag.")
            return

        print(
            yf_is.loc["Interest Expense"]
            .dropna()
            .sort_index(ascending=False)
            .head(20)
        )

    def _print_sec_interest_tags(self, us_gaap, frequency):
        print("\n==============================")
        print(f"SEC AMZN INTEREST TAGS / {frequency.upper()}")
        print("==============================")

        for tag in AMZN_INTEREST_TAGS_TO_CHECK:

            payload = us_gaap.get(tag)

            if not payload:
                print(f"\nTAG: {tag}")
                print("NOT FOUND")
                continue

            series = self._get_sec_series(
                us_gaap=us_gaap,
                tag=tag,
                frequency=frequency,
            )

            print(f"\nTAG: {tag}")
            print(f"LABEL: {payload.get('label')}")
            print(f"UNITS: {list(payload.get('units', {}).keys())}")

            if series is None or series.dropna().empty:
                print("Keine verwertbare Serie.")
                continue

            clean = (
                series.dropna()
                .sort_index(ascending=False)
            )

            print(clean.head(25))

            years = sorted({
                pd.Timestamp(idx).year
                for idx in clean.index
            })

            if years:
                expected_years = set(
                    range(
                        min(years),
                        max(years) + 1,
                    )
                )

                missing_years = sorted(
                    expected_years - set(years)
                )

                print("Jahre:", years)
                print("Fehlende Jahre:", missing_years)

    def _print_interest_sum_check(
            self,
            us_gaap,
            frequency,
    ):
        print("\n==============================")
        print(f"INTEREST EXPENSE SUM CHECK / {frequency.upper()}")
        print("==============================")

        series_list = []

        for tag in [
            "InterestExpenseNonoperating",
            "InterestExpense",
            "InterestExpenseDebt",
            "FinanceLeaseInterestExpense",
            "InterestCostsIncurred",
        ]:

            series = self._get_sec_series(
                us_gaap,
                tag,
                frequency,
            )

            if series is None:
                continue

            series = series.dropna()

            if series.empty:
                continue

            print(f"\n{tag}")

            print(
                series
                .sort_index(ascending=False)
                .head(15)
            )

            series_list.append(
                series.rename(tag)
            )

        if not series_list:
            print("\nKeine verwertbaren Interest-Tags gefunden.")
            return

        combined = pd.concat(
            series_list,
            axis=1,
        )

        print("\nCOMBINED TABLE")
        print(
            combined
            .sort_index(ascending=False)
            .head(15)
        )

        summed = combined.sum(
            axis=1,
            skipna=True,
        )

        print("\nSUMME ALLER INTEREST TAGS")
        print(
            summed
            .dropna()
            .sort_index(ascending=False)
            .head(15)
        )

    def _print_current_sec_result(self, frequency):

        sec_is = self.sec.get_stock_financials(
            symbol="AMZN",
            frequency=frequency,
            use_cache=False,
            scope="core",
        )

        print("\n==============================")
        print(f"CURRENT SEC INCOME RESULT / {frequency.upper()}")
        print("==============================")

        self.assertIsInstance(
            sec_is,
            pd.DataFrame,
        )

        for row in [
            "Interest Expense",
            "EBIT",
            "EBITDA",
            "Operating Income",
            "Net Income",
        ]:

            if row in sec_is.index:

                print(f"\n{row}:")

                print(
                    sec_is.loc[row]
                    .dropna()
                    .sort_index(ascending=False)
                    .head(25)
                )

            else:

                print(f"\n{row}: FEHLT")

    def _run_amzn_debug(self):

        facts = self.sec.get_company_facts(
            "AMZN",
            use_cache=False,
        )

        self.assertIsInstance(
            facts,
            dict,
        )

        self.assertNotIn(
            "error",
            facts,
        )

        us_gaap = (
            facts.get("facts", {})
            .get("us-gaap", {})
        )

        self.assertTrue(us_gaap)

        for frequency in [
            "annual",
            "quarterly",
        ]:

            print(
                f"\n\n########## AMZN INTEREST DEBUG / "
                f"{frequency.upper()} ##########"
            )

            self._print_yahoo_interest(
                frequency
            )

            self._print_sec_interest_tags(
                us_gaap,
                frequency,
            )

            self._print_interest_sum_check(
                us_gaap,
                frequency,
            )

            self._print_current_sec_result(
                frequency
            )

    def test_amzn_interest_debug(self):
        self._run_amzn_debug()


if __name__ == "__main__":
    unittest.main()