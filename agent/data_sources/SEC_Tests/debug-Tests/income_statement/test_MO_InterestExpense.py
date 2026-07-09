import unittest
import pandas as pd

from data_sources.sec_source import SecSource


MO_INTEREST_TAGS_TO_CHECK = [
    "InterestExpense",
    "InterestExpenseNonoperating",
]


class TestMOInterestExpenseDebug(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sec = SecSource(
            user_agent="deine-email@example.com"
        )

    def _print_raw_interest_facts(
            self,
            us_gaap,
            tag,
    ):
        payload = us_gaap.get(tag)

        if not payload:
            print(f"\n{tag}: NOT FOUND")
            return

        values = (
            payload.get("units", {})
            .get("USD")
        )

        if not values:
            print(f"\n{tag}: NO USD FACTS")
            return

        print("\n========================================")
        print(f"RAW FACTS: {tag}")
        print("========================================")

        for fact in values:

            print({
                "end": fact.get("end"),
                "start": fact.get("start"),
                "fy": fact.get("fy"),
                "fp": fact.get("fp"),
                "form": fact.get("form"),
                "frame": fact.get("frame"),
                "filed": fact.get("filed"),
                "val": fact.get("val"),
            })

    def _print_quarterly_dataframe(
            self,
            us_gaap,
            tag,
    ):
        payload = us_gaap.get(tag)

        if not payload:
            return

        values = (
            payload.get("units", {})
            .get("USD")
        )

        if not values:
            return

        print("\n========================================")
        print(f"DATAFRAME RESULT: {tag}")
        print("========================================")

        df = self.sec._fact_values_to_dataframe(
            values=values,
            frequency="quarterly",
            statement_type="income",
        )

        if df is None:
            print("RESULT: None")
            return

        print(
            df[
                [
                    "date",
                    "start",
                    "duration_days",
                    "value",
                    "fy",
                    "fp",
                    "form",
                    "frame",
                ]
            ]
            .sort_values(
                "date",
                ascending=False,
            )
            .head(50)
        )

    def _print_quarterly_series(
            self,
            us_gaap,
            tag,
    ):
        payload = us_gaap.get(tag)

        if not payload:
            return

        values = (
            payload.get("units", {})
            .get("USD")
        )

        if not values:
            return

        print("\n========================================")
        print(f"SERIES RESULT: {tag}")
        print("========================================")

        series = self.sec._fact_values_to_series(
            values=values,
            frequency="quarterly",
            statement_type="income",
        )

        if series is None:
            print("SERIES: None")
            return

        print(
            series
            .dropna()
            .sort_index(ascending=False)
            .head(50)
        )

    def test_mo_interest_quarterly_debug(self):

        facts = self.sec.get_company_facts(
            "MO",
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

        for tag in MO_INTEREST_TAGS_TO_CHECK:

            self._print_raw_interest_facts(
                us_gaap,
                tag,
            )

            self._print_quarterly_dataframe(
                us_gaap,
                tag,
            )

            self._print_quarterly_series(
                us_gaap,
                tag,
            )


if __name__ == "__main__":
    unittest.main()