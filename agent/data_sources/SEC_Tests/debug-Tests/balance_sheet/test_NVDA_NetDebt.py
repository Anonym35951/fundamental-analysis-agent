import unittest
import pandas as pd

from data_sources.sec_source import SecSource


INVESTMENT_TAGS = [
    "MarketableSecuritiesCurrent",
    "MarketableSecurities",
    "OtherLongTermInvestments",

    "AvailableForSaleSecurities",
    "AvailableForSaleSecuritiesCurrent",
    "AvailableForSaleSecuritiesDebtSecurities",
    "AvailableForSaleSecuritiesDebtSecuritiesCurrent",

    "DebtSecuritiesCurrent",

    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
]


class TestNVDAInvestments(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sec = SecSource(
            user_agent="deine-email@example.com"
        )

    def test_nvda_investment_tags(self):

        symbol = "NVDA"

        facts = self.sec.get_company_facts(
            symbol,
            use_cache=False,
        )

        self.assertFalse(
            isinstance(facts, dict) and "error" in facts,
            msg=facts,
        )

        us_gaap = facts["facts"]["us-gaap"]

        print("\n")
        print("=" * 120)
        print("NVDA INVESTMENT TAG ANALYSIS")
        print("=" * 120)

        for tag in INVESTMENT_TAGS:

            print("\n")
            print("-" * 120)
            print(tag)
            print("-" * 120)

            if tag not in us_gaap:
                print("TAG NOT FOUND")
                continue

            #
            # Annual
            #
            annual_series, annual_meta = self.sec._first_available_series(
                us_gaap=us_gaap,
                sec_tags=[tag],
                frequency="annual",
                unit_preference=["USD"],
                return_metadata=True,
            )

            if annual_series is None or annual_series.empty:
                print("ANNUAL: NO USABLE SERIES")
            else:
                annual_latest_date = annual_series.dropna().index.max()
                annual_latest_value = annual_series.loc[annual_latest_date]

                print("\nANNUAL")
                print(f"Latest Date     : {annual_latest_date}")
                print(f"Latest Value    : {annual_latest_value:,.0f}")
                print(f"Data Points     : {len(annual_series.dropna())}")
                print(f"Selected Unit   : {annual_meta.get('unit')}")
                print(f"Tag Used        : {annual_meta.get('tag')}")

                print("\nLast 10 annual values:")

                print(
                    annual_series
                    .dropna()
                    .sort_index(ascending=False)
                    .head(10)
                )

            #
            # Quarterly
            #
            quarterly_series, quarterly_meta = self.sec._first_available_series(
                us_gaap=us_gaap,
                sec_tags=[tag],
                frequency="quarterly",
                unit_preference=["USD"],
                return_metadata=True,
            )

            if quarterly_series is None or quarterly_series.empty:
                print("\nQUARTERLY: NO USABLE SERIES")
            else:
                quarterly_latest_date = quarterly_series.dropna().index.max()
                quarterly_latest_value = quarterly_series.loc[quarterly_latest_date]

                print("\nQUARTERLY")
                print(f"Latest Date     : {quarterly_latest_date}")
                print(f"Latest Value    : {quarterly_latest_value:,.0f}")
                print(f"Data Points     : {len(quarterly_series.dropna())}")
                print(f"Selected Unit   : {quarterly_meta.get('unit')}")
                print(f"Tag Used        : {quarterly_meta.get('tag')}")

                print("\nLast 10 quarterly values:")

                print(
                    quarterly_series
                    .dropna()
                    .sort_index(ascending=False)
                    .head(10)
                )

        print("\n")
        print("=" * 120)
        print("OTHER SHORT TERM INVESTMENTS MAPPING TEST")
        print("=" * 120)

        mapping_tags = [
            "ShortTermInvestments",
            "OtherShortTermInvestments",
            "MarketableSecuritiesCurrent",
            "MarketableSecurities",
            "AvailableForSaleSecuritiesCurrent",
            "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
            "AvailableForSaleSecurities",
            "AvailableForSaleSecuritiesDebtSecurities",
            "DebtSecuritiesCurrent",
        ]

        annual_series, annual_meta = self.sec._first_available_series(
            us_gaap=us_gaap,
            sec_tags=mapping_tags,
            frequency="annual",
            unit_preference=["USD"],
            return_metadata=True,
        )

        print("\nANNUAL SELECTION")
        print(annual_meta)

        if annual_series is not None and not annual_series.empty:

            latest_date = annual_series.dropna().index.max()
            latest_value = annual_series.loc[latest_date]

            print(f"\nChosen Latest Date  : {latest_date}")
            print(f"Chosen Latest Value : {latest_value:,.0f}")

            print("\nHistory:")

            print(
                annual_series
                .dropna()
                .sort_index(ascending=False)
                .head(10)
            )

        quarterly_series, quarterly_meta = self.sec._first_available_series(
            us_gaap=us_gaap,
            sec_tags=mapping_tags,
            frequency="quarterly",
            unit_preference=["USD"],
            return_metadata=True,
        )

        print("\n")
        print("QUARTERLY SELECTION")
        print(quarterly_meta)

        if quarterly_series is not None and not quarterly_series.empty:

            latest_date = quarterly_series.dropna().index.max()
            latest_value = quarterly_series.loc[latest_date]

            print(f"\nChosen Latest Date  : {latest_date}")
            print(f"Chosen Latest Value : {latest_value:,.0f}")

            print("\nHistory:")

            print(
                quarterly_series
                .dropna()
                .sort_index(ascending=False)
                .head(10)
            )


if __name__ == "__main__":
    unittest.main()