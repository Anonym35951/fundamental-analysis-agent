import unittest
import pandas as pd

from data_sources.sec_source import SecSource


INTANGIBLE_TAGS = [
    "Goodwill",
    "FiniteLivedIntangibleAssetsNet",
    "IntangibleAssetsNetExcludingGoodwill",
    "OtherIntangibleAssets",
    "FiniteLivedIntangibleAssetsGross",
    "FiniteLivedIntangibleAssetsAccumulatedAmortization",
]

DEBT_TAGS = [
    "DebtInstrumentCarryingAmount",
    "DebtCurrent",
    "DebtNoncurrent",

    "LongTermDebt",
    "LongTermDebtCurrent",
    "LongTermDebtNoncurrent",

    "ShortTermBorrowings",
    "ShortTermDebt",
    "ShortTermDebtCurrent",

    "CommercialPaper",
    "CommercialPaperCurrent",

    "ConvertibleDebt",
    "ConvertibleDebtCurrent",
    "ConvertibleDebtNoncurrent",

    "LongTermDebtAndCapitalLeaseObligations",
    "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
    "LongTermDebtAndFinanceLeaseObligationsCurrentAndNoncurrent",

    "FinanceLeaseLiability",
    "FinanceLeaseLiabilityCurrent",
    "FinanceLeaseLiabilityNoncurrent",

    "OperatingLeaseLiability",
    "OperatingLeaseLiabilityCurrent",
    "OperatingLeaseLiabilityNoncurrent",
]


class TestAAPLIntangibles(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sec = SecSource(
            user_agent="deine-email@example.com"
        )

    def test_aapl_intangibles(self):

        symbol = "AAPL"

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
        print("AAPL INTANGIBLE ASSET ANALYSIS")
        print("=" * 120)

        for tag in INTANGIBLE_TAGS:

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
                print("\nANNUAL: NO USABLE SERIES")
            else:

                latest_date = annual_series.dropna().index.max()
                latest_value = annual_series.loc[latest_date]

                print("\nANNUAL")
                print(f"Latest Date     : {latest_date}")
                print(f"Latest Value    : {latest_value:,.0f}")
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

                latest_date = quarterly_series.dropna().index.max()
                latest_value = quarterly_series.loc[latest_date]

                print("\nQUARTERLY")
                print(f"Latest Date     : {latest_date}")
                print(f"Latest Value    : {latest_value:,.0f}")
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
        print("OTHER INTANGIBLE ASSETS MAPPING TEST")
        print("=" * 120)

        mapping_tags = [
            "OtherIntangibleAssets",
            "IntangibleAssetsNetExcludingGoodwill",
            "FiniteLivedIntangibleAssetsNet",
        ]

        #
        # Annual
        #

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

        #
        # Quarterly
        #

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

        #
        # DEBT DEEP DIVE
        #

        print("\n")
        print("=" * 120)
        print("AAPL DEBT ANALYSIS")
        print("=" * 120)

        for tag in DEBT_TAGS:

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

                print("\nANNUAL: NO USABLE SERIES")

            else:

                latest_date = annual_series.dropna().index.max()
                latest_value = annual_series.loc[latest_date]

                print("\nANNUAL")
                print(f"Latest Date     : {latest_date}")
                print(f"Latest Value    : {latest_value:,.0f}")
                print(f"Tag Used        : {annual_meta.get('tag')}")

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

                latest_date = quarterly_series.dropna().index.max()
                latest_value = quarterly_series.loc[latest_date]

                print("\nQUARTERLY")
                print(f"Latest Date     : {latest_date}")
                print(f"Latest Value    : {latest_value:,.0f}")
                print(f"Tag Used        : {quarterly_meta.get('tag')}")

                print("\nLast 5 quarterly values:")

                print(
                    quarterly_series
                    .dropna()
                    .sort_index(ascending=False)
                    .head(5)
                )

        #
        # TOTAL DEBT RECONCILIATION
        #

        print("\n")
        print("=" * 120)
        print("TOTAL DEBT RECONCILIATION")
        print("=" * 120)

        reconciliation_tags = [
            "DebtInstrumentCarryingAmount",
            "LongTermDebt",
            "LongTermDebtCurrent",
            "LongTermDebtNoncurrent",
            "DebtCurrent",
            "CommercialPaper",
            "FinanceLeaseLiability",
            "OperatingLeaseLiability",
        ]

        values = {}

        for tag in reconciliation_tags:

            series, meta = self.sec._first_available_series(
                us_gaap=us_gaap,
                sec_tags=[tag],
                frequency="quarterly",
                unit_preference=["USD"],
                return_metadata=True,
            )

            if series is None or series.empty:
                continue

            latest_date = series.dropna().index.max()

            values[tag] = float(series.loc[latest_date])

        for k, v in values.items():
            print(f"{k:<45} {v:>20,.0f}")

        print("\n")

        if (
            "LongTermDebtNoncurrent" in values
            and "LongTermDebtCurrent" in values
        ):

            total = (
                values["LongTermDebtNoncurrent"]
                + values["LongTermDebtCurrent"]
            )

            print(
                f"LongTermDebtNoncurrent + LongTermDebtCurrent = "
                f"{total:,.0f}"
            )

        if (
            "DebtInstrumentCarryingAmount" in values
            and "FinanceLeaseLiability" in values
        ):

            total = (
                values["DebtInstrumentCarryingAmount"]
                + values["FinanceLeaseLiability"]
            )

            print(
                f"DebtInstrumentCarryingAmount + FinanceLeaseLiability = "
                f"{total:,.0f}"
            )

        if (
            "DebtInstrumentCarryingAmount" in values
            and "OperatingLeaseLiability" in values
        ):

            total = (
                values["DebtInstrumentCarryingAmount"]
                + values["OperatingLeaseLiability"]
            )

            print(
                f"DebtInstrumentCarryingAmount + OperatingLeaseLiability = "
                f"{total:,.0f}"
            )


if __name__ == "__main__":
    unittest.main()