import unittest
import pandas as pd

from data_sources.sec_source import SecSource


DEBT_TAGS = [
    "DebtLongtermAndShorttermCombinedAmount",
    "DebtCurrentAndNoncurrent",
    "DebtInstrumentCarryingAmount",

    "LongTermDebtAndFinanceLeaseObligationsCurrentAndNoncurrent",
    "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
    "LongTermDebtAndCapitalLeaseObligations",

    "LongTermDebt",
    "LongTermDebtNoncurrent",
    "LongTermDebtCurrent",

    "ShortTermDebtCurrent",
    "ShortTermBorrowings",
    "CommercialPaper",

    "FinanceLeaseLiability",
    "FinanceLeaseLiabilityNoncurrent",
]


LEASE_TAGS = [
    "FinanceLeaseLiability",
    "FinanceLeaseLiabilityNoncurrent",

    "OperatingLeaseLiability",
    "OperatingLeaseLiabilityNoncurrent",

    "CapitalLeaseObligationsIncurred",

    "LesseeOperatingLeaseLiabilityPaymentsDue",
]


class TestMSFTDebtTags(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sec = SecSource(
            user_agent="deine-email@example.com"
        )

    def test_msft_debt_tags(self):

        symbol = "MSFT"

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
        print("MSFT DEBT TAG ANALYSIS")
        print("=" * 120)

        #
        # Annual Debt Tags
        #

        for tag in DEBT_TAGS:

            print("\n")
            print("-" * 120)
            print(tag)
            print("-" * 120)

            if tag not in us_gaap:
                print("TAG NOT FOUND")
                continue

            series, meta = self.sec._first_available_series(
                us_gaap=us_gaap,
                sec_tags=[tag],
                frequency="annual",
                unit_preference=["USD"],
                return_metadata=True,
            )

            if series is None or series.empty:
                print("NO USABLE SERIES")
                continue

            latest_date = series.dropna().index.max()
            latest_value = series.loc[latest_date]

            print(f"Latest Date     : {latest_date}")
            print(f"Latest Value    : {latest_value:,.0f}")
            print(f"Data Points     : {len(series.dropna())}")
            print(f"Selected Unit   : {meta.get('unit')}")
            print(f"Tag Used        : {meta.get('tag')}")

            print("\nLast 10 values:")

            print(
                series
                .dropna()
                .sort_index(ascending=False)
                .head(10)
            )

        #
        # Total Debt Mapping
        #

        print("\n")
        print("=" * 120)
        print("TOTAL DEBT MAPPING TEST")
        print("=" * 120)

        total_debt_tags = [
            "DebtLongtermAndShorttermCombinedAmount",
            "DebtCurrentAndNoncurrent",
            "LongTermDebtAndFinanceLeaseObligationsCurrentAndNoncurrent",
            "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
            "LongTermDebtAndCapitalLeaseObligations",
            "DebtInstrumentCarryingAmount",
        ]

        series, meta = self.sec._first_available_series(
            us_gaap=us_gaap,
            sec_tags=total_debt_tags,
            frequency="annual",
            unit_preference=["USD"],
            return_metadata=True,
        )

        print("\nSelected by _first_available_series():")
        print(meta)

        if series is not None and not series.empty:

            latest_date = series.dropna().index.max()
            latest_value = series.loc[latest_date]

            print(f"\nChosen Latest Date  : {latest_date}")
            print(f"Chosen Latest Value : {latest_value:,.0f}")

            print("\nHistory:")

            print(
                series
                .dropna()
                .sort_index(ascending=False)
                .head(10)
            )

        #
        # Quarterly Debt Components
        #

        print("\n")
        print("=" * 120)
        print("QUARTERLY DEBT COMPONENT ANALYSIS")
        print("=" * 120)

        quarterly_tags = [
            "DebtInstrumentCarryingAmount",
            "FinanceLeaseLiability",
            "LongTermDebt",
            "LongTermDebtNoncurrent",
            "LongTermDebtCurrent",
        ]

        for tag in quarterly_tags:

            print("\n")
            print("-" * 120)
            print(tag)
            print("-" * 120)

            result = self.sec.get_balance_sheet_line_item(
                symbol="MSFT",
                line_item=tag,
                frequency="quarterly",
                scope="raw",
                use_cache=False,
            )

            if isinstance(result, dict) and "error" in result:
                print(result)
                continue

            print(f"Latest Date : {result['date']}")
            print(f"Latest Value: {result['value']:,.0f}")

            print("\nLast 10 periods:")

            series = pd.Series(result["series"], dtype=float)

            print(
                series
                .sort_index(ascending=False)
                .head(10)
            )

        #
        # Lease Analysis
        #

        print("\n")
        print("=" * 120)
        print("LEASE LIABILITY ANALYSIS")
        print("=" * 120)

        for tag in LEASE_TAGS:

            print("\n")
            print("-" * 120)
            print(tag)
            print("-" * 120)

            if tag not in us_gaap:
                print("TAG NOT FOUND")
                continue

            series, meta = self.sec._first_available_series(
                us_gaap=us_gaap,
                sec_tags=[tag],
                frequency="quarterly",
                unit_preference=["USD"],
                return_metadata=True,
            )

            if series is None or series.empty:
                print("NO USABLE SERIES")
                continue

            latest_date = series.dropna().index.max()
            latest_value = series.loc[latest_date]

            print(f"Latest Date     : {latest_date}")
            print(f"Latest Value    : {latest_value:,.0f}")
            print(f"Data Points     : {len(series.dropna())}")
            print(f"Selected Unit   : {meta.get('unit')}")
            print(f"Tag Used        : {meta.get('tag')}")

            print("\nLast 10 values:")

            print(
                series
                .dropna()
                .sort_index(ascending=False)
                .head(10)
            )

        #
        # Debt + Lease Reconciliation
        #

        print("\n")
        print("=" * 120)
        print("DEBT + LEASE RECONCILIATION")
        print("=" * 120)

        reconciliation_tags = [
            "DebtInstrumentCarryingAmount",
            "FinanceLeaseLiability",
            "OperatingLeaseLiability",
            "LongTermDebtNoncurrent",
            "LongTermDebtCurrent",
        ]

        values = {}

        for tag in reconciliation_tags:

            result = self.sec.get_balance_sheet_line_item(
                symbol="MSFT",
                line_item=tag,
                frequency="quarterly",
                scope="raw",
                use_cache=False,
            )

            if isinstance(result, dict) and "error" in result:
                print(f"{tag:<40} NOT AVAILABLE")
                continue

            value = result["value"]

            values[tag] = value

            print(f"{tag:<40} {value:>20,.0f}")

        print("\n")
        print("-" * 120)

        debt = values.get("DebtInstrumentCarryingAmount", 0)
        finance_lease = values.get("FinanceLeaseLiability", 0)
        operating_lease = values.get("OperatingLeaseLiability", 0)

        print(f"{'DebtInstrumentCarryingAmount':<40} {debt:>20,.0f}")
        print(f"{'+ FinanceLeaseLiability':<40} {finance_lease:>20,.0f}")
        print(f"{'= Combined':<40} {(debt + finance_lease):>20,.0f}")

        print()

        print(f"{'DebtInstrumentCarryingAmount':<40} {debt:>20,.0f}")
        print(f"{'+ OperatingLeaseLiability':<40} {operating_lease:>20,.0f}")
        print(f"{'= Combined':<40} {(debt + operating_lease):>20,.0f}")

        print()

        print(f"{'DebtInstrumentCarryingAmount':<40} {debt:>20,.0f}")
        print(f"{'+ FinanceLeaseLiability':<40} {finance_lease:>20,.0f}")
        print(f"{'+ OperatingLeaseLiability':<40} {operating_lease:>20,.0f}")
        print(
            f"{'= Combined':<40} "
            f"{(debt + finance_lease + operating_lease):>20,.0f}"
        )


if __name__ == "__main__":
    unittest.main()