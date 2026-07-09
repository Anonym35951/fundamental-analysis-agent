import unittest
import pandas as pd

from data_sources.sec_source import SecSource


DEBT_TAGS = [
    "DebtInstrumentCarryingAmount",
    "LongTermDebt",
    "LongTermDebtCurrent",
    "LongTermDebtNoncurrent",
    "DebtCurrent",
    "DebtNoncurrent",
    "CommercialPaper",
    "CommercialPaperCurrent",
]


class TestDebtStructureAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sec = SecSource(
            user_agent="deine-email@example.com"
        )

    def _latest_value(self, us_gaap, tag, frequency="annual"):

        if tag not in us_gaap:
            return None

        series, meta = self.sec._first_available_series(
            us_gaap=us_gaap,
            sec_tags=[tag],
            frequency=frequency,
            unit_preference=["USD"],
            return_metadata=True,
        )

        if series is None or series.dropna().empty:
            return None

        latest_date = series.dropna().index.max()

        return float(series.loc[latest_date])

    def test_aapl_debt_structure(self):

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
        print(f"{symbol} DEBT STRUCTURE ANALYSIS")
        print("=" * 120)

        for frequency in ["annual", "quarterly"]:

            print("\n")
            print("#" * 120)
            print(f"{frequency.upper()}")
            print("#" * 120)

            values = {}

            for tag in DEBT_TAGS:

                value = self._latest_value(
                    us_gaap,
                    tag,
                    frequency=frequency,
                )

                values[tag] = value

                if value is None:
                    print(f"{tag:<40} NOT FOUND")
                else:
                    print(f"{tag:<40} {value:>20,.0f}")

            print("\n")
            print("-" * 120)
            print("RECONCILIATION TESTS")
            print("-" * 120)

            ltd = values.get("LongTermDebt")
            ltd_current = values.get("LongTermDebtCurrent")
            ltd_noncurrent = values.get("LongTermDebtNoncurrent")

            debt_current = values.get("DebtCurrent")
            debt_noncurrent = values.get("DebtNoncurrent")

            debt_total = values.get("DebtInstrumentCarryingAmount")

            commercial_paper = values.get("CommercialPaper")
            commercial_paper_current = values.get(
                "CommercialPaperCurrent"
            )

            #
            # Test 1
            #

            if (
                ltd is not None
                and ltd_current is not None
                and ltd_noncurrent is not None
            ):

                reconstructed = (
                    ltd_current
                    + ltd_noncurrent
                )

                diff = ltd - reconstructed

                print(
                    f"\nLongTermDebt vs "
                    f"(Current + Noncurrent)"
                )

                print(
                    f"LongTermDebt                 : "
                    f"{ltd:,.0f}"
                )

                print(
                    f"Current + Noncurrent         : "
                    f"{reconstructed:,.0f}"
                )

                print(
                    f"Difference                   : "
                    f"{diff:,.0f}"
                )

                if abs(diff) < 1:
                    print(
                        "RESULT: LongTermDebt already "
                        "includes Current portion"
                    )
                else:
                    print(
                        "RESULT: LongTermDebt appears "
                        "to be Noncurrent only"
                    )

            #
            # Test 2
            #

            if (
                debt_total is not None
                and ltd_noncurrent is not None
                and ltd_current is not None
            ):

                reconstructed = (
                    ltd_noncurrent
                    + ltd_current
                )

                diff = debt_total - reconstructed

                print(
                    f"\nDebtInstrumentCarryingAmount "
                    f"vs "
                    f"(LongTermDebtCurrent + "
                    f"LongTermDebtNoncurrent)"
                )

                print(
                    f"DebtInstrumentCarryingAmount : "
                    f"{debt_total:,.0f}"
                )

                print(
                    f"Reconstructed                : "
                    f"{reconstructed:,.0f}"
                )

                print(
                    f"Difference                   : "
                    f"{diff:,.0f}"
                )

            #
            # Test 3
            #

            if (
                debt_total is not None
                and commercial_paper is not None
            ):

                reconstructed = (
                    debt_total
                    + commercial_paper
                )

                print(
                    f"\nDebtInstrumentCarryingAmount "
                    f"+ CommercialPaper"
                )

                print(
                    f"{debt_total:,.0f} + "
                    f"{commercial_paper:,.0f}"
                )

                print(
                    f"= {reconstructed:,.0f}"
                )

            #
            # Test 4
            #

            if (
                debt_current is not None
                and debt_noncurrent is not None
            ):

                reconstructed = (
                    debt_current
                    + debt_noncurrent
                )

                print(
                    f"\nDebtCurrent + DebtNoncurrent"
                )

                print(
                    f"{debt_current:,.0f} + "
                    f"{debt_noncurrent:,.0f}"
                )

                print(
                    f"= {reconstructed:,.0f}"
                )

                if debt_total is not None:

                    print(
                        f"Difference to "
                        f"DebtInstrumentCarryingAmount: "
                        f"{debt_total - reconstructed:,.0f}"
                    )

            #
            # Test 5
            #

            if (
                commercial_paper is not None
                and commercial_paper_current is not None
            ):

                print(
                    "\nCommercialPaper vs "
                    "CommercialPaperCurrent"
                )

                print(
                    f"CommercialPaper          : "
                    f"{commercial_paper:,.0f}"
                )

                print(
                    f"CommercialPaperCurrent   : "
                    f"{commercial_paper_current:,.0f}"
                )

                print(
                    f"Difference               : "
                    f"{commercial_paper - commercial_paper_current:,.0f}"
                )

        print("\n")
        print("=" * 120)
        print("INTERPRETATION")
        print("=" * 120)

        print("""
Wenn gilt:

LongTermDebt
=
LongTermDebtCurrent
+
LongTermDebtNoncurrent

dann darf LongTermDebtCurrent NICHT nochmals
zu LongTermDebt addiert werden.

Wenn gilt:

LongTermDebt
≈ LongTermDebtNoncurrent

dann muss Current Debt separat addiert werden.

Genau dadurch lassen sich AAPL und NVDA sauber unterscheiden.
""")


if __name__ == "__main__":
    unittest.main()