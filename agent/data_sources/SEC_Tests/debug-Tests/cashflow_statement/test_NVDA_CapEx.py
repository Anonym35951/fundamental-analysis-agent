import unittest
import pandas as pd
import yfinance as yf

from data_sources.sec_source import SecSource


NVDA_CAPEX_TAGS_TO_CHECK = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
    "PaymentsToAcquireOtherPropertyPlantAndEquipment",
    "PaymentsToAcquirePropertyPlantEquipmentAndOtherProductiveAssets",
    "CapitalExpenditures",
    "PropertyAndEquipmentAdditions",
    "CapitalSpending",
    "SegmentExpenditureAdditionToLongLivedAssets",
]


class TestNVDACapexDebug(unittest.TestCase):

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
            statement_type="cashflow",
        )

    def _print_yahoo_capex(self, frequency):
        ticker = yf.Ticker("NVDA")

        yf_cf = (
            ticker.cashflow
            if frequency == "annual"
            else ticker.quarterly_cashflow
        )

        print("\n==============================")
        print(f"YAHOO NVDA CAPEX / {frequency.upper()}")
        print("==============================")

        if "Capital Expenditure" not in yf_cf.index:
            print("Yahoo hat keinen CapEx-Eintrag.")
            return

        print(
            yf_cf.loc["Capital Expenditure"]
            .dropna()
            .sort_index(ascending=False)
            .head(20)
        )

    def _print_sec_capex_tags(self, us_gaap, frequency):
        print("\n==============================")
        print(f"SEC NVDA CAPEX TAGS / {frequency.upper()}")
        print("==============================")

        for tag in NVDA_CAPEX_TAGS_TO_CHECK:
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
                expected_years = set(range(min(years), max(years) + 1))
                missing_years = sorted(expected_years - set(years))

                print("Jahre:", years)
                print("Fehlende Jahre:", missing_years)

    def _print_current_sec_result(self, frequency):
        sec_cf = self.sec.get_cashflow_statement(
            symbol="NVDA",
            frequency=frequency,
            use_cache=False,
            scope="core",
        )

        print("\n==============================")
        print(f"CURRENT SEC CASHFLOW RESULT / {frequency.upper()}")
        print("==============================")

        self.assertIsInstance(sec_cf, pd.DataFrame)

        for row in [
            "Operating Cash Flow",
            "Capital Expenditure",
            "Purchase Of PPE",
            "Capital Expenditure Reported",
            "Free Cash Flow",
        ]:
            if row in sec_cf.index:
                print(f"\n{row}:")
                print(
                    sec_cf.loc[row]
                    .dropna()
                    .sort_index(ascending=False)
                    .head(25)
                )
            else:
                print(f"\n{row}: FEHLT")

        print("\nSEC ATTRIBUTES")
        for attr in [
            "capex_quality",
            "capex_latest_date",
            "capex_source_tag",
            "capex_source_unit",
            "capex_data_points",
            "capex_missing_years",
            "capex_history_complete",
            "capex_warning",
            "fcf_quality",
            "fcf_latest_date",
            "fcf_source",
            "fcf_warning",
        ]:
            print(f"{attr}: {sec_cf.attrs.get(attr)}")

    def _run_nvda_debug(self):
        facts = self.sec.get_company_facts(
            "NVDA",
            use_cache=False,
        )

        self.assertIsInstance(facts, dict)
        self.assertNotIn("error", facts)

        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        self.assertTrue(us_gaap)

        for frequency in ["annual", "quarterly"]:
            print(
                f"\n\n########## NVDA CAPEX DEBUG / "
                f"{frequency.upper()} ##########"
            )

            self._print_yahoo_capex(frequency)
            self._print_sec_capex_tags(us_gaap, frequency)
            self._print_current_sec_result(frequency)

    def test_nvda_capex_debug(self):
        self._run_nvda_debug()


if __name__ == "__main__":
    unittest.main()