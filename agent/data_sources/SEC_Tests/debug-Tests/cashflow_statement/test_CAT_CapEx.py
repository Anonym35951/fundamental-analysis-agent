import unittest
import yfinance as yf

from data_sources.sec_source import SecSource


CAPEX_TAGS_TO_CHECK = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
    "SegmentExpenditureAdditionToLongLivedAssets",
    "PaymentsToAcquireEquipmentOnLease",
    "CapitalExpenditures",
    "PropertyAndEquipmentAdditions",
    "CapitalSpending",
]


class TestCapexDiscovery(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sec = SecSource(
            user_agent="deine-email@example.com"
        )

    def _print_yahoo_capex(self, symbol, frequency):

        ticker = yf.Ticker(symbol)

        yf_cf = (
            ticker.cashflow
            if frequency == "annual"
            else ticker.quarterly_cashflow
        )

        print("\n==============================")
        print("YAHOO CAPEX")
        print("==============================")

        if "Capital Expenditure" in yf_cf.index:

            print(
                yf_cf.loc["Capital Expenditure"]
                .dropna()
                .head(10)
            )

        else:
            print("Yahoo hat keinen CapEx-Eintrag.")

    def _print_specific_capex_tags(self, symbol, frequency):

        facts = self.sec.get_company_facts(
            symbol,
            use_cache=False,
        )

        us_gaap = facts["facts"]["us-gaap"]

        print("\n==============================")
        print("SEC CAPEX TAGS")
        print("==============================")

        for tag in CAPEX_TAGS_TO_CHECK:

            payload = us_gaap.get(tag)

            if not payload:
                continue

            values = payload.get("units", {}).get("USD")

            if not values:
                continue

            series = self.sec._fact_values_to_series(
                values=values,
                frequency=frequency,
                statement_type="cashflow",
            )

            if series is None or series.dropna().empty:
                continue

            print(f"\nTAG: {tag}")
            print(f"LABEL: {payload.get('label', '')}")

            print(
                series.dropna()
                .sort_index(ascending=False)
                .head(10)
            )

    def _print_actual_sec_capex(self, symbol, frequency):

        print("\n==============================")
        print("SEC CASHFLOW RESULT")
        print("==============================")

        cf = self.sec.get_cashflow_statement(
            symbol=symbol,
            frequency=frequency,
            use_cache=False,
        )

        if isinstance(cf, dict):
            print(cf)
            return

        if "Capital Expenditure" in cf.index:

            print("\nCapital Expenditure:")

            print(
                cf.loc["Capital Expenditure"]
                .dropna()
                .head(10)
            )

        else:
            print("Capital Expenditure nicht vorhanden.")

        if "Free Cash Flow" in cf.index:

            print("\nFree Cash Flow:")

            print(
                cf.loc["Free Cash Flow"]
                .dropna()
                .head(10)
            )

        print("\nSEC ATTRIBUTES")

        for key in [
            "capex_quality",
            "capex_latest_date",
            "capex_source_tag",
            "capex_source_unit",
            "capex_data_points",
            "capex_warning",
            "fcf_quality",
            "fcf_latest_date",
            "fcf_source",
            "fcf_warning",
        ]:
            print(f"{key}: {cf.attrs.get(key)}")

    def _run(self, symbol):

        for frequency in ["annual", "quarterly"]:

            print(
                f"\n\n########## "
                f"{symbol} / {frequency.upper()} "
                f"##########"
            )

            self._print_yahoo_capex(
                symbol,
                frequency,
            )

            self._print_specific_capex_tags(
                symbol,
                frequency,
            )

            self._print_actual_sec_capex(
                symbol,
                frequency,
            )

    def test_cat(self):
        self._run("CAT")

    def test_de(self):
        self._run("DE")


if __name__ == "__main__":
    unittest.main()