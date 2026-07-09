import unittest
from pprint import pprint

from data_sources.sec_source import SecSource


CAPEX_TAGS = [
    "PaymentsToAcquireProductiveAssets",
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireOtherPropertyPlantAndEquipment",
    "PaymentsToAcquirePropertyPlantEquipmentAndOtherProductiveAssets",
    "CapitalExpenditures",
]


class TestCapexTagCoverage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sec = SecSource(
            user_agent="deine-email@example.com"
        )

    def _inspect_symbol(self, symbol):

        facts = self.sec.get_company_facts(
            symbol,
            use_cache=False,
        )

        self.assertNotIn(
            "error",
            facts,
            f"SEC Fehler für {symbol}"
        )

        us_gaap = facts["facts"]["us-gaap"]

        print("\n")
        print("=" * 80)
        print(symbol)
        print("=" * 80)

        best_tag = None
        best_date = None

        for tag in CAPEX_TAGS:

            print("\n")
            print("-" * 80)
            print(tag)
            print("-" * 80)

            if tag not in us_gaap:
                print("NICHT VORHANDEN")
                continue

            units = us_gaap[tag].get("units", {})
            usd_data = units.get("USD", [])

            if not usd_data:
                print("KEINE USD DATEN")
                continue

            latest = max(
                usd_data,
                key=lambda x: x.get("end", "")
            )

            latest_date = latest.get("end")
            latest_value = latest.get("val")

            print(f"Latest Date  : {latest_date}")
            print(f"Latest Value : {latest_value:,}")

            if (
                best_date is None
                or latest_date > best_date
            ):
                best_tag = tag
                best_date = latest_date

            #
            # Letzte 10 Datensätze anzeigen
            #

            sorted_rows = sorted(
                usd_data,
                key=lambda x: x.get("end", ""),
                reverse=True,
            )

            print("\nLETZTE EINTRÄGE:")

            for row in sorted_rows[:10]:

                print(
                    f"{row.get('end')} | "
                    f"{row.get('form')} | "
                    f"{row.get('fp')} | "
                    f"{row.get('val'):,}"
                )

        print("\n")
        print("=" * 80)
        print("BESTER TAG")
        print("=" * 80)
        print(f"Tag        : {best_tag}")
        print(f"Latest Date: {best_date}")

        self.assertIsNotNone(
            best_tag,
            f"Kein CapEx Tag gefunden für {symbol}"
        )

    #
    # Problemfall NVDA
    #

    def test_nvda_capex(self):
        self._inspect_symbol("NVDA")

    #
    # Weitere Vergleichswerte
    #

    def test_aapl_capex(self):
        self._inspect_symbol("AAPL")

    def test_msft_capex(self):
        self._inspect_symbol("MSFT")

    def test_cat_capex(self):
        self._inspect_symbol("CAT")

    def test_meta_capex(self):
        self._inspect_symbol("META")


if __name__ == "__main__":
    unittest.main()