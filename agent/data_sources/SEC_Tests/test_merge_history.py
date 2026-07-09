import unittest

from data_sources.sec_source import SecSource


class TestCapExMergedHistory(unittest.TestCase):

    SYMBOLS = [
        "NVDA",
        "AAPL",
        "MSFT",
        "META",
        "CAT",
    ]

    def setUp(self):
        self.sec = SecSource(
            user_agent="gecen.efe1308@gmail.com"
        )

    def test_capex_history_merge(self):

        for symbol in self.SYMBOLS:

            print("\n")
            print("=" * 100)
            print(symbol)
            print("=" * 100)

            df = self.sec.get_cashflow_statement(
                symbol=symbol,
                frequency="annual",
                use_cache=True,
                scope="core",
            )

            self.assertFalse(
                isinstance(df, dict),
                msg=f"Fehler für {symbol}: {df}"
            )

            self.assertFalse(
                df.empty,
                msg=f"Leeres DataFrame für {symbol}"
            )

            self.assertIn(
                "Capital Expenditure",
                df.index,
                msg=f"Capital Expenditure fehlt für {symbol}"
            )

            capex = df.loc["Capital Expenditure"].dropna()

            self.assertFalse(
                capex.empty,
                msg=f"Keine CapEx-Werte für {symbol}"
            )

            latest_date = capex.index.max()
            oldest_date = capex.index.min()

            print("\nCAPEX-HISTORIE")
            print("-" * 80)

            print(
                f"Neuester Wert : {latest_date.date()} "
                f"| {capex.loc[latest_date]:,.0f}"
            )

            print(
                f"Ältester Wert : {oldest_date.date()} "
                f"| {capex.loc[oldest_date]:,.0f}"
            )

            print(
                f"Anzahl Perioden : {len(capex)}"
            )

            print("\nDATAFRAME ATTRIBUTES")
            print("-" * 80)

            print(
                "capex_quality:",
                df.attrs.get("capex_quality")
            )

            print(
                "capex_source_tag:",
                df.attrs.get("capex_source_tag")
            )

            print(
                "capex_source_unit:",
                df.attrs.get("capex_source_unit")
            )

            print(
                "capex_latest_date:",
                df.attrs.get("capex_latest_date")
            )

            print(
                "capex_data_points:",
                df.attrs.get("capex_data_points")
            )

            print(
                "capex_warning:",
                df.attrs.get("capex_warning")
            )

            print("\nLETZTE 20 PERIODEN")
            print("-" * 80)

            for date, value in capex.head(20).items():
                print(
                    f"{date.date()} | {value:,.0f}"
                )

    def test_nvda_full_history(self):

        df = self.sec.get_cashflow_statement(
            symbol="NVDA",
            frequency="annual",
            use_cache=True,
            scope="core",
        )

        self.assertFalse(
            isinstance(df, dict),
            msg=f"Fehler für NVDA: {df}"
        )

        capex = df.loc["Capital Expenditure"].dropna()

        print("\n")
        print("=" * 100)
        print("NVDA DETAILTEST")
        print("=" * 100)

        print(
            "Neuester Wert:",
            capex.index.max().date()
        )

        print(
            "Ältester Wert:",
            capex.index.min().date()
        )

        print(
            "Anzahl Werte:",
            len(capex)
        )

        print("\nGESAMTE HISTORIE")
        print("-" * 80)

        for date, value in capex.items():
            print(
                f"{date.date()} | {value:,.0f}"
            )

        #
        # Die neue Merge-Logik soll
        # deutlich weiter zurückreichen
        # als der ProductiveAssets-Tag alleine.
        #

        self.assertLess(
            capex.index.min().year,
            2020,
            msg="NVDA-Historie wurde nicht erweitert."
        )

    def test_nvda_source_tags(self):

        facts = self.sec.get_company_facts(
            "NVDA",
            use_cache=True,
        )

        self.assertNotIn(
            "error",
            facts,
            msg=str(facts)
        )

        us_gaap = facts["facts"]["us-gaap"]

        tags = [
            "PaymentsToAcquireProductiveAssets",
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireOtherPropertyPlantAndEquipment",
            "PaymentsToAcquirePropertyPlantEquipmentAndOtherProductiveAssets",
            "CapitalExpenditures",
        ]

        print("\n")
        print("=" * 100)
        print("NVDA SOURCE TAG ANALYSE")
        print("=" * 100)

        for tag in tags:

            if tag not in us_gaap:
                print(f"\n{tag}: NICHT VORHANDEN")
                continue

            series = self.sec._merge_available_series(
                us_gaap=us_gaap,
                sec_tags=[tag],
                frequency="annual",
                unit_preference=["USD"],
                statement_type="cashflow",
            )

            if series is None or series.empty:
                print(f"\n{tag}: LEER")
                continue

            print("\n")
            print("-" * 80)
            print(tag)
            print("-" * 80)

            print(
                "Von:",
                series.index.min().date()
            )

            print(
                "Bis:",
                series.index.max().date()
            )

            print(
                "Punkte:",
                len(series)
            )

            print("\nLETZTE 10 PERIODEN")

            for date, value in series.head(10).items():
                print(
                    f"{date.date()} | {value:,.0f}"
                )

    def test_nvda_merged_series_directly(self):

        facts = self.sec.get_company_facts(
            "NVDA",
            use_cache=True,
        )

        us_gaap = facts["facts"]["us-gaap"]

        merged, metadata = self.sec._merge_available_series(
            us_gaap=us_gaap,
            sec_tags=[
                "PaymentsToAcquireProductiveAssets",
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "PaymentsToAcquireOtherPropertyPlantAndEquipment",
                "PaymentsToAcquirePropertyPlantEquipmentAndOtherProductiveAssets",
                "CapitalExpenditures",
            ],
            frequency="annual",
            unit_preference=["USD"],
            statement_type="cashflow",
            return_metadata=True,
        )

        self.assertIsNotNone(merged)
        self.assertFalse(merged.empty)

        print("\n")
        print("=" * 100)
        print("NVDA MERGED SERIES")
        print("=" * 100)

        print(
            "Von:",
            merged.index.min().date()
        )

        print(
            "Bis:",
            merged.index.max().date()
        )

        print(
            "Punkte:",
            len(merged)
        )

        print("\nMETADATA")
        print("-" * 80)

        print(metadata)

        print("\nGESAMTE MERGED HISTORIE")
        print("-" * 80)

        for date, value in merged.items():
            print(
                f"{date.date()} | {value:,.0f}"
            )

        self.assertLess(
            merged.index.min().year,
            2020,
            msg="Merge hat die Historie nicht erweitert."
        )


if __name__ == "__main__":
    unittest.main()