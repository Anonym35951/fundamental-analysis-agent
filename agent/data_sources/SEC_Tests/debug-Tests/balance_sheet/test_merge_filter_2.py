import unittest
import pandas as pd

from agent.data_sources.sec_source import SecSource


class TestNVDAParser(unittest.TestCase):

    def setUp(self):
        self.sec = SecSource(
            user_agent="Dein Name deine@email.de"
        )

    def test_parser_output(self):

        facts = self.sec.get_company_facts("NVDA")

        tag = "PaymentsToAcquirePropertyPlantAndEquipment"

        values = (
            facts["facts"]["us-gaap"][tag]
            .get("units", {})
            .get("USD", [])
        )

        print("\n")
        print("=" * 120)
        print("RAW INPUT VALUES")
        print("=" * 120)
        print(f"ANZAHL ROH-EINTRÄGE: {len(values)}")
        print()

        for item in values:

            print(
                f"END={item.get('end')} | "
                f"START={item.get('start')} | "
                f"FY={item.get('fy')} | "
                f"FP={item.get('fp')} | "
                f"FORM={item.get('form')} | "
                f"FILED={item.get('filed')} | "
                f"VAL={item.get('val'):,}"
                if item.get("val") is not None
                else f"END={item.get('end')}"
            )

        #
        # Nur FY-Einträge separat anzeigen
        #

        print("\n")
        print("=" * 120)
        print("NUR FY-EINTRÄGE")
        print("=" * 120)

        fy_rows = [
            x for x in values
            if x.get("fp") == "FY"
        ]

        print(f"ANZAHL FY-EINTRÄGE: {len(fy_rows)}")
        print()

        for item in sorted(
                fy_rows,
                key=lambda x: x.get("end", "")
        ):
            print(
                f"END={item.get('end')} | "
                f"FY={item.get('fy')} | "
                f"FORM={item.get('form')} | "
                f"FILED={item.get('filed')} | "
                f"VAL={item.get('val'):,}"
            )

        #
        # Parser aufrufen
        #

        df = self.sec._fact_values_to_dataframe(
            values=values,
            frequency="annual",
            statement_type="cashflow",
        )

        print("\n")
        print("=" * 120)
        print("NACH _fact_values_to_dataframe()")
        print("=" * 120)

        if df is None or df.empty:
            print("LEER")
            return

        print(f"ANZAHL ZEILEN: {len(df)}")
        print()

        print(df.to_string())

        #
        # Nur relevante Spalten
        #

        print("\n")
        print("=" * 120)
        print("KOMPAKTE SICHT")
        print("=" * 120)

        cols = [
            c for c in [
                "date",
                "start",
                "duration_days",
                "value",
                "fy",
                "fp",
                "form",
                "filed",
            ]
            if c in df.columns
        ]

        print(df[cols].to_string())

        #
        # Jahresübersicht
        #

        print("\n")
        print("=" * 120)
        print("JAHRESÜBERSICHT")
        print("=" * 120)

        for _, row in df.sort_values("date").iterrows():

            date = row["date"]

            if pd.notna(date):
                year = date.year
            else:
                year = "N/A"

            print(
                f"{year} | "
                f"{row.get('fy')} | "
                f"{row.get('fp')} | "
                f"{row.get('form')} | "
                f"{row.get('value'):,.0f}"
            )


if __name__ == "__main__":
    unittest.main()