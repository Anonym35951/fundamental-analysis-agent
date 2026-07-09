import unittest

from agent.data_sources.sec_source import SecSource


class TestNVDACapexTags(unittest.TestCase):

    def setUp(self):
        self.sec = SecSource(
            user_agent="Dein Name deine@email.de"
        )

    def test_find_missing_capex_history(self):

        facts = self.sec.get_company_facts("NVDA")

        us_gaap = facts["facts"]["us-gaap"]

        capex_candidates = [
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets",
            "PaymentsToAcquireOtherPropertyPlantAndEquipment",
            "PaymentsToAcquirePropertyPlantEquipmentAndOtherProductiveAssets",
            "CapitalExpenditures",
            "PropertyAndEquipmentAdditions",
            "CapitalSpending",
            "CapitalExpendituresIncurredButNotYetPaid",
        ]

        print("\n")
        print("=" * 120)
        print("NVDA CAPEX TAG DISCOVERY")
        print("=" * 120)

        for tag in capex_candidates:

            print("\n")
            print("-" * 120)
            print(tag)
            print("-" * 120)

            fact = us_gaap.get(tag)

            if not fact:
                print("NICHT VORHANDEN")
                continue

            units = fact.get("units", {})

            if not units:
                print("KEINE UNITS")
                continue

            for unit_name, values in units.items():

                print(f"\nUNIT: {unit_name}")
                print(f"ROH-EINTRÄGE: {len(values)}")

                fy_rows = [
                    x for x in values
                    if x.get("fp") == "FY"
                ]

                print(f"FY-EINTRÄGE: {len(fy_rows)}")

                if not fy_rows:
                    continue

                fy_rows = sorted(
                    fy_rows,
                    key=lambda x: x.get("end", "")
                )

                first = fy_rows[0]
                last = fy_rows[-1]

                print(
                    f"ERSTER FY: {first.get('end')} "
                    f"| VAL={first.get('val'):,}"
                )

                print(
                    f"LETZTER FY: {last.get('end')} "
                    f"| VAL={last.get('val'):,}"
                )

                print("\nALLE FY-WERTE")

                for row in fy_rows:

                    val = row.get("val")

                    try:
                        val_str = f"{float(val):,.0f}"
                    except Exception:
                        val_str = str(val)

                    print(
                        f"END={row.get('end')} | "
                        f"FY={row.get('fy')} | "
                        f"FORM={row.get('form')} | "
                        f"FILED={row.get('filed')} | "
                        f"VAL={val_str}"
                    )

    def test_find_all_capex_like_tags(self):

        facts = self.sec.get_company_facts("NVDA")

        us_gaap = facts["facts"]["us-gaap"]

        keywords = [
            "Capital",
            "Expenditure",
            "Property",
            "Plant",
            "Equipment",
            "Productive",
            "Acquire",
            "Addition",
            "Capex",
        ]

        print("\n")
        print("=" * 120)
        print("ALLE POTENTIELLEN CAPEX TAGS")
        print("=" * 120)

        matches = []

        for tag in sorted(us_gaap.keys()):

            tag_lower = tag.lower()

            if any(
                keyword.lower() in tag_lower
                for keyword in keywords
            ):
                matches.append(tag)

        for tag in matches:
            print(tag)

        print("\n")
        print(f"ANZAHL GEFUNDENER TAGS: {len(matches)}")

    def test_find_q4_entries(self):

        facts = self.sec.get_company_facts("NVDA")

        us_gaap = facts["facts"]["us-gaap"]

        tags = [
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets",
        ]

        print("\n")
        print("=" * 120)
        print("NVDA Q4 ANALYSE")
        print("=" * 120)

        for tag in tags:

            print("\n")
            print("-" * 120)
            print(tag)
            print("-" * 120)

            fact = us_gaap.get(tag)

            if not fact:
                print("NICHT VORHANDEN")
                continue

            units = fact.get("units", {})

            values = units.get("USD", [])

            q4_rows = [
                row for row in values
                if row.get("fp") == "Q4"
            ]

            print(f"Q4 EINTRÄGE: {len(q4_rows)}")

            if not q4_rows:
                continue

            q4_rows = sorted(
                q4_rows,
                key=lambda x: x.get("end", "")
            )

            for row in q4_rows:

                val = row.get("val")

                try:
                    val_str = f"{float(val):,.0f}"
                except Exception:
                    val_str = str(val)

                print(
                    f"END={row.get('end')} | "
                    f"START={row.get('start')} | "
                    f"FY={row.get('fy')} | "
                    f"FP={row.get('fp')} | "
                    f"FORM={row.get('form')} | "
                    f"FILED={row.get('filed')} | "
                    f"VAL={val_str}"
                )


if __name__ == "__main__":
    unittest.main()