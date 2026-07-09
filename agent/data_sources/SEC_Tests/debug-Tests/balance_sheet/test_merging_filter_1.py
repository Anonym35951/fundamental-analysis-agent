import unittest

from agent.data_sources.sec_source import SecSource


class TestNVDARawCapex(unittest.TestCase):

    def setUp(self):
        self.sec = SecSource(
            user_agent="Dein Name deine@email.de"
        )

    def test_nvda_raw_capex_tags(self):

        facts = self.sec.get_company_facts("NVDA")

        us_gaap = facts["facts"]["us-gaap"]

        tags = [
            "PaymentsToAcquireProductiveAssets",
            "PaymentsToAcquirePropertyPlantAndEquipment",
        ]

        for tag in tags:

            print("\n" + "=" * 120)
            print(tag)
            print("=" * 120)

            if tag not in us_gaap:
                print("NICHT VORHANDEN")
                continue

            units = us_gaap[tag].get("units", {})

            if "USD" not in units:
                print("KEINE USD DATEN")
                continue

            values = units["USD"]

            print(f"ANZAHL ROH-EINTRÄGE: {len(values)}")

            values = sorted(
                values,
                key=lambda x: x.get("end", ""),
                reverse=True,
            )

            for row in values:

                print(
                    f"END={row.get('end')} | "
                    f"START={row.get('start')} | "
                    f"FY={row.get('fy')} | "
                    f"FP={row.get('fp')} | "
                    f"FORM={row.get('form')} | "
                    f"FILED={row.get('filed')} | "
                    f"VAL={row.get('val'):,}"
                )


if __name__ == "__main__":
    unittest.main()