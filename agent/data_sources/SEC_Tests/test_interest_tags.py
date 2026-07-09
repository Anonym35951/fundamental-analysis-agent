from data_sources.sec_source import SecSource


def inspect_interest_tags():
    sec = SecSource(user_agent="gecen.efe1308@gmail.com")

    symbols = ["AAPL", "MSFT", "LCID"]

    interest_tags = [
        "InterestExpense",
        "InterestExpenseDebt",
        "InterestExpenseNonoperating",
        "FinanceLeaseInterestExpense",
        "InterestCostsIncurred",
        "InterestPaid",
        "InterestPaidNet",
    ]

    for symbol in symbols:
        print(f"\n==================== {symbol} ====================")

        facts = sec.get_company_facts(symbol, use_cache=False)

        if isinstance(facts, dict) and "error" in facts:
            print(facts)
            continue

        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        for tag in interest_tags:
            if tag not in us_gaap:
                continue

            print(f"\n--- {tag} ---")

            fact = us_gaap[tag]
            units = fact.get("units", {})

            for unit, values in units.items():
                series = sec._fact_values_to_series(values, frequency="annual")

                if series is not None and not series.dropna().empty:
                    latest_date = series.dropna().index[0]
                    latest_value = series.dropna().iloc[0]

                    print(
                        f"Annual | Unit: {unit} | "
                        f"Latest Date: {latest_date.date()} | "
                        f"Value: {latest_value}"
                    )

                q_series = sec._fact_values_to_series(values, frequency="quarterly")

                if q_series is not None and not q_series.dropna().empty:
                    latest_date = q_series.dropna().index[0]
                    latest_value = q_series.dropna().iloc[0]

                    print(
                        f"Quarterly | Unit: {unit} | "
                        f"Latest Date: {latest_date.date()} | "
                        f"Value: {latest_value}"
                    )


if __name__ == "__main__":
    inspect_interest_tags()