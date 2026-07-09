import pandas as pd

from agent.data_sources.sec_source import SecSource


TICKER = "JPM"

sec = SecSource(
    user_agent="Efe Gecen efe@example.com"
)


CASH_RELATED_TAGS = [
    "CashAndCashEquivalentsAtCarryingValue",
    "CashAndDueFromBanks",
    "InterestBearingDepositsInBanks",
    "AvailableForSaleSecurities",
    "HeldToMaturitySecurities",

    # neue Kandidaten
    "DebtSecuritiesAvailableForSaleAndHeldToMaturityFairValue",
    "DebtSecuritiesAvailableForSaleAndHeldToMaturity",
    "DebtSecuritiesAvailableForSaleExcludingAccruedInterest",
    "AvailableForSaleSecuritiesDebtSecurities",
    "HeldToMaturitySecuritiesFairValue",
]


def print_separator(title):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def inspect_tag(us_gaap, tag):

    print_separator(f"TAG: {tag}")

    if tag not in us_gaap:
        print("NOT FOUND")
        return

    payload = us_gaap[tag]

    print("Label:")
    print(payload.get("label"))

    print("\nUnits:")
    print(list(payload.get("units", {}).keys()))

    for unit_name, values in payload.get("units", {}).items():

        print(f"\n--- RAW FACTS ({unit_name}) ---")

        df = pd.DataFrame(values)

        cols = [
            c for c in [
                "end",
                "val",
                "form",
                "fp",
                "fy",
                "filed",
            ]
            if c in df.columns
        ]

        print(
            df[cols]
            .sort_values("end", ascending=False)
            .head(15)
            .to_string(index=False)
        )

    if "USD" in payload.get("units", {}):

        series = sec._fact_values_to_series(
            values=payload["units"]["USD"],
            frequency="annual",
            statement_type=None,
        )

        print("\n--- ANNUAL SERIES ---")

        if series is None:
            print("None")
        else:

            print(series.head(10))

            latest = series.dropna()

            if not latest.empty:

                print(
                    f"\nLATEST = "
                    f"{latest.index[0].date()} -> "
                    f"{latest.iloc[0]:,.0f}"
                )


def print_single_series(us_gaap, tag):

    print_separator(f"SERIES CHECK: {tag}")

    if tag not in us_gaap:
        print("NOT FOUND")
        return

    payload = us_gaap[tag]

    if "USD" not in payload.get("units", {}):
        print("NO USD UNIT")
        return

    series = sec._fact_values_to_series(
        values=payload["units"]["USD"],
        frequency="annual",
        statement_type=None,
    )

    if series is None:
        print("None")
        return

    print(series.head(20))

    latest = series.dropna()

    if not latest.empty:

        print(
            "\nLATEST:",
            latest.index[0].date(),
            f"{latest.iloc[0]:,.0f}"
        )


def main():

    facts = sec.get_company_facts(
        TICKER,
        use_cache=False,
    )

    if isinstance(facts, dict) and "error" in facts:
        print(facts)
        return

    us_gaap = facts["facts"]["us-gaap"]

    #
    # Einzelanalyse
    #

    for tag in CASH_RELATED_TAGS:
        inspect_tag(us_gaap, tag)

    #
    # Spezifische Yahoo-Vergleichsanalyse
    #

    print_separator("YAHOO SECURITIES CANDIDATES")

    candidate_tags = [

        "DebtSecuritiesAvailableForSaleAndHeldToMaturityFairValue",

        "DebtSecuritiesAvailableForSaleAndHeldToMaturity",

        "DebtSecuritiesAvailableForSaleExcludingAccruedInterest",

        "AvailableForSaleSecurities",

        "AvailableForSaleSecuritiesDebtSecurities",

        "HeldToMaturitySecurities",

        "HeldToMaturitySecuritiesFairValue",
    ]

    for tag in candidate_tags:
        print_single_series(us_gaap, tag)

    #
    # Aktuelle Mapping-Ergebnisse
    #

    print_separator("CURRENT FINANCIAL MAPPING RESULT")

    for tag in [
        "DebtSecuritiesAvailableForSaleAndHeldToMaturityFairValue",
        "DebtSecuritiesAvailableForSaleAndHeldToMaturity",
        "AvailableForSaleSecurities",
        "HeldToMaturitySecurities",
    ]:

        series, meta = sec._first_available_series(
            us_gaap=us_gaap,
            sec_tags=[tag],
            frequency="annual",
            unit_preference=["USD"],
            return_metadata=True,
        )

        print("\n")
        print(tag)

        print("META:")
        print(meta)

        print("SERIES:")
        print(series)

    #
    # Balance Sheet
    #

    print_separator("CURRENT BALANCE SHEET OUTPUT")

    bs = sec.get_balance_sheet(
        TICKER,
        frequency="annual",
        use_cache=False,
    )

    if isinstance(bs, dict):
        print(bs)
        return

    for row in [
        "Cash And Cash Equivalents",
        "Other Short Term Investments",
        "Cash Cash Equivalents And Short Term Investments",
        "Long Term Debt",
        "Current Debt",
        "Total Debt",
    ]:

        print(f"\n--- {row} ---")

        if row in bs.index:
            print(bs.loc[row])
        else:
            print("NOT FOUND")


if __name__ == "__main__":
    main()