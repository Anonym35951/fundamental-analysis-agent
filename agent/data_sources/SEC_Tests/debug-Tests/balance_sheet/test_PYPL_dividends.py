# agent/data_sources/SEC_Tests/debug_tests/test_pypl_dividends.py

import pandas as pd

from agent.data_sources.sec_source import SecSource


TICKER = "PYPL"

sec = SecSource(
    user_agent="Efe Gecen efe@example.com"
)

DIVIDEND_TAGS = [
    "PaymentsOfDividends",
    "PaymentsOfDividendsCommonStock",
    "PaymentsOfDividendsPreferredStockAndPreferenceStock",
    "DividendsCash",
    "DividendsCommonStockCash",
]

CASHFLOW_ROWS = [
    "Dividends Paid",
    "Operating Cash Flow",
    "Free Cash Flow",
    "Capital Expenditure",
]


def print_raw_sec_tag(facts, tag):

    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    print("\n" + "=" * 120)
    print(f"SEC TAG: {tag}")
    print("=" * 120)

    if tag not in us_gaap:
        print("NOT FOUND")
        return

    payload = us_gaap[tag]

    print(f"Label       : {payload.get('label')}")
    print(f"Description : {payload.get('description')}")

    units = payload.get("units", {})

    for unit_name, values in units.items():

        print("\n" + "-" * 120)
        print(f"UNIT: {unit_name}")
        print("-" * 120)

        rows = []

        for item in values:

            rows.append({
                "end": item.get("end"),
                "start": item.get("start"),
                "val": item.get("val"),
                "fy": item.get("fy"),
                "fp": item.get("fp"),
                "form": item.get("form"),
                "filed": item.get("filed"),
            })

        if not rows:
            continue

        df = pd.DataFrame(rows)

        if "end" in df.columns:
            df["end"] = pd.to_datetime(df["end"], errors="coerce")

        df = df.sort_values(
            "end",
            ascending=False,
        )

        print(df.to_string(index=False))


def print_sec_cashflow_core():

    print("\n" + "=" * 120)
    print("SEC CASHFLOW CORE")
    print("=" * 120)

    df = sec.get_cashflow_statement(
        TICKER,
        frequency="annual",
        use_cache=False,
        scope="core",
    )

    if isinstance(df, dict):
        print(df)
        return

    for row in CASHFLOW_ROWS:

        if row not in df.index:
            continue

        print("\n" + "-" * 120)
        print(row)
        print("-" * 120)

        print(df.loc[row].dropna())


def print_sec_cashflow_raw_dividend_tags():

    print("\n" + "=" * 120)
    print("SEC RAW DIVIDEND TAGS")
    print("=" * 120)

    result = sec.get_cashflow_statement_raw_labeled(
        TICKER,
        frequency="annual",
        use_cache=False,
    )

    if isinstance(result, dict):
        print(result)
        return

    df, metadata = result

    for row in df.index:

        if "dividend" not in row.lower():
            continue

        print("\n" + "-" * 120)
        print(row)
        print("-" * 120)

        print(df.loc[row].dropna())

        print("\nMetadata:")
        print(metadata.get(row))


def print_companyfacts_dividend_tags():

    facts = sec.get_company_facts(
        TICKER,
        use_cache=False,
    )

    if isinstance(facts, dict) and "error" in facts:
        print(facts)
        return

    print("\n" + "=" * 120)
    print("COMPANYFACTS DIVIDEND TAGS")
    print("=" * 120)

    for tag in DIVIDEND_TAGS:
        print_raw_sec_tag(facts, tag)


def main():

    print("\n")
    print("=" * 120)
    print(f"DIVIDEND DEBUG TEST FOR {TICKER}")
    print("=" * 120)

    print_companyfacts_dividend_tags()

    print_sec_cashflow_core()

    print_sec_cashflow_raw_dividend_tags()


if __name__ == "__main__":
    main()