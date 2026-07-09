# agent/data_sources/SEC_Tests/debug_tests/test_jpm_vs_yahoo.py

import pandas as pd
import yfinance as yf

from agent.data_sources.sec_source import SecSource


TICKER = "JPM"

sec = SecSource(
    user_agent="Efe Gecen efe@example.com"
)

TEST_TAGS = [
    # Cash
    "CashAndCashEquivalentsAtCarryingValue",
    "CashAndDueFromBanks",
    "InterestBearingDepositsInBanks",

    # Securities
    "AvailableForSaleSecurities",
    "HeldToMaturitySecurities",

    # Repo Assets
    "FederalFundsSoldAndSecuritiesPurchasedUnderAgreementsToResell",
    "SecuritiesPurchasedUnderAgreementsToResell",

    # Loans
    "LoansReceivableNet",
    "LoansAndLeasesReceivableNetReportedAmount",

    # Funding
    "Deposits",

    # Debt
    "LongTermDebt",
    "OtherBorrowings",
    "ShortTermBorrowings",

    # Core BS
    "Assets",
    "Liabilities",
    "StockholdersEquity",
]


def get_latest_fact_value(facts, tag):

    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    if tag not in us_gaap:
        return None

    payload = us_gaap[tag]
    units = payload.get("units", {})

    best_value = None
    best_date = None

    for unit_values in units.values():

        for item in unit_values:

            end = item.get("end")
            value = item.get("val")

            if end is None or value is None:
                continue

            try:
                end_date = pd.to_datetime(end)
            except Exception:
                continue

            if best_date is None or end_date > best_date:
                best_date = end_date
                best_value = value

    return best_value


def print_yahoo_balance_sheet(ticker):

    stock = yf.Ticker(ticker)

    print("\n" + "=" * 100)
    print("YAHOO BALANCE SHEET (LATEST PERIOD)")
    print("=" * 100)

    try:
        bs = stock.balance_sheet

        if bs.empty:
            print("No Yahoo balance sheet found.")
            return

        latest_col = bs.columns[0]

        interesting_rows = [
            "Cash And Cash Equivalents",
            "Cash Cash Equivalents And Short Term Investments",
            "Other Short Term Investments",
            "Available For Sale Securities",
            "Held To Maturity Securities",
            "Net Loans",
            "Loans",
            "Deposits",
            "Total Assets",
            "Total Liabilities Net Minority Interest",
            "Stockholders Equity",
            "Long Term Debt",
        ]

        for row in interesting_rows:

            if row not in bs.index:
                continue

            value = bs.loc[row, latest_col]

            print(
                f"{row:<60}"
                f"{value:>20,.0f}"
            )

    except Exception as e:
        print(f"Yahoo error: {e}")


def main():

    facts = sec.get_company_facts(
        TICKER,
        use_cache=False,
    )

    if isinstance(facts, dict) and "error" in facts:
        print(facts)
        return

    print("\n" + "=" * 100)
    print(f"SEC COMPANYFACTS TAGS ({TICKER})")
    print("=" * 100)

    sec_values = {}

    for tag in TEST_TAGS:

        value = get_latest_fact_value(
            facts=facts,
            tag=tag,
        )

        sec_values[tag] = value

        if value is None:

            print(
                f"{tag:<80}"
                f"NOT FOUND"
            )

        else:

            print(
                f"{tag:<80}"
                f"{value:>20,.0f}"
            )

    print("\n" + "=" * 100)
    print("DERIVED COMBINATIONS")
    print("=" * 100)

    cash_combo = (
        (sec_values.get("CashAndCashEquivalentsAtCarryingValue") or 0)
        + (sec_values.get("CashAndDueFromBanks") or 0)
        + (sec_values.get("InterestBearingDepositsInBanks") or 0)
    )

    securities_combo = (
        (sec_values.get("AvailableForSaleSecurities") or 0)
        + (sec_values.get("HeldToMaturitySecurities") or 0)
    )

    cash_plus_securities = (
        cash_combo
        + securities_combo
    )

    repo_assets = (
        (sec_values.get(
            "FederalFundsSoldAndSecuritiesPurchasedUnderAgreementsToResell"
        ) or 0)
        + (sec_values.get(
            "SecuritiesPurchasedUnderAgreementsToResell"
        ) or 0)
    )

    print(
        f"{'Cash Combo':<60}"
        f"{cash_combo:>20,.0f}"
    )

    print(
        f"{'Securities Combo':<60}"
        f"{securities_combo:>20,.0f}"
    )

    print(
        f"{'Cash + Securities':<60}"
        f"{cash_plus_securities:>20,.0f}"
    )

    print(
        f"{'Repo Assets':<60}"
        f"{repo_assets:>20,.0f}"
    )

    print_yahoo_balance_sheet(TICKER)


if __name__ == "__main__":
    main()