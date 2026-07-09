# test_cashflow_robustness.py

from agent.data_sources.sec_source import SecSource


def print_line_item(df, line_item):
    if line_item not in df.index:
        print(f"{line_item}: FEHLT")
        return

    series = df.loc[line_item].dropna()

    if series.empty:
        print(f"{line_item}: LEER")
        return

    latest_date = series.index[0]
    latest_value = series.iloc[0]

    oldest_date = series.index[-1]
    oldest_value = series.iloc[-1]

    print(
        f"{line_item}: "
        f"Latest={latest_value} | Latest Date={latest_date.date()} | "
        f"Oldest={oldest_value} | Oldest Date={oldest_date.date()} | "
        f"Items={len(series)}"
    )


def validate_fcf(df):
    if "Operating Cash Flow" not in df.index:
        print("FCF Check: Operating Cash Flow fehlt")
        return

    if "Capital Expenditure" not in df.index:
        print("FCF Check: Capital Expenditure fehlt")
        return

    if "Free Cash Flow" not in df.index:
        print("FCF Check: Free Cash Flow fehlt")
        return

    ocf = df.loc["Operating Cash Flow"].dropna()
    capex = df.loc["Capital Expenditure"].dropna()
    fcf = df.loc["Free Cash Flow"].dropna()

    common_dates = list(
        set(ocf.index).intersection(set(capex.index)).intersection(set(fcf.index))
    )

    if not common_dates:
        print("FCF Check: Keine gemeinsamen Datenpunkte gefunden")
        return

    latest_date = sorted(common_dates, reverse=True)[0]

    expected_fcf = ocf.loc[latest_date] - capex.loc[latest_date]
    actual_fcf = fcf.loc[latest_date]

    print(
        f"FCF Check {latest_date.date()}: "
        f"{ocf.loc[latest_date]} - {capex.loc[latest_date]} = {expected_fcf} "
        f"| Actual={actual_fcf}"
    )


def test_symbol(symbol, frequency):
    print(f"\n{'=' * 20} {symbol} | {frequency.upper()} {'=' * 20}")

    sec = SecSource(
        user_agent="Your Name your@email.com"
    )

    df = sec.get_cashflow_statement(
        symbol=symbol,
        frequency=frequency,
        use_cache=False,
        scope="core",
    )

    if isinstance(df, dict) and "error" in df:
        print(df["error"])
        return

    print("\n===== CASHFLOW CORE =====")
    print(df.head())
    print(f"\nRows: {len(df.index)}")
    print(f"Columns: {len(df.columns)}")

    important_rows = [
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
        "Capital Expenditure",
        "Purchase Of PPE",
        "Capital Expenditure Reported",
        "Free Cash Flow",
        "Dividends Paid",
        "Repurchase Of Capital Stock",
        "Net Cash From Investing",
        "Net Cash From Financing",
        "Acquisitions",
        "Debt Issuance",
        "Debt Repayment",
    ]

    print("\n--- Cashflow Robustness Test ---")

    for row in important_rows:
        print_line_item(df, row)

    print("\n--- Free Cash Flow Validation ---")
    validate_fcf(df)

    print("\n--- CapEx Source Check ---")

    raw_df = sec.get_cashflow_statement(
        symbol=symbol,
        frequency=frequency,
        use_cache=False,
        scope="raw",
    )

    capex_tags = [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireOtherPropertyPlantAndEquipment",
        "CapitalExpenditures",
        "PaymentsToAcquireBusinessesNetOfCashAcquired",
        "PaymentsToAcquireIntangibleAssets",
        "PaymentsToAcquireInterestInSubsidiariesAndAffiliates",
        "PaymentsToAcquireEquityMethodInvestments",
    ]

    for tag in capex_tags:
        if tag in raw_df.index:
            series = raw_df.loc[tag].dropna()
            if not series.empty:
                print(
                    f"{tag}: FOUND | "
                    f"Latest={series.iloc[0]} | "
                    f"Date={series.index[0].date()} | "
                    f"Items={len(series)}"
                )
            else:
                print(f"{tag}: LEER")
        else:
            print(f"{tag}: NICHT VORHANDEN")


if __name__ == "__main__":
    test_cases = [
        ("BABA", "annual"),
        ("BABA", "quarterly"),
        ("AMZN", "annual"),
        ("GOOGL", "annual"),
        ("META", "annual"),
    ]

    for symbol, frequency in test_cases:
        test_symbol(symbol, frequency)

    print("\n✅ Cashflow Robustness Test abgeschlossen.")