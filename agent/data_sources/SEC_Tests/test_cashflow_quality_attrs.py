import pandas as pd

from sec_source import SecSource


USER_AGENT = "Dein Name deine.email@example.com"


def print_series_summary(df: pd.DataFrame, row: str):
    if row not in df.index:
        print(f"{row}: FEHLT")
        return

    series = df.loc[row].dropna()

    if series.empty:
        print(f"{row}: LEER")
        return

    print(
        f"{row}: Latest={series.iloc[0]} | "
        f"Latest Date={series.index[0].strftime('%Y-%m-%d')} | "
        f"Oldest={series.iloc[-1]} | "
        f"Oldest Date={series.index[-1].strftime('%Y-%m-%d')} | "
        f"Items={len(series)}"
    )


def print_attrs(df: pd.DataFrame):
    print("\n--- ATTRS ---")

    attr_keys = [
        "capex_quality",
        "capex_latest_date",
        "capex_source_tag",
        "capex_source_unit",
        "capex_data_points",
        "capex_warning",
        "fcf_quality",
        "fcf_latest_date",
        "fcf_source",
        "fcf_warning",
    ]

    for key in attr_keys:
        print(f"{key}: {df.attrs.get(key)}")


def validate_attrs(df: pd.DataFrame):
    print("\n--- ATTR VALIDATION ---")

    required_attrs = [
        "capex_quality",
        "capex_latest_date",
        "capex_source_tag",
        "capex_source_unit",
        "capex_data_points",
    ]

    for attr in required_attrs:
        value = df.attrs.get(attr)

        if value is None:
            print(f"❌ {attr}: FEHLT")
        else:
            print(f"✅ {attr}: {value}")


def validate_capex_source(df: pd.DataFrame):
    print("\n--- CAPEX SOURCE VALIDATION ---")

    quality = df.attrs.get("capex_quality")
    source_tag = df.attrs.get("capex_source_tag")
    source_unit = df.attrs.get("capex_source_unit")
    data_points = df.attrs.get("capex_data_points")
    latest_date = df.attrs.get("capex_latest_date")
    warning = df.attrs.get("capex_warning")

    if "Capital Expenditure" not in df.index:
        print("❌ Capital Expenditure Row fehlt")
        return

    capex = df.loc["Capital Expenditure"].dropna()

    if capex.empty:
        print("❌ Capital Expenditure ist leer")
        return

    row_latest_date = capex.index[0].strftime("%Y-%m-%d")

    print(f"CapEx Quality: {quality}")
    print(f"CapEx Source Tag: {source_tag}")
    print(f"CapEx Source Unit: {source_unit}")
    print(f"CapEx Data Points: {data_points}")
    print(f"CapEx Latest Date Attr: {latest_date}")
    print(f"CapEx Latest Date Row: {row_latest_date}")
    print(f"CapEx Warning: {warning}")

    print("✅ Source Tag wurde korrekt mitgegeben" if source_tag else "❌ Source Tag fehlt")
    print("✅ Source Unit wurde korrekt mitgegeben" if source_unit else "❌ Source Unit fehlt")
    print("✅ Data Points wurden korrekt mitgegeben" if data_points is not None else "❌ Data Points fehlen")

    if latest_date == row_latest_date:
        print("✅ Latest Date stimmt mit der Row überein")
    else:
        print("⚠️ Latest Date weicht von der Row ab")


def validate_fcf_attrs(df: pd.DataFrame):
    print("\n--- FCF ATTR VALIDATION ---")

    fcf_quality = df.attrs.get("fcf_quality")
    fcf_latest_date = df.attrs.get("fcf_latest_date")
    fcf_source = df.attrs.get("fcf_source")
    fcf_warning = df.attrs.get("fcf_warning")

    print(f"FCF Quality: {fcf_quality}")
    print(f"FCF Latest Date: {fcf_latest_date}")
    print(f"FCF Source: {fcf_source}")
    print(f"FCF Warning: {fcf_warning}")

    if "Free Cash Flow" not in df.index:
        print("❌ Free Cash Flow Row fehlt")
        return

    fcf = df.loc["Free Cash Flow"].dropna()

    if fcf.empty:
        print("❌ Free Cash Flow ist leer")
        return

    row_latest_date = fcf.index[0].strftime("%Y-%m-%d")

    if fcf_latest_date == row_latest_date:
        print("✅ FCF Latest Date stimmt mit der Row überein")
    else:
        print("⚠️ FCF Latest Date weicht von der Row ab")

    if fcf_quality:
        print("✅ FCF Quality wurde gesetzt")
    else:
        print("❌ FCF Quality fehlt")

    if fcf_source:
        print("✅ FCF Source wurde gesetzt")
    else:
        print("❌ FCF Source fehlt")


def validate_fcf(df: pd.DataFrame):
    print("\n--- FCF CHECK ---")

    required_rows = [
        "Operating Cash Flow",
        "Capital Expenditure",
        "Free Cash Flow",
    ]

    for row in required_rows:
        if row not in df.index:
            print(f"❌ {row}: FEHLT")
            return

    ocf = df.loc["Operating Cash Flow"].dropna()
    capex = df.loc["Capital Expenditure"].dropna()
    fcf = df.loc["Free Cash Flow"].dropna()

    common_dates = ocf.index.intersection(capex.index).intersection(fcf.index)

    if len(common_dates) == 0:
        print("⚠️ Keine gemeinsamen Daten für OCF, CapEx und FCF")
        return

    date = common_dates[0]
    calculated = ocf.loc[date] - capex.loc[date]
    actual = fcf.loc[date]
    diff = abs(calculated - actual)

    print(
        f"{date.strftime('%Y-%m-%d')}: "
        f"{ocf.loc[date]} - {capex.loc[date]} = {calculated} | "
        f"Actual={actual} | Diff={diff}"
    )

    if diff < 1:
        print("✅ FCF stimmt exakt")
    else:
        print("⚠️ FCF weicht ab")


def validate_quality_warning(df: pd.DataFrame):
    print("\n--- QUALITY WARNING VALIDATION ---")

    quality = df.attrs.get("capex_quality")
    warning = df.attrs.get("capex_warning")

    if quality == "HIGH":
        print("✅ HIGH Quality ohne Warnung" if warning is None else f"⚠️ HIGH Quality hat Warnung: {warning}")
    elif quality == "MEDIUM":
        print(f"✅ MEDIUM Quality mit Warnung: {warning}" if warning else "⚠️ MEDIUM Quality ohne Warnung")
    elif quality == "LOW":
        print(f"⚠️ LOW Quality: {warning}")
    elif quality == "NONE":
        print("❌ Keine CapEx-Daten gefunden")
    else:
        print(f"⚠️ Unbekannte CapEx Quality: {quality}")


def print_cashflow_quality(sec: SecSource, symbol: str, frequency: str):
    print(f"\n\n==================== {symbol} | {frequency.upper()} ====================")

    df = sec.get_cashflow_statement(
        symbol=symbol,
        frequency=frequency,
        use_cache=False,
        scope="core",
    )

    if isinstance(df, dict) and "error" in df:
        print(df)
        return

    print("\n===== CASHFLOW CORE =====")
    print(df.head(10))

    print_attrs(df)
    validate_attrs(df)

    print("\n--- IMPORTANT ROWS ---")
    for row in [
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
    ]:
        print_series_summary(df, row)

    validate_capex_source(df)
    validate_quality_warning(df)
    validate_fcf_attrs(df)
    validate_fcf(df)


def main():
    sec = SecSource(user_agent=USER_AGENT)

    tests = [
        ("BABA", "annual"),
        ("BABA", "quarterly"),
        ("AAPL", "annual"),
        ("AAPL", "quarterly"),
        ("GOOGL", "annual"),
        ("META", "annual"),
        ("AMZN", "annual"),
    ]

    for symbol, frequency in tests:
        print_cashflow_quality(sec, symbol, frequency)

    print("\n\n✅ Cashflow Quality Attrs Test abgeschlossen.")


if __name__ == "__main__":
    main()