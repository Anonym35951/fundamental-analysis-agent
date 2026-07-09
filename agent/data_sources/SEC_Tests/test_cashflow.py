from data_sources.sec_source import SecSource


def print_line_item_result(result):
    if isinstance(result, dict) and "error" in result:
        print(result)
        return

    print("Symbol:", result.get("symbol"))
    print("Frequency:", result.get("frequency"))
    print("Scope:", result.get("scope"))
    print("Line Item:", result.get("line_item"))
    print("Tag:", result.get("tag"))
    print("Label:", result.get("label"))
    print("Units:", result.get("units"))
    print("Date:", result.get("date"))
    print("Value:", result.get("value"))

    series = result.get("series", {})
    print("Series Items:", len(series))


def print_series_diagnostics(df, row_name):
    if row_name not in df.index:
        print(f"{row_name}: FEHLT")
        return

    series = df.loc[row_name].dropna()

    if series.empty:
        print(f"{row_name}: VORHANDEN, aber keine Werte")
        return

    latest_date = series.index[0]
    latest_value = series.iloc[0]

    oldest_date = series.index[-1]
    oldest_value = series.iloc[-1]

    print(
        f"{row_name}: "
        f"Latest={latest_value} | Latest Date={latest_date.strftime('%Y-%m-%d')} | "
        f"Oldest={oldest_value} | Oldest Date={oldest_date.strftime('%Y-%m-%d')} | "
        f"Items={len(series)}"
    )

    assert latest_date >= oldest_date


def assert_columns_are_descending_dates(df):
    columns = list(df.columns)

    if len(columns) <= 1:
        return

    for i in range(len(columns) - 1):
        assert columns[i] >= columns[i + 1], (
            f"Spalten sind nicht absteigend sortiert: "
            f"{columns[i]} < {columns[i + 1]}"
        )


def assert_latest_value_is_first(df, row_name):
    if row_name not in df.index:
        return

    series = df.loc[row_name].dropna()

    if len(series) <= 1:
        return

    assert series.index[0] == max(series.index), (
        f"Latest Value steht nicht vorne für '{row_name}'. "
        f"Erster Wert: {series.index[0]}, Max Date: {max(series.index)}"
    )


def test_fact_values_to_dataframe(sec, symbol, frequency, tag):
    print(f"\n--- FACT VALUES DATAFRAME TEST | {symbol} | {frequency} | {tag} ---")

    facts = sec.get_company_facts(symbol, use_cache=False)

    if isinstance(facts, dict) and "error" in facts:
        print(facts)
        return

    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    fact = us_gaap.get(tag)

    if not fact:
        print(f"Tag nicht gefunden: {tag}")
        return

    units = fact.get("units", {})

    for unit in ["USD", "USD/shares", "shares", "pure", "CNY"]:
        values = units.get(unit)

        if not values:
            continue

        df = sec._fact_values_to_dataframe(values, frequency)

        if df is None or df.empty:
            continue

        print(f"Unit: {unit}")
        print(df.head(10))

        assert "date" in df.columns
        assert "value" in df.columns
        assert "filed" in df.columns
        assert "fy" in df.columns
        assert "fp" in df.columns
        assert "form" in df.columns
        assert "accn" in df.columns
        assert "frame" in df.columns

        assert df["date"].iloc[0] == df["date"].max()

        duplicated_dates = df["date"].duplicated().sum()
        assert duplicated_dates == 0, f"Doppelte Dates nach Bereinigung gefunden: {duplicated_dates}"

        print("✅ _fact_values_to_dataframe funktioniert für diesen Tag.")
        return

    print(f"Keine passende Unit mit Daten gefunden für Tag: {tag}")


def test_sec_cashflow_statement():
    sec = SecSource(user_agent="gecen.efe1308@gmail.com")

    symbols = ["BABA"]

    core_rows_to_check = [
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
        "Capital Expenditure",
        "Purchase Of PPE",
        "Capital Expenditure Reported",
        "Free Cash Flow",
        "Dividends Paid",
        "Common Stock Dividends Paid",
        "Repurchase Of Capital Stock",
    ]

    line_item_tests = [
        {
            "title": "CASHFLOW CORE Line Item",
            "line_item": "Operating Cash Flow",
            "scope": "core",
            "by": "index",
        },
        {
            "title": "CASHFLOW CORE Free Cash Flow",
            "line_item": "Free Cash Flow",
            "scope": "core",
            "by": "index",
        },
        {
            "title": "CASHFLOW RAW Line Item by Tag",
            "line_item": "NetCashProvidedByUsedInOperatingActivities",
            "scope": "raw",
            "by": "tag",
        },
        {
            "title": "CASHFLOW LABELED Line Item by Tag",
            "line_item": "NetCashProvidedByUsedInOperatingActivities",
            "scope": "labeled",
            "by": "tag",
        },
    ]

    fact_dataframe_tags_to_test = [
        "NetCashProvidedByUsedInOperatingActivities",
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForRepurchaseOfCommonStock",
    ]

    for symbol in symbols:
        for frequency in ["annual", "quarterly"]:
            print(f"\n\n==================== {symbol} | {frequency.upper()} ====================")

            print(f"\n===== {symbol} | CASHFLOW | CORE =====")

            core_df = sec.get_cashflow_statement(
                symbol=symbol,
                frequency=frequency,
                use_cache=False,
                scope="core",
            )

            if isinstance(core_df, dict) and "error" in core_df:
                print(core_df)
            else:
                print(core_df.head())
                print("\nCore Rows:", len(core_df))
                print("Core Columns:", len(core_df.columns))

                assert not core_df.empty
                assert_columns_are_descending_dates(core_df)

                print("\n--- Wichtige Cashflow-Zeilen ---")
                for row in core_rows_to_check:
                    print_series_diagnostics(core_df, row)
                    assert_latest_value_is_first(core_df, row)

                if (
                    "Operating Cash Flow" in core_df.index
                    and "Capital Expenditure" in core_df.index
                    and "Free Cash Flow" in core_df.index
                ):
                    operating = core_df.loc["Operating Cash Flow"].dropna()
                    capex = core_df.loc["Capital Expenditure"].dropna()
                    fcf = core_df.loc["Free Cash Flow"].dropna()

                    common_dates = operating.index.intersection(capex.index).intersection(fcf.index)

                    if len(common_dates) > 0:
                        latest_common_date = common_dates[0]
                        expected_fcf = operating.loc[latest_common_date] - capex.loc[latest_common_date]
                        actual_fcf = fcf.loc[latest_common_date]

                        print(
                            f"\nFCF Check {latest_common_date.strftime('%Y-%m-%d')}: "
                            f"{operating.loc[latest_common_date]} - {capex.loc[latest_common_date]} = "
                            f"{expected_fcf} | Actual={actual_fcf}"
                        )

                        assert abs(expected_fcf - actual_fcf) < 1e-6

            print(f"\n===== {symbol} | CASHFLOW | RAW =====")

            raw_df = sec.get_cashflow_statement(
                symbol=symbol,
                frequency=frequency,
                use_cache=False,
                scope="raw",
            )

            if isinstance(raw_df, dict) and "error" in raw_df:
                print(raw_df)
                continue

            print(raw_df.head())
            print("\nRaw Rows:", len(raw_df))
            print("Raw Columns:", len(raw_df.columns))

            assert not raw_df.empty
            assert_columns_are_descending_dates(raw_df)

            print("\n--- Raw SEC Tags Beispiel ---")
            for tag in list(raw_df.index[:25]):
                print(tag)

            assert "NetCashProvidedByUsedInOperatingActivities" in raw_df.index

            print(f"\n===== {symbol} | CASHFLOW | RAW LABELED =====")

            labeled_result = sec.get_cashflow_statement_raw_labeled(
                symbol=symbol,
                frequency=frequency,
                use_cache=False,
            )

            if isinstance(labeled_result, dict) and "error" in labeled_result:
                print(labeled_result)
                continue

            labeled_df, metadata = labeled_result

            print(labeled_df.head())
            print("\nLabeled Rows:", len(labeled_df))
            print("Labeled Columns:", len(labeled_df.columns))
            print("Metadata Items:", len(metadata))

            assert not labeled_df.empty
            assert len(metadata) > 0
            assert len(labeled_df.index) == len(metadata)
            assert len(set(labeled_df.index)) == len(labeled_df.index)

            print("\n--- Labeled SEC Rows Beispiel ---")
            for label in list(labeled_df.index[:25]):
                print(label)

            print("\n--- Metadata Beispiel ---")
            for label in list(metadata.keys())[:10]:
                print(label, "->", metadata[label])

                assert "tag" in metadata[label]
                assert "label" in metadata[label]
                assert "description" in metadata[label]
                assert "units" in metadata[label]

            print(f"\n===== {symbol} | CASHFLOW | LINE ITEM TESTS =====")

            for test in line_item_tests:
                print(f"\n--- {test['title']} ---")

                result = sec.get_cashflow_statement_line_item(
                    symbol=symbol,
                    line_item=test["line_item"],
                    frequency=frequency,
                    scope=test["scope"],
                    by=test["by"],
                    use_cache=False,
                )

                print_line_item_result(result)

                if isinstance(result, dict) and "error" in result:
                    print("Hinweis: Nicht jedes Unternehmen nutzt jeden SEC-Tag gleich.")
                    continue

                assert result.get("symbol") == symbol.upper()
                assert result.get("frequency") == frequency
                assert result.get("value") is not None
                assert result.get("date") is not None
                assert len(result.get("series", {})) > 0

                series_dates = list(result.get("series", {}).keys())

                if len(series_dates) > 1:
                    assert series_dates[0] >= series_dates[-1], (
                        f"Series ist nicht absteigend sortiert: "
                        f"{series_dates[0]} < {series_dates[-1]}"
                    )

            print(f"\n===== {symbol} | FACT VALUES DATAFRAME TESTS =====")

            for tag in fact_dataframe_tags_to_test:
                test_fact_values_to_dataframe(
                    sec=sec,
                    symbol=symbol,
                    frequency=frequency,
                    tag=tag,
                )

    print("\n\n✅ Cashflow SEC-Test abgeschlossen.")


if __name__ == "__main__":
    test_sec_cashflow_statement()