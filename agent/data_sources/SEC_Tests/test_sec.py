from data_sources.sec_source import SecSource


def print_line_item_result(result: dict):
    if isinstance(result, dict) and "error" in result:
        print(result)
        return

    print("Symbol:", result.get("symbol"))
    print("Scope:", result.get("scope"))
    print("Line Item:", result.get("line_item"))
    print("Tag:", result.get("tag"))
    print("Label:", result.get("label"))
    print("Units:", result.get("units"))
    print("Date:", result.get("date"))
    print("Value:", result.get("value"))

    series = result.get("series", {})
    print("Series Items:", len(series))

    assert result.get("value") is not None
    assert len(series) > 0


def test_sec_balance_sheet_and_income_statement():
    sec = SecSource(user_agent="dein_email@example.com")

    symbols = ["AAPL", "PYPL", "MO"]

    balance_rows_to_check = [
        "Total Debt",
        "Long Term Debt",
        "Short Term Debt",
        "Current Debt",
        "Cash And Cash Equivalents",
        "Net Debt",
        "Working Capital",
        "Invested Capital",
        "Tangible Book Value",
    ]

    income_rows_to_check = [
        "Total Revenue",
        "Revenue",
        "Cost Of Revenue",
        "Gross Profit",
        "Operating Income",
        "EBIT",
        "EBITDA",
        "Interest Expense",
        "Net Income",
        "Net Income Common Stockholders",
        "Diluted EPS",
        "Basic EPS",
        "Depreciation And Amortization",
    ]

    balance_line_item_tests = [
        {
            "title": "BALANCE CORE Line Item",
            "line_item": "Cash And Cash Equivalents",
            "scope": "core",
            "by": "index",
        },
        {
            "title": "BALANCE RAW Line Item by Tag",
            "line_item": "Assets",
            "scope": "raw",
            "by": "tag",
        },
        {
            "title": "BALANCE LABELED Line Item by Label",
            "line_item": "Assets",
            "scope": "labeled",
            "by": "label",
        },
        {
            "title": "BALANCE LABELED Line Item by Tag",
            "line_item": "AccountsPayableCurrent",
            "scope": "labeled",
            "by": "tag",
        },
    ]

    income_line_item_tests = [
        {
            "title": "INCOME CORE Line Item",
            "line_item": "Net Income",
            "scope": "core",
            "by": "index",
        },
        {
            "title": "INCOME CORE Revenue",
            "line_item": "Total Revenue",
            "scope": "core",
            "by": "index",
        },
        {
            "title": "INCOME RAW Line Item by Tag",
            "line_item": "Revenues",
            "scope": "raw",
            "by": "tag",
        },
        {
            "title": "INCOME LABELED Line Item by Label",
            "line_item": "Revenues",
            "scope": "labeled",
            "by": "label",
        },
        {
            "title": "INCOME LABELED Line Item by Tag",
            "line_item": "NetIncomeLoss",
            "scope": "labeled",
            "by": "tag",
        },
    ]

    for symbol in symbols:
        # =====================================================
        # BALANCE SHEET CORE
        # =====================================================

        print(f"\n===== {symbol} | BALANCE SHEET | CORE =====")

        core_df = sec.get_balance_sheet(
            symbol,
            frequency="annual",
            scope="core",
            use_cache=True,
        )

        if isinstance(core_df, dict) and "error" in core_df:
            print(core_df)
        else:
            print(core_df.head())
            print("\nCore Rows:", len(core_df))
            print("Core Columns:", len(core_df.columns))

            assert not core_df.empty

            print("\n--- Wichtige Bilanz-Zeilen ---")
            for row in balance_rows_to_check:
                if row in core_df.index:
                    print(f"{row}: {core_df.loc[row].iloc[0]}")
                else:
                    print(f"{row}: FEHLT")

        # =====================================================
        # BALANCE SHEET RAW
        # =====================================================

        print(f"\n===== {symbol} | BALANCE SHEET | RAW =====")

        raw_df = sec.get_balance_sheet(
            symbol,
            frequency="annual",
            scope="raw",
            use_cache=True,
        )

        if isinstance(raw_df, dict) and "error" in raw_df:
            print(raw_df)
            continue

        print(raw_df.head())
        print("\nRaw Rows:", len(raw_df))
        print("Raw Columns:", len(raw_df.columns))

        assert not raw_df.empty

        print("\n--- Raw SEC Tags Beispiel ---")
        for tag in list(raw_df.index[:25]):
            print(tag)

        # =====================================================
        # TAG MAP
        # =====================================================

        print(f"\n===== {symbol} | TAG MAP =====")

        tag_map = sec.get_us_gaap_tag_map(symbol, use_cache=True)

        if isinstance(tag_map, dict) and "error" in tag_map:
            print(tag_map)
            continue

        print("Mapped Tags:", len(tag_map))

        print("\n--- Tag Map Beispiel ---")
        for tag in list(tag_map.keys())[:25]:
            info = tag_map[tag]
            print(
                f"{tag} -> "
                f"Label: {info.get('label')} | "
                f"Units: {info.get('units')}"
            )

        # =====================================================
        # BALANCE SHEET RAW LABELED
        # =====================================================

        print(f"\n===== {symbol} | BALANCE SHEET | RAW LABELED =====")

        labeled_result = sec.get_balance_sheet_raw_labeled(
            symbol,
            frequency="annual",
            use_cache=True,
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

        print("\n--- Labeled SEC Rows Beispiel ---")
        for label in list(labeled_df.index[:25]):
            print(label)

        print("\n--- Metadata Beispiel ---")
        for label in list(metadata.keys())[:10]:
            print(label, "->", metadata[label])

        # =====================================================
        # BALANCE SHEET LINE ITEMS
        # =====================================================

        print(f"\n===== {symbol} | BALANCE SHEET | LINE ITEM TESTS =====")

        for test in balance_line_item_tests:
            print(f"\n--- {test['title']} ---")

            result = sec.get_balance_sheet_line_item(
                symbol=symbol,
                line_item=test["line_item"],
                frequency="annual",
                scope=test["scope"],
                by=test["by"],
                use_cache=True,
            )

            print_line_item_result(result)

        # =====================================================
        # INCOME STATEMENT CORE
        # =====================================================

        print(f"\n===== {symbol} | INCOME STATEMENT | CORE =====")

        income_core_df = sec.get_stock_financials(
            symbol,
            frequency="annual",
            scope="core",
            use_cache=True,
        )

        if isinstance(income_core_df, dict) and "error" in income_core_df:
            print(income_core_df)
        else:
            print(income_core_df.head())
            print("\nIncome Core Rows:", len(income_core_df))
            print("Income Core Columns:", len(income_core_df.columns))

            assert not income_core_df.empty

            print("\n--- Wichtige GuV-Zeilen ---")
            for row in income_rows_to_check:
                if row in income_core_df.index:
                    print(f"{row}: {income_core_df.loc[row].iloc[0]}")
                else:
                    print(f"{row}: FEHLT")

        # =====================================================
        # INCOME STATEMENT RAW
        # =====================================================

        print(f"\n===== {symbol} | INCOME STATEMENT | RAW =====")

        income_raw_df = sec.get_stock_financials(
            symbol,
            frequency="annual",
            scope="raw",
            use_cache=True,
        )

        if isinstance(income_raw_df, dict) and "error" in income_raw_df:
            print(income_raw_df)
            continue

        print(income_raw_df.head())
        print("\nIncome Raw Rows:", len(income_raw_df))
        print("Income Raw Columns:", len(income_raw_df.columns))

        assert not income_raw_df.empty

        print("\n--- Income Raw SEC Tags Beispiel ---")
        for tag in list(income_raw_df.index[:25]):
            print(tag)

        # =====================================================
        # INCOME STATEMENT RAW LABELED
        # =====================================================

        print(f"\n===== {symbol} | INCOME STATEMENT | RAW LABELED =====")

        income_labeled_result = sec.get_stock_financials_raw_labeled(
            symbol,
            frequency="annual",
            use_cache=True,
        )

        if isinstance(income_labeled_result, dict) and "error" in income_labeled_result:
            print(income_labeled_result)
            continue

        income_labeled_df, income_metadata = income_labeled_result

        print(income_labeled_df.head())
        print("\nIncome Labeled Rows:", len(income_labeled_df))
        print("Income Labeled Columns:", len(income_labeled_df.columns))
        print("Income Metadata Items:", len(income_metadata))

        assert not income_labeled_df.empty
        assert len(income_metadata) > 0

        print("\n--- Income Labeled SEC Rows Beispiel ---")
        for label in list(income_labeled_df.index[:25]):
            print(label)

        print("\n--- Income Metadata Beispiel ---")
        for label in list(income_metadata.keys())[:10]:
            print(label, "->", income_metadata[label])

        # =====================================================
        # INCOME STATEMENT LINE ITEMS
        # =====================================================

        print(f"\n===== {symbol} | INCOME STATEMENT | LINE ITEM TESTS =====")

        for test in income_line_item_tests:
            print(f"\n--- {test['title']} ---")

            result = sec.get_stock_financials_line_item(
                symbol=symbol,
                line_item=test["line_item"],
                frequency="annual",
                scope=test["scope"],
                by=test["by"],
                use_cache=True,
            )

            print_line_item_result(result)


if __name__ == "__main__":
    test_sec_balance_sheet_and_income_statement()