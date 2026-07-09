from data_sources.sec_source import SecSource


def test_baba_all_sec_tags():
    sec = SecSource(user_agent="gecen.efe1308@gmail.com")

    symbol = "BABA"

    for frequency in ["annual", "quarterly"]:
        print(f"\n\n==================== {symbol} | {frequency.upper()} | ALL RAW SEC TAGS ====================")

        raw_df = sec.get_cashflow_statement(
            symbol=symbol,
            frequency=frequency,
            use_cache=False,
            scope="raw",
        )

        if isinstance(raw_df, dict) and "error" in raw_df:
            print(raw_df)
            continue

        print("\nRaw Rows:", len(raw_df))
        print("Raw Columns:", len(raw_df.columns))

        print("\n--- ALLE RAW SEC TAGS ---")
        for i, tag in enumerate(raw_df.index, start=1):
            print(f"{i}. {tag}")

    print("\n\n✅ BABA All-Raw-Tags-Test abgeschlossen.")


if __name__ == "__main__":
    test_baba_all_sec_tags()