from data_sources.sec_source import SecSource


def list_all_sec_tags():
    sec = SecSource(user_agent="gecen.efe1308@gmail.com")

    symbols = ["JPM"]  # erstmal nur eine Aktie

    for symbol in symbols:
        print(f"\n\n==================== {symbol} ====================")

        facts = sec.get_company_facts(symbol, use_cache=True)

        if isinstance(facts, dict) and "error" in facts:
            print(facts)
            continue

        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        tags = sorted(us_gaap.keys())

        print(f"TOTAL TAGS: {len(tags)}\n")

        for tag in tags:
            print(tag)


if __name__ == "__main__":
    list_all_sec_tags()