from data_sources.sec_source import SecSource


def inspect_interest_tags():
    sec = SecSource(user_agent="dein_email@example.com")

    symbols = ["AAPL", "MSFT", "LCID", "MO", "PYPL", "BABA"]

    search_terms = [
        "interest",
        "Interest",
        "debt",
        "Debt",
        "borrow",
        "Borrow",
    ]

    for symbol in symbols:
        print(f"\n==================== {symbol} ====================")

        tag_map = sec.get_us_gaap_tag_map(symbol, use_cache=True)

        if isinstance(tag_map, dict) and "error" in tag_map:
            print(tag_map)
            continue

        matches = {}

        for tag, info in tag_map.items():
            label = info.get("label", "")
            description = info.get("description", "")

            text = f"{tag} {label} {description}"

            if any(term in text for term in search_terms):
                matches[tag] = info

        print("MATCHING TAGS:", len(matches))

        for tag, info in matches.items():
            print(
                f"{tag} -> "
                f"Label: {info.get('label')} | "
                f"Units: {info.get('units')}"
            )


if __name__ == "__main__":
    inspect_interest_tags()