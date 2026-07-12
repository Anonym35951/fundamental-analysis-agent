import json
import logging
import os
from datetime import datetime, timedelta
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed, RetryError

from agent.cache_object_storage import ObjectStorageCacheSync


# Formtypen, die für die "neues Filing verfügbar"-Benachrichtigung relevant
# sind (Favoriten-Alerts, siehe api/services/filing_alert_service.py). 8-K &
# Co. bewusst ausgeklammert — die sind zu häufig und würden aus einer
# seltenen, werthaltigen Benachrichtigung eine Spam-Quelle machen.
RELEVANT_FILING_FORMS = {"10-K", "10-Q"}


def describe_exception(e: Exception) -> str:
    """Unwraps tenacity's RetryError to the actual exception that caused the
    final retry attempt to fail, so user-facing error messages show the real
    cause (e.g. a timeout or rate limit) instead of the unreadable
    "RetryError[<Future at ... raised ...>]" wrapper text."""
    if isinstance(e, RetryError):
        try:
            underlying = e.last_attempt.exception()
            if underlying is not None:
                return str(underlying)
        except Exception:
            pass
    return str(e)


class SecSource:
    def __init__(self, user_agent: str, cache_dir: str = "cache/sec"):
        self.user_agent = user_agent
        self.cache_dir = cache_dir
        self.base_url = "https://data.sec.gov/api/xbrl/companyfacts"
        self.ticker_url = "https://www.sec.gov/files/company_tickers.json"

        os.makedirs(self.cache_dir, exist_ok=True)
        # Eigene ObjectStorageCacheSync-Instanz (statt eine von DataLoader
        # durchgereicht zu bekommen): SecSource wird auch eigenständig
        # instanziiert (siehe api/services/filing_alert_service.py), muss also
        # unabhängig funktionieren. "sec/"-Präfix trennt die Keys von
        # DataLoader, falls beide denselben Bucket nutzen.
        self._cache_sync = ObjectStorageCacheSync()
        self._cache_sync_key_prefix = "sec"

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_balance_sheet(
            self,
            symbol: str,
            frequency: str = "annual",
            use_cache: bool = True,
            scope: str = "core",
    ):
        if frequency not in ["annual", "quarterly"]:
            return {
                "error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.",
                "symbol": symbol,
            }

        if scope not in ["core", "raw"]:
            return {
                "error": f"Ungültiger Scope: {scope}. Verwende 'core' oder 'raw'.",
                "symbol": symbol,
            }

        cache_key = f"{symbol.upper()}_sec_balance_sheet_{frequency}_{scope}"

        if use_cache:
            cached = self._load_cached_data(cache_key)
            if cached is not None:
                return cached

        try:
            facts = self.get_company_facts(symbol, use_cache=use_cache)

            if isinstance(facts, dict) and "error" in facts:
                return facts

            #
            # WICHTIG:
            # Foreign Private Issuers / ADRs liefern bei SEC
            # häufig keine konsistenten Quarterly-Daten.
            #
            # Deshalb Quarterly hier bewusst blockieren.
            #

            is_foreign_issuer = self._is_foreign_private_issuer(facts)

            if frequency == "quarterly" and is_foreign_issuer:
                return {
                    "error": (
                        f"{symbol.upper()} ist ein im Ausland gelistetes Unternehmen (20-F-Filer). "
                        f"SEC-Quarterly-Bilanzdaten sind für solche Unternehmen "
                        f"nicht zuverlässig verfügbar."
                    ),
                    "symbol": symbol,
                    "frequency": frequency,
                    "scope": scope,
                    "foreign_issuer": True,
                }

            #
            # Financials (Banken / Versicherungen)
            #

            is_financial = (
                    symbol.upper() in FINANCIAL_TICKERS
            )

            if scope == "raw":
                df = self._build_raw_us_gaap_dataframe(
                    facts=facts,
                    frequency=frequency,
                    unit_preference=["USD", "USD/shares", "shares", "pure"],
                )

            else:

                #
                # Für Banken / Versicherungen eigenes Mapping verwenden.
                # Alle anderen Unternehmen nutzen weiterhin das Standard-Mapping.
                #

                mapping = (
                    FINANCIAL_BALANCE_SHEET_MAP
                    if is_financial
                    else BALANCE_SHEET_MAP
                )

                df = self._build_statement_dataframe(
                    facts=facts,
                    mapping=mapping,
                    frequency=frequency,
                    unit_preference=["USD", "shares"],
                )

                if not df.empty:
                    #
                    # Financial-Flag für spätere Derived Rows
                    # (Net Debt, Invested Capital usw.)
                    #

                    df.attrs["is_financial"] = is_financial

                    df = self._add_balance_sheet_derived_rows(df)

            if df.empty:
                return {
                    "error": f"Keine SEC-Bilanzdaten für {symbol} ({frequency}, {scope}) gefunden.",
                    "symbol": symbol,
                    "foreign_issuer": is_foreign_issuer,
                }

            if use_cache:
                self._cache_data(df, cache_key)

            return df

        except Exception as e:
            return {
                "error": f"Fehler beim Abrufen der SEC-Bilanzdaten für {symbol} ({frequency}, {scope}): {describe_exception(e)}",
                "symbol": symbol,
            }

    def get_stock_financials(
            self,
            symbol: str,
            frequency: str = "annual",
            use_cache: bool = True,
            scope: str = "core",
    ):

        if frequency not in ["annual", "quarterly"]:
            return {
                "error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.",
                "symbol": symbol,
            }

        if scope not in ["core", "raw"]:
            return {
                "error": f"Ungültiger Scope: {scope}. Verwende 'core' oder 'raw'.",
                "symbol": symbol,
            }

        cache_key = f"{symbol.upper()}_sec_stock_financials_{frequency}_{scope}"

        if use_cache:
            cached = self._load_cached_data(cache_key)
            if cached is not None:
                return cached

        try:
            facts = self.get_company_facts(symbol, use_cache=use_cache)

            if isinstance(facts, dict) and "error" in facts:
                return facts

            #
            # WICHTIG:
            # Foreign Private Issuers / ADRs liefern bei SEC
            # häufig keine konsistenten Quarterly-Daten.
            #
            # Deshalb Quarterly hier bewusst blockieren.
            #

            is_foreign_issuer = self._is_foreign_private_issuer(facts)

            if frequency == "quarterly" and is_foreign_issuer:
                return {
                    "error": (
                        f"{symbol.upper()} ist ein im Ausland gelistetes Unternehmen (20-F-Filer). "
                        f"SEC-Quarterly-GuV-Daten sind für solche Unternehmen "
                        f"nicht zuverlässig verfügbar."
                    ),
                    "symbol": symbol,
                    "frequency": frequency,
                    "scope": scope,
                    "foreign_issuer": True,
                }

            if scope == "raw":
                df = self._build_raw_us_gaap_dataframe(
                    facts=facts,
                    frequency=frequency,
                    unit_preference=["USD", "USD/shares", "shares", "pure"],
                )

            else:
                df = self._build_statement_dataframe(
                    facts=facts,
                    mapping=INCOME_STATEMENT_MAP,
                    frequency=frequency,
                    unit_preference=["USD", "USD/shares"],
                    statement_type="income",
                )

                if not df.empty:
                    df = self._add_income_statement_derived_rows(df, symbol=symbol)

            if df.empty:
                return {
                    "error": f"Keine SEC-GuV-Daten für {symbol} ({frequency}, {scope}) gefunden.",
                    "symbol": symbol,
                    "foreign_issuer": is_foreign_issuer,
                }

            if use_cache:
                self._cache_data(df, cache_key)

            return df

        except Exception as e:
            return {
                "error": f"Fehler beim Abrufen der SEC-GuV-Daten für {symbol} ({frequency}, {scope}): {describe_exception(e)}",
                "symbol": symbol,
            }

    def get_stock_financials_raw_labeled(
            self,
            symbol: str,
            frequency: str = "annual",
            use_cache: bool = True,
    ):
        df = self.get_stock_financials(
            symbol,
            frequency=frequency,
            use_cache=use_cache,
            scope="raw",
        )

        if isinstance(df, dict) and "error" in df:
            return df

        tag_map = self.get_us_gaap_tag_map(symbol, use_cache=use_cache)

        if isinstance(tag_map, dict) and "error" in tag_map:
            return tag_map

        df = df.copy()

        new_index = []
        metadata = {}

        for tag in df.index:
            info = tag_map.get(tag, {})

            base_label = info.get("label", tag)
            label = base_label

            if label in metadata:
                label = f"{base_label} [{tag}]"

            new_index.append(label)

            metadata[label] = {
                "tag": tag,
                "label": base_label,
                "description": info.get("description", ""),
                "units": info.get("units", []),
            }

        df.index = new_index

        return df, metadata

    def get_stock_financials_line_item(
            self,
            symbol: str,
            line_item: str,
            frequency: str = "annual",
            scope: str = "core",
            by: str = "index",
            use_cache: bool = True,
    ):
        if frequency not in ["annual", "quarterly"]:
            return {"error": f"Ungültige Frequenz: {frequency}.", "symbol": symbol}

        if scope not in ["core", "raw", "labeled"]:
            return {"error": f"Ungültiger Scope: {scope}.", "symbol": symbol}

        if by not in ["index", "tag", "label"]:
            return {"error": f"Ungültiger by-Wert: {by}.", "symbol": symbol}

        try:
            if scope == "labeled" or by == "label":
                labeled_result = self.get_stock_financials_raw_labeled(
                    symbol=symbol,
                    frequency=frequency,
                    use_cache=use_cache,
                )

                if isinstance(labeled_result, dict) and "error" in labeled_result:
                    return labeled_result

                df, metadata = labeled_result

                if by == "tag":
                    matching_label = None

                    for label, meta in metadata.items():
                        if meta.get("tag") == line_item:
                            matching_label = label
                            break

                    if not matching_label:
                        return {"error": f"Tag '{line_item}' wurde nicht gefunden.", "symbol": symbol}

                    row_key = matching_label
                else:
                    row_key = line_item

                if row_key not in df.index:
                    return {"error": f"Line Item '{line_item}' wurde nicht gefunden.", "symbol": symbol}

                series = df.loc[row_key].dropna()

                if series.empty:
                    return {"error": f"Keine Werte für '{line_item}' gefunden.", "symbol": symbol}

                latest_date = series.index[0]
                latest_value = series.iloc[0]
                meta = metadata.get(row_key, {})

                return {
                    "symbol": symbol.upper(),
                    "frequency": frequency,
                    "scope": "labeled",
                    "line_item": row_key,
                    "tag": meta.get("tag"),
                    "label": row_key,
                    "description": meta.get("description", ""),
                    "units": meta.get("units", []),
                    "date": latest_date.strftime("%Y-%m-%d"),
                    "value": float(latest_value),
                    "series": {
                        date.strftime("%Y-%m-%d"): float(value)
                        for date, value in series.items()
                    },
                }

            df = self.get_stock_financials(
                symbol=symbol,
                frequency=frequency,
                use_cache=use_cache,
                scope=scope,
            )

            if isinstance(df, dict) and "error" in df:
                return df

            if line_item not in df.index:
                return {"error": f"Line Item '{line_item}' wurde nicht gefunden.", "symbol": symbol}

            series = df.loc[line_item].dropna()

            if series.empty:
                return {"error": f"Keine Werte für '{line_item}' gefunden.", "symbol": symbol}

            latest_date = series.index[0]
            latest_value = series.iloc[0]

            return {
                "symbol": symbol.upper(),
                "frequency": frequency,
                "scope": scope,
                "line_item": line_item,
                "tag": line_item if scope == "raw" else None,
                "label": line_item,
                "date": latest_date.strftime("%Y-%m-%d"),
                "value": float(latest_value),
                "series": {
                    date.strftime("%Y-%m-%d"): float(value)
                    for date, value in series.items()
                },
            }

        except Exception as e:
            return {
                "error": f"Fehler beim Abrufen von '{line_item}' für {symbol}: {describe_exception(e)}",
                "symbol": symbol,
            }

    def get_cashflow_statement(
            self,
            symbol: str,
            frequency: str = "annual",
            use_cache: bool = True,
            scope: str = "core",
    ):
        if frequency not in ["annual", "quarterly"]:
            return {
                "error": f"Ungültige Frequenz: {frequency}. Verwende 'annual' oder 'quarterly'.",
                "symbol": symbol,
            }

        if scope not in ["core", "raw"]:
            return {
                "error": f"Ungültiger Scope: {scope}. Verwende 'core' oder 'raw'.",
                "symbol": symbol,
            }

        cache_key = f"{symbol.upper()}_sec_cashflow_{frequency}_{scope}"

        if use_cache:
            cached = self._load_cached_data(cache_key)
            if cached is not None:
                return cached

        try:
            facts = self.get_company_facts(symbol, use_cache=use_cache)

            if isinstance(facts, dict) and "error" in facts:
                return facts

            #
            # WICHTIG:
            # Foreign Private Issuers / ADRs liefern bei SEC
            # häufig keine konsistenten Quarterly-Daten.
            #
            # Deshalb Quarterly hier bewusst blockieren.
            #

            is_foreign_issuer = self._is_foreign_private_issuer(facts)

            if frequency == "quarterly" and is_foreign_issuer:
                return {
                    "error": (
                        f"{symbol.upper()} ist ein im Ausland gelistetes Unternehmen (20-F-Filer). "
                        f"SEC-Quarterly-Cashflow-Daten sind für solche Unternehmen "
                        f"nicht zuverlässig verfügbar."
                    ),
                    "symbol": symbol,
                    "frequency": frequency,
                    "scope": scope,
                    "foreign_issuer": True,
                }

            if scope == "raw":
                df = self._build_raw_us_gaap_dataframe(
                    facts=facts,
                    frequency=frequency,
                    unit_preference=["USD", "USD/shares", "shares", "pure"],
                )

            else:
                # WICHTIG:
                # Hier jetzt NICHT mehr _build_statement_dataframe(),
                # sondern die robuste Cashflow-Methode verwenden
                df = self._build_cashflow_core_df(
                    facts=facts,
                    frequency=frequency,
                )

                if not df.empty:
                    df = self._add_cashflow_derived_rows(df)

            if df.empty:
                return {
                    "error": f"Keine SEC-Cashflow-Daten für {symbol} ({frequency}, {scope}) gefunden.",
                    "symbol": symbol,
                    "foreign_issuer": is_foreign_issuer,
                }

            if use_cache:
                self._cache_data(df, cache_key)

            return df

        except Exception as e:
            return {
                "error": f"Fehler beim Abrufen der SEC-Cashflow-Daten für {symbol} ({frequency}, {scope}): {describe_exception(e)}",
                "symbol": symbol,
            }

    def get_cashflow_statement_raw_labeled(
            self,
            symbol: str,
            frequency: str = "annual",
            use_cache: bool = True,
    ):
        df = self.get_cashflow_statement(
            symbol,
            frequency=frequency,
            use_cache=use_cache,
            scope="raw",
        )

        if isinstance(df, dict) and "error" in df:
            return df

        tag_map = self.get_us_gaap_tag_map(symbol, use_cache=use_cache)

        if isinstance(tag_map, dict) and "error" in tag_map:
            return tag_map

        df = df.copy()

        new_index = []
        metadata = {}

        for tag in df.index:
            info = tag_map.get(tag, {})

            base_label = info.get("label", tag)
            label = base_label

            if label in metadata:
                label = f"{base_label} [{tag}]"

            new_index.append(label)

            metadata[label] = {
                "tag": tag,
                "label": base_label,
                "description": info.get("description", ""),
                "units": info.get("units", []),
            }

        df.index = new_index

        return df, metadata

    def get_cashflow_statement_line_item(
            self,
            symbol: str,
            line_item: str,
            frequency: str = "annual",
            scope: str = "core",
            by: str = "index",
            use_cache: bool = True,
    ):
        if frequency not in ["annual", "quarterly"]:
            return {"error": f"Ungültige Frequenz: {frequency}.", "symbol": symbol}

        if scope not in ["core", "raw", "labeled"]:
            return {"error": f"Ungültiger Scope: {scope}.", "symbol": symbol}

        if by not in ["index", "tag", "label"]:
            return {"error": f"Ungültiger by-Wert: {by}.", "symbol": symbol}

        try:
            if scope == "labeled" or by == "label":
                labeled_result = self.get_cashflow_statement_raw_labeled(
                    symbol=symbol,
                    frequency=frequency,
                    use_cache=use_cache,
                )

                if isinstance(labeled_result, dict) and "error" in labeled_result:
                    return labeled_result

                df, metadata = labeled_result

                if by == "tag":
                    matching_label = None

                    for label, meta in metadata.items():
                        if meta.get("tag") == line_item:
                            matching_label = label
                            break

                    if not matching_label:
                        return {
                            "error": f"Tag '{line_item}' wurde nicht gefunden.",
                            "symbol": symbol,
                            "line_item": line_item,
                        }

                    row_key = matching_label
                else:
                    row_key = line_item

                if row_key not in df.index:
                    return {
                        "error": f"Line Item '{line_item}' wurde nicht gefunden.",
                        "symbol": symbol,
                        "line_item": line_item,
                    }

                series = df.loc[row_key].dropna()

                if series.empty:
                    return {
                        "error": f"Keine Werte für '{line_item}' gefunden.",
                        "symbol": symbol,
                        "line_item": line_item,
                    }

                latest_date = series.index[0]
                latest_value = series.iloc[0]
                meta = metadata.get(row_key, {})

                return {
                    "symbol": symbol.upper(),
                    "frequency": frequency,
                    "scope": "labeled",
                    "line_item": row_key,
                    "tag": meta.get("tag"),
                    "label": row_key,
                    "description": meta.get("description", ""),
                    "units": meta.get("units", []),
                    "date": latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, "strftime") else str(latest_date),
                    "value": float(latest_value),
                    "series": {
                        date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date): float(value)
                        for date, value in series.items()
                    },
                }

            df = self.get_cashflow_statement(
                symbol=symbol,
                frequency=frequency,
                use_cache=use_cache,
                scope=scope,
            )

            if isinstance(df, dict) and "error" in df:
                return df

            row_key = line_item

            if row_key not in df.index:
                return {
                    "error": f"Line Item '{line_item}' wurde nicht gefunden.",
                    "symbol": symbol,
                    "line_item": line_item,
                    "scope": scope,
                }

            series = df.loc[row_key].dropna()

            if series.empty:
                return {
                    "error": f"Keine Werte für '{line_item}' gefunden.",
                    "symbol": symbol,
                    "line_item": line_item,
                    "scope": scope,
                }

            latest_date = series.index[0]
            latest_value = series.iloc[0]

            return {
                "symbol": symbol.upper(),
                "frequency": frequency,
                "scope": scope,
                "line_item": row_key,
                "tag": row_key if scope == "raw" else None,
                "label": row_key,
                "date": latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, "strftime") else str(latest_date),
                "value": float(latest_value),
                "series": {
                    date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date): float(value)
                    for date, value in series.items()
                },
            }

        except Exception as e:
            return {
                "error": f"Fehler beim Abrufen von '{line_item}' für {symbol}: {describe_exception(e)}",
                "symbol": symbol,
                "line_item": line_item,
            }

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_company_facts(self, symbol: str, use_cache: bool = True):
        """LAUNCH_AUDIT.md P2-12: @retry war hier wirkungslos, weil ein
        umschließendes try/except JEDE Exception (auch transiente
        Netzwerkfehler auf dem echten requests.get-Call) abfing, bevor der
        Decorator sie je sehen konnte - der Choke-Point hinter praktisch
        jedem SEC-Datenpfad (get_balance_sheet, get_stock_financials,
        get_cashflow_statement usw. rufen alle transitiv hierher durch).
        Jetzt propagieren echte Fehler zum Decorator; nur das bewusste
        "CIK nicht gefunden"-Dict von get_cik bleibt ein normaler
        Rückgabewert (kein transienter Fehler, kein Retry sinnvoll). Aufrufer
        weiter oben (z. B. get_balance_sheet) haben weiterhin ihr eigenes
        try/except und liefern nach erschöpften Retries unverändert ein
        {"error": ...}-Dict, kein rohes RetryError."""
        symbol = symbol.upper().strip()
        cache_key = f"{symbol}_companyfacts"

        if use_cache:
            cached = self._load_cached_data(cache_key, max_age=timedelta(days=7))
            if cached is not None:
                return cached

        cik = self.get_cik(symbol)
        if isinstance(cik, dict) and "error" in cik:
            return cik

        url = f"{self.base_url}/CIK{cik}.json"
        response = requests.get(url, headers=self._headers(), timeout=30)
        response.raise_for_status()

        data = response.json()

        if use_cache:
            self._cache_data(data, cache_key)

        return data

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_cik(self, symbol: str):
        """Gleiches P2-12-Muster wie get_company_facts: der echte
        Netzwerk-Call (Ticker->CIK-Mapping) propagiert jetzt zu @retry; nur
        "Symbol nicht in der Liste" bleibt ein Dict statt einer Exception."""
        symbol = symbol.upper().strip()
        cache_key = "sec_company_tickers"

        cached = self._load_cached_data(cache_key, max_age=timedelta(days=30))

        if cached is None:
            response = requests.get(self.ticker_url, headers=self._headers(), timeout=30)
            response.raise_for_status()
            cached = response.json()
            self._cache_data(cached, cache_key)

        for item in cached.values():
            if item.get("ticker", "").upper() == symbol:
                return str(item["cik_str"]).zfill(10)

        return {"error": f"Kein SEC-CIK für Symbol {symbol} gefunden.", "symbol": symbol}

    def get_latest_filing(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        """Liefert die zuletzt eingereichte 10-K/10-Q-Meldung für ein Symbol —
        Grundlage für die Filing-Alerts auf Favoriten
        (api/services/filing_alert_service.py).

        Returns:
            {"symbol", "form", "filing_date", "accession_number"} bei Erfolg,
            {"error": ..., "symbol": ...} bei Fehler oder wenn keine 10-K/
            10-Q-Meldung gefunden wurde (z. B. bei Foreign Private Issuers,
            die stattdessen 20-F/6-K einreichen).
        """
        symbol = symbol.upper().strip()
        cache_key = f"{symbol}_latest_filing"

        if use_cache:
            cached = self._load_cached_data(cache_key, max_age=timedelta(hours=6))
            if cached is not None:
                return cached

        try:
            cik = self.get_cik(symbol)
            if isinstance(cik, dict) and "error" in cik:
                return cik

            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            response = requests.get(url, headers=self._headers(), timeout=30)
            response.raise_for_status()
            data = response.json()

            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            filing_dates = recent.get("filingDate", [])
            accession_numbers = recent.get("accessionNumber", [])

            # SEC liefert "recent" bereits neueste zuerst.
            for form, filing_date, accession_number in zip(forms, filing_dates, accession_numbers):
                if form in RELEVANT_FILING_FORMS:
                    result = {
                        "symbol": symbol,
                        "form": form,
                        "filing_date": filing_date,
                        "accession_number": accession_number,
                    }
                    if use_cache:
                        self._cache_data(result, cache_key)
                    return result

            return {"error": f"Keine 10-K/10-Q-Meldung für {symbol} gefunden.", "symbol": symbol}

        except Exception as e:
            return {
                "error": f"Fehler beim Abrufen der aktuellen SEC-Filings für {symbol}: {describe_exception(e)}",
                "symbol": symbol,
            }

    def _build_statement_dataframe(
            self,
            facts: Dict[str, Any],
            mapping: Dict[str, List[str]],
            frequency: str,
            unit_preference: List[str],
            statement_type: Optional[str] = None,
    ) -> pd.DataFrame:

        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        rows: Dict[str, pd.Series] = {}

        for yahoo_label, sec_tags in mapping.items():

            #
            # WICHTIG:
            # Diese Kennzahl wird robuster in
            # _add_balance_sheet_derived_rows()
            # aus Cash + Other Short Term Investments berechnet.
            #

            if yahoo_label == "Cash Cash Equivalents And Short Term Investments":
                continue

            #
            # Zeilen, die aus mehreren SEC-Tags
            # addiert werden sollen.
            #

            sum_rows = [
                "Current Debt",
                "Short Term Debt",
                "Depreciation And Amortization",
                "Depreciation Depletion And Amortization",
            ]

            #
            # Zeilen, deren Historie aus mehreren
            # Tags zusammengeführt werden soll.
            #
            # Beispiel:
            # InterestExpenseNonoperating (neu)
            # + InterestExpense (alt)
            #

            merge_rows = [
                "Interest Expense",
            ]

            #
            # Financials:
            # Bei Banken / Versicherungen bestehen Cash und Securities
            # häufig aus mehreren Bilanz-Tags.
            #

            financial_sum_rows = [
                "Cash And Cash Equivalents",
            ]

            should_sum = (
                    yahoo_label in sum_rows
                    or (
                            mapping is FINANCIAL_BALANCE_SHEET_MAP
                            and yahoo_label in financial_sum_rows
                    )
            )

            should_merge = (
                    yahoo_label in merge_rows
            )

            #
            # SPEZIALFALL:
            # Other Short Term Investments bei Banken.
            #

            if (
                    mapping is FINANCIAL_BALANCE_SHEET_MAP
                    and yahoo_label == "Other Short Term Investments"
            ):

                series = self._first_available_series(
                    us_gaap=us_gaap,
                    sec_tags=sec_tags,
                    frequency=frequency,
                    unit_preference=unit_preference,
                    statement_type=statement_type,
                )

                if (
                        series is None
                        or series.empty
                        or series.dropna().empty
                ):
                    series = self._sum_available_series(
                        us_gaap=us_gaap,
                        sec_tags=sec_tags,
                        frequency=frequency,
                        unit_preference=unit_preference,
                        statement_type=statement_type,
                    )

            elif should_sum:

                series = self._sum_available_series(
                    us_gaap=us_gaap,
                    sec_tags=sec_tags,
                    frequency=frequency,
                    unit_preference=unit_preference,
                    statement_type=statement_type,
                )

            elif should_merge:

                series = self._merge_available_series(
                    us_gaap=us_gaap,
                    sec_tags=sec_tags,
                    frequency=frequency,
                    unit_preference=unit_preference,
                    statement_type=statement_type,
                )

            else:

                series = self._first_available_series(
                    us_gaap=us_gaap,
                    sec_tags=sec_tags,
                    frequency=frequency,
                    unit_preference=unit_preference,
                    statement_type=statement_type,
                )

            if series is not None and not series.empty:
                rows[yahoo_label] = series

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).T

        df.columns = pd.to_datetime(df.columns)

        df = df.sort_index(axis=1, ascending=False)

        df = df.apply(pd.to_numeric, errors="coerce")

        return df

    def _sum_available_series(
            self,
            us_gaap: Dict[str, Any],
            sec_tags: List[str],
            frequency: str,
            unit_preference: List[str],
            statement_type: Optional[str] = None,
    ):
        collected_series = []

        for tag in sec_tags:
            fact = us_gaap.get(tag)

            if not fact:
                continue

            units = fact.get("units", {})

            best_series = None
            best_latest_date = None
            best_data_points = -1

            for unit in unit_preference:
                values = units.get(unit)

                if not values:
                    continue

                series = self._fact_values_to_series(
                    values=values,
                    frequency=frequency,
                    statement_type=statement_type,
                )

                if series is None or series.empty:
                    continue

                valid_series = series.dropna()

                if valid_series.empty:
                    continue

                latest_date = valid_series.index.max()
                data_points = len(valid_series)

                if (
                        best_series is None
                        or latest_date > best_latest_date
                        or (
                        latest_date == best_latest_date
                        and data_points > best_data_points
                )
                ):
                    best_series = series
                    best_latest_date = latest_date
                    best_data_points = data_points

            if best_series is not None:
                collected_series.append(best_series)

        if not collected_series:
            return None

        combined = pd.concat(collected_series, axis=1)

        if combined.empty:
            return None

        summed_series = combined.sum(axis=1, skipna=True)
        summed_series = summed_series.dropna()

        if summed_series.empty:
            return None

        return summed_series

    def _first_available_series(
            self,
            us_gaap: Dict[str, Any],
            sec_tags: List[str],
            frequency: str,
            unit_preference: List[str],
            return_metadata: bool = False,
            strict_tag_priority: bool = False,
            statement_type: Optional[str] = None,
    ):
        candidates = []

        for tag_priority, tag in enumerate(sec_tags):
            fact = us_gaap.get(tag)

            if not fact:
                continue

            units = fact.get("units", {})

            for unit_priority, unit in enumerate(unit_preference):
                values = units.get(unit)

                if not values:
                    continue

                series = self._fact_values_to_series(
                    values=values,
                    frequency=frequency,
                    statement_type=statement_type,
                )

                if series is None or series.empty:
                    continue

                valid_series = series.dropna()

                if valid_series.empty:
                    continue

                candidates.append({
                    "series": series,
                    "latest_date": valid_series.index.max(),
                    "data_points": len(valid_series),
                    "tag_priority": tag_priority,
                    "unit_priority": unit_priority,
                    "tag": tag,
                    "unit": unit,
                })

        if not candidates:
            return None if not return_metadata else (None, None)

        if strict_tag_priority:
            best_candidate = sorted(
                candidates,
                key=lambda item: (
                    item["tag_priority"],
                    -item["unit_priority"],
                    item["latest_date"],
                    item["data_points"],
                ),
            )[0]
        else:
            best_candidate = sorted(
                candidates,
                key=lambda item: (
                    item["latest_date"],
                    item["data_points"],
                    -item["tag_priority"],
                    -item["unit_priority"],
                ),
                reverse=True,
            )[0]

        if return_metadata:
            return best_candidate["series"], {
                "tag": best_candidate["tag"],
                "unit": best_candidate["unit"],
                "latest_date": best_candidate["latest_date"],
                "data_points": best_candidate["data_points"],
                "strict_tag_priority": strict_tag_priority,
                "statement_type": statement_type,
            }

        return best_candidate["series"]

    def _merge_available_series(
            self,
            us_gaap: Dict[str, Any],
            sec_tags: List[str],
            frequency: str,
            unit_preference: List[str],
            statement_type: Optional[str] = None,
            return_metadata: bool = False,
    ):
        """
        Baut eine Historie aus mehreren SEC-Tags auf.

        Priorität:
        1. Tag-Reihenfolge in sec_tags
        2. Neuere Tags überschreiben ältere Tags
        3. Fehlende Perioden werden aus älteren Tags ergänzt

        Beispiel NVDA:

        PaymentsToAcquireProductiveAssets
            2021-2026

        PaymentsToAcquirePropertyPlantAndEquipment
            2010-2020

        Ergebnis:
            2010-2026 vollständig
        """

        merged = pd.Series(dtype=float)

        used_tags = []
        used_units = []

        #
        # Rückwärts durchlaufen:
        #
        # letzter Tag = niedrigste Priorität
        # erster Tag = höchste Priorität
        #

        for tag in reversed(sec_tags):

            fact = us_gaap.get(tag)

            if not fact:
                continue

            units = fact.get("units", {})

            best_series = None
            best_latest_date = None
            best_data_points = -1
            best_unit = None

            for unit in unit_preference:

                values = units.get(unit)

                if not values:
                    continue

                series = self._fact_values_to_series(
                    values=values,
                    frequency=frequency,
                    statement_type=statement_type,
                )

                if series is None or series.empty:
                    continue

                valid = series.dropna()

                if valid.empty:
                    continue

                latest_date = valid.index.max()
                data_points = len(valid)

                if (
                        best_series is None
                        or latest_date > best_latest_date
                        or (
                        latest_date == best_latest_date
                        and data_points > best_data_points
                )
                ):
                    best_series = series
                    best_latest_date = latest_date
                    best_data_points = data_points
                    best_unit = unit

            if best_series is None:
                continue

            used_tags.append(tag)

            if best_unit:
                used_units.append(best_unit)

            #
            # combine_first:
            #
            # vorhandene Werte bleiben erhalten
            # fehlende werden ergänzt
            #

            merged = merged.combine_first(best_series)

        if merged.empty:
            return (None, None) if return_metadata else None

        merged = merged.sort_index(ascending=False)

        metadata = {
            "source_tags": used_tags,
            "units": sorted(list(set(used_units))),
            "data_points": len(merged.dropna()),
            "latest_date": (
                merged.dropna().index.max()
                if not merged.dropna().empty
                else None
            ),
        }

        if return_metadata:
            return merged, metadata

        return merged

    def _build_cashflow_core_df(
            self,
            facts: Dict[str, Any],
            frequency: str,
    ) -> pd.DataFrame:
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        rows: Dict[str, pd.Series] = {}

        unit_preference = ["USD", "USD/shares", "shares", "pure"]

        for yahoo_label, sec_tags in CASHFLOW_MAP.items():
            series = self._merge_available_series(
                us_gaap=us_gaap,
                sec_tags=sec_tags,
                frequency=frequency,
                unit_preference=unit_preference,
                statement_type="cashflow",
            )

            if series is not None and not series.empty:
                rows[yahoo_label] = series

        capex_primary_tags = [
            "PaymentsToAcquireProductiveAssets",
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireOtherPropertyPlantAndEquipment",
            "PaymentsToAcquirePropertyPlantEquipmentAndOtherProductiveAssets",
            "CapitalExpenditures",
            "PropertyAndEquipmentAdditions",
            "CapitalSpending",
        ]

        #
        # CAT / DE usw.
        #
        # Wenn SegmentExpenditureAdditionToLongLivedAssets existiert,
        # dann die aktuellen Jahre daraus verwenden,
        # fehlende Historie aber aus den klassischen CapEx-Tags ergänzen.
        #

        preferred_capex_tags = [
            "SegmentExpenditureAdditionToLongLivedAssets",
        ]

        preferred_capex, preferred_meta = self._first_available_series(
            us_gaap=us_gaap,
            sec_tags=preferred_capex_tags,
            frequency=frequency,
            unit_preference=["USD"],
            return_metadata=True,
            strict_tag_priority=True,
            statement_type="cashflow",
        )

        fallback_capex, fallback_meta = self._merge_available_series(
            us_gaap=us_gaap,
            sec_tags=capex_primary_tags,
            frequency=frequency,
            unit_preference=["USD"],
            return_metadata=True,
            statement_type="cashflow",
        )

        capex_series = None
        capex_meta = None

        if (
                preferred_capex is not None
                and not preferred_capex.empty
        ):

            #
            # Neueste Jahre aus SegmentExpenditure...
            # Historie aus den übrigen Tags ergänzen.
            #

            if fallback_capex is not None:
                capex_series = preferred_capex.combine_first(
                    fallback_capex
                )
            else:
                capex_series = preferred_capex

            source_tags = []

            if preferred_meta:
                source_tags.append(
                    preferred_meta.get("tag")
                )

            if fallback_meta:
                source_tags.extend(
                    fallback_meta.get("source_tags", [])
                )

            #
            # Duplikate entfernen
            #

            source_tags = list(
                dict.fromkeys(source_tags)
            )

            capex_meta = {
                "source_tags": source_tags,
                "units": ["USD"],
                "data_points": len(
                    capex_series.dropna()
                ),
            }

        else:

            capex_series = fallback_capex
            capex_meta = fallback_meta

        capex_quality = "NONE"
        capex_source_tag = None
        capex_source_unit = None
        capex_data_points = None
        capex_latest_date = None
        capex_warning = None

        capex_missing_years = []
        capex_history_complete = None

        if capex_series is not None and not capex_series.empty:
            valid_capex = capex_series.dropna()

            if not valid_capex.empty:
                capex_latest_date = valid_capex.index.max()

                #
                # Historien-Vollständigkeit prüfen
                #

                years = sorted(
                    {
                        pd.Timestamp(idx).year
                        for idx in valid_capex.index
                    }
                )

                if len(years) >= 2:

                    expected_years = set(
                        range(
                            min(years),
                            max(years) + 1
                        )
                    )

                    actual_years = set(years)

                    capex_missing_years = sorted(
                        expected_years - actual_years
                    )

                    capex_history_complete = (
                            len(capex_missing_years) == 0
                    )

                else:
                    capex_history_complete = True

                if capex_meta:
                    capex_source_tag = capex_meta.get("source_tags")
                    capex_source_unit = capex_meta.get("units")
                    capex_data_points = capex_meta.get("data_points")

                today = pd.Timestamp.today().normalize()
                max_age_days = 730 if frequency == "annual" else 450
                age_days = (today - capex_latest_date).days

                if age_days <= max_age_days:

                    if capex_missing_years:

                        capex_quality = "PARTIAL_HISTORY"

                        capex_warning = (
                            "SEC liefert keine vollständige FY-Historie. "
                            f"Fehlende Jahre: {capex_missing_years}. "
                            "Historie wurde bewusst nicht rekonstruiert."
                        )

                    else:
                        capex_quality = "HIGH"

                else:

                    capex_quality = "MEDIUM"

                    capex_warning = (
                        f"CapEx-Daten sind veraltet. Letztes echtes CapEx-Datum: "
                        f"{capex_latest_date.strftime('%Y-%m-%d')}."
                    )

                rows["Capital Expenditure"] = capex_series
                rows["Purchase Of PPE"] = capex_series
                rows["Capital Expenditure Reported"] = capex_series

        fcf_series, fcf_meta = self._merge_available_series(
            us_gaap=us_gaap,
            sec_tags=["FreeCashFlow"],
            frequency=frequency,
            unit_preference=["USD"],
            return_metadata=True,
            statement_type="cashflow",
        )

        fcf_quality = None
        fcf_latest_date = None
        fcf_source = None
        fcf_warning = None

        if fcf_series is not None and not fcf_series.empty:
            rows["Free Cash Flow"] = fcf_series

            valid_fcf = fcf_series.dropna()

            if not valid_fcf.empty:
                fcf_latest_date = valid_fcf.index.max()
                fcf_source = (
                    fcf_meta.get("source_tags")
                    if fcf_meta
                    else ["FreeCashFlow"]
                )
                fcf_quality = "HIGH"

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).T
        df.columns = pd.to_datetime(df.columns)
        df = df.sort_index(axis=1, ascending=False)
        df = df.apply(pd.to_numeric, errors="coerce")

        df.attrs["capex_quality"] = capex_quality

        df.attrs["capex_latest_date"] = (
            capex_latest_date.strftime("%Y-%m-%d")
            if capex_latest_date is not None and not pd.isna(capex_latest_date)
            else None
        )

        df.attrs["capex_source_tag"] = capex_source_tag
        df.attrs["capex_source_unit"] = capex_source_unit
        df.attrs["capex_data_points"] = capex_data_points
        df.attrs["capex_warning"] = capex_warning

        df.attrs["capex_missing_years"] = capex_missing_years

        df.attrs["capex_history_complete"] = (
            capex_history_complete
        )

        df.attrs["fcf_quality"] = fcf_quality

        df.attrs["fcf_latest_date"] = (
            fcf_latest_date.strftime("%Y-%m-%d")
            if fcf_latest_date is not None and not pd.isna(fcf_latest_date)
            else None
        )

        df.attrs["fcf_source"] = fcf_source
        df.attrs["fcf_warning"] = fcf_warning

        return df

    def _reconstruct_annual_from_quarterly_series(
            self,
            us_gaap: Dict[str, Any],
            sec_tags: List[str],
            unit_preference: List[str],
            statement_type: str,
    ) -> Tuple[Optional[pd.Series], Optional[Dict[str, Any]]]:

        annual_values = {}
        reconstructed_years = set()
        source_tags_used = set()

        for tag in sec_tags:

            if tag not in us_gaap:
                continue

            tag_data = us_gaap[tag]
            units = tag_data.get("units", {})

            selected_unit = None

            for unit in unit_preference:
                if unit in units:
                    selected_unit = unit
                    break

            if selected_unit is None:
                continue

            facts = units[selected_unit]

            fiscal_year_data = {}

            for fact in facts:

                fp = str(
                    fact.get("fp", "")
                ).upper()

                fy = fact.get("fy")

                value = fact.get("val")

                form = str(
                    fact.get("form", "")
                ).upper()

                end_date = fact.get("end")

                #
                # Nur echte Quartalsberichte
                #

                if form not in {
                    "10-Q",
                    "10-Q/A",
                }:
                    continue

                #
                # Nur echte Quartale
                #

                if fp not in {
                    "Q1",
                    "Q2",
                    "Q3",
                    "Q4",
                }:
                    continue

                if fy is None:
                    continue

                if value is None:
                    continue

                if end_date is None:
                    continue

                fy = int(fy)

                fiscal_year_data.setdefault(
                    fy,
                    {}
                )

                fiscal_year_data[fy][fp] = {
                    "value": float(value),
                    "end": pd.Timestamp(end_date),
                }

            #
            # Annuals rekonstruieren
            #

            for fy, quarter_map in fiscal_year_data.items():

                required_quarters = {
                    "Q1",
                    "Q2",
                    "Q3",
                    "Q4",
                }

                if set(quarter_map.keys()) != required_quarters:
                    continue

                annual_value = (
                        quarter_map["Q1"]["value"]
                        + quarter_map["Q2"]["value"]
                        + quarter_map["Q3"]["value"]
                        + quarter_map["Q4"]["value"]
                )

                #
                # Echtes Geschäftsjahresende
                #

                annual_date = (
                    quarter_map["Q4"]["end"]
                )

                annual_values[annual_date] = annual_value

                reconstructed_years.add(fy)
                source_tags_used.add(tag)

        if not annual_values:
            return None, None

        reconstructed_series = (
            pd.Series(annual_values)
            .sort_index()
            .astype(float)
        )

        meta = {
            "source": "quarterly_reconstruction",
            "source_tags": sorted(
                source_tags_used
            ),
            "units": unit_preference,
            "data_points": len(
                reconstructed_series
            ),
            "years": sorted(
                reconstructed_years
            ),
        }

        return (
            reconstructed_series,
            meta,
        )

    def _fact_values_to_dataframe(
            self,
            values: List[Dict[str, Any]],
            frequency: str,
            statement_type: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:

        parsed = []

        annual_primary_forms = ["10-K", "20-F", "40-F"]
        annual_fallback_forms = ["8-K", "6-K"]

        quarterly_primary_forms = ["10-Q"]
        quarterly_fallback_forms = ["8-K", "6-K"]

        annual_fps = ["FY"]
        quarterly_fps = ["Q1", "Q2", "Q3", "Q4"]

        for item in values:
            value = item.get("val")
            end = item.get("end")
            start = item.get("start")
            form = item.get("form")
            fp = item.get("fp")

            if value is None or end is None:
                continue

            if frequency == "annual":
                if fp not in annual_fps:
                    continue

                if form in annual_primary_forms:
                    form_priority = 3
                elif form in annual_fallback_forms:
                    form_priority = 2
                else:
                    form_priority = 1

            else:
                if fp not in quarterly_fps:
                    continue

                if form in quarterly_primary_forms:
                    form_priority = 3
                elif form in quarterly_fallback_forms:
                    form_priority = 2
                else:
                    form_priority = 1

            try:
                date = pd.to_datetime(end)
                numeric_value = float(value)
                start_date = pd.to_datetime(start) if start else pd.NaT
            except Exception:
                continue

            duration_days = None

            if pd.notna(start_date):
                duration_days = (date - start_date).days

            parsed.append({
                "date": date,
                "start": start_date,
                "duration_days": duration_days,
                "value": numeric_value,
                "filed": item.get("filed"),
                "fy": item.get("fy"),
                "fp": fp,
                "form": form,
                "form_priority": form_priority,
                "accn": item.get("accn"),
                "frame": item.get("frame"),
            })

        if not parsed:
            return None

        df = pd.DataFrame(parsed)
        df["filed"] = pd.to_datetime(df["filed"], errors="coerce")

        #
        # WICHTIG:
        # Bei Income/Cashflow-Statements können SEC Facts für Quarterly entweder
        # echte Quartalswerte oder kumulative YTD-Werte enthalten.
        #
        # Deshalb werden für quarterly bevorzugt echte ca. 3-Monats-Perioden
        # genommen. Nur wenn keine solchen Perioden vorhanden sind, wird mit
        # YTD-Differenzen normalisiert.
        #
        # Balance Sheet bleibt unverändert, da Bilanzwerte Point-in-Time sind.
        #

        if (
                frequency == "quarterly"
                and statement_type in ["cashflow", "income"]
                and not df.empty
        ):
            df["is_quarter_duration"] = df["duration_days"].between(70, 115, inclusive="both")
            df["is_ytd_duration"] = df["duration_days"].between(116, 370, inclusive="both")

            #
            # Pro Datum zuerst echte Quartalsperiode bevorzugen.
            # Falls nicht vorhanden, besten verfügbaren YTD-Wert behalten.
            #

            df = df.sort_values(
                [
                    "date",
                    "is_quarter_duration",
                    "form_priority",
                    "filed",
                    "duration_days",
                ],
                ascending=[False, False, False, False, True],
            )

            df = df.drop_duplicates(subset=["date"], keep="first")

            #
            # Wenn wir überwiegend echte Quartalswerte haben,
            # KEINE YTD-Normalisierung durchführen.
            #

            valid_duration_rows = df[df["duration_days"].notna()]

            quarter_rows = valid_duration_rows[
                valid_duration_rows["is_quarter_duration"]
            ]

            if not quarter_rows.empty and len(quarter_rows) >= max(1, len(valid_duration_rows) // 2):
                df = df.sort_values("date", ascending=False)
                return df

            #
            # Fallback:
            # Wenn keine ausreichenden echten Quartalswerte vorhanden sind,
            # YTD-Werte in Quartalswerte umrechnen.
            #

            df = df.sort_values(["fy", "date"])

            normalized_rows = []

            for fy, group in df.groupby("fy", dropna=False):
                group = group.sort_values("date")

                previous_ytd_value = None

                for _, row in group.iterrows():
                    normalized_row = row.copy()
                    fp = normalized_row["fp"]
                    current_ytd_value = normalized_row["value"]

                    if fp == "Q1":
                        normalized_row["value"] = current_ytd_value

                    elif fp in ["Q2", "Q3", "Q4"]:
                        if previous_ytd_value is not None:
                            normalized_row["value"] = current_ytd_value - previous_ytd_value
                        else:
                            normalized_row["value"] = current_ytd_value

                    previous_ytd_value = current_ytd_value
                    normalized_rows.append(normalized_row)

            if normalized_rows:
                df = pd.DataFrame(normalized_rows)

            df = df.sort_values("date", ascending=False)

            return df

        df = df.sort_values(
            ["date", "form_priority", "filed"],
            ascending=[False, False, False],
        )

        df = df.drop_duplicates(subset=["date"], keep="first")

        df = df.sort_values("date", ascending=False)

        return df

    def _fact_values_to_series(
            self,
            values: List[Dict[str, Any]],
            frequency: str,
            statement_type: Optional[str] = None,
    ) -> Optional[pd.Series]:

        df = self._fact_values_to_dataframe(
            values=values,
            frequency=frequency,
            statement_type=statement_type,
        )

        if df is None or df.empty:
            return None

        return pd.Series(
            data=df["value"].values,
            index=pd.to_datetime(df["date"].values),
        )

    def _add_balance_sheet_derived_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        is_financial = bool(df.attrs.get("is_financial", False))

        #
        # Total Debt
        #

        derived_total_debt = None

        if (
                "Long Term Debt" in df.index
                and "Current Debt" in df.index
                and "Long Term Debt Noncurrent" in df.index
        ):
            long_term_debt = df.loc["Long Term Debt"]
            current_debt = df.loc["Current Debt"]
            noncurrent_debt = df.loc["Long Term Debt Noncurrent"]

            reconstructed_long_term_debt = (
                    current_debt.fillna(0)
                    + noncurrent_debt.fillna(0)
            )

            difference = (
                    long_term_debt.fillna(0)
                    - reconstructed_long_term_debt
            ).abs()

            tolerance = (
                    reconstructed_long_term_debt.abs()
                    * 0.01
            )

            long_term_debt_already_includes_current = (
                    difference <= tolerance
            )

            derived_total_debt = pd.Series(
                index=df.columns,
                dtype=float,
            )

            derived_total_debt.loc[
                long_term_debt_already_includes_current
            ] = long_term_debt.loc[
                long_term_debt_already_includes_current
            ]

            derived_total_debt.loc[
                ~long_term_debt_already_includes_current
            ] = (
                    long_term_debt.loc[
                        ~long_term_debt_already_includes_current
                    ].fillna(0)
                    + current_debt.loc[
                        ~long_term_debt_already_includes_current
                    ].fillna(0)
            )

        elif (
                "Long Term Debt Noncurrent" in df.index
                and "Current Debt" in df.index
        ):
            derived_total_debt = (
                    df.loc["Long Term Debt Noncurrent"].fillna(0)
                    + df.loc["Current Debt"].fillna(0)
            )

        elif (
                "Long Term Debt" in df.index
                and "Current Debt" in df.index
        ):
            derived_total_debt = (
                    df.loc["Long Term Debt"].fillna(0)
                    + df.loc["Current Debt"].fillna(0)
            )

        elif "Long Term Debt" in df.index:
            derived_total_debt = df.loc["Long Term Debt"]

        if derived_total_debt is not None:

            #
            # Financials:
            # Long Term Debt enthält häufig nur die
            # langfristige Komponente.
            #
            # Deshalb bevorzugen wir die selbst
            # rekonstruierte Total-Debt-Zahl.
            #

            if is_financial:

                df.loc["Total Debt"] = derived_total_debt

            else:

                if "Total Debt" not in df.index:
                    df.loc["Total Debt"] = derived_total_debt
                else:
                    df.loc["Total Debt"] = (
                        df.loc["Total Debt"]
                        .combine_first(derived_total_debt)
                    )

        df = self._ensure_row(
            df,
            "Cash And Cash Equivalents",
            ["Cash And Cash Equivalents"],
            mode="first",
        )

        if "Cash And Cash Equivalents" in df.index:
            cash_series = df.loc["Cash And Cash Equivalents"].fillna(0)

            sti_label = self._first_existing_row(
                df,
                ["Other Short Term Investments"],
            )

            if sti_label:
                df.loc["Cash Cash Equivalents And Short Term Investments"] = (
                        cash_series + df.loc[sti_label].fillna(0)
                )
            else:
                df.loc["Cash Cash Equivalents And Short Term Investments"] = cash_series

        #
        # Net Debt
        # Für Banken / Versicherungen bewusst nicht berechnen,
        # weil Cash, Investments und Debt dort operative Bilanzbestandteile sind.
        #

        if not is_financial:
            if (
                    "Total Debt" in df.index
                    and "Cash Cash Equivalents And Short Term Investments" in df.index
            ):
                df.loc["Net Debt"] = (
                        df.loc["Total Debt"]
                        - df.loc["Cash Cash Equivalents And Short Term Investments"]
                )

            elif (
                    "Total Debt" in df.index
                    and "Cash And Cash Equivalents" in df.index
            ):
                df.loc["Net Debt"] = (
                        df.loc["Total Debt"]
                        - df.loc["Cash And Cash Equivalents"]
                )

        if "Current Assets" in df.index and "Current Liabilities" in df.index:
            df.loc["Working Capital"] = (
                    df.loc["Current Assets"] - df.loc["Current Liabilities"]
            )

        equity_label = self._first_existing_row(
            df,
            [
                "Common Stock Equity",
                "Stockholders Equity",
                "Total Stockholders Equity",
                "Total Equity",
            ],
        )

        goodwill = (
            df.loc["Goodwill"].fillna(0)
            if "Goodwill" in df.index
            else 0
        )

        intangibles = (
            df.loc["Other Intangible Assets"].fillna(0)
            if "Other Intangible Assets" in df.index
            else 0
        )

        if equity_label:
            if "Common Stock Equity" not in df.index:
                df.loc["Common Stock Equity"] = df.loc[equity_label]

            #
            # Invested Capital
            # Für Financials bewusst nicht berechnen,
            # da Debt dort nicht klassisches Finanzierungskapital ist.
            #

            if not is_financial and "Total Debt" in df.index:
                df.loc["Invested Capital"] = (
                        df.loc[equity_label] + df.loc["Total Debt"]
                )

            df.loc["Tangible Book Value"] = (
                    df.loc[equity_label] - goodwill - intangibles
            )

            df.loc["Net Tangible Assets"] = (
                    df.loc[equity_label] - goodwill - intangibles
            )

        liabilities_label = self._first_existing_row(
            df,
            ["Total Liabilities Net Minority Interest"],
        )

        if "Net Tangible Assets" not in df.index and "Total Assets" in df.index:
            if liabilities_label:
                df.loc["Net Tangible Assets"] = (
                        df.loc["Total Assets"]
                        - df.loc[liabilities_label]
                        - goodwill
                        - intangibles
                )

            elif equity_label:
                df.loc["Net Tangible Assets"] = (
                        df.loc[equity_label]
                        - goodwill
                        - intangibles
                )

        if "Net Tangible Assets" not in df.index and "Tangible Book Value" in df.index:
            df.loc["Net Tangible Assets"] = df.loc["Tangible Book Value"]

        return df.sort_index(axis=1, ascending=False)

    def _add_income_statement_derived_rows(
            self,
            df: pd.DataFrame,
            symbol: str | None = None,
    ) -> pd.DataFrame:

        df = df.copy()

        #
        # Gross Profit
        #

        revenue_label = self._first_existing_row(
            df,
            ["Total Revenue", "Revenue"],
        )

        cost_label = self._first_existing_row(
            df,
            ["Cost Of Revenue"],
        )

        if revenue_label and cost_label:

            derived_gross_profit = (
                    df.loc[revenue_label]
                    - df.loc[cost_label]
            )

            if "Gross Profit" not in df.index:

                df.loc["Gross Profit"] = derived_gross_profit

            elif df.loc["Gross Profit"].dropna().empty:

                df.loc["Gross Profit"] = derived_gross_profit

            else:

                df.loc["Gross Profit"] = (
                    df.loc["Gross Profit"]
                    .combine_first(derived_gross_profit)
                )

        #
        # EBIT
        #
        # Priorität:
        #
        # 1) echtes SEC-EBIT
        # 2) Pretax Income + Interest Expense
        # 3) Operating Income
        #

        if (
                "EBIT" not in df.index
                or df.loc["EBIT"].dropna().empty
        ):

            pretax_label = self._first_existing_row(
                df,
                ["Pretax Income"],
            )

            interest_label = self._first_existing_row(
                df,
                ["Interest Expense"],
            )

            if pretax_label and interest_label:

                derived_ebit = (
                        df.loc[pretax_label]
                        + df.loc[interest_label].fillna(0)
                )

                # Pretax Income can be missing/NaN for individual periods
                # (e.g. the most recent filing) even though the row exists
                # overall — fall back to tier 3 (Operating Income) per
                # period instead of losing the whole row to NaN.
                if "Operating Income" in df.index:
                    derived_ebit = derived_ebit.combine_first(df.loc["Operating Income"])

                df.loc["EBIT"] = derived_ebit

            elif "Operating Income" in df.index:

                df.loc["EBIT"] = df.loc["Operating Income"]

        #
        # D&A Synchronisierung
        #

        da_label = self._first_existing_row(
            df,
            [
                "Depreciation And Amortization",
                "Depreciation Depletion And Amortization",
            ],
        )

        if da_label:

            if (
                    "Depreciation And Amortization"
                    not in df.index
            ):
                df.loc[
                    "Depreciation And Amortization"
                ] = df.loc[da_label]

            if (
                    "Depreciation Depletion And Amortization"
                    not in df.index
            ):
                df.loc[
                    "Depreciation Depletion And Amortization"
                ] = df.loc[da_label]

        #
        # EBITDA
        #

        if "EBIT" in df.index:

            depreciation_label = self._first_existing_row(
                df,
                [
                    "Depreciation And Amortization",
                    "Depreciation Depletion And Amortization",
                ],
            )

            if depreciation_label:

                derived_ebitda = (
                        df.loc["EBIT"]
                        + df.loc[depreciation_label].fillna(0)
                )

                if "EBITDA" not in df.index:

                    df.loc["EBITDA"] = derived_ebitda

                elif df.loc["EBITDA"].dropna().empty:

                    df.loc["EBITDA"] = derived_ebitda

                else:

                    df.loc["EBITDA"] = (
                        df.loc["EBITDA"]
                        .combine_first(derived_ebitda)
                    )

        #
        # Net Income Common Stockholders
        #

        if (
                "Net Income Common Stockholders"
                not in df.index
                and "Net Income" in df.index
        ):
            df.loc[
                "Net Income Common Stockholders"
            ] = df.loc["Net Income"]

        return df.sort_index(
            axis=1,
            ascending=False,
        )

    def _add_cashflow_derived_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        #
        # Alias-Fallback:
        # Wenn Continuing Operating CF leer ist, aber Operating CF vorhanden ist,
        # übernehmen wir Operating CF.
        #

        if (
                "Operating Cash Flow" in df.index
                and "Cash Flow From Continuing Operating Activities" in df.index
        ):
            df.loc["Cash Flow From Continuing Operating Activities"] = (
                df.loc["Cash Flow From Continuing Operating Activities"]
                .combine_first(df.loc["Operating Cash Flow"])
            )

        elif (
                "Operating Cash Flow" in df.index
                and "Cash Flow From Continuing Operating Activities" not in df.index
        ):
            df.loc["Cash Flow From Continuing Operating Activities"] = (
                df.loc["Operating Cash Flow"]
            )

        elif (
                "Cash Flow From Continuing Operating Activities" in df.index
                and "Operating Cash Flow" not in df.index
        ):
            df.loc["Operating Cash Flow"] = (
                df.loc["Cash Flow From Continuing Operating Activities"]
            )

        #
        # Yahoo-kompatible Vorzeichenlogik:
        # Cash-Outflows negativ darstellen.
        #

        outflow_rows = [
            "Capital Expenditure",
            "Purchase Of PPE",
            "Capital Expenditure Reported",
            "Dividends Paid",
            "Repurchase Of Capital Stock",
            "Acquisitions",
            "Debt Repayment",
        ]

        for row in outflow_rows:
            if row in df.index:
                df.loc[row] = -df.loc[row].abs()

        #
        # Free Cash Flow
        #

        operating_label = self._first_existing_row(
            df,
            [
                "Operating Cash Flow",
                "Cash Flow From Continuing Operating Activities",
            ],
        )

        capex_label = self._first_existing_row(
            df,
            [
                "Capital Expenditure",
                "Purchase Of PPE",
                "Capital Expenditure Reported",
            ],
        )

        if operating_label and capex_label:
            #
            # Nach Vorzeichen-Normalisierung ist CapEx negativ.
            # Daher:
            # FCF = OCF + CapEx
            #

            df.loc["Free Cash Flow"] = (
                    df.loc[operating_label] + df.loc[capex_label]
            )

            fcf_series = df.loc["Free Cash Flow"].dropna()
            capex_quality = df.attrs.get("capex_quality")
            capex_latest_date = df.attrs.get("capex_latest_date")
            capex_warning = df.attrs.get("capex_warning")

            if not fcf_series.empty:
                fcf_latest_date = fcf_series.index.max()

                df.attrs["fcf_quality"] = capex_quality or "UNKNOWN"
                df.attrs["fcf_latest_date"] = fcf_latest_date.strftime("%Y-%m-%d")
                df.attrs["fcf_source"] = "derived_from_operating_cash_flow_plus_negative_capex"
                df.attrs["fcf_warning"] = capex_warning

                if capex_quality in ["MEDIUM", "LOW", "NONE"]:
                    df.attrs["fcf_warning"] = (
                        f"Free Cash Flow wurde aus OCF + negativem CapEx berechnet, "
                        f"aber die CapEx-Qualität ist {capex_quality}. "
                        f"Letztes CapEx-Datum: {capex_latest_date}."
                    )

        elif "Free Cash Flow" in df.index:
            fcf_series = df.loc["Free Cash Flow"].dropna()

            if not fcf_series.empty and df.attrs.get("fcf_quality") is None:
                fcf_latest_date = fcf_series.index.max()

                df.attrs["fcf_quality"] = "HIGH"
                df.attrs["fcf_latest_date"] = fcf_latest_date.strftime("%Y-%m-%d")
                df.attrs["fcf_source"] = "reported"
                df.attrs["fcf_warning"] = None

        return df.sort_index(axis=1, ascending=False)

    def _ensure_row(self, df: pd.DataFrame, target_row: str, source_rows: List[str], mode: str) -> pd.DataFrame:
        if target_row in df.index:
            return df

        existing = [row for row in source_rows if row in df.index]

        if not existing:
            return df

        if mode == "sum":
            df.loc[target_row] = df.loc[existing].sum(axis=0, skipna=True)
        elif mode == "first":
            df.loc[target_row] = df.loc[existing[0]]

        return df

    def _first_existing_row(self, df: pd.DataFrame, rows: List[str]) -> Optional[str]:
        for row in rows:
            if row in df.index:
                return row
        return None

    def _load_cached_data(self, cache_key: str, max_age: timedelta = timedelta(days=1)):
        filepath = os.path.join(self.cache_dir, f"{cache_key}.json")

        if not os.path.exists(filepath):
            self._cache_sync.warm(filepath, f"{self._cache_sync_key_prefix}/{cache_key}.json")

        if not os.path.exists(filepath):
            return None

        if datetime.now() - datetime.fromtimestamp(os.path.getmtime(filepath)) > max_age:
            return None

        with open(filepath, "r") as file:
            data = json.load(file)

        return self._convert_from_json(data)

    def _cache_data(self, data: Any, cache_key: str):
        filepath = os.path.join(self.cache_dir, f"{cache_key}.json")

        with open(filepath, "w") as file:
            json.dump(self._convert_to_json(data), file)
        self._cache_sync.persist(filepath, f"{self._cache_sync_key_prefix}/{cache_key}.json")

    def _convert_to_json(self, data: Any):
        if isinstance(data, pd.DataFrame):
            return {"__type__": "dataframe", "data": data.to_json(date_format="iso")}

        if isinstance(data, pd.Series):
            return {"__type__": "series", "data": data.to_json(date_format="iso")}

        if isinstance(data, dict):
            return {key: self._convert_to_json(value) for key, value in data.items()}

        if isinstance(data, list):
            return [self._convert_to_json(item) for item in data]

        return data

    def _convert_from_json(self, data: Any):
        if isinstance(data, dict) and data.get("__type__") == "dataframe":
            df = pd.read_json(StringIO(data["data"]))
            df.columns = pd.to_datetime(df.columns)
            return df

        if isinstance(data, dict) and data.get("__type__") == "series":
            series = pd.read_json(StringIO(data["data"]), typ="series")
            series.index = pd.to_datetime(series.index)
            return series

        if isinstance(data, dict):
            return {key: self._convert_from_json(value) for key, value in data.items()}

        if isinstance(data, list):
            return [self._convert_from_json(item) for item in data]

        return data

    def _headers(self):
        return {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

    def _build_raw_us_gaap_dataframe(
            self,
            facts: Dict[str, Any],
            frequency: str,
            unit_preference: List[str],
    ) -> pd.DataFrame:
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        rows: Dict[str, pd.Series] = {}

        for tag in sorted(us_gaap.keys()):
            series = self._first_available_series(
                us_gaap=us_gaap,
                sec_tags=[tag],
                frequency=frequency,
                unit_preference=unit_preference,
            )

            if series is not None and not series.empty:
                rows[tag] = series

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).T
        df.columns = pd.to_datetime(df.columns)
        df = df.sort_index(axis=1, ascending=False)
        df = df.apply(pd.to_numeric, errors="coerce")

        return df

    def get_us_gaap_tag_map(self, symbol: str, use_cache: bool = True) -> dict:
        facts = self.get_company_facts(symbol, use_cache=use_cache)

        if isinstance(facts, dict) and "error" in facts:
            return facts

        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        tag_map = {}

        for tag, payload in sorted(us_gaap.items()):
            tag_map[tag] = {
                "tag": tag,
                "label": payload.get("label") or tag,
                "description": payload.get("description") or "",
                "units": list(payload.get("units", {}).keys()),
            }

        return tag_map

    def get_balance_sheet_raw_labeled(
            self,
            symbol: str,
            frequency: str = "annual",
            use_cache: bool = True,
    ):
        df = self.get_balance_sheet(symbol, frequency=frequency, use_cache=use_cache, scope="raw")

        if isinstance(df, dict) and "error" in df:
            return df

        tag_map = self.get_us_gaap_tag_map(symbol, use_cache=use_cache)

        # Neue Index Labels bauen
        new_index = []
        metadata = {}

        for tag in df.index:
            info = tag_map.get(tag, {})

            label = info.get("label", tag)

            new_index.append(label)

            metadata[label] = {
                "tag": tag,
                "description": info.get("description", ""),
                "units": info.get("units", []),
            }

        df.index = new_index

        return df, metadata

    def get_balance_sheet_line_item(
            self,
            symbol: str,
            line_item: str,
            frequency: str = "annual",
            scope: str = "core",
            by: str = "index",
            use_cache: bool = True,
    ):
        if frequency not in ["annual", "quarterly"]:
            return {"error": f"Ungültige Frequenz: {frequency}.", "symbol": symbol}

        if scope not in ["core", "raw", "labeled"]:
            return {"error": f"Ungültiger Scope: {scope}.", "symbol": symbol}

        if by not in ["index", "tag", "label"]:
            return {"error": f"Ungültiger by-Wert: {by}. Verwende 'index', 'tag' oder 'label'.", "symbol": symbol}

        try:
            if scope == "labeled" or by == "label":
                labeled_result = self.get_balance_sheet_raw_labeled(
                    symbol=symbol,
                    frequency=frequency,
                    use_cache=use_cache,
                )

                if isinstance(labeled_result, dict) and "error" in labeled_result:
                    return labeled_result

                df, metadata = labeled_result

                if by == "tag":
                    matching_label = None

                    for label, meta in metadata.items():
                        if meta.get("tag") == line_item:
                            matching_label = label
                            break

                    if not matching_label:
                        return {
                            "error": f"Tag '{line_item}' wurde nicht gefunden.",
                            "symbol": symbol,
                            "line_item": line_item,
                        }

                    row_key = matching_label
                else:
                    row_key = line_item

                if row_key not in df.index:
                    return {
                        "error": f"Line Item '{line_item}' wurde nicht gefunden.",
                        "symbol": symbol,
                        "line_item": line_item,
                    }

                series = df.loc[row_key].dropna()

                if series.empty:
                    return {
                        "error": f"Keine Werte für '{line_item}' gefunden.",
                        "symbol": symbol,
                        "line_item": line_item,
                    }

                latest_date = series.index[0]
                latest_value = series.iloc[0]
                meta = metadata.get(row_key, {})

                return {
                    "symbol": symbol.upper(),
                    "frequency": frequency,
                    "scope": "labeled",
                    "line_item": row_key,
                    "tag": meta.get("tag"),
                    "label": row_key,
                    "description": meta.get("description", ""),
                    "units": meta.get("units", []),
                    "date": latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, "strftime") else str(latest_date),
                    "value": float(latest_value),
                    "series": {
                        date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date): float(value)
                        for date, value in series.items()
                    },
                }

            df = self.get_balance_sheet(
                symbol=symbol,
                frequency=frequency,
                use_cache=use_cache,
                scope=scope,
            )

            if isinstance(df, dict) and "error" in df:
                return df

            row_key = line_item

            if row_key not in df.index:
                return {
                    "error": f"Line Item '{line_item}' wurde nicht gefunden.",
                    "symbol": symbol,
                    "line_item": line_item,
                    "scope": scope,
                }

            series = df.loc[row_key].dropna()

            if series.empty:
                return {
                    "error": f"Keine Werte für '{line_item}' gefunden.",
                    "symbol": symbol,
                    "line_item": line_item,
                    "scope": scope,
                }

            latest_date = series.index[0]
            latest_value = series.iloc[0]

            return {
                "symbol": symbol.upper(),
                "frequency": frequency,
                "scope": scope,
                "line_item": row_key,
                "tag": row_key if scope == "raw" else None,
                "label": row_key,
                "date": latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, "strftime") else str(latest_date),
                "value": float(latest_value),
                "series": {
                    date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date): float(value)
                    for date, value in series.items()
                },
            }

        except Exception as e:
            return {
                "error": f"Fehler beim Abrufen von '{line_item}' für {symbol}: {describe_exception(e)}",
                "symbol": symbol,
                "line_item": line_item,
            }

    def is_foreign_private_issuer(self, symbol: str, use_cache: bool = True) -> bool:
        """Öffentlicher Zugriff auf die FPI-Erkennung für Aufrufer außerhalb
        von SecSource (z. B. DataLoader.get_shares_outstanding)."""
        try:
            facts = self.get_company_facts(symbol, use_cache=use_cache)
            if isinstance(facts, dict) and "error" in facts:
                return False
            return self._is_foreign_private_issuer(facts)
        except Exception:
            return False

    def _is_foreign_private_issuer(self, facts: Dict[str, Any]) -> bool:
        """
        Erkennt Foreign Private Issuers / ADRs anhand typischer SEC-Forms.

        Hintergrund:
        - 20-F  -> Annual Report für Foreign Private Issuers
        - 6-K   -> laufende Reports / Zwischenmeldungen für Foreign Private Issuers
        - 40-F  -> kanadische Foreign Issuers

        US-Unternehmen verwenden diese Forms normalerweise nicht.
        """

        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        foreign_forms = {
            "20-F",
            "6-K",
            "40-F",
        }

        for tag_payload in us_gaap.values():

            units = tag_payload.get("units", {})

            for values in units.values():

                if not isinstance(values, list):
                    continue

                for item in values:

                    form = item.get("form")

                    if form in foreign_forms:
                        return True

        return False


BALANCE_SHEET_MAP = {
    "Total Assets": ["Assets"],
    "Current Assets": ["AssetsCurrent"],
    "Total Liabilities Net Minority Interest": ["Liabilities"],
    "Current Liabilities": ["LiabilitiesCurrent"],
    "Stockholders Equity": ["StockholdersEquity",
                            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "Common Stock Equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
        "CommonStocksIncludingAdditionalPaidInCapital",
    ],
    "Total Stockholders Equity": ["StockholdersEquity"],
    "Treasury Stock": ["TreasuryStockValue", "TreasuryStockValueAcquiredCostMethod"],
    "Retained Earnings": ["RetainedEarningsAccumulatedDeficit"],
    "Additional Paid In Capital": ["AdditionalPaidInCapital"],
    "Common Stock": [
        "CommonStocksIncludingAdditionalPaidInCapital",
        "CommonStockValue",
        "CommonStockNoParValue",
    ],
    "Preferred Stock": [
        "PreferredStocksIncludingAdditionalPaidInCapital",
        "PreferredStockValue",
    ],
    "Cash And Cash Equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "Cash",
        "CashEquivalentsAtCarryingValue",
    ],
    "Cash Cash Equivalents And Short Term Investments": [
        "CashCashEquivalentsAndShortTermInvestments",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashAndCashEquivalentsAtCarryingValue",
        "Cash",
        "CashEquivalentsAtCarryingValue",
        "ShortTermInvestments",
        "OtherShortTermInvestments",
        "MarketableSecuritiesCurrent",
        "AvailableForSaleSecuritiesCurrent",
    ],
    "Other Short Term Investments": [
    "ShortTermInvestments",
    "OtherShortTermInvestments",
    "MarketableSecuritiesCurrent",
    "MarketableSecurities",
    "AvailableForSaleSecuritiesCurrent",
    "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
    # NVDA
    "AvailableForSaleSecuritiesDebtSecurities",
    # neue SEC-Filer
    "DebtSecuritiesCurrent",
    ],
    "Inventory": [
        "InventoryNet",
        "Inventory",
        "InventoryFinishedGoodsNetOfReserves",
        "InventoryPartsAndComponentsNetOfReserves",
        "InventoryRawMaterialsAndPurchasedPartsNetOfReserves",
        "InventoryWorkInProcessAndRawMaterialsNetOfReserves",
    ],
    "Total Debt": [
    "DebtLongtermAndShorttermCombinedAmount",
    "DebtCurrentAndNoncurrent",
    "LongTermDebtAndFinanceLeaseObligationsCurrentAndNoncurrent",
    "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
    "LongTermDebtAndCapitalLeaseObligations",
    "DebtInstrumentCarryingAmount",
    ],
    "Long Term Debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "OtherLongTermDebt",
        "LongTermDebtAndFinanceLeaseObligationsCurrentAndNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
    ],
    "Short Term Debt": [
        "ShortTermBorrowings",
        "ShortTermDebtCurrent",
        "LongTermDebtCurrent",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "CommercialPaper",
        "OtherShortTermBorrowings",
        "NotesAndLoansPayableCurrent",
    ],
    "Current Debt": [
        "ShortTermBorrowings",
        "ShortTermDebtCurrent",
        "LongTermDebtCurrent",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "CommercialPaper",
        "OtherShortTermBorrowings",
        "NotesAndLoansPayableCurrent",
    ],
    "Long Term Debt And Capital Lease Obligation": [
        "LongTermDebtAndFinanceLeaseObligationsCurrentAndNoncurrent",
        "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "OtherLongTermDebt",
        "FinanceLeaseLiability",
        "FinanceLeaseLiabilityNoncurrent",
    ],
    "Goodwill": [
        "Goodwill",
        "BusinessAcquisitionPurchasePriceAllocationGoodwillAmount",
    ],
    "Other Intangible Assets": [
    "OtherIntangibleAssetsNet",
    "FiniteLivedIntangibleAssetsNet",
    "IndefiniteLivedTrademarks",
    "IndefiniteLivedFranchiseRights",
    "OtherIndefiniteLivedIntangibleAssets",
    "IndefiniteLivedIntangibleAssetsExcludingGoodwill",
    "IndefiniteLivedIntangibleAssetsExcludingGoodwillFairValueDisclosure",
    "IntangibleAssetsNetExcludingGoodwill",
    "BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibleAssetsOtherThanGoodwill",
    ],
    "Net PPE": [
        "PropertyPlantAndEquipmentNet",
        "FinanceLeaseRightOfUseAsset",
        "OperatingLeaseRightOfUseAsset",
    ],
    "Accounts Receivable": [
        "AccountsReceivableNetCurrent",
        "NontradeReceivablesCurrent",
    ],
    "Receivables": [
        "AccountsReceivableNetCurrent",
        "NontradeReceivablesCurrent",
    ],
    "Accounts Payable": [
        "AccountsPayableCurrent",
        "AccountsPayable",
        "AccountsPayableTradeCurrent",
        "AccountsPayableOtherCurrent",
        "AccountsPayableAndAccruedLiabilitiesCurrent",
    ],
    "Minority Interest": [
        "MinorityInterest",
        "NoncontrollingInterestInConsolidatedEntity",
    ],
    "Ordinary Shares Number": [
        "EntityCommonStockSharesOutstanding",
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ],
    "Share Issued": [
        "CommonStockSharesIssued",
        "EntityCommonStockSharesIssued",
    ],
}

INCOME_STATEMENT_MAP = {
    "Total Revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],

    "Revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],

    #
    # Banken
    #

    "Interest Income": [
        "InterestIncomeExpenseNet",
        "InterestAndDividendIncomeOperating",
        "InterestAndFeeIncomeLoansAndLeases",
        "InterestIncomeOperating",
        "InterestIncomeOther",
        "InterestIncomeFederalFundsSoldAndSecuritiesPurchasedUnderAgreementsToResell",
        "InterestIncomeDepositsWithFinancialInstitutions",
        "InterestIncomeSecuritiesTaxable",
        "InterestIncomeSecuritiesTaxExempt",
    ],

    "Net Interest Income": [
        "InterestIncomeExpenseNet",
        "InterestIncomeExpenseAfterProvisionForLoanLoss",
    ],

    "Noninterest Income": [
        "NoninterestIncome",
        "NoninterestIncomeOther",
    ],

    "Noninterest Expense": [
        "NoninterestExpense",
        "OtherNoninterestExpense",
    ],

    #
    # Klassisches Income Statement
    #

    "Cost Of Revenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
    ],

    "Gross Profit": [
        "GrossProfit",
    ],

    "Operating Income": [
        "OperatingIncomeLoss",
    ],

    #
    # Für EBIT-Fallback
    #

    "Pretax Income": [
        "IncomeBeforeTaxExpenseBenefit",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],

    #
    # Nur echte EBIT-Tags
    #

    "EBIT": [
        "EarningsBeforeInterestAndTaxes",
    ],

    "EBITDA": [
        "EarningsBeforeInterestTaxesDepreciationAndAmortization",
    ],

    "Interest Expense": [
        "FinanceLeaseInterestExpense",
        "InterestCostsIncurred",
        "InterestExpenseDebt",
        "InterestExpense",
        "InterestExpenseNonoperating",

        # zusätzliche Varianten
        "InterestAndDebtExpense",
        "InterestExpenseAndDebtExpense",
        "InterestExpenseDebtExcludingAmortization",
        "InterestExpenseBorrowings",
        "InterestExpenseDeposits",
        "InterestExpenseOther",
        "InterestExpenseAndOther",
        "InterestExpenseRelatedParty",
        "InterestExpenseLongTermDebt",
        "InterestExpenseShortTermBorrowings",
        "InterestExpenseCapitalLease",
    ],

    "Net Income": [
        "NetIncomeLoss",
    ],

    "Net Income Common Stockholders": [
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "NetIncomeLoss",
    ],

    "Diluted EPS": [
        "EarningsPerShareDiluted",
    ],

    "Basic EPS": [
        "EarningsPerShareBasic",
    ],

    #
    # D&A
    #
    # Wird über _sum_available_series() aufgebaut
    #

    "Depreciation And Amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "Depreciation",
        "AmortizationOfIntangibleAssets",
        "FiniteLivedIntangibleAssetsAmortizationExpense",
    ],

    "Depreciation Depletion And Amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "Depreciation",
        "AmortizationOfIntangibleAssets",
        "FiniteLivedIntangibleAssetsAmortizationExpense",
    ],
}

CASHFLOW_MAP = {
    "Operating Cash Flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],

    "Cash Flow From Continuing Operating Activities": [
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        "NetCashProvidedByUsedInOperatingActivities",
    ],

    "Capital Expenditure": [
        # häufig bestes aggregiertes CapEx-Tag
        "SegmentExpenditureAdditionToLongLivedAssets",

        # klassische CapEx-Tags
        "PaymentsToAcquireProductiveAssets",
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireOtherPropertyPlantAndEquipment",
        "PaymentsToAcquirePropertyPlantEquipmentAndOtherProductiveAssets",
        "CapitalExpenditures",

        # häufig bei Leasing-/Equipment-Finanzierern
        "PaymentsToAcquireEquipmentOnLease",
    ],

    "Purchase Of PPE": [
        "SegmentExpenditureAdditionToLongLivedAssets",

        "PaymentsToAcquireProductiveAssets",
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireOtherPropertyPlantAndEquipment",
        "PaymentsToAcquirePropertyPlantEquipmentAndOtherProductiveAssets",

        "PaymentsToAcquireEquipmentOnLease",
    ],

    "Capital Expenditure Reported": [
        "SegmentExpenditureAdditionToLongLivedAssets",

        "PaymentsToAcquireProductiveAssets",
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireOtherPropertyPlantAndEquipment",
        "PaymentsToAcquirePropertyPlantEquipmentAndOtherProductiveAssets",

        "PaymentsToAcquireEquipmentOnLease",
    ],

    "Free Cash Flow": [
        "FreeCashFlow",
    ],

    "Dividends Paid": [
        "PaymentsOfDividends",
        "PaymentsOfDividendsCommonStock",
        "PaymentsOfDividendsPreferredStockAndPreferenceStock",
        "DividendsCash",
        "DividendsCommonStockCash",
    ],

    "Repurchase Of Capital Stock": [
        "PaymentsForRepurchaseOfCommonStock",
        "StockRepurchasedDuringPeriodValue",
        "StockRepurchasedAndRetiredDuringPeriodValue",
        "TreasuryStockValueAcquiredCostMethod",
    ],

    "Net Cash From Investing": [
        "NetCashProvidedByUsedInInvestingActivities",
        "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations",
    ],

    "Net Cash From Financing": [
        "NetCashProvidedByUsedInFinancingActivities",
        "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations",
    ],

    "Acquisitions": [
        "PaymentsToAcquireBusinessesNetOfCashAcquired",
        "PaymentsToAcquireBusinessTwoNetOfCashAcquired",
        "PaymentsToAcquireInterestInSubsidiariesAndAffiliates",
        "PaymentsToAcquireEquityMethodInvestments",
    ],

    "Investments In Securities": [
        "PaymentsToAcquireMarketableSecurities",
        "PaymentsToAcquireAvailableForSaleSecurities",
        "PaymentsToAcquireAvailableForSaleSecuritiesDebt",
        "PaymentsToAcquireEquitySecuritiesFvNi",
        "PaymentsToAcquireOtherInvestments",

        "ProceedsFromSaleAndMaturityOfMarketableSecurities",
        "ProceedsFromSaleOfAvailableForSaleSecurities",
        "ProceedsFromSaleOfAvailableForSaleSecuritiesDebt",
        "ProceedsFromSaleOfEquitySecuritiesFvNi",
        "ProceedsFromMaturitiesPrepaymentsAndCallsOfAvailableForSaleSecurities",
    ],

    "Debt Issuance": [
        "ProceedsFromBankDebt",
        "ProceedsFromConvertibleDebt",
        "ProceedsFromIssuanceOfSeniorLongTermDebt",
        "ProceedsFromIssuanceOfDebt",
        "ProceedsFromIssuanceOfLongTermDebt",
        "ProceedsFromDebtNetOfIssuanceCosts",
    ],

    "Debt Repayment": [
        "RepaymentsOfBankDebt",
        "RepaymentsOfSeniorDebt",
        "RepaymentsOfDebt",
        "RepaymentsOfConvertibleDebt",
    ],

    "Stock Issuance": [
        "ProceedsFromIssuanceOfCommonStock",
        "ProceedsFromStockOptionsExercised",
        "ProceedsFromStockPlans",
        "ProceedsFromIssuanceOfWarrants",
    ],
}

FINANCIAL_BALANCE_SHEET_MAP = dict(BALANCE_SHEET_MAP)

FINANCIAL_BALANCE_SHEET_MAP.update({

    "Cash And Cash Equivalents": [
        "CashAndDueFromBanks",
        "InterestBearingDepositsInBanks",
    ],

    "Other Short Term Investments": [
    "DebtSecuritiesAvailableForSaleAndHeldToMaturityFairValue",
    "DebtSecuritiesAvailableForSaleAndHeldToMaturity",
    "AvailableForSaleSecurities",
    "HeldToMaturitySecurities",
    ],
})


FINANCIAL_TICKERS = {
    "BAC",
    "JPM",
    "WFC",
    "C",
    "GS",
    "MS",
    "SCHW",
    "PGR",
    "AIG",
    "CB",
    "TRV",
    "ALL",
}