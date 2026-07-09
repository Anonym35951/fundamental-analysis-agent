import unittest

from sec_source import SecSource


class TestForeignIssuerQuarterlyBlocking(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sec = SecSource(
            user_agent="YourName your@email.com"
        )

    def _success(self, message: str):
        print(f"\n✅ SUCCESS: {message}")

    #
    # ============================================================
    # FOREIGN ISSUER DETECTION
    # ============================================================
    #

    def test_baba_is_foreign_private_issuer(self):
        facts = self.sec.get_company_facts("BABA")

        self.assertFalse(
            isinstance(facts, dict) and "error" in facts,
            msg=f"Fehler beim Laden der Company Facts: {facts}"
        )

        is_foreign = self.sec._is_foreign_private_issuer(facts)

        self.assertTrue(
            is_foreign,
            msg="BABA sollte als Foreign Private Issuer erkannt werden."
        )

        self._success("BABA wurde korrekt als Foreign Private Issuer erkannt.")

    def test_aapl_is_not_foreign_private_issuer(self):
        facts = self.sec.get_company_facts("AAPL")

        self.assertFalse(
            isinstance(facts, dict) and "error" in facts,
            msg=f"Fehler beim Laden der Company Facts: {facts}"
        )

        is_foreign = self.sec._is_foreign_private_issuer(facts)

        self.assertFalse(
            is_foreign,
            msg="AAPL sollte KEIN Foreign Private Issuer sein."
        )

        self._success("AAPL wurde korrekt NICHT als Foreign Private Issuer erkannt.")

    #
    # ============================================================
    # CASHFLOW BLOCKING
    # ============================================================
    #

    def test_baba_quarterly_cashflow_is_blocked(self):
        result = self.sec.get_cashflow_statement(
            symbol="BABA",
            frequency="quarterly",
        )

        self.assertIsInstance(result, dict)

        self.assertIn("error", result)

        self.assertTrue(
            result.get("foreign_issuer"),
            msg="foreign_issuer sollte True sein."
        )

        self._success("BABA Quarterly Cashflow wurde korrekt blockiert.")

    def test_baba_annual_cashflow_still_works(self):
        result = self.sec.get_cashflow_statement(
            symbol="BABA",
            frequency="annual",
        )

        self.assertFalse(
            isinstance(result, dict) and "error" in result,
            msg=f"Annual Cashflow sollte funktionieren: {result}"
        )

        self._success("BABA Annual Cashflow funktioniert weiterhin.")

    #
    # ============================================================
    # INCOME STATEMENT BLOCKING
    # ============================================================
    #

    def test_baba_quarterly_income_statement_is_blocked(self):
        result = self.sec.get_stock_financials(
            symbol="BABA",
            frequency="quarterly",
        )

        self.assertIsInstance(result, dict)

        self.assertIn("error", result)

        self.assertTrue(
            result.get("foreign_issuer"),
            msg="foreign_issuer sollte True sein."
        )

        self._success("BABA Quarterly Income Statement wurde korrekt blockiert.")

    def test_baba_annual_income_statement_still_works(self):
        result = self.sec.get_stock_financials(
            symbol="BABA",
            frequency="annual",
        )

        self.assertFalse(
            isinstance(result, dict) and "error" in result,
            msg=f"Annual Income Statement sollte funktionieren: {result}"
        )

        self._success("BABA Annual Income Statement funktioniert weiterhin.")

    #
    # ============================================================
    # BALANCE SHEET BLOCKING
    # ============================================================
    #

    def test_baba_quarterly_balance_sheet_is_blocked(self):
        result = self.sec.get_balance_sheet(
            symbol="BABA",
            frequency="quarterly",
        )

        self.assertIsInstance(result, dict)

        self.assertIn("error", result)

        self.assertTrue(
            result.get("foreign_issuer"),
            msg="foreign_issuer sollte True sein."
        )

        self._success("BABA Quarterly Balance Sheet wurde korrekt blockiert.")

    def test_baba_annual_balance_sheet_still_works(self):
        result = self.sec.get_balance_sheet(
            symbol="BABA",
            frequency="annual",
        )

        self.assertFalse(
            isinstance(result, dict) and "error" in result,
            msg=f"Annual Balance Sheet sollte funktionieren: {result}"
        )

        self._success("BABA Annual Balance Sheet funktioniert weiterhin.")

    #
    # ============================================================
    # US COMPANY SHOULD STILL WORK QUARTERLY
    # ============================================================
    #

    def test_aapl_quarterly_cashflow_still_works(self):
        result = self.sec.get_cashflow_statement(
            symbol="AAPL",
            frequency="quarterly",
        )

        self.assertFalse(
            isinstance(result, dict) and "error" in result,
            msg=f"AAPL Quarterly Cashflow sollte funktionieren: {result}"
        )

        self._success("AAPL Quarterly Cashflow funktioniert weiterhin.")

    def test_aapl_quarterly_income_statement_still_works(self):
        result = self.sec.get_stock_financials(
            symbol="AAPL",
            frequency="quarterly",
        )

        self.assertFalse(
            isinstance(result, dict) and "error" in result,
            msg=f"AAPL Quarterly Income Statement sollte funktionieren: {result}"
        )

        self._success("AAPL Quarterly Income Statement funktioniert weiterhin.")

    def test_aapl_quarterly_balance_sheet_still_works(self):
        result = self.sec.get_balance_sheet(
            symbol="AAPL",
            frequency="quarterly",
        )

        self.assertFalse(
            isinstance(result, dict) and "error" in result,
            msg=f"AAPL Quarterly Balance Sheet sollte funktionieren: {result}"
        )

        self._success("AAPL Quarterly Balance Sheet funktioniert weiterhin.")


if __name__ == "__main__":
    unittest.main()