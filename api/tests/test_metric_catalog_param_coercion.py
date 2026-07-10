"""Regressionstest für den Live-Bug (2026-07-10): eine gespeicherte
individuelle Analyse-Definition mit einem numerischen Metrik-Parameter
(z. B. analyze_payout_ratio's "threshold") crashte mit

    ufunc 'greater' did not contain a loop with signature matching types
    (Float64DType, StrDType) -> None

weil der Parameter als JSON-String (nicht als Zahl) gespeichert war -
MetricSelection.params ist ein untypisiertes dict, das ungeprüft an die
Model.py-Methode durchgereicht wird. Der Frontend-Fix (Number()-Coercion im
Eingabefeld) verhindert das nur für NEU eingegebene Werte, nicht für bereits
in der DB gespeicherte Definitionen - _generic_call muss selbst koerzieren.
"""
from unittest.mock import MagicMock

from api.services.metric_catalog import _generic_call


class _FakeModel:
    def calc_with_float_param(self, symbol, threshold: float = 75.0):
        return {"symbol": symbol, "threshold": threshold, "threshold_type": type(threshold).__name__}

    def calc_with_int_param(self, symbol, years: int = 10):
        return {"years": years, "years_type": type(years).__name__}

    def calc_with_string_param(self, symbol, frequency: str = "annual"):
        return {"frequency": frequency, "frequency_type": type(frequency).__name__}

    def calc_no_annotation(self, symbol, mystery=None):
        return {"mystery": mystery, "mystery_type": type(mystery).__name__}


def _fake_action():
    action = MagicMock()
    action.model = _FakeModel()
    return action


def test_string_number_param_is_coerced_to_float():
    """Reproduziert exakt den Live-Bug: ein als String gespeicherter
    "threshold"-Parameter (z. B. aus einer vor dem Frontend-Fix gespeicherten
    Definition) muss trotzdem als float bei der Model.py-Methode ankommen."""
    result = _generic_call(_fake_action(), "PYPL", {"threshold": "50"}, "calc_with_float_param", True)

    assert result["threshold"] == 50.0
    assert result["threshold_type"] == "float"


def test_string_number_param_is_coerced_to_int():
    result = _generic_call(_fake_action(), "PYPL", {"years": "5"}, "calc_with_int_param", True)

    assert result["years"] == 5
    assert result["years_type"] == "int"


def test_real_json_number_param_passes_through_unchanged():
    """Der reguläre, bereits korrekte Fall (Frontend sendet jetzt eine echte
    JSON-Zahl) darf durch die Koerzierung nicht verändert werden."""
    result = _generic_call(_fake_action(), "PYPL", {"threshold": 60.0}, "calc_with_float_param", True)

    assert result["threshold"] == 60.0
    assert result["threshold_type"] == "float"


def test_string_typed_param_is_left_as_string():
    """Ein Parameter, dessen Model.py-Signatur str erwartet (z. B.
    frequency), darf NICHT numerisch koerziert werden."""
    result = _generic_call(_fake_action(), "PYPL", {"frequency": "quarterly"}, "calc_with_string_param", True)

    assert result["frequency"] == "quarterly"
    assert result["frequency_type"] == "str"


def test_non_numeric_string_for_float_param_is_left_unchanged():
    """Ein nicht-numerischer String darf nicht crashen - float(...) schlägt
    fehl, der Rohwert bleibt unverändert (verhält sich dann wie vor diesem
    Fix, statt eine neue Exception in der Koerzierung selbst zu werfen)."""
    result = _generic_call(_fake_action(), "PYPL", {"threshold": "not-a-number"}, "calc_with_float_param", True)

    assert result["threshold"] == "not-a-number"


def test_param_without_type_annotation_is_left_unchanged():
    result = _generic_call(_fake_action(), "PYPL", {"mystery": "42"}, "calc_no_annotation", True)

    assert result["mystery"] == "42"
    assert result["mystery_type"] == "str"
