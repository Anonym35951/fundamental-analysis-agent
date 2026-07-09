"""Verifiziert, dass agent/industry_multiples.py nur Multiples referenziert,
die AgentAction.MULTIPLE_METHOD_MAP tatsaechlich kennt - ein Tippfehler oder
ein veraltetes Multiple in der Zuordnungstabelle waere sonst ein stiller
Laufzeitfehler erst zur Analysezeit."""
from agent.AgentAction import AgentAction
from agent.industry_multiples import (
    GLOBAL_FALLBACK_MULTIPLES,
    INDUSTRY_MULTIPLES_OVERRIDES,
    SECTOR_DEFAULT_MULTIPLES,
    resolve_multiples,
)

VALID_MULTIPLES = set(AgentAction.MULTIPLE_METHOD_MAP.keys())


def test_global_fallback_multiples_are_valid():
    assert set(GLOBAL_FALLBACK_MULTIPLES) <= VALID_MULTIPLES


def test_sector_default_multiples_are_valid():
    for sector, multiples in SECTOR_DEFAULT_MULTIPLES.items():
        assert set(multiples) <= VALID_MULTIPLES, f"Ungueltiges Multiple fuer Sektor {sector!r}"
        assert len(multiples) == 2, f"Sektor {sector!r} sollte genau 2 Multiples haben"


def test_industry_override_multiples_are_valid():
    for industry, multiples in INDUSTRY_MULTIPLES_OVERRIDES.items():
        assert set(multiples) <= VALID_MULTIPLES, f"Ungueltiges Multiple fuer Industrie {industry!r}"
        # Tabak ist die einzige bekannte 3-Multiples-Ausnahme (Legacy-Verhalten).
        assert len(multiples) in (2, 3), f"Industrie {industry!r} sollte 2 (oder 3) Multiples haben"


def test_resolve_multiples_prefers_industry_over_sector():
    multiples, source = resolve_multiples(sector="Healthcare", industry="Biotechnology")
    assert source == "industry"
    assert multiples == INDUSTRY_MULTIPLES_OVERRIDES["Biotechnology"]


def test_resolve_multiples_falls_back_to_sector():
    multiples, source = resolve_multiples(sector="Technology", industry="Some Unknown Industry")
    assert source == "sector"
    assert multiples == SECTOR_DEFAULT_MULTIPLES["Technology"]


def test_resolve_multiples_falls_back_to_global():
    multiples, source = resolve_multiples(sector=None, industry=None)
    assert source == "fallback"
    assert multiples == GLOBAL_FALLBACK_MULTIPLES
