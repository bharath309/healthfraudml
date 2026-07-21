"""
Tests for the coding audit (code-vs-description check).

These exercise the optional [rag] path and are skipped when chromadb is absent.
The index is built once into a temp dir shared by the module.
"""

import pytest

from healthfraudml.auditor.coding_audit import (
    CANNOT_VALIDATE,
    CHROMADB_AVAILABLE,
    MATCH,
    NOT_VALIDATABLE_EM,
    POSSIBLE_MISCODING,
    CodingAuditor,
    is_em_code,
)

pytestmark = pytest.mark.skipif(
    not CHROMADB_AVAILABLE, reason="requires the [rag] extra (chromadb)"
)


@pytest.fixture(scope="module")
def auditor(tmp_path_factory):
    index_dir = tmp_path_factory.mktemp("coding_index")
    return CodingAuditor(index_dir=str(index_dir))


def test_em_code_detection():
    assert is_em_code("99213")
    assert is_em_code("99285")
    assert not is_em_code("G0008")
    assert not is_em_code("71046")


def test_match_verdict(auditor):
    result = auditor.audit_line("G0008", "administration of influenza vaccine")
    assert result["verdict"] == MATCH
    assert result["resolved_code"] == "G0008"
    assert result["similarity"] > 0.5


def test_possible_miscoding_verdict(auditor):
    """Billed pneumococcal admin, described as influenza admin -> flag for review."""
    result = auditor.audit_line("G0009", "administration of influenza vaccine")
    assert result["verdict"] == POSSIBLE_MISCODING
    assert result["resolved_code"] == "G0008"
    assert "review" in result["note"]


def test_cannot_validate_unresolvable_description(auditor):
    result = auditor.audit_line("G0008", "qwerty zxcvb nonsense text")
    assert result["verdict"] == CANNOT_VALIDATE
    assert result["resolved_code"] is None


def test_cannot_validate_missing_description(auditor):
    result = auditor.audit_line("G0008", "")
    assert result["verdict"] == CANNOT_VALIDATE
    assert "no description" in result["note"]


def test_em_line_not_validatable(auditor):
    result = auditor.audit_line("99214", "office visit established patient")
    assert result["verdict"] == NOT_VALIDATABLE_EM
    assert "documentation" in result["note"]


def test_code_outside_vocabulary_cannot_validate(auditor):
    """No name on file for the billed code -> never manufacture a miscoding."""
    result = auditor.audit_line("71046", "chest x-ray 2 views")
    assert result["verdict"] == CANNOT_VALIDATE
    assert "no plain-language name on file" in result["note"]


def test_audit_bill_covers_every_line(auditor):
    rows = auditor.audit_bill([
        {"cpt_code": "G0008", "description": "administration of influenza vaccine"},
        {"cpt_code": "99214", "description": "office visit"},
    ])
    assert len(rows) == 2
    assert rows[0]["verdict"] == MATCH
    assert rows[1]["verdict"] == NOT_VALIDATABLE_EM
