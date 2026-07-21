"""
Unit tests for the Patient Billing Auditor package.
"""

import os
import pytest
from healthfraudml.auditor import BillingAuditor, LLMBillParser


@pytest.fixture
def sample_bill_items():
    return [
        {"cpt_code": "56420", "amount": 709.00, "description": "ED Proc Minor"},
        {"cpt_code": "99285", "amount": 6672.00, "description": "ED Proc Level 5 W/Proc"},
    ]


def test_billing_auditor_upcoding(sample_bill_items):
    auditor = BillingAuditor(provider_name="Example Health System")
    report = auditor.audit_bill(sample_bill_items)

    assert report["provider_name"] == "Example Health System"
    assert report["total_billed"] == 7381.00
    assert report["risk_level"] == "High"

    # Savings should be 6672.0 - 1200.0 (fair_max for CPT 99283) = 5472.0
    assert report["suggested_savings"] == 5472.0

    findings_types = [f["type"] for f in report["findings"]]
    assert "Upcoding" in findings_types
    assert "Unbundling" in findings_types
    assert report["dispute_letter"] != ""


def test_expanded_cms_benchmark_loaded():
    """The CMS PFS benchmark should extend coverage well beyond the built-ins."""
    ref = BillingAuditor.CPT_REFERENCE
    # Far more than the 10 curated built-ins.
    assert len(ref) > 1000
    # A common expanded code (CT head) is present and price-only (no severity).
    assert "70450" in ref
    assert "severity" not in ref["70450"]
    assert ref["70450"]["fair_max"] > ref["70450"]["medicare_max"]
    # Built-ins retain their curated metadata.
    assert ref["99285"]["severity"] == 5


def test_expanded_benchmark_flags_overpricing_without_severity():
    """A price-only expanded code overpriced -> Overpricing flag, no crash, no upcoding."""
    auditor = BillingAuditor(provider_name="Imaging Center")
    report = auditor.audit_bill([
        {"cpt_code": "70450", "amount": 2400.00, "description": "CT head without contrast"},
    ])
    types = [f["type"] for f in report["findings"]]
    assert "Overpricing" in types
    assert "Upcoding" not in types  # no severity metadata -> never accused of upcoding
    assert report["audited_items"][0]["status"] == "Overpriced"


def test_unbundling_not_triggered_by_diagnostics():
    """Regression: E/M + imaging must NOT flag unbundling (v0.3.0 fix).

    Bundling rules concern minor surgical procedures with a global period,
    not diagnostics like a chest X-ray.
    """
    auditor = BillingAuditor(provider_name="Clinic")
    report = auditor.audit_bill([
        {"cpt_code": "99213", "amount": 210.00, "description": "Office visit"},
        {"cpt_code": "71046", "amount": 120.00, "description": "Chest X-ray 2 views"},
    ])
    assert "Unbundling" not in [f["type"] for f in report["findings"]]


def test_unbundling_preserved_for_curated_procedure():
    """Regression: E/M + curated minor surgical procedure still flags unbundling."""
    auditor = BillingAuditor(provider_name="Hospital")
    report = auditor.audit_bill([
        {"cpt_code": "99285", "amount": 6672.00, "description": "ED Visit Level 5"},
        {"cpt_code": "56420", "amount": 709.00, "description": "Bartholin Cyst I&D"},
    ])
    types = [f["type"] for f in report["findings"]]
    assert "Unbundling" in types
    assert "Upcoding" in types
    assert report["suggested_savings"] == 5472.0


def test_letter_asserts_only_present_findings():
    """A bill with only an overpricing issue must not allege coding violations."""
    auditor = BillingAuditor(provider_name="Imaging Center")
    report = auditor.audit_bill([
        {"cpt_code": "70450", "amount": 2400.00, "description": "CT head"},
    ])
    letter = report["dispute_letter"]
    assert "Overpricing" in [f["type"] for f in report["findings"]]
    # No E/M, upcoding or unbundling language on a single imaging line.
    for phrase in ["99285", "upcoding", "Evaluation and Management", "bundled"]:
        assert phrase not in letter, f"letter should not mention {phrase!r}"
    assert "one item was identified" in letter


def test_letter_includes_rules_when_findings_present():
    """Coding-rule paragraphs appear only when the matching finding was raised."""
    auditor = BillingAuditor(provider_name="Example Hospital")
    letter = auditor.audit_bill([
        {"cpt_code": "99285", "amount": 6672.00, "description": "ED Visit Level 5"},
        {"cpt_code": "56420", "amount": 709.00, "description": "Bartholin Cyst I&D"},
    ])["dispute_letter"]
    assert "Evaluation and Management" in letter
    assert "CPT 99285" in letter


def test_letter_makes_no_regulatory_compliance_claim():
    """No statutory/regulatory assertions ship without the counsel-review gate."""
    auditor = BillingAuditor(provider_name="Example Hospital")
    letter = auditor.audit_bill([
        {"cpt_code": "99285", "amount": 6672.00, "description": "ED Visit Level 5"},
        {"cpt_code": "56420", "amount": 709.00, "description": "Bartholin Cyst I&D"},
    ])["dispute_letter"]
    for claim in ["NCCI", "National Correct Coding", "does not comply", "501(r)",
                  "No Surprises", "violation", "unlawful", "illegal"]:
        assert claim not in letter, f"letter must not assert {claim!r} before counsel review"


def test_code_names_loaded_and_labelled():
    """HCPCS Level II names are shown as-is; authored CPT names are marked."""
    assert len(BillingAuditor.CODE_NAMES) > 1000
    l2 = BillingAuditor.code_name("G0008")
    assert l2 and "unofficial" not in l2
    authored = BillingAuditor.code_name("99285")
    assert authored and authored.endswith("(unofficial name)")
    # No AMA CPT descriptors: numeric codes never come from the HCPCS file.
    for code, entry in BillingAuditor.CODE_NAMES.items():
        if entry["source"] == "cms_hcpcs_l2":
            assert code[0].isalpha() and code[0].upper() != "D"


def test_description_display_order():
    """Partner description wins; otherwise fall back to a name, else say so."""
    auditor = BillingAuditor(provider_name="Clinic")
    report = auditor.audit_bill([
        {"cpt_code": "G0008", "amount": 50.0, "description": "Partner's own wording"},
        {"cpt_code": "G0008", "amount": 50.0},
        {"cpt_code": "ZZZZZ", "amount": 50.0},
    ])
    items = report["audited_items"]
    assert items[0]["description"] == "Partner's own wording"
    assert "influenza" in items[1]["description"].lower()
    assert items[2]["description"] == "no description available"


def test_unknown_code_bypasses_benchmark_not_dropped():
    auditor = BillingAuditor(provider_name="Clinic")
    report = auditor.audit_bill([
        {"cpt_code": "0000X", "amount": 500.00, "description": "Unlisted proprietary"},
    ])
    assert len(report["audited_items"]) == 1
    assert "bypassed price benchmarking" in report["audited_items"][0]["notes"]


def test_billing_auditor_overpricing_only():
    items = [
        {"cpt_code": "99214", "amount": 900.00, "description": "Office Visit"}
    ]
    auditor = BillingAuditor(provider_name="Example Health System")
    report = auditor.audit_bill(items)

    assert report["risk_level"] == "Medium"  # Overpricing only
    # CPT 99214 fair_max is 450.0. Savings should be 900.0 - 450.0 = 450.0
    assert report["suggested_savings"] == 450.0
    
    findings_types = [f["type"] for f in report["findings"]]
    assert "Overpricing" in findings_types


def test_billing_auditor_clear():
    items = [
        {"cpt_code": "99282", "amount": 250.00, "description": "Low Severity ED Visit"}
    ]
    auditor = BillingAuditor(provider_name="Community Clinic")
    report = auditor.audit_bill(items)

    assert report["risk_level"] == "Low"
    assert report["suggested_savings"] == 0.0
    assert len(report["findings"]) == 0
    assert report["dispute_letter"] == ""


def test_llm_bill_parser_regex_fallback():
    parser = LLMBillParser(api_key=None)  # Force fallback
    text = """
    Bill from Example Health System hospital:
    Urgent Care visit Level 4 (99284): $1,500.00
    Minor stitches (12001): $450.00
    """
    data = parser.parse_bill_text(text)
    
    assert "Example Health System" in data["provider_name"]
    assert len(data["items"]) == 2
    
    cpts = [item["cpt_code"] for item in data["items"]]
    amounts = [item["amount"] for item in data["items"]]
    
    assert "99284" in cpts
    assert "12001" in cpts
    assert 1500.0 in amounts
    assert 450.0 in amounts


def test_llm_bill_parser_pdf_extraction(tmp_path):
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj <</Type/Catalog/Pages 2 0 R>> endobj\n"
        b"2 0 obj <</Type/Pages/Kids[3 0 R]/Count 1>> endobj\n"
        b"3 0 obj <</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>> endobj\n"
        b"4 0 obj <</Type/Font/Subtype/Type1/BaseFont/Helvetica>> endobj\n"
        b"5 0 obj <</Length 120>> stream\n"
        b"BT /F1 12 Tf 72 712 Td (Example Health System Hospital Bill Details:) Tj\n"
        b"0 -20 Td (ED Proc Minor CPT 56420: $709.00) Tj\n"
        b"0 -20 Td (ED Proc Level 5 W/Proc - 99285: $6672.00) Tj ET\n"
        b"endstream\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f\n0000000009 00000 n\n0000000056 00000 n\n0000000111 00000 n\n0000000250 00000 n\n0000000318 00000 n\n"
        b"trailer <</Size 6/Root 1 0 R>>\n"
        b"startxref\n488\n%%EOF\n"
    )
    
    pdf_file = tmp_path / "test_bill.pdf"
    pdf_file.write_bytes(pdf_content)
    
    parser = LLMBillParser()
    extracted_text = parser.extract_text_from_pdf(str(pdf_file))
    
    assert "Example Health System" in extracted_text
    assert "56420" in extracted_text
    assert "99285" in extracted_text
