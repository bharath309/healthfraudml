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
    auditor = BillingAuditor(provider_name="Sutter Health")
    report = auditor.audit_bill(sample_bill_items)

    assert report["provider_name"] == "Sutter Health"
    assert report["total_billed"] == 7381.00
    assert report["risk_level"] == "High"
    
    # Savings should be 6672.0 - 1200.0 (fair_max for CPT 99283) = 5472.0
    assert report["suggested_savings"] == 5472.0
    
    findings_types = [f["type"] for f in report["findings"]]
    assert "Upcoding" in findings_types
    assert "Unbundling" in findings_types
    assert report["dispute_letter"] != ""


def test_billing_auditor_overpricing_only():
    items = [
        {"cpt_code": "99214", "amount": 900.00, "description": "Office Visit"}
    ]
    auditor = BillingAuditor(provider_name="Sutter Health")
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
    Bill from UC Health hospital:
    Urgent Care visit Level 4 (99284): $1,500.00
    Minor stitches (12001): $450.00
    """
    data = parser.parse_bill_text(text)
    
    assert "UC Health" in data["provider_name"]
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
        b"BT /F1 12 Tf 72 712 Td (Sutter Health Hospital Bill Details:) Tj\n"
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
    
    assert "Sutter Health" in extracted_text
    assert "56420" in extracted_text
    assert "99285" in extracted_text
