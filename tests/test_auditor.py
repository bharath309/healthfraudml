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

    # Savings = 6672.00 - 341.25 (CMS-derived review ceiling for 99283).
    # Was 5472.00 while hand-written prices overrode CMS; see
    # docs/medicare_benchmark_design.md.
    assert report["suggested_savings"] == 6330.75

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
    assert report["suggested_savings"] == 6330.75


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
    item = report["audited_items"][0]
    assert len(report["audited_items"]) == 1
    assert "not benchmarked" in item["notes"]
    assert item["status"] == "Not price-checked"


def test_billing_auditor_overpricing_only():
    items = [
        {"cpt_code": "99214", "amount": 900.00, "description": "Office Visit"}
    ]
    auditor = BillingAuditor(provider_name="Example Health System")
    report = auditor.audit_bill(items)

    assert report["risk_level"] == "Medium"  # Overpricing only
    # 99214 review ceiling is 5 x CMS 125.18 = 625.90; 900.00 - 625.90 = 274.10
    assert report["suggested_savings"] == 274.10
    
    findings_types = [f["type"] for f in report["findings"]]
    assert "Overpricing" in findings_types


def test_billing_auditor_clear():
    items = [
        # 99282 review ceiling is 5 x CMS 40.43 = 202.15, so stay under it.
        {"cpt_code": "99282", "amount": 150.00, "description": "Low Severity ED Visit"}
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


def test_unbenchmarked_line_is_not_reported_as_clear():
    """A line we never price-checked must not read as having passed a check."""
    auditor = BillingAuditor(provider_name="Lab")
    report = auditor.audit_bill([
        {"cpt_code": "80053", "amount": 340.00, "description": "Comprehensive metabolic panel"},
    ])
    item = report["audited_items"][0]
    assert item["status"] == "Not price-checked"
    assert item["status"] != "Clear"
    assert "not benchmarked" in item["notes"]


def test_hcpcs_codes_are_not_called_cpt():
    """J- and G-codes are HCPCS Level II; calling them CPT discredits the letter."""
    assert BillingAuditor.code_system("J1885") == "HCPCS"
    assert BillingAuditor.code_system("G0009") == "HCPCS"
    assert BillingAuditor.code_system("99285") == "CPT"

    auditor = BillingAuditor(provider_name="Clinic")
    report = auditor.audit_bill([
        {"cpt_code": "99285", "amount": 6672.00, "description": "ED Visit Level 5"},
        {"cpt_code": "J1885", "amount": 180.00, "description": "Ketorolac injection"},
    ])
    letter = report["dispute_letter"]
    assert "CPT J1885" not in letter
    assert "HCPCS J1885" in letter


def test_letter_states_code_meaning_not_the_bills_claim():
    """A miscoded line must not have its error repeated back in our letter."""
    auditor = BillingAuditor(provider_name="Clinic")
    letter = auditor.audit_bill([
        {"cpt_code": "G0009", "amount": 90.00,
         "description": "administration of influenza vaccine"},
    ])["dispute_letter"] or ""
    # No findings on this bill, so no letter is generated; assert via the item.
    report = auditor.audit_bill([
        {"cpt_code": "G0009", "amount": 90.00,
         "description": "administration of influenza vaccine"},
        {"cpt_code": "70450", "amount": 2400.00, "description": "CT head"},
    ])
    letter = report["dispute_letter"]
    # The code's actual meaning is stated...
    assert "Administration of pneumococcal vaccine" in letter
    # ...and the bill's differing wording is quoted, not asserted as the meaning.
    assert 'described on the bill as: "administration of influenza vaccine"' in letter


def test_audited_item_carries_code_meaning_separate_from_bill_wording():
    """Two distinct facts per line: what the code means, what the bill called it."""
    auditor = BillingAuditor(provider_name="Clinic")
    item = auditor.audit_bill([
        {"cpt_code": "G0009", "amount": 90.00,
         "description": "administration of influenza vaccine"},
    ])["audited_items"][0]
    assert item["description"] == "administration of influenza vaccine"   # bill's claim
    assert item["description_source"] == "bill"
    assert item["code_name"] == "Administration of pneumococcal vaccine"  # code's meaning
    assert item["code_name_source"] == "cms_hcpcs_l2"


def test_code_meaning_is_none_when_unknown():
    auditor = BillingAuditor(provider_name="Clinic")
    item = auditor.audit_bill([
        {"cpt_code": "70450", "amount": 2400.00, "description": "CT head"},
    ])["audited_items"][0]
    assert item["code_name"] is None
    assert item["code_name_source"] is None


def test_prices_come_from_cms_not_handwritten():
    """Curated built-ins supply metadata only; CMS supplies every price."""
    for meta in BillingAuditor._BUILTIN_CPT_REFERENCE.values():
        assert "medicare_max" not in meta and "fair_max" not in meta
    ref = BillingAuditor.CPT_REFERENCE["99285"]
    assert ref["medicare_max"] == 168.85          # CMS 2025: 5.22 RVU x 32.3465
    assert ref["fair_max"] == 844.25              # 5x CMS
    assert ref["severity"] == 5                   # metadata still applied


def test_diagnostics_do_not_erase_an_upcoding_flag():
    """Regression: adding an X-ray must not suppress a real upcoding finding."""
    auditor = BillingAuditor(provider_name="Hospital")
    base = [
        {"cpt_code": "99285", "amount": 6672.00, "description": "ED Visit Level 5"},
        {"cpt_code": "56420", "amount": 709.00, "description": "Bartholin Cyst I&D"},
    ]
    assert "Upcoding" in [f["type"] for f in auditor.audit_bill(base)["findings"]]
    with_xray = base + [{"cpt_code": "71046", "amount": 890.00, "description": "Chest X-ray"}]
    types = [f["type"] for f in auditor.audit_bill(with_xray)["findings"]]
    assert "Upcoding" in types, "unclassified diagnostics must be ignored, not block the check"


def test_coding_audit_reaches_the_letter_as_a_question():
    """A possible miscoding is requested for confirmation, never alleged."""
    auditor = BillingAuditor(provider_name="Clinic")
    coding = [{"cpt_code": "G0009", "verdict": "POSSIBLE MISCODING",
               "description": "administration of influenza vaccine",
               "resolved_code": "G0008"}]
    letter = auditor.audit_bill(
        [{"cpt_code": "G0009", "amount": 90.00,
          "description": "administration of influenza vaccine"},
         {"cpt_code": "70450", "amount": 2400.00, "description": "CT head"}],
        coding_audit=coding,
    )["dispute_letter"]
    assert "Confirmation of the service billed under HCPCS G0009" in letter
    for accusation in ["miscoded", "wrong code", "incorrect code"]:
        assert accusation not in letter.lower()


def test_uncoded_charge_is_reported_never_inferred():
    """A charge with no code becomes a finding; the code is never guessed."""
    auditor = BillingAuditor(provider_name="Hospital")
    report = auditor.audit_bill([
        {"cpt_code": "99285", "amount": 6672.00, "description": "ED Level 5 W/Proc"},
        {"cpt_code": "", "amount": 709.00, "description": "ED Proc Minor"},
    ])
    uncoded = report["audited_items"][1]
    assert uncoded["status"] == "No code disclosed"
    assert uncoded["cpt_code"] == ""          # never filled in
    types = [f["type"] for f in report["findings"]]
    assert "Missing Code" in types
    # An uncoded line cannot support a severity-based finding.
    assert "Upcoding" not in types and "Unbundling" not in types
    # And it contributes no claimed saving - we cannot value what we cannot identify.
    finding = next(f for f in report["findings"] if f["type"] == "Missing Code")
    assert "709.00" in finding["message"] and "Request an itemised bill" in finding["message"]


def test_uncoded_charge_alone_still_raises_risk():
    auditor = BillingAuditor(provider_name="Hospital")
    report = auditor.audit_bill([
        {"cpt_code": "", "amount": 709.00, "description": "ED Proc Minor"},
    ])
    assert report["risk_level"] == "Medium"
    assert "Missing Code" in [f["type"] for f in report["findings"]]
    assert "itemised statement showing the CPT or HCPCS code" in report["dispute_letter"]
