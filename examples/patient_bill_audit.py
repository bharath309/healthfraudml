"""
Example showing how to use the Patient Billing Auditor.

Demonstrates parsing unstructured text (e.g. from an email) and a PDF bill,
running the audit, flagging upcoding and unbundling, and generating a dispute letter.
"""

import os
from healthfraudml.auditor import BillingAuditor, LLMBillParser


def create_dummy_pdf(filename: str):
    """Write a minimal valid PDF containing bill text for demonstration."""
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
    with open(filename, "wb") as f:
        f.write(pdf_content)


def main():
    # 1. Unstructured Email Text Example
    email_text = """
    From: billing@sutterhealth.org
    To: patient@example.com
    Subject: Your Sutter Health Medical Bill #12345
    
    Dear Patient,
    Here is a summary of the services provided during your Emergency Department
    visit on 2026-06-25 for a Bartholin Cyst Treatment:
    
    - ED Proc Minor (CPT 56420): $709.00
    - ED Proc Level 5 W/Proc (CPT 99285): $6,672.00
    
    Total Balance Due: $7,381.00
    """

    print("=" * 60)
    print("DEMO 1: Auditing Unstructured Email Text")
    print("=" * 60)

    # Initialize the LLM Bill Parser (will fallback to regex if GEMINI_API_KEY is not set)
    parser = LLMBillParser()
    extracted_data = parser.parse_bill_text(email_text)

    print(f"Extracted Provider: {extracted_data['provider_name']}")
    print("Extracted Items:")
    for item in extracted_data["items"]:
        print(f"  - CPT {item['cpt_code']} ({item['description']}): ${item['amount']:.2f}")
    print()

    # Initialize the Billing Auditor with the extracted provider
    auditor = BillingAuditor(provider_name=extracted_data["provider_name"])
    report = auditor.audit_bill(extracted_data["items"])

    print(f"Audit Risk Level: {report['risk_level']}")
    print(f"Total Billed: ${report['total_billed']:.2f}")
    print(f"Suggested Audit Savings: ${report['suggested_savings']:.2f}")
    print("\nFindings:")
    for i, finding in enumerate(report["findings"], 1):
        print(f"  {i}. [{finding['type']} - {finding['severity']} Severity]: {finding['message']}")

    print("\n" + "=" * 60)
    print("DEMO 2: Auditing PDF Bill Document")
    print("=" * 60)

    pdf_filename = "sample_bill.pdf"
    create_dummy_pdf(pdf_filename)

    try:
        # Extract text from the generated dummy PDF
        pdf_text = parser.extract_text_from_pdf(pdf_filename)
        print("Raw text extracted from PDF:")
        print(pdf_text.strip())
        print()

        # Parse extracted PDF text
        pdf_extracted = parser.parse_bill_text(pdf_text)
        print(f"Extracted Provider from PDF: {pdf_extracted['provider_name']}")
        print("Extracted Items from PDF:")
        for item in pdf_extracted["items"]:
            print(f"  - CPT {item['cpt_code']} ({item['description']}): ${item['amount']:.2f}")
        
        # Run audit on PDF items
        pdf_auditor = BillingAuditor(provider_name=pdf_extracted["provider_name"])
        pdf_report = pdf_auditor.audit_bill(pdf_extracted["items"])
        
        print(f"\nPDF Audit Risk Level: {pdf_report['risk_level']}")
        print(f"Suggested Savings on PDF Bill: ${pdf_report['suggested_savings']:.2f}")
        
        if pdf_report["dispute_letter"]:
            print("\nGenerated Dispute Letter for Patient:")
            print("-" * 50)
            print(pdf_report["dispute_letter"])
            print("-" * 50)

    finally:
        # Clean up sample PDF file
        if os.path.exists(pdf_filename):
            os.remove(pdf_filename)


if __name__ == "__main__":
    main()
