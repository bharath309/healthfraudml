"""
Example showing CPT vector database indexing, description mapping, and RAG auditing.
"""

import os
from healthfraudml.auditor import CPTDatabase, BillingAuditor, RAGBillAuditor


def main():
    print("=" * 60)
    print("INITIALIZING CPT VECTOR DATABASE (Chroma DB)")
    print("=" * 60)

    # Initialize the local persistent CPT database
    db = CPTDatabase()
    
    # Check populated collection size
    count = db.collection.count()
    print(f"CPT Rules Database initialized. Rules indexed: {count}")
    print()

    print("=" * 60)
    print("DEMO 1: Mapping Descriptions to CPT Codes using Semantic Vector Search")
    print("=" * 60)

    # Run queries using loose descriptions to resolve CPT codes
    queries = [
        "emergency department visit with high complexity level 5 care",
        "laceration repair of superficial wound under 2.5 cm on arm",
        "bartholin gland cyst incision and drainage"
    ]

    for q in queries:
        print(f"Loose Description: '{q}'")
        matches = db.query_by_description(q, n_results=1)
        if matches:
            match = matches[0]
            print(f"  Matched Code: CPT {match['cpt_code']}")
            print(f"  Official Desc: {match['description']}")
            print(f"  Fair Max Price: ${match['fair_max']:.2f}")
            print(f"  Cosine Distance: {match['similarity_distance']:.4f}")
        else:
            print("  No match found.")
        print("-" * 40)
    print()

    print("=" * 60)
    print("DEMO 2: Auditing Billed items with Vector DB Rules (Rule-Based)")
    print("=" * 60)

    # Example bill with some items missing CPT codes (only descriptions present)
    # The preprocessor/auditor should resolve descriptions to standard CPT codes and verify prices
    bill_items = [
        # Billed Level 5 ER Visit (Upcoding check candidate)
        {"cpt_code": "99285", "amount": 6500.00, "description": "ED Proc Level 5"},
        # Loose description (should resolve semantically to CPT 56420: Bartholin gland drainage)
        {"cpt_code": "", "amount": 950.00, "description": "incision and drainage of Bartholin cyst"},
    ]

    # Audit using the database
    auditor = BillingAuditor(provider_name="Sutter Health", db=db)
    report = auditor.audit_bill(bill_items)

    print(f"Provider: {report['provider_name']}")
    print(f"Risk Level: {report['risk_level']}")
    print(f"Total Billed: ${report['total_billed']:.2f}")
    print(f"Suggested Savings: ${report['suggested_savings']:.2f}")
    print("\nAudited Items:")
    for item in report["audited_items"]:
        print(f"  CPT {item['cpt_code']} ({item['description']}): ${item['billed_amount']:.2f} | Status: {item['status']}")
        if item["notes"]:
            print(f"    Note: {item['notes']}")

    print()
    print("=" * 60)
    print("DEMO 3: RAG Auditing Using Gemini LLM and local database context")
    print("=" * 60)

    # RAG auditor checks
    rag_auditor = RAGBillAuditor(db=db)
    
    if not rag_auditor.api_key:
        print("GEMINI_API_KEY environment variable not set. RAG auditor will automatically")
        print("fall back to rule-based CPT auditing. Set GEMINI_API_KEY to test LLM RAG audit.")
        print()
    
    rag_report = rag_auditor.audit_bill(bill_items, provider_name="Sutter Health")
    
    print(f"Provider: {rag_report['provider_name']}")
    print(f"Risk Level: {rag_report['risk_level']}")
    print(f"Suggested Savings: ${rag_report['suggested_savings']:.2f}")
    print("\nFindings:")
    for finding in rag_report.get("findings", []):
        print(f"  - [{finding['type']} - {finding['severity']} Severity]: {finding['message']}")

    print("\nGenerated Dispute Letter Snippet:")
    print("-" * 50)
    letter = rag_report.get("dispute_letter", "")
    lines = letter.split("\n")
    # Show first 15 lines of the letter
    print("\n".join(lines[:18]))
    print("...")
    print("-" * 50)


if __name__ == "__main__":
    main()
