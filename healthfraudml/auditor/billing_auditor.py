"""
Patient Billing Auditor for HealthFraudML.

Analyzes patient bills for upcoding, unbundling, and price gouging.
Provides human-readable explanations and generates a dispute letter.
"""

from typing import List, Dict, Any, Optional


class BillingAuditor:
    """
    Audits a list of billed line items for potential billing anomalies.
    """

    # Reference Database of common CPT codes and pricing benchmarks
    # Columns: CPT code -> {description, medicare_min, medicare_max, fair_min, fair_max, severity}
    # Severity ranges from 1 (minor) to 5 (critical/severe)
    CPT_REFERENCE = {
        "99285": {
            "description": "Emergency Department Visit, Level 5 (High Severity/Complexity)",
            "medicare_min": 150.0,
            "medicare_max": 250.0,
            "fair_min": 1500.0,
            "fair_max": 3500.0,
            "severity": 5,
        },
        "99284": {
            "description": "Emergency Department Visit, Level 4 (High/Moderate Severity)",
            "medicare_min": 120.0,
            "medicare_max": 180.0,
            "fair_min": 1000.0,
            "fair_max": 2200.0,
            "severity": 4,
        },
        "99283": {
            "description": "Emergency Department Visit, Level 3 (Moderate Severity)",
            "medicare_min": 80.0,
            "medicare_max": 120.0,
            "fair_min": 500.0,
            "fair_max": 1200.0,
            "severity": 3,
        },
        "99282": {
            "description": "Emergency Department Visit, Level 2 (Low/Moderate Severity)",
            "medicare_min": 50.0,
            "medicare_max": 80.0,
            "fair_min": 300.0,
            "fair_max": 700.0,
            "severity": 2,
        },
        "99281": {
            "description": "Emergency Department Visit, Level 1 (Low Severity)",
            "medicare_min": 30.0,
            "medicare_max": 50.0,
            "fair_min": 150.0,
            "fair_max": 400.0,
            "severity": 1,
        },
        "56420": {
            "description": "Incision and Drainage of Bartholin's Gland Abscess/Cyst",
            "medicare_min": 115.0,
            "medicare_max": 200.0,
            "fair_min": 400.0,
            "fair_max": 1200.0,
            "severity": 2,
        },
        "12001": {
            "description": "Simple Repair of Superficial Wound (2.5 cm or less)",
            "medicare_min": 80.0,
            "medicare_max": 130.0,
            "fair_min": 300.0,
            "fair_max": 800.0,
            "severity": 1,
        },
        "12002": {
            "description": "Simple Repair of Superficial Wound (2.6 cm to 7.5 cm)",
            "medicare_min": 100.0,
            "medicare_max": 160.0,
            "fair_min": 400.0,
            "fair_max": 1000.0,
            "severity": 2,
        },
        "99214": {
            "description": "Office/Outpatient Visit, Established Patient, 30-39 minutes",
            "medicare_min": 100.0,
            "medicare_max": 140.0,
            "fair_min": 200.0,
            "fair_max": 450.0,
            "severity": 3,
        },
        "99215": {
            "description": "Office/Outpatient Visit, Established Patient, 40-54 minutes",
            "medicare_min": 150.0,
            "medicare_max": 200.0,
            "fair_min": 300.0,
            "fair_max": 600.0,
            "severity": 4,
        },
    }

    def __init__(self, provider_name: str = "Unknown Provider", db: Optional[Any] = None):
        self.provider_name = provider_name
        self.db = db

    def audit_bill(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Audit a list of billed items.

        Parameters
        ----------
        items : list of dict
            Each dict should have:
            - 'cpt_code': str (e.g. '99285')
            - 'amount': float (the billed charge, e.g. 6672.0)
            - 'description': str, optional

        Returns
        -------
        dict
            Contains audited items, detected flags, overall risk,
            explanations, and a dispute letter.
        """
        if not items:
            return {
                "provider_name": self.provider_name,
                "total_billed": 0.0,
                "suggested_savings": 0.0,
                "risk_level": "Low",
                "audited_items": [],
                "findings": [],
                "dispute_letter": "",
            }

        audited_items = []
        findings = []
        has_upcoding = False
        has_unbundling = False
        has_overpricing = False
        total_billed = sum(float(item.get("amount", 0.0)) for item in items)
        
        # Track savings per index to prevent double counting
        item_savings = [0.0] * len(items)

        # Separate E/M visits and procedures
        em_items = []
        procedure_items = []

        for idx, item in enumerate(items):
            cpt = str(item.get("cpt_code", "")).strip()
            amount = float(item.get("amount", 0.0))
            desc = item.get("description", "")

            ref = None
            # Try database lookup
            if self.db:
                db_rule = self.db.get_rule(cpt)
                if db_rule:
                    ref = db_rule
                elif desc:
                    # Semantic search to resolve unknown/missing code to CPT
                    matches = self.db.query_by_description(desc, n_results=1)
                    if matches and matches[0]["similarity_distance"] < 0.5:
                        ref = matches[0]
                        cpt = ref["cpt_code"]  # Update CPT code to resolved one

            # Fallback to static reference
            if not ref:
                ref = self.CPT_REFERENCE.get(cpt)

            resolved_desc = desc or (ref["description"] if ref else "Unknown Procedure")
            
            audit_entry = {
                "cpt_code": cpt,
                "description": resolved_desc,
                "billed_amount": amount,
                "status": "Clear",
                "notes": "",
                "fair_max_ref": ref["fair_max"] if ref else None,
            }

            if ref:
                # Check for extreme pricing
                if amount > ref["fair_max"]:
                    audit_entry["status"] = "Overpriced"
                    diff = amount - ref["fair_max"]
                    audit_entry["notes"] = f"Billed amount exceeds fair ceiling of ${ref['fair_max']:.2f} by ${diff:.2f}."
                    has_overpricing = True
                    item_savings[idx] = max(item_savings[idx], diff)
                
                # Classify E/M codes (Evaluation and Management visits) vs procedures
                if cpt.startswith("992"):  # E/M codes
                    em_items.append((cpt, amount, ref, idx))
                else:
                    procedure_items.append((cpt, amount, ref, idx))
            else:
                audit_entry["notes"] = "CPT code not in reference database; bypassed price benchmarking."

            audited_items.append(audit_entry)

        # Audit Check: Upcoding (High E/M visit billed alongside only minor procedures)
        for em_cpt, em_amount, em_ref, em_idx in em_items:
            # If it's a high-severity visit (Level 5 or Level 4)
            if em_ref["severity"] >= 4:
                # Check if all other procedures are low severity (<= 2)
                if procedure_items and all(p_ref["severity"] <= 2 for _, _, p_ref, _ in procedure_items):
                    has_upcoding = True
                    # Find a suggested lower E/M code (e.g. Level 3)
                    suggested_cpt = "99283" if em_cpt.startswith("9928") else "99214"
                    suggested_ref = None
                    if self.db:
                        suggested_ref = self.db.get_rule(suggested_cpt)
                    if not suggested_ref:
                        suggested_ref = self.CPT_REFERENCE[suggested_cpt]
                    
                    # Calculate potential savings based on downcoding to fair max of Level 3
                    current_fair_max = em_ref["fair_max"]
                    downcoded_fair_max = suggested_ref["fair_max"]
                    savings = max(em_amount - downcoded_fair_max, 0.0)
                    item_savings[em_idx] = max(item_savings[em_idx], savings)

                    findings.append({
                        "type": "Upcoding",
                        "severity": "High",
                        "message": (
                            f"Potential Upcoding on E/M visit code {em_cpt} (${em_amount:.2f}). "
                            f"The performed procedures are classified as minor/moderate, which does not "
                            f"support a Level {em_ref['severity']} visit. Recommended to downcode to "
                            f"{suggested_cpt} ({suggested_ref['description']}), saving up to ${savings:.2f}."
                        )
                    })
                    
                    # Update status of the E/M code in audited items
                    for item in audited_items:
                        if item["cpt_code"] == em_cpt:
                            item["status"] = "Potential Upcoding"
                            item["notes"] = f"Suspected upcoding. Recommended CPT: {suggested_cpt}."

        # Audit Check: Unbundling (Separate charges for same-day evaluation & minor procedure)
        if em_items and procedure_items:
            has_unbundling = True
            findings.append({
                "type": "Unbundling",
                "severity": "Medium",
                "message": (
                    "Potential Unbundling/Double Billing detected. Evaluation & Management visits (E/M) "
                    "are generally bundled within the global package of minor surgical procedures. "
                    "Charging separately for both the visit and the procedure on the same day is often duplicate billing."
                )
            })
            for em_cpt, _, _, _ in em_items:
                for item in audited_items:
                    if item["cpt_code"] == em_cpt:
                        if item["status"] == "Clear":
                            item["status"] = "Potential Unbundling"
                            item["notes"] = "E/M visit billed same day as procedure. Check bundling rules."

        # Check for standalone overpriced items that weren't caught by upcoding
        for idx, item in enumerate(audited_items):
            if item["status"] == "Overpriced":
                # Only report as overpricing finding if not already handled by upcoding
                if not any(f["type"] == "Upcoding" and item["cpt_code"] in f["message"] for f in findings):
                    findings.append({
                        "type": "Overpricing",
                        "severity": "High",
                        "message": f"CPT {item['cpt_code']} is overpriced at ${item['billed_amount']:.2f}. Fair market value max is ${item['fair_max_ref']:.2f}."
                    })

        suggested_savings = sum(item_savings)

        # Calculate final overall risk level
        if has_upcoding or (has_overpricing and suggested_savings > 1000):
            risk_level = "High"
        elif has_unbundling or has_overpricing:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Generate dispute letter if issues found
        dispute_letter = ""
        if risk_level != "Low":
            dispute_letter = self._generate_dispute_letter(items, audited_items, findings, total_billed, suggested_savings)

        return {
            "provider_name": self.provider_name,
            "total_billed": total_billed,
            "suggested_savings": suggested_savings,
            "risk_level": risk_level,
            "audited_items": audited_items,
            "findings": findings,
            "dispute_letter": dispute_letter,
        }

    def _generate_dispute_letter(
        self,
        original_items: List[Dict[str, Any]],
        audited_items: List[Dict[str, Any]],
        findings: List[Dict[str, Any]],
        total_billed: float,
        suggested_savings: float,
    ) -> str:
        """Helper to generate a formatted dispute letter."""
        findings_bullets = ""
        for idx, f in enumerate(findings, 1):
            findings_bullets += f"  {idx}. [{f['type']} - {f['severity']} Risk]: {f['message']}\n"

        bill_table = ""
        for item in audited_items:
            ref_str = f" (Fair Max: ${item['fair_max_ref']:.2f})" if item["fair_max_ref"] else ""
            bill_table += f"  - CPT {item['cpt_code']}: {item['description']} - ${item['billed_amount']:.2f}{ref_str} [{item['status']}]\n"

        letter = f"""Subject: Formal Billing Dispute & Audit Request - Urgent

To: {self.provider_name} Billing Department / Patients Accounts
From: Patient Advocacy / Insured Patient
Date: [Insert Date]

Dear {self.provider_name} Billing Representative,

I am writing to formally dispute the charges on my recent bill of ${total_billed:.2f} from your facility, {self.provider_name}. 

Following a review of the billed itemized CPT codes using healthcare billing standards, several critical discrepancies have been identified indicating potential billing errors, upcoding, and unbundled charges:

Summary of Audit Findings:
{findings_bullets}
Itemized Bill Breakdown:
{bill_table}
Under standard medical coding rules:
1. Evaluation and Management (E/M) services are generally bundled into the procedure charge when performed on the same day. 
2. Visit levels must match the clinical severity of the condition. Billing a Level 5 visit (CPT 99285) for minor outpatient procedures is considered upcoding and does not comply with National Correct Coding Initiative (NCCI) guidelines.

Based on these findings, I request:
- A formal coding audit of my chart and physician notes to adjust the E/M codes to the correct level of care.
- Correction or removal of duplicate/unbundled same-day fees.
- Adjustment of any overpriced fees exceeding fair market value.

I request that this account be placed on "Dispute Status" and all collection actions suspended while this audit is performed. I look forward to receiving a revised, corrected statement.

Sincerely,

[Your Name]
[Account/Billing Number]
[Contact Information]
"""
        return letter.strip()
