"""
LLM and PDF Integration for HealthFraudML Billing Auditor.

Extracts structured billing data from PDF files or unstructured email text
using pypdf and the Gemini LLM API (with a robust regex fallback).
"""

import json
import re
import os
from typing import List, Dict, Any, Optional

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class LLMBillParser:
    """
    Parses bills from unstructured sources (PDF text or raw emails)
    and extracts structured billing items using Gemini or a regex fallback.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = None
        if GEMINI_AVAILABLE and self.api_key:
            try:
                # Initialize GenAI Client
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all raw text from a PDF file."""
        if not PYPDF_AVAILABLE:
            raise ImportError(
                "pypdf package is not installed. Please run "
                "pip install pypdf"
            )

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts)

    def parse_bill_text(self, text: str) -> Dict[str, Any]:
        """
        Parse unstructured text (from email or PDF) to extract billing items.

        Returns
        -------
        dict
            Contains 'provider_name' and a list of 'items',
            where each item is a dict with CPT, description, and amount.
        """
        # If Gemini client is configured, use LLM extraction
        if self.client:
            try:
                return self._parse_with_gemini(text)
            except Exception as e:
                # Fallback to regex on LLM failure
                pass

        # Fallback regex-based extraction
        return self._parse_with_regex(text)

    def _parse_with_gemini(self, text: str) -> Dict[str, Any]:
        """Use Gemini 2.5 Flash to extract structured billing data."""
        prompt = f"""
Analyze the following unstructured medical bill text (could be from an email, scan, or PDF) and extract the billing line items and the healthcare provider/hospital name.

Bill Text:
---
{text}
---

Return your response strictly as a JSON object with the following schema:
{{
  "provider_name": "string (the hospital or clinic name, e.g. 'Example Health System')",
  "items": [
    {{
      "cpt_code": "string (the 5-digit CPT code if found, e.g., '99285', '56420'. If not explicit, map standard codes like Level 5 ER -> '99285', minor I&D -> '56420')",
      "description": "string (description of the procedure or charge)",
      "amount": float (the billed charge amount, e.g., 6672.0)
    }}
  ]
}}
"""
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        
        # Load and validate JSON response
        data = json.loads(response.text)
        if "items" not in data:
            data["items"] = []
        if "provider_name" not in data:
            data["provider_name"] = "Unknown Provider"
            
        return data

    def _parse_with_regex(self, text: str) -> Dict[str, Any]:
        """Robust regex fallback parser to extract common CPT codes and prices."""
        items = []
        provider_name = "Unknown Provider"

        # Attempt to find provider name
        provider_matches = [
            r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:Health\s+System|Health|Hospital|Medical\s+Center|Clinic)",
        ]
        for pat in provider_matches:
            match = re.search(pat, text)
            if match:
                provider_name = match.group(0)
                break

        # Search line-by-line for CPT codes and associated dollar amounts
        lines = text.split("\n")
        for line in lines:
            # Look for 5-digit numbers (potential CPT codes)
            cpt_match = re.search(r"\b(9928[1-5]|9921[4-5]|56420|1200[1-2])\b", line)
            # Look for dollar amounts (e.g. $6,672, $709, 1200.00)
            amount_match = re.search(r"\$\s*([\d,]+(?:\.\d{2})?)|\b([\d,]+\.\d{2})\b", line)

            if cpt_match and amount_match:
                cpt = cpt_match.group(1)
                amount_str = amount_match.group(1) or amount_match.group(2)
                # Clean amount string
                amount_cleaned = float(amount_str.replace(",", ""))
                
                # Make up a description if none found
                description = "Procedure"
                if "level 5" in line.lower() or "99285" in line:
                    description = "ED Level 5 Visit"
                elif "minor" in line.lower():
                    description = "Minor Procedure"
                elif "bartholin" in line.lower() or "56420" in line:
                    description = "Bartholin Cyst I&D"
                elif "suture" in line.lower() or "wound" in line.lower():
                    description = "Superficial Wound Repair"

                items.append({
                    "cpt_code": cpt,
                    "description": description,
                    "amount": amount_cleaned
                })

        # If no explicit CPT matches were found, try matching descriptions + amounts
        if not items:
            # Check for "$6,672" and "Level 5"
            l5_match = re.search(r"(?i)level\s*5.*?\$\s*([\d,]+(?:\.\d{2})?)", text)
            if l5_match:
                items.append({
                    "cpt_code": "99285",
                    "description": "ED Level 5 Visit",
                    "amount": float(l5_match.group(1).replace(",", ""))
                })
            # Check for minor procedure amount
            minor_match = re.search(r"(?i)minor.*?\$\s*([\d,]+(?:\.\d{2})?)", text)
            if minor_match:
                items.append({
                    "cpt_code": "56420",
                    "description": "ED Proc Minor (Bartholin Cyst)",
                    "amount": float(minor_match.group(1).replace(",", ""))
                })

        return {
            "provider_name": provider_name,
            "items": items
        }


class RAGBillAuditor:
    """
    RAG-based Medical Billing Auditor using local vector database rules
    and Gemini LLM to detect upcoding, unbundling, and price gouging.

    Parameters
    ----------
    db : CPTDatabase
        An initialized CPTDatabase instance for rule retrieval and semantic search.
    api_key : str, optional
        Gemini API key. Falls back to GEMINI_API_KEY env var, then to rule-based audit.
    """

    def __init__(self, db: "Any", api_key: Optional[str] = None):
        self.db = db
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = None
        if GEMINI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None

    def audit_bill(self, items: List[Dict[str, Any]], provider_name: str = "Unknown Provider") -> Dict[str, Any]:
        """
        Audits billing items by retrieving rules from Chroma DB (RAG) and prompting Gemini.
        Falls back to rule-based BillingAuditor on failure or if Gemini is unavailable.
        """
        # Retrieve rules for each billed item from CPTDatabase
        rules_context = []
        for idx, item in enumerate(items):
            cpt = str(item.get("cpt_code", "")).strip()
            desc = item.get("description", "")
            amount = item.get("amount", 0.0)

            # Retrieve rule
            ref = None
            if self.db:
                ref = self.db.get_rule(cpt)
                if not ref and desc:
                    matches = self.db.query_by_description(desc, n_results=1)
                    if matches and matches[0]["similarity_distance"] < 0.5:
                        ref = matches[0]

            if ref:
                rules_context.append({
                    "cpt_code": ref.get("cpt_code"),
                    "description": ref.get("description"),
                    "medicare_min": ref.get("medicare_min"),
                    "medicare_max": ref.get("medicare_max"),
                    "fair_min": ref.get("fair_min"),
                    "fair_max": ref.get("fair_max"),
                    "severity": ref.get("severity"),
                    "billed_amount": amount
                })

        if self.client and rules_context:
            try:
                return self._audit_with_gemini(items, rules_context, provider_name)
            except Exception:
                # Fallback to BillingAuditor
                pass

        # Fallback to standard BillingAuditor
        from healthfraudml.auditor.billing_auditor import BillingAuditor
        auditor = BillingAuditor(provider_name=provider_name, db=self.db)
        return auditor.audit_bill(items)

    def _audit_with_gemini(self, items: List[Dict[str, Any]], rules_context: List[Dict[str, Any]], provider_name: str) -> Dict[str, Any]:
        """Generate audit report and dispute letter using Gemini with RAG rules."""
        prompt = f"""
You are an expert healthcare financial auditor representing a patient.
Analyze the following patient bill and the official CPT pricing rules retrieved from our database.

Provider Name: {provider_name}
Billed Items:
{json.dumps(items, indent=2)}

Retrieved Database CPT Rules & pricing guidelines:
{json.dumps(rules_context, indent=2)}

Audit Guidelines:
1. Price Gouging/Overpricing: Compare billed amount to the "fair_max" parameter of the corresponding CPT rule. If billed amount exceeds "fair_max", flag it as Overpriced.
2. Upcoding: If there is a high-severity E/M visit code (severity >= 4, e.g. CPT 99285 or 99284) billed alongside only low-severity procedures (severity <= 2, e.g. simple suture CPT 12001 or gland drainage CPT 56420), flag it as suspected Upcoding and recommend downcoding to Level 3 (CPT 99283 or 99214).
3. Unbundling: If an E/M visit code (CPT starting with 992) is billed on the same day as a minor procedure (e.g., CPT 56420), flag it as suspected Unbundling/Double billing, as the visit is generally bundled into the procedure charge.

Return your response strictly as a JSON object matching this schema:
{{
  "provider_name": "string",
  "total_billed": float,
  "suggested_savings": float,
  "risk_level": "string ('Low', 'Medium', 'High')",
  "audited_items": [
    {{
      "cpt_code": "string",
      "description": "string",
      "billed_amount": float,
      "status": "string ('Clear', 'Overpriced', 'Potential Upcoding', 'Potential Unbundling')",
      "notes": "string detailing the reason"
    }}
  ],
  "findings": [
    {{
      "type": "string ('Upcoding', 'Unbundling', 'Overpricing')",
      "severity": "string ('High', 'Medium', 'Low')",
      "message": "string"
    }}
  ],
  "dispute_letter": "string (a professional, customized patient dispute letter addressed to the provider disputing the flagged items based on the rules and audit findings)"
}}
"""
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        
        return json.loads(response.text)
