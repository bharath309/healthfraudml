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

    #: Placeholder emitted when no provider can be identified with confidence.
    #: Deliberately looks like a blank to fill in, not like a real value.
    PROVIDER_PLACEHOLDER = "[PROVIDER NAME]"

    #: Lines carrying these markers describe the PATIENT, never the facility.
    _PATIENT_HEADER_MARKERS = (
        "name:", "dob", "date of birth", "patient", "mrn", "member",
        "account", "guarantor", "subscriber", "legal name",
    )

    @classmethod
    def _find_provider_name(cls, text: str):
        """Identify the facility, or decline. Returns (name, source).

        Matching is per line - the previous pattern let ``\\s+`` span newlines,
        so a patient name on one line followed by "Hospital" on the next was
        captured as the provider and then addressed in the dispute letter.
        """
        pattern = re.compile(
            r"[A-Z][A-Za-z'&.-]+(?:[ \t]+[A-Z][A-Za-z'&.-]+)*"
            r"[ \t]+(?:Health\s*System|Health|Hospital|Medical\s+Center|Clinic)"
        )
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Skip anything that identifies the patient rather than the facility.
            if any(m in stripped.lower() for m in cls._PATIENT_HEADER_MARKERS):
                continue
            match = pattern.search(stripped)
            if match:
                return match.group(0).strip(), "parsed"
        return cls.PROVIDER_PLACEHOLDER, "not_found"

    #: Lines that summarise the account rather than describe a charge.
    _SUMMARY_MARKERS = (
        "total", "balance", "insurance", "covered", "deductible", "coinsurance",
        "copay", "responsibility", "payment", "adjustment", "billed to",
        "amount due", "statement", "subtotal", "previous", "credit",
    )

    #: A CPT (5 digits) or HCPCS Level II (letter + 4 digits) code. The guards
    #: keep it out of NDC segments like 0409-4276-01, quantities and amounts.
    _CODE_RE = re.compile(r"(?<![\w$.,-])([A-Z]\d{4}|\d{5})(?![\w.,-])")

    #: A dollar amount, with or without the sign.
    _AMOUNT_RE = re.compile(r"\$\s*([\d,]+\.\d{2})|(?<![\d.,-])([\d,]+\.\d{2})(?![\d])")

    #: A credit or payment: the amount is preceded by a minus sign.
    _NEGATIVE_RE = re.compile(r"-\s*\$?\s*[\d,]+\.\d{2}")

    @classmethod
    def _clean_description(cls, line: str, code: str) -> str:
        """The bill's own wording for a line, with code/system/amount removed."""
        text = cls._AMOUNT_RE.sub(" ", line)
        if code:
            text = re.sub(rf"(?<![\w]){re.escape(code)}(?![\w])", " ", text)
        text = re.sub(r"\((?:CPT|HCPCS|CDT|HCPCS Level II)[^)]*\)", " ", text, flags=re.I)
        text = re.sub(r"[\s\-–—:|]+$", "", text.strip())
        text = re.sub(r"^[\s\-–—:|]+", "", text)
        return re.sub(r"\s{2,}", " ", text).strip()

    @staticmethod
    def _drop_subtotals(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove section headers whose amount is the sum of the lines beneath.

        Portal bills print "Emergency Room $7,381.00" above the charges that
        make it up. Counting both triples the total. Detected by arithmetic
        rather than a list of section names, which would not survive a
        different hospital's wording.

        Subtotals nest: a grand total sits above section totals, which sit above
        the charges. Running the pass once cannot see the grand total, because
        its children still include the section totals and the sum overshoots.
        So repeat until stable - innermost sections drop first, then the level
        above them matches what remains.
        """
        while True:
            drop = set()
            for i, item in enumerate(items):
                running = 0.0
                for j in range(i + 1, len(items)):
                    running = round(running + items[j]["amount"], 2)
                    if running > item["amount"] + 0.005:
                        break
                    if abs(running - item["amount"]) < 0.005:
                        drop.add(i)
                        break
                if i in drop:
                    break
            if not drop:
                return items
            items = [it for k, it in enumerate(items) if k not in drop]

    def _parse_with_regex(self, text: str) -> Dict[str, Any]:
        """Extract billed lines from bill text.

        Reads only what is printed. Codes are never inferred from wording - an
        earlier version mapped "minor" to 56420 and "level 5" to 99285, which
        invents findings, since a code carries severity the auditor keys on.
        Lines that cannot be parsed are reported rather than dropped.
        """
        provider_name, provider_name_source = self._find_provider_name(text)

        items: List[Dict[str, Any]] = []
        unparsed: List[str] = []
        lines = [l.strip() for l in text.splitlines()]

        i = 0
        while i < len(lines):
            line = lines[i]
            i += 1
            if not line:
                continue
            low = line.lower()
            if any(m in low for m in self._SUMMARY_MARKERS):
                continue
            # Credits and insurance payments reduce the balance; they are not charges.
            if self._NEGATIVE_RE.search(line):
                continue

            code_m = self._CODE_RE.search(line)
            amt_m = self._AMOUNT_RE.search(line)

            # Amount may sit on the following line (common in portal exports).
            if code_m and not amt_m and i < len(lines):
                nxt = lines[i]
                if nxt and not any(m in nxt.lower() for m in self._SUMMARY_MARKERS):
                    nxt_amt = self._AMOUNT_RE.search(nxt)
                    if nxt_amt and not self._CODE_RE.search(nxt):
                        amt_m = nxt_amt
                        i += 1

            if not amt_m:
                if code_m:
                    unparsed.append(line)
                continue

            amount = float((amt_m.group(1) or amt_m.group(2)).replace(",", ""))
            code = code_m.group(1) if code_m else ""
            items.append({
                "cpt_code": code,
                "description": self._clean_description(line, code),
                "amount": amount,
            })

        items = self._drop_subtotals(items)

        return {
            "provider_name": provider_name,
            "provider_name_source": provider_name_source,
            "unparsed_lines": unparsed,
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
