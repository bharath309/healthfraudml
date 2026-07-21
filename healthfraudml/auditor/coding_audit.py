"""
Coding audit — "is the code right for this service?"

Where :class:`BillingAuditor` asks whether the *price* is defensible, this
module asks whether the *code* matches the service the line describes. It
resolves the line's own description against the plain-language code vocabulary
(``code_names.csv``) using local embeddings, then compares the best matches to
the code that was actually billed.

Verdicts
--------
``MATCH``
    The billed code is among the top matches for the described service.
``POSSIBLE MISCODING``
    The description resolves strongly to a *different* code. Surfaced for human
    review — never a coding determination.
``CANNOT VALIDATE``
    The description resolves to nothing in the vocabulary (or is missing). We
    say so rather than guessing.
``NOT VALIDATABLE (E/M)``
    Evaluation & Management level selection (e.g. 99213 vs 99214 vs 99215)
    depends on documentation, not on a one-line description. Never guessed.

Requires the ``[rag]`` extra (``pip install "healthfraudml[rag]"``). The first
run downloads a small local embedding model (~80MB); everything after that is
fully offline. Without the extra the auditor degrades gracefully and reports
that the coding audit is unavailable.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:  # optional dependency, ships in the [rag] extra
    import chromadb

    CHROMADB_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    CHROMADB_AVAILABLE = False

from healthfraudml.auditor.billing_auditor import BillingAuditor

# Verdict constants
MATCH = "MATCH"
POSSIBLE_MISCODING = "POSSIBLE MISCODING"
CANNOT_VALIDATE = "CANNOT VALIDATE"
NOT_VALIDATABLE_EM = "NOT VALIDATABLE (E/M)"

# Evaluation & Management code families whose level cannot be validated from a
# one-line description.
_EM_PREFIXES = ("992", "993")

#: Minimum cosine similarity for a resolution to be considered meaningful.
DEFAULT_SIM_THRESHOLD = 0.35


def _default_index_dir() -> str:
    return os.environ.get(
        "HEALTHFRAUDML_CODING_INDEX",
        os.path.join(os.path.expanduser("~"), ".cache", "healthfraudml", "coding_index"),
    )


def is_em_code(code: str) -> bool:
    """True for Evaluation & Management codes (level not validatable here)."""
    code = str(code).strip()
    return code.startswith(_EM_PREFIXES)


class CodingAuditor:
    """Semantic code-vs-description checker.

    Parameters
    ----------
    index_dir:
        Where the local vector index is persisted. Defaults to
        ``~/.cache/healthfraudml/coding_index`` (override with the
        ``HEALTHFRAUDML_CODING_INDEX`` environment variable).
    sim_threshold:
        Minimum cosine similarity for a match to count as meaningful.
    top_k:
        How many candidate codes to retrieve per description.
    """

    def __init__(self, index_dir: Optional[str] = None,
                 sim_threshold: float = DEFAULT_SIM_THRESHOLD,
                 top_k: int = 5):
        self.sim_threshold = sim_threshold
        self.top_k = top_k
        self.available = CHROMADB_AVAILABLE
        self._collection = None
        if not self.available:
            return
        index_dir = index_dir or _default_index_dir()
        os.makedirs(index_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=index_dir)
        self._collection = client.get_or_create_collection(
            name="code_names", metadata={"hnsw:space": "cosine"}
        )
        # Rebuild when the persisted index does not match the current
        # vocabulary, otherwise a stale cache silently keeps answering with
        # codes that are no longer indexable.
        expected = len(self.indexable_codes())
        if self._collection.count() != expected:
            if self._collection.count():
                client.delete_collection("code_names")
                self._collection = client.get_or_create_collection(
                    name="code_names", metadata={"hnsw:space": "cosine"}
                )
            self._build_index()

    @staticmethod
    def indexable_codes() -> Dict[str, Dict[str, str]]:
        """The vocabulary the matcher is allowed to resolve descriptions against.

        Only official CMS HCPCS Level II descriptions qualify. Project-authored
        paraphrases are deliberately excluded: matching a bill against wording
        this project invented would let an unofficial name drive a coding
        suggestion. Those codes still get their name displayed - they just
        cannot be the answer to "what does this description sound like".
        """
        return {
            code: entry
            for code, entry in BillingAuditor.CODE_NAMES.items()
            if entry.get("source") == "cms_hcpcs_l2"
        }

    def _build_index(self) -> None:
        """Embed the matchable code vocabulary (one-time, then cached)."""
        names = self.indexable_codes()
        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        for code, entry in names.items():
            ids.append(code)
            docs.append(entry["plain_name"])
            metas.append({"code": code, "source": entry.get("source", "")})
        # Chroma caps batch size; add in chunks.
        for start in range(0, len(ids), 2000):
            stop = start + 2000
            self._collection.add(
                ids=ids[start:stop], documents=docs[start:stop], metadatas=metas[start:stop]
            )

    def resolve(self, description: str) -> List[Dict[str, Any]]:
        """Return top-k candidate codes for a free-text description."""
        if not self.available or not description or not description.strip():
            return []
        res = self._collection.query(query_texts=[description], n_results=self.top_k)
        out: List[Dict[str, Any]] = []
        if res and res.get("ids") and res["ids"][0]:
            for i, code in enumerate(res["ids"][0]):
                distance = res["distances"][0][i] if res.get("distances") else 1.0
                out.append({
                    "code": code,
                    "plain_name": res["documents"][0][i] if res.get("documents") else "",
                    "similarity": round(1.0 - float(distance), 4),
                })
        return out

    def audit_line(self, cpt_code: str, description: str) -> Dict[str, Any]:
        """Audit a single billed line: does the code fit the description?"""
        billed = str(cpt_code).strip()
        result: Dict[str, Any] = {
            "cpt_code": billed,
            "billed_name": BillingAuditor.code_name(billed),
            "billed_name_source": BillingAuditor.code_name_source(billed),
            "description": description or "",
            "verdict": CANNOT_VALIDATE,
            "resolved_code": None,
            "resolved_name": None,
            "similarity": None,
            "note": "",
        }

        if not self.available:
            result["note"] = (
                'coding audit not available — install healthfraudml[rag]'
            )
            return result

        if is_em_code(billed):
            result["verdict"] = NOT_VALIDATABLE_EM
            result["note"] = (
                "E/M level cannot be validated from a one-line description; "
                "requires documentation review."
            )
            return result

        if not description or not description.strip():
            result["note"] = "no description provided on this line."
            return result

        # A comparison is only meaningful if the billed code is in the *indexed*
        # vocabulary. Otherwise the description can only ever resolve to some
        # other code, manufacturing a false "possible miscoding". Note this is
        # stricter than "has a name": a code with only a project-authored name
        # is displayable but not matchable.
        if billed not in self.indexable_codes():
            if billed in BillingAuditor.CODE_NAMES:
                result["note"] = (
                    f"billed code {billed} has only an unofficial name on file, "
                    "which is not used for matching, so the description cannot "
                    "be compared against it."
                )
            else:
                result["note"] = (
                    f"no plain-language name on file for billed code {billed}, "
                    "so the description cannot be compared against it."
                )
            return result

        candidates = self.resolve(description)
        strong = [c for c in candidates if c["similarity"] >= self.sim_threshold]
        if not strong:
            result["note"] = (
                "described service did not resolve to any code in the vocabulary."
            )
            return result

        best = strong[0]
        result["similarity"] = best["similarity"]
        if any(c["code"] == billed for c in strong):
            hit = next(c for c in strong if c["code"] == billed)
            result["verdict"] = MATCH
            result["resolved_code"] = billed
            result["resolved_name"] = hit["plain_name"]
            result["similarity"] = hit["similarity"]
            result["note"] = "billed code is among the top matches for the description."
        else:
            result["verdict"] = POSSIBLE_MISCODING
            result["resolved_code"] = best["code"]
            result["resolved_name"] = best["plain_name"]
            result["note"] = (
                f"described service resolves to {best['code']}; "
                f"billed as {billed} — review."
            )
        return result

    def audit_bill(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run :meth:`audit_line` across every line of a bill."""
        return [
            self.audit_line(item.get("cpt_code", ""), item.get("description", ""))
            for item in items
        ]


#: Printed verbatim in reports so the limits travel with the output.
CODING_AUDIT_LIMITS = (
    "This is a prompt to ask questions, not a coding decision. Anything flagged "
    "here is worth raising with the provider or your insurer, not proof of an error."
)


#: Short status shown in the leftmost column of the report.
_STATUS_LABELS = {
    MATCH: "Looks right",
    POSSIBLE_MISCODING: "Ask about it",
    NOT_VALIDATABLE_EM: "Needs notes",
    CANNOT_VALIDATE: "Not checked",
}


def status_label(row: Dict[str, Any]) -> str:
    """Two-word status for the report's status column."""
    return _STATUS_LABELS.get(row.get("verdict"), "Not checked")


def plain_detail(row: Dict[str, Any]) -> str:
    """The one-line explanation shown beside the status."""
    verdict = row.get("verdict")

    if verdict == MATCH:
        return "Matches the service described on the bill."

    if verdict == POSSIBLE_MISCODING:
        other = row.get("resolved_code")
        other_name = row.get("resolved_name") or ""
        target = f"{other} ({other_name})" if other_name else str(other)
        return f"The description sounds like {target}."

    if verdict == NOT_VALIDATABLE_EM:
        return "Visit-level codes depend on the doctor's notes, not the bill."

    # CANNOT VALIDATE covers several distinct situations; say which one.
    note = (row.get("note") or "").lower()
    if "not available" in note:
        return "Needs the optional install (see setup notes)."
    if "unofficial name" in note:
        return ("We only match against official code descriptions, and this code "
                "has none on file yet.")
    if "no plain-language name" in note:
        return "No description on file for this code yet."
    if "no description provided" in note:
        return "This line has no service description on the bill."
    return "The description doesn't match any service we recognise."


def plain_verdict(row: Dict[str, Any]) -> str:
    """Status and explanation as a single sentence, for narrow layouts."""
    return f"{status_label(row)} - {plain_detail(row)}"


def name_sources_legend(rows: List[Dict[str, Any]]) -> List[str]:
    """Explain where the displayed names came from, for the sources in use.

    A reader should never have to ask whether a description is their own
    wording, an official CMS one, or something this project wrote.
    """
    used = {r.get("billed_name_source") for r in rows if r.get("billed_name_source")}
    legend = []
    if "cms_hcpcs_l2" in used:
        legend.append(
            "Names without a marker are official CMS HCPCS Level II descriptions."
        )
    if "authored" in used:
        legend.append(
            "Names marked (unofficial name) were written by this project, not taken "
            "from the official code book."
        )
    return legend


def coverage_explanation(rows: List[Dict[str, Any]]) -> List[str]:
    """Say *why* lines went unchecked, when the reason is the licensing cap.

    Without this the report shows the symptom ("no description on file") and a
    reader reasonably assumes an incomplete database that will fill in over
    time. For numeric CPT codes that is not true: the descriptions are licensed,
    so those lines stay unchecked until a licensed vocabulary is supplied.
    """
    unnamed_cpt = [
        r for r in rows
        if r.get("verdict") == CANNOT_VALIDATE
        and not r.get("billed_name")
        and str(r.get("cpt_code", "")).strip()[:1].isdigit()
    ]
    if not unnamed_cpt:
        return []
    codes = ", ".join(sorted({r["cpt_code"] for r in unnamed_cpt}))
    return [
        f"Codes not checked ({codes}) are CPT codes. Their official descriptions are",
        "licensed by the AMA and are not distributed with this tool, so these lines",
        "cannot be matched unless a licensed vocabulary is supplied locally.",
        "Their prices are still checked - see the findings above.",
    ]


def coverage_summary(rows: List[Dict[str, Any]]) -> str:
    """One line telling the reader how much of the bill could be checked."""
    total = len(rows)
    checked = sum(1 for r in rows if r.get("verdict") in (MATCH, POSSIBLE_MISCODING))
    em = sum(1 for r in rows if r.get("verdict") == NOT_VALIDATABLE_EM)
    unchecked = total - checked - em

    parts = []
    if em:
        parts.append(f"{em} {'needs' if em == 1 else 'need'} the doctor's notes")
    if unchecked:
        parts.append(f"{unchecked} {'has' if unchecked == 1 else 'have'} no description on file")
    detail = f" - {'; '.join(parts)}" if parts else ""
    line_word = "line" if total == 1 else "lines"
    return f"Checked {checked} of {total} {line_word}{detail}."
