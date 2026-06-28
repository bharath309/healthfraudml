"""
CPT Reference Database wrapper using Chroma DB.
"""

import os
from typing import Dict, Any, List, Optional

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class CPTDatabase:
    """
    CPT Reference Database using Chroma DB vector storage.
    Enables metadata lookups by CPT code and vector searches by description.
    """

    DEFAULT_RULES = {
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

    def __init__(self, persist_dir: Optional[str] = None):
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb package is not installed. Please run "
                "pip install chromadb"
            )

        if not persist_dir:
            # Default to data/chroma_db under framework folder
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            persist_dir = os.path.join(base_dir, "data", "chroma_db")

        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="cpt_rules",
            metadata={"hnsw:space": "cosine"}
        )

        # Populate if empty
        if self.collection.count() == 0:
            self._populate_database()

    def _populate_database(self):
        """Populate Chroma collection with default CPT reference rules."""
        ids = []
        documents = []
        metadatas = []

        for code, details in self.DEFAULT_RULES.items():
            doc = (
                f"CPT {code}: {details['description']}. "
                f"Medicare pricing ranges from ${details['medicare_min']} to ${details['medicare_max']}. "
                f"Fair market pricing ranges from ${details['fair_min']} to ${details['fair_max']}. "
                f"Severity level is {details['severity']}."
            )
            ids.append(code)
            documents.append(doc)
            metadatas.append({
                "cpt_code": code,
                "description": details["description"],
                "medicare_min": details["medicare_min"],
                "medicare_max": details["medicare_max"],
                "fair_min": details["fair_min"],
                "fair_max": details["fair_max"],
                "severity": details["severity"]
            })

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def get_rule(self, cpt_code: str) -> Optional[Dict[str, Any]]:
        """
        Query metadata directly for a specific CPT code.
        """
        cpt_code = str(cpt_code).strip()
        result = self.collection.get(ids=[cpt_code])
        if result and result["metadatas"]:
            return result["metadatas"][0]
        return None

    def query_by_description(self, query: str, n_results: int = 1) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search on descriptions to resolve loose text to CPT rules.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        output = []
        if results and results["metadatas"] and results["metadatas"][0]:
            for i in range(len(results["metadatas"][0])):
                meta = results["metadatas"][0][i]
                dist = results["distances"][0][i] if "distances" in results else 0.0
                doc = results["documents"][0][i] if "documents" in results else ""
                
                # Combine
                meta_copy = dict(meta)
                meta_copy["similarity_distance"] = dist
                meta_copy["document_summary"] = doc
                output.append(meta_copy)
        return output
