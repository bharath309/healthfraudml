"""
Unit tests for CPT Chroma DB integration.
"""

import pytest
import os
import shutil
from healthfraudml.auditor import CPTDatabase, BillingAuditor


@pytest.fixture
def temp_db(tmpdir):
    """Fixture to create a temporary Chroma DB instance."""
    db_dir = os.path.join(tmpdir, "test_chroma_db")
    db = CPTDatabase(persist_dir=db_dir)
    yield db
    # Clean up
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)


def test_db_populate(temp_db):
    """Verify that the database populates with CPT rules on startup."""
    count = temp_db.collection.count()
    assert count > 0
    assert count == len(temp_db.DEFAULT_RULES)


def test_db_get_rule(temp_db):
    """Test direct metadata lookup by CPT code."""
    # Exist
    rule = temp_db.get_rule("99285")
    assert rule is not None
    assert rule["cpt_code"] == "99285"
    assert rule["severity"] == 5
    assert rule["fair_max"] == 3500.0

    # Non-exist
    non_exist = temp_db.get_rule("99999")
    assert non_exist is None


def test_db_query_description(temp_db):
    """Test description search and semantic matching."""
    # Search Bartholin cyst drainage
    results = temp_db.query_by_description("Bartholin Cyst incision and drainage", n_results=1)
    assert len(results) == 1
    assert results[0]["cpt_code"] == "56420"
    assert "Bartholin" in results[0]["description"]

    # Search level 5 ER visit
    results2 = temp_db.query_by_description("high complexity emergency department visit level 5", n_results=1)
    assert len(results2) == 1
    assert results2[0]["cpt_code"] == "99285"


def test_billing_auditor_with_db(temp_db):
    """Test that BillingAuditor can query and resolve codes dynamically from the DB."""
    # Billed item with missing CPT code, only description
    items = [
        {"cpt_code": "99285", "amount": 6500.00, "description": "ED Visit Level 5"},
        {"cpt_code": "", "amount": 900.00, "description": "drainage of Bartholin gland abscess"},
    ]

    # Initialize auditor with temp db
    auditor = BillingAuditor(provider_name="Test Clinic", db=temp_db)
    report = auditor.audit_bill(items)

    assert report["provider_name"] == "Test Clinic"
    assert report["total_billed"] == 7400.00
    # CPT 56420 has fair_max = 1200.00. Billed was 900.00 (not overpriced).
    # CPT 99285 has fair_max = 3500.00. Billed was 6500.00 (overpriced).
    # Expected overprice savings on CPT 99285 = 6500 - 3500 = 3000
    # Expected upcoding savings downcoded to CPT 99283 (fair max 1200) = 6500 - 1200 = 5300
    # Max savings = 5300.00
    assert report["suggested_savings"] == 5300.00
    
    # Verify semantic code resolution updated the empty CPT item to 56420
    resolved_item = report["audited_items"][1]
    assert resolved_item["cpt_code"] == "56420"
    assert resolved_item["description"] == "drainage of Bartholin gland abscess"
