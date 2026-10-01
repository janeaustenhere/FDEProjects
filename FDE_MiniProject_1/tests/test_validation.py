import pandas as pd
import pytest

from src.validators import InputValidationError, validate_dataframe


def test_validation_retains_blank_comments_but_rejects_duplicate_ids():
    frame = pd.DataFrame([
        {"return_id": "R1", "sku_id": "S1", "category": "Top", "return_reason": "Other", "return_comment": ""},
        {"return_id": "R2", "sku_id": "S2", "category": "Dress", "return_reason": "Other", "return_comment": "tight"},
        {"return_id": "R2", "sku_id": "S3", "category": "Kurti", "return_reason": "Other", "return_comment": "loose"},
    ])
    records, report = validate_dataframe(frame)
    assert [record.return_id for record in records] == ["R1"]
    assert report.blank_comments == 1
    assert report.duplicate_ids == 2


def test_missing_column_is_visible():
    with pytest.raises(InputValidationError, match="return_comment"):
        validate_dataframe(pd.DataFrame({"return_id": ["R1"]}))


def test_vendor_is_preserved_or_safely_defaulted():
    base = {"return_id": "R1", "sku_id": "S1", "category": "Top", "return_reason": "Other", "return_comment": "tight"}
    with_vendor, _ = validate_dataframe(pd.DataFrame([{**base, "vendor": "Urban Loom"}]))
    without_vendor, _ = validate_dataframe(pd.DataFrame([base]))
    assert with_vendor[0].vendor == "Urban Loom"
    assert without_vendor[0].vendor == "Unknown Vendor"
