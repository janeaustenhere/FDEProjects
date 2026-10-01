from src.reason_registry import business_category_for, is_other_reason, structured_reason_mapping
from src.taxonomy import PrimaryReason, SubReason


def test_reason_aliases_and_structured_mapping_are_centralized():
    assert is_other_reason(" Other reason ")
    assert structured_reason_mapping("Defective Pieces") == (
        PrimaryReason.DAMAGED_OR_DEFECTIVE,
        SubReason.DEFECT_OTHER,
    )


def test_business_category_uses_discovered_label_for_new_reason():
    assert business_category_for(PrimaryReason.OTHER_KNOWN, discovered_category="Color Bleeding") == "Color Bleeding"
