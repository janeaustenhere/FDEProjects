from __future__ import annotations

import pandas as pd

from .models import ReturnRecord, ValidationIssue, ValidationReport
from .reason_registry import DEFAULT_VENDOR

REQUIRED_COLUMNS = ["return_id", "sku_id", "category", "return_reason", "return_comment"]


class InputValidationError(ValueError):
    """Raised when an uploaded dataset cannot satisfy the input contract."""

    error_type = "INPUT_VALIDATION_ERROR"

    def __init__(self, message: str, *, column: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.column = column

    def as_dict(self) -> dict[str, str | None]:
        return {"error_type": self.error_type, "message": self.message, "column": self.column}


def validate_dataframe(frame: pd.DataFrame) -> tuple[list[ReturnRecord], ValidationReport]:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise InputValidationError(f"Missing required columns: {', '.join(missing)}")

    selected_columns = REQUIRED_COLUMNS + (["vendor"] if "vendor" in frame.columns else [])
    data = frame[selected_columns].fillna("").astype(str).copy()
    if "vendor" not in data.columns:
        data["vendor"] = DEFAULT_VENDOR
    else:
        data["vendor"] = data["vendor"].str.strip().replace("", DEFAULT_VENDOR)
    duplicate_mask = data["return_id"].duplicated(keep=False) & data["return_id"].str.strip().ne("")
    issues: list[ValidationIssue] = []
    records: list[ReturnRecord] = []

    for index, row in data.iterrows():
        row_number = int(index) + 2 if isinstance(index, int) else None
        return_id = row["return_id"].strip()
        if not return_id:
            issues.append(ValidationIssue(row_number=row_number, issue="Blank return_id"))
            continue
        if duplicate_mask.loc[index]:
            issues.append(ValidationIssue(row_number=row_number, return_id=return_id, issue="Duplicate return_id"))
            continue
        if not row["sku_id"].strip() or not row["category"].strip():
            issues.append(ValidationIssue(row_number=row_number, return_id=return_id, issue="Blank sku_id or category"))
            continue
        records.append(ReturnRecord(**row.to_dict()))

    report = ValidationReport(
        total_rows=len(data),
        valid_rows=len(records),
        blank_comments=int(data["return_comment"].str.strip().eq("").sum()),
        duplicate_ids=int(duplicate_mask.sum()),
        issues=issues,
    )
    return records, report


def read_csv(source) -> pd.DataFrame:
    try:
        return pd.read_csv(source, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise InputValidationError(f"Could not read CSV: {exc}") from exc
