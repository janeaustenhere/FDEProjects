import re


def normalize_comment(comment: str) -> str:
    return re.sub(r"\s+", " ", str(comment)).strip()


def contains_multiple_issues(comment: str) -> bool:
    text = f" {comment.lower()} "
    connectors = (" and ", " aur ", " but ", " lekin ", ",")
    return any(item in text for item in connectors)

