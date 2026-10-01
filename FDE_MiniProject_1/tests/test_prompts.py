from src.config import ROOT


def test_bulk_prompt_guards_critical_regressions():
    prompt = (ROOT / "prompts" / "bulk_classifier.txt").read_text().lower()
    assert '"acha nahi laga"' in prompt and "must not automatically become quality" in prompt
    assert '"fit good hai but colour different hai"' in prompt and "product_mismatch / colour_mismatch" in prompt
    assert "size_information" in prompt
    assert "uncertain" in prompt
    assert "classify every record independently" in prompt
    assert "exactly one item per input record" in prompt


def test_evaluator_prompt_defines_batch_isolation_and_coverage():
    prompt = (ROOT / "prompts" / "evaluator.txt").read_text().lower()
    assert "evaluate every candidate independently" in prompt
    assert "exactly one item per candidate" in prompt
