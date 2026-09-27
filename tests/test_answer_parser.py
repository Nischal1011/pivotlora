"""Checks that the released numeric evaluator behaves as documented.

These are the handcrafted cases the reproducibility checklist calls for, plus
the boundary cases that distinguish the GSM8K/SVAMP scorer from a naive
"find a number" parser. No model outputs are re-evaluated here.

Run with ``pytest tests/`` or directly with ``python tests/test_answer_parser.py``.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from activation_collection import response_predictor_mask, response_position_pool  # noqa: E402
from evaluation import (  # noqa: E402
    extract_answer,
    normalize_numeric_answer,
    score_predictions,
)


# ---------------------------------------------------------------------------
# The documented handcrafted cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("#### 42", "42"),
        ("The answer is 1,234", "1234"),
        ("-3.5", "-3.5"),
        ("An invalid response.", ""),
    ],
)
def test_documented_cases(text, expected):
    assert extract_answer(text) == expected


# ---------------------------------------------------------------------------
# Delimiter handling
# ---------------------------------------------------------------------------


def test_uses_the_text_after_the_final_delimiter():
    text = "First attempt:\n#### 7\nOn reflection:\n#### 9"
    assert extract_answer(text) == "9"


def test_takes_the_first_number_after_the_delimiter():
    """Trailing commentary after the answer must not override it."""
    assert extract_answer("#### 18\nwhich is 3 x 6") == "18"


def test_falls_back_to_the_last_number_without_a_delimiter():
    assert extract_answer("She had 5 apples, ate 2, and kept 3") == "3"


def test_delimiter_with_no_following_number_falls_back_to_the_text():
    """A bare '####' is not an answer; the fallback scan still applies."""
    assert extract_answer("The total is 12 ####") == "12"


def test_gsm8k_style_reference_answer():
    reference = (
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
        "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n"
        "#### 72"
    )
    assert extract_answer(reference) == "72"


def test_svamp_style_completion():
    assert extract_answer("Equation: ( 62.0 - 35.0 )\n#### 27") == "27"


# ---------------------------------------------------------------------------
# Number formats: what is and is not supported
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("#### -17", "-17"),
        ("#### 3.25", "3.25"),
        ("#### 1,234,567", "1234567"),
        ("#### 0", "0"),
    ],
)
def test_supported_number_formats(text, expected):
    assert extract_answer(text) == expected


def test_fractions_and_percentages_are_not_converted():
    """Documented limitation: no fraction or percentage arithmetic is done."""
    assert extract_answer("#### 3/4") == "3"      # first number only
    assert extract_answer("#### 50%") == "50"     # the '%' is dropped, not applied


def test_empty_and_whitespace_input():
    assert extract_answer("") == ""
    assert extract_answer("   \n  ") == ""


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("42", "42"),
        ("42.0", "42"),        # integral floats collapse, so 42.0 scores as 42
        ("-0.0", "0"),
        ("1,234", "1234"),
        ("3.50", "3.5"),
        ("", ""),
        ("not a number", "not a number"),  # unparsable input is left alone
    ],
)
def test_normalization(raw, expected):
    assert normalize_numeric_answer(raw) == expected


def test_normalization_does_not_coerce_infinity():
    assert normalize_numeric_answer("inf") == "inf"
    assert normalize_numeric_answer("nan") == "nan"


# ---------------------------------------------------------------------------
# Aggregate scoring
# ---------------------------------------------------------------------------


def test_score_predictions_counts_invalid_as_incorrect():
    generations = ["#### 5", "#### 7", "I cannot answer that."]
    references = ["#### 5", "#### 6", "#### 8"]

    summary = score_predictions(generations, references)

    assert summary.total == 3
    assert summary.correct == 1
    assert summary.invalid == 1
    assert summary.accuracy_pct == pytest.approx(100.0 / 3.0)
    assert summary.parse_rate_pct == pytest.approx(200.0 / 3.0)


def test_score_predictions_reproduces_a_reported_parse_rate():
    """GSM8K LoRA seed 43: 306/1319 correct, 1 invalid, 99.92% parse rate."""
    total, correct, invalid = 1319, 306, 1

    generations = (
        ["#### 1"] * correct
        + ["#### 2"] * (total - correct - invalid)
        + ["no answer"] * invalid
    )
    references = ["#### 1"] * total

    summary = score_predictions(generations, references)

    assert summary.correct == correct
    assert summary.invalid == invalid
    assert round(summary.accuracy_pct, 2) == 23.20
    assert round(summary.parse_rate_pct, 2) == 99.92


def test_score_predictions_rejects_length_mismatch():
    with pytest.raises(ValueError):
        score_predictions(["#### 1"], ["#### 1", "#### 2"])


# ---------------------------------------------------------------------------
# Response-position masking (the causal shift the evaluator depends on)
# ---------------------------------------------------------------------------


def test_response_predictor_mask_shifts_one_position_left():
    """Hidden state t is selected when label t+1 is supervised."""
    import torch

    #                     prompt        completion
    labels = torch.tensor([[-100, -100, 10, 11, 12]])
    mask = response_predictor_mask(labels)

    # Labels 10, 11, 12 sit at positions 2, 3, 4, so their predictors are 1, 2, 3.
    assert mask.tolist() == [[False, True, True, True, False]]


def test_response_predictor_mask_excludes_padding():
    import torch

    labels = torch.tensor([[-100, 10, 11, -100]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])

    mask = response_predictor_mask(labels, attention_mask)
    assert mask.tolist() == [[True, True, False, False]]


def test_response_position_pool_averages_within_the_example():
    """One vector per example, so a long response is not over-weighted."""
    import torch

    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [5.0, 5.0], [99.0, 99.0]]])
    mask = torch.tensor([[True, True, True, False]])

    pooled = response_position_pool(hidden, mask)

    assert pooled.shape == (1, 2)
    assert torch.allclose(pooled, torch.tensor([[3.0, 3.0]]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
