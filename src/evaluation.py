"""
Numeric-answer evaluation for the GSM8K reasoning experiment.

``train_reasoning_gsm8k.py`` contains byte-identical copies of
``NUMBER_PATTERN``, ``extract_answer``, and ``normalize_numeric_answer``. This
module is the single documented copy, exercised by
``tests/test_answer_parser.py``. StrategyQA uses a yes/no scorer defined in
``train_reasoning_strategyqa.py``.

Procedure (Appendix "Generation and Evaluation")
------------------------------------------------
1. Take the text after the **final** ``####`` delimiter and return the
   **first** number found there.
2. If there is no ``####``, return the **last** number anywhere in the text.
3. Normalize prediction and reference to a canonical numeric string and
   compare by exact equality.

Supported: signed integers, comma-grouped numbers, and decimals. **Not**
converted: fractions and percentages. An output with no valid numeric value is
counted as incorrect and contributes to the reported "invalid" count, from
which the parse rate is derived.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "NUMBER_PATTERN",
    "extract_answer",
    "normalize_numeric_answer",
    "score_predictions",
    "ScoreSummary",
]


# Signed integer, optional comma groups, optional decimal part.
NUMBER_PATTERN = re.compile(r"-?\d+(?:,\d+)*(?:\.\d+)?")


def extract_answer(text: str) -> str:
    """Pull the numeric answer out of a reference or generated string.

    Returns the empty string when no number is present, which the caller
    treats as an unparsable (and therefore incorrect) response.

    >>> extract_answer("#### 42")
    '42'
    >>> extract_answer("The answer is 1,234")
    '1234'
    >>> extract_answer("-3.5")
    '-3.5'
    >>> extract_answer("An invalid response.")
    ''
    """
    if "####" in text:
        answer_portion = text.split("####")[-1].strip()
        numbers = NUMBER_PATTERN.findall(answer_portion)
        if numbers:
            return numbers[0].replace(",", "")

    numbers = NUMBER_PATTERN.findall(text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def normalize_numeric_answer(answer: str) -> str:
    """Canonicalize a numeric string so that equal values compare equal.

    Commas are stripped, integral floats collapse to their integer spelling
    (``"42.0"`` -> ``"42"``, so a model that writes ``42.0`` is not penalized),
    and non-integral values are rendered with ``%.15g``. Non-finite values and
    strings that do not parse as a float are returned unchanged rather than
    coerced, so a malformed answer stays malformed instead of silently
    becoming ``"0"``.
    """
    answer = str(answer).strip().replace(",", "")

    if not answer:
        return ""

    try:
        numeric_value = float(answer)
    except ValueError:
        return answer

    if not math.isfinite(numeric_value):
        return answer

    if numeric_value.is_integer():
        return str(int(numeric_value))

    return format(numeric_value, ".15g")


@dataclass
class ScoreSummary:
    """Aggregate scorer output matching the columns in ``results/reasoning.csv``."""

    total: int
    correct: int
    invalid: int

    @property
    def accuracy(self) -> float:
        """Fraction correct over *all* examples; invalid outputs count as wrong."""
        return self.correct / self.total if self.total else 0.0

    @property
    def accuracy_pct(self) -> float:
        return 100.0 * self.accuracy

    @property
    def parse_rate_pct(self) -> float:
        """Percentage of generations that yielded any numeric value."""
        if not self.total:
            return 0.0
        return 100.0 * (self.total - self.invalid) / self.total


def score_predictions(
    generations: Sequence[str],
    references: Sequence[str],
    references_are_raw: bool = True,
) -> ScoreSummary:
    """Score generated completions against reference answers.

    ``references_are_raw`` should stay ``True`` when the references are raw
    dataset answer fields (e.g. a GSM8K rationale ending in ``#### 18``); set
    it to ``False`` if the references have already been reduced to bare
    numeric strings.
    """
    if len(generations) != len(references):
        raise ValueError(
            f"length mismatch: {len(generations)} generations vs {len(references)} references"
        )

    correct = 0
    invalid = 0

    for generation, reference in zip(generations, references):
        predicted = normalize_numeric_answer(extract_answer(generation))

        if references_are_raw:
            target = normalize_numeric_answer(extract_answer(reference))
        else:
            target = normalize_numeric_answer(reference)

        if not predicted:
            invalid += 1
            continue

        if predicted == target:
            correct += 1

    return ScoreSummary(total=len(generations), correct=correct, invalid=invalid)
