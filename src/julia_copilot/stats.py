"""Compare two benchmark runs with McNemar's test.

The two models are evaluated on the same HumanEval-jl problems, so the pass/fail
outcomes are *paired*. McNemar's test conditions on the discordant pairs -- the
problems exactly one model solved -- which is the right null for this design;
an unpaired proportion test would ignore that pairing and overstate the
evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class McNemarResult:
    """Outcome of a paired comparison between two models."""

    both_passed: int
    only_first: int
    only_second: int
    both_failed: int
    p_value: float

    @property
    def n_discordant(self) -> int:
        """Number of problems the two models disagreed on."""
        return self.only_first + self.only_second

    def render(self, alpha: float = 0.05) -> str:
        verdict = "significant" if self.p_value < alpha else "not significant"
        return "\n".join(
            [
                "McNemar's test (exact)",
                f"  both passed:        {self.both_passed}",
                f"  only model 1:       {self.only_first}",
                f"  only model 2:       {self.only_second}",
                f"  both failed:        {self.both_failed}",
                f"  discordant pairs:   {self.n_discordant}",
                f"  p-value:            {self.p_value:.6g}",
                f"  at alpha={alpha}:    difference is {verdict}",
            ]
        )


def load_passed_results(file_path: Path | str) -> list[bool]:
    """Read the ``passed`` flags out of an extracted results file."""
    data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    return [bool(item["passed"]) for item in data]


def mcnemar_test(first: list[bool], second: list[bool], *, exact: bool = True) -> McNemarResult:
    """Run McNemar's test over two aligned pass/fail vectors.

    Args:
        first: Per-problem outcomes for model 1.
        second: Per-problem outcomes for model 2, in the same problem order.
        exact: Use the exact binomial test rather than the chi-square
            approximation. Exact is the right default: the discordant count is
            usually small on a 161-problem benchmark.

    Raises:
        ValueError: If the two vectors differ in length, which would mean the
            models were not evaluated on the same problems.
    """
    from statsmodels.stats.contingency_tables import mcnemar

    if len(first) != len(second):
        raise ValueError(
            f"Both models must be evaluated on the same problems "
            f"(got {len(first)} and {len(second)} results)."
        )

    pairs = list(zip(first, second, strict=True))
    both_passed = sum(1 for a, b in pairs if a and b)
    only_first = sum(1 for a, b in pairs if a and not b)
    only_second = sum(1 for a, b in pairs if not a and b)
    both_failed = sum(1 for a, b in pairs if not a and not b)

    result = mcnemar([[both_passed, only_first], [only_second, both_failed]], exact=exact)

    return McNemarResult(
        both_passed=both_passed,
        only_first=only_first,
        only_second=only_second,
        both_failed=both_failed,
        p_value=float(result.pvalue),
    )
