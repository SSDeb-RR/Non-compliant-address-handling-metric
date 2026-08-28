import json
from pathlib import Path

import pytest

from non_compliant_address_handling import (
    evaluate_non_compliant_address_handling,
)


BASE = Path(__file__).resolve().parent
DATA = BASE / "august_ground_truth.json"

SOURCE_TO_EXPECTED = {
    "true": "true",
    "false": "false",
    "not_applicable": "false",
    "pass": "false",
    "fail": "true",
}


with open(DATA, "r", encoding="utf-8") as f:
    CASES = json.load(f)


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[case["case_id"] for case in CASES],
)
def test_source_transcript_matches_ground_truth(case):
    result = evaluate_non_compliant_address_handling(
        case["transcript"]
    )

    expected = SOURCE_TO_EXPECTED[case["metric_result"]]

    assert result["value"] == expected, (
        f"Case {case['case_id']} ({case['call_id']}): "
        f"ground_truth={case['metric_result']} "
        f"expected_binary={expected} "
        f"actual={result['value']} "
        f"reason={result['reason']}"
    )
