import json
from collections import Counter
from pathlib import Path

from non_compliant_address_handling import (
    evaluate_non_compliant_address_handling,
)


BASE = Path(__file__).resolve().parent
DATA = BASE / "august_ground_truth.json"
OUTPUT = BASE / "august_comparison_report.json"


def main():
    with open(DATA, "r", encoding="utf-8") as f:
        cases = json.load(f)

    counts = Counter()
    results = []

    for case in cases:
        actual = evaluate_non_compliant_address_handling(
            case["transcript"]
        )

        gt = case["metric_result"]
        expected = {
            "true": "true",
            "false": "false",
            "not_applicable": "false",
            "pass": "false",
            "fail": "true",
        }[gt]

        match = actual["value"] == expected

        counts["matches" if match else "mismatches"] += 1

        results.append({
            "case_id": case["case_id"],
            "call_started_at_ist": case["call_started_at_ist"],
            "call_id": case["call_id"],
            "ground_truth_metric_result": gt,
            "expected_binary_output": expected,
            "actual_output": actual["value"],
            "match": match,
            "actual_result": actual,
        })

    total = len(results)
    report = {
        "total_cases": total,
        "matches": counts["matches"],
        "mismatches": counts["mismatches"],
        "accuracy": counts["matches"] / total if total else None,
        "results": results,
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Non-Compliant Address Handling")
    print(f"Total cases: {total}")
    print(f"Matches:     {counts['matches']}")
    print(f"Mismatches:  {counts['mismatches']}")
    print(
        f"Accuracy:    {report['accuracy']:.2%}"
        if total else "Accuracy: N/A"
    )
    print(f"Report:      {OUTPUT}")

    for item in results:
        if not item["match"]:
            print("-" * 80)
            print("Case:", item["case_id"])
            print("Ground truth:", item["ground_truth_metric_result"])
            print("Expected:", item["expected_binary_output"])
            print("Actual:", item["actual_output"])
            print("Reason:", item["actual_result"]["reason"])
            if item["actual_result"]["analyses"]:
                print(
                    "Analyses:",
                    json.dumps(
                        item["actual_result"]["analyses"],
                        ensure_ascii=False,
                    ),
                )


if __name__ == "__main__":
    main()
