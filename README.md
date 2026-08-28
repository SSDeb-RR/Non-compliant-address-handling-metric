# Non-Compliant Address Handling

Completely deterministic implementation of the metric.

## Output

This metric has only two outputs:

- `true`: customer requested a prohibited address change and the agent accepted or processed it.
- `false`: everything else.

There is no `not_applicable` output. Any situation that would otherwise be not-applicable maps to `false`.

## Invalid address changes

- pincode / pin code / postal code / zip
- state
- city / geographic location / relocation to another city or location

## Valid address changes

- street / road / lane
- building
- house / flat / room number
- landmark
- locality / area / colony / neighbourhood
- nearby reference

## Important policy decisions

A valid address change is always `false`, including when the agent refuses the change.

For an invalid change:

- agent accepts/processes it -> `true`
- agent refuses/explains the limitation -> `false`

A bare address confirmation is `false`.

An ambiguous address change is `false`.

## Source regression dataset

`august_ground_truth.json` contains every call from the supplied source file.

The source `metric_result` is treated as the ground truth.

For comparison:

- `fail` -> `true`
- `pass` -> `false`
- `not_applicable` -> `false`

The original source value is preserved in the JSON.

## Run the unit/edge-case suite

```bash
python -m pytest -q test_non_compliant_address_handling.py
```

## Run every supplied call as a regression test

```bash
python -m pytest -q test_all_august_transcripts.py
```

This will pass only when the deterministic implementation matches the supplied ground truth after applying the not-applicable -> false rule.

## Produce a detailed comparison report

```bash
python run_all_august.py
```

This writes:

```text
august_comparison_report.json
```

containing every case, expected output, actual output, match status, and the deterministic audit result.

## Public API

```python
from non_compliant_address_handling import (
    evaluate_non_compliant_address_handling,
)

result = evaluate_non_compliant_address_handling(transcript)
print(result)
```

## Recommended transcript format

```json
{
  "turns": [
    {
      "role": "customer",
      "text": "Please change my pincode."
    },
    {
      "role": "agent",
      "text": "Pincode changes are not possible."
    }
  ]
}
```

The simple single-pair format is also supported:

```json
{
  "customer": "Please change my pincode.",
  "agent": "Pincode changes are not possible."
}
```
