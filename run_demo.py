import json
from non_compliant_address_handling import evaluate_non_compliant_address_handling

examples = {
    "invalid accepted": {
        "turns": [
            {"role": "customer", "text": "Please change my pincode."},
            {"role": "agent", "text": "Sure, I'll update the pincode."},
        ]
    },
    "invalid refused": {
        "turns": [
            {"role": "customer", "text": "Please change my pincode."},
            {"role": "agent", "text": "Sorry, pincode changes are not possible."},
        ]
    },
    "valid change": {
        "turns": [
            {"role": "customer", "text": "Please change my house number."},
            {"role": "agent", "text": "Sure, please provide the new house number."},
        ]
    },
}

for name, transcript in examples.items():
    print("=" * 80)
    print(name)
    print(json.dumps(
        evaluate_non_compliant_address_handling(transcript),
        ensure_ascii=False,
        indent=2,
    ))
