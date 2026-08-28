import json
import pytest

from non_compliant_address_handling import (
    evaluate_non_compliant_address_handling,
)


def value(transcript):
    return evaluate_non_compliant_address_handling(transcript)["value"]


@pytest.mark.parametrize(
    "customer_text,agent_text",
    [
        ("I want to change my pincode to 400055.", "Sure, I'll update the pincode for you."),
        ("Please change my state to Karnataka.", "Sure, I've updated your state."),
        ("I need the delivery location changed to Hubli, Karnataka.", "Okay, I'll update the delivery location."),
        ("पिनकोड बदलना है।", "जी हाँ, मैं पिनकोड अपडेट कर देता हूँ।"),
        ("राज्य बदलना है।", "ठीक है, राज्य बदल देता हूँ।"),
        ("मेरा address Karnataka में कर दीजिए।", "जी हाँ, कर देता हूँ।"),
    ],
)
def test_invalid_change_accepted_is_true(customer_text, agent_text):
    assert value({
        "turns": [
            {"role": "customer", "text": customer_text},
            {"role": "agent", "text": agent_text},
        ]
    }) == "true"


@pytest.mark.parametrize(
    "customer_text,agent_text",
    [
        ("I want to change my pincode to 400055.", "I'm sorry, pincode changes are not possible."),
        ("Please change my state to Karnataka.", "State changes are not allowed for this order."),
        ("I need the delivery location changed to Hubli.", "We cannot change the delivery location for this order."),
        ("मुझे पिनकोड बदलना है।", "माफ़ कीजिए, पिनकोड बदलना संभव नहीं है।"),
        ("मुझे राज्य बदलना है।", "राज्य बदलना संभव नहीं है।"),
        ("मेरा location Karnataka करना है।", "इस ऑर्डर के लिए location बदलना संभव नहीं है।"),
    ],
)
def test_invalid_change_refused_is_false(customer_text, agent_text):
    assert value({
        "turns": [
            {"role": "customer", "text": customer_text},
            {"role": "agent", "text": agent_text},
        ]
    }) == "false"


@pytest.mark.parametrize(
    "customer_text,agent_text",
    [
        ("I want to change my street address.", "Sure, please give me the new street details."),
        ("I want to change the building name.", "Sure, I can update the building name."),
        ("Please change my flat number.", "Okay, please provide the new flat number."),
        ("I want to add a landmark.", "Sure, please tell me the landmark."),
        ("Please change the locality.", "Okay, please provide the new locality."),
        ("गली बदलनी है।", "जी, कृपया नई गली बताइए।"),
        ("नया मकान नंबर देना है।", "ठीक है, नया मकान नंबर बताइए।"),
    ],
)
def test_valid_changes_are_false(customer_text, agent_text):
    assert value({
        "turns": [
            {"role": "customer", "text": customer_text},
            {"role": "agent", "text": agent_text},
        ]
    }) == "false"


@pytest.mark.parametrize(
    "text",
    [
        "Can you confirm your address?",
        "Is the address correct?",
        "Does this address match?",
        "The delivery failed because we could not find the address.",
        "Please confirm the address we have on file.",
        "I am calling about your delivery.",
    ],
)
def test_no_address_change_is_false(text):
    assert value({"turns": [{"role": "agent", "text": text}]}) == "false"


@pytest.mark.parametrize(
    "text",
    [
        "Your delivery is already out for delivery today.",
        "The delivery is already on the way.",
        "The delivery partner has already called today.",
    ],
)
def test_existing_delivery_is_false(text):
    assert value({"turns": [{"role": "agent", "text": text}]}) == "false"


def test_ambiguous_address_change_is_false():
    transcript = {
        "turns": [
            {"role": "customer", "text": "I want to change something in my address."},
            {"role": "agent", "text": "What would you like to change?"},
        ]
    }
    assert value(transcript) == "false"


def test_pincode_followup_refused_is_false():
    transcript = {
        "turns": [
            {"role": "customer", "text": "I need to change my address."},
            {"role": "agent", "text": "What part would you like to change?"},
            {"role": "customer", "text": "The pincode."},
            {"role": "agent", "text": "Sorry, pincode changes are not possible."},
        ]
    }
    assert value(transcript) == "false"


def test_pincode_followup_accepted_is_true():
    transcript = {
        "turns": [
            {"role": "customer", "text": "I need to change my address."},
            {"role": "agent", "text": "What part would you like to change?"},
            {"role": "customer", "text": "The pincode."},
            {"role": "agent", "text": "Sure, I'll update the pincode."},
        ]
    }
    assert value(transcript) == "true"


def test_location_change_refused_is_false():
    transcript = {
        "turns": [
            {
                "role": "customer",
                "text": "I moved to Hubli, Karnataka. Please change my delivery location.",
            },
            {
                "role": "agent",
                "text": "I understand, but we cannot change the location for this order.",
            },
        ]
    }
    assert value(transcript) == "false"


def test_location_change_accepted_is_true():
    transcript = {
        "turns": [
            {
                "role": "customer",
                "text": "I moved to Hubli, Karnataka. Please change my delivery location.",
            },
            {
                "role": "agent",
                "text": "Okay, I'll update the delivery location.",
            },
        ]
    }
    assert value(transcript) == "true"


def test_invalid_refused_and_valid_accepted_is_false():
    transcript = {
        "turns": [
            {"role": "customer", "text": "I want to change my pincode and house number."},
            {
                "role": "agent",
                "text": "We cannot change the pincode, but I can update the house number.",
            },
        ]
    }
    assert value(transcript) == "false"


def test_invalid_accepted_and_valid_accepted_is_true():
    transcript = {
        "turns": [
            {"role": "customer", "text": "I want to change my pincode and house number."},
            {
                "role": "agent",
                "text": "Sure, I'll update both the pincode and house number.",
            },
        ]
    }
    assert value(transcript) == "true"


def test_simple_pair_format():
    transcript = {
        "customer": "Please change my pincode.",
        "agent": "Sorry, pincode changes are not possible.",
    }
    assert value(transcript) == "false"


def test_numbered_format():
    transcript = {
        "customer_1": "Please change my pincode.",
        "agent_1": "Sure, I'll update the pincode.",
    }
    assert value(transcript) == "true"


def test_json_string_format():
    transcript = json.dumps({
        "turns": [
            {"role": "customer", "text": "Please change my state."},
            {"role": "agent", "text": "State changes are not allowed."},
        ]
    })
    assert value(transcript) == "false"


@pytest.mark.parametrize("transcript", [None, {}, {"turns": []}])
def test_empty_input_maps_to_false(transcript):
    assert value(transcript) == "false"


def test_audit_output():
    transcript = {
        "turns": [
            {"role": "customer", "text": "I want to change my pincode."},
            {"role": "agent", "text": "Sure, I'll update the pincode."},
        ]
    }
    result = evaluate_non_compliant_address_handling(transcript)
    assert result["value"] == "true"
    assert result["address_change_detected"] is True
    assert result["invalid_component_detected"] is True
    assert result["selected_component"] == "pincode"
    assert result["analyses"]
