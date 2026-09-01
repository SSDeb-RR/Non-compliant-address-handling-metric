"""
Deterministic metric: Non-Compliant Address Handling

Output:
    true  -> customer requested a prohibited address change and the agent
             accepted/processed it.
    false -> every other case.

There is intentionally no not_applicable output.

Deterministic only:
- regex/pattern matching
- conversation state
- address component classification
- agent accept/refuse classification
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence


# ============================================================================
# ENUMS / DATA CLASSES
# ============================================================================

class MetricValue(str, Enum):
    TRUE = "true"
    FALSE = "false"


class TurnRole(str, Enum):
    AGENT = "agent"
    CUSTOMER = "customer"


class AddressComponent(str, Enum):
    PINCODE = "pincode"
    STATE = "state"
    CITY_OR_LOCATION = "city_or_location"

    VALID_STREET = "street"
    VALID_BUILDING = "building"
    VALID_HOUSE_FLAT = "house_flat"
    VALID_LANDMARK = "landmark"
    VALID_LOCALITY = "locality"
    VALID_NEARBY_REFERENCE = "nearby_reference"

    UNKNOWN = "unknown"


class AddressIntent(str, Enum):
    CHANGE_REQUEST = "change_request"
    CONFIRMATION_ONLY = "confirmation_only"
    NONE = "none"
    AMBIGUOUS = "ambiguous"


class AgentAction(str, Enum):
    ACCEPT = "accept"
    REFUSE = "refuse"
    CLARIFY = "clarify"
    UNKNOWN = "unknown"


@dataclass
class Turn:
    index: int
    role: TurnRole
    text: str


@dataclass
class AddressAnalysis:
    customer_turn_index: Optional[int]
    agent_turn_index: Optional[int]
    intent: AddressIntent
    component: AddressComponent
    agent_action: AgentAction
    result: MetricValue
    customer_text: Optional[str]
    agent_text: Optional[str]
    reason: str


@dataclass
class EvaluationResult:
    metric: str
    value: str

    address_change_detected: bool
    invalid_component_detected: bool

    selected_component: Optional[str]
    selected_customer_turn_index: Optional[int]
    selected_agent_turn_index: Optional[int]

    reason: str
    analyses: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

class TextNormalizer:
    TRANSLATION = str.maketrans(
        {
            "’": "'",
            "‘": "'",
            "“": '"',
            "”": '"',
            "–": "-",
            "—": "-",
            "\u00a0": " ",
        }
    )

    @classmethod
    def normalize(cls, text: Any) -> str:
        if text is None:
            return ""

        text = str(text).translate(cls.TRANSLATION)
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip().lower()


# ============================================================================
# TRANSCRIPT PARSER
# ============================================================================

class TranscriptParser:
    ROLE_ALIASES = {
        "agent": TurnRole.AGENT,
        "assistant": TurnRole.AGENT,
        "bot": TurnRole.AGENT,
        "customer": TurnRole.CUSTOMER,
        "user": TurnRole.CUSTOMER,
        "caller": TurnRole.CUSTOMER,
    }

    @classmethod
    def parse(cls, payload: Any) -> List[Turn]:
        if payload is None:
            return []

        if isinstance(payload, str):
            payload = json.loads(payload)

        if isinstance(payload, list):
            return cls._parse_list(payload)

        if not isinstance(payload, dict):
            raise TypeError("Transcript must be a dict/list or JSON string.")

        if "turns" in payload:
            return cls._parse_turns(payload["turns"])

        numbered = cls._parse_numbered(payload)
        if numbered:
            return numbered

        arrays = cls._parse_arrays(payload)
        if arrays:
            return arrays

        return cls._parse_simple(payload)

    @classmethod
    def _parse_turns(cls, items: Any) -> List[Turn]:
        if not isinstance(items, list):
            raise TypeError("'turns' must be a list.")

        result: List[Turn] = []

        for item in items:
            if not isinstance(item, dict):
                continue

            role = item.get("role") or item.get("speaker") or item.get("type")
            text = item.get("text", item.get("content", ""))

            parsed_role = (
                cls.ROLE_ALIASES.get(str(role).strip().lower())
                if role is not None
                else None
            )

            if parsed_role is None:
                continue

            result.append(
                Turn(
                    index=len(result),
                    role=parsed_role,
                    text=TextNormalizer.normalize(text),
                )
            )

        return result

    @classmethod
    def _parse_list(cls, items: List[Any]) -> List[Turn]:
        result: List[Turn] = []

        for item in items:
            if not isinstance(item, dict):
                continue

            role = item.get("role") or item.get("speaker") or item.get("type")
            text = item.get("text", item.get("content", ""))

            if role is None:
                if "agent" in item:
                    role, text = "agent", item["agent"]
                elif "customer" in item:
                    role, text = "customer", item["customer"]

            parsed_role = (
                cls.ROLE_ALIASES.get(str(role).strip().lower())
                if role is not None
                else None
            )

            if parsed_role is None:
                continue

            result.append(
                Turn(
                    index=len(result),
                    role=parsed_role,
                    text=TextNormalizer.normalize(text),
                )
            )

        return result

    @classmethod
    def _parse_numbered(cls, payload: Dict[str, Any]) -> List[Turn]:
        pattern = re.compile(
            r"^(agent|assistant|customer|user|caller|bot)[_\-\s]?(\d+)$",
            re.I,
        )

        items = []
        keys = list(payload.keys())

        for key, value in payload.items():
            match = pattern.match(str(key))

            if not match:
                continue

            role = cls.ROLE_ALIASES.get(match.group(1).lower())

            if role is None:
                continue

            items.append(
                (
                    int(match.group(2)),
                    keys.index(key),
                    role,
                    TextNormalizer.normalize(value),
                )
            )

        items.sort(key=lambda x: (x[0], x[1]))

        return [
            Turn(index=i, role=item[2], text=item[3])
            for i, item in enumerate(items)
        ]

    @classmethod
    def _parse_arrays(cls, payload: Dict[str, Any]) -> List[Turn]:
        if not any(
            isinstance(payload.get(k), list)
            for k in ("agent", "customer")
        ):
            return []

        agents = payload.get("agent", [])
        customers = payload.get("customer", [])

        if not isinstance(agents, list):
            agents = [agents]

        if not isinstance(customers, list):
            customers = [customers]

        result = []

        for i in range(max(len(agents), len(customers))):
            if i < len(agents):
                result.append(
                    Turn(
                        index=len(result),
                        role=TurnRole.AGENT,
                        text=TextNormalizer.normalize(agents[i]),
                    )
                )

            if i < len(customers):
                result.append(
                    Turn(
                        index=len(result),
                        role=TurnRole.CUSTOMER,
                        text=TextNormalizer.normalize(customers[i]),
                    )
                )

        return result

    @classmethod
    def _parse_simple(cls, payload: Dict[str, Any]) -> List[Turn]:
        """
        Supports:
            {"customer": "...", "agent": "..."}

        For a full call, `turns` is preferred because chronology matters.
        """
        result = []

        if "customer" in payload:
            result.append(
                Turn(
                    index=len(result),
                    role=TurnRole.CUSTOMER,
                    text=TextNormalizer.normalize(payload["customer"]),
                )
            )

        if "agent" in payload:
            result.append(
                Turn(
                    index=len(result),
                    role=TurnRole.AGENT,
                    text=TextNormalizer.normalize(payload["agent"]),
                )
            )

        return result


# ============================================================================
# ADDRESS COMPONENT / CHANGE-INTENT DETECTION
# ============================================================================

class AddressChangeDetector:
    """
    Important design rule:

    A request can be a change request even when the word "address" does not
    occur.

    Examples:
        "I want to change my pincode."
        "Please change my state."
        "Change the delivery location to Hubli."
        "पिनकोड बदलना है।"

    Known address component + change verb is therefore enough to establish
    CHANGE_REQUEST.
    """

    # General change/update language.
    CHANGE_VERB_PATTERNS = [
        r"\bchange(?:d|s|ing)?\b",
        r"\bupdate(?:d|s|ing)?\b",
        r"\bmodify(?:d|s|ing)?\b",
        r"\bedit(?:ed|s|ing)?\b",
        r"\bcorrect(?:ed|s|ing)?\b",
        r"\breplace(?:d|s|ing)?\b",
        r"\bmove(?:d|s|ing)?\b",
        r"\bshift(?:ed|s|ing)?\b",
        r"\bchangeover\b",
        r"कर दीजिए",
        r"कर दीजिये",
        r"कर दो",
        r"कर दें",
        r"कर देना",
        r"करना है",
        r"करना चाह",
        r"बदल",
        r"अपडेट",
        r"बदलाव",
        r"सुधार",
        r"संशोधन",
        r"हटाकर",
        r"नया",
    ]

    # Explicit negation of a change request must never be interpreted as a
    # change request. These patterns are checked before normal component/change
    # detection.
    NEGATED_CHANGE_PATTERNS = [
        # English
        r"\bdo not change\b",
        r"\bdon't change\b",
        r"\bdo not update\b",
        r"\bdon't update\b",
        r"\bdo not modify\b",
        r"\bdon't modify\b",
        r"\bi do not want to change\b",
        r"\bi don't want to change\b",
        r"\bi do not want.*\bchange\b",
        r"\bi don't want.*\bchange\b",
        r"\bno change\b",
        r"\bno update\b",
        r"\bnot change\b",
        r"\bnot to change\b",
        r"\bchange nahi\b",

        # Hindi / Hinglish
        r"नहीं बदलना",
        r"नहीं बदलना है",
        r"नहीं बदलना चाहता",
        r"नहीं बदलना चाहती",
        r"नहीं बदलना चाहते",
        r"नहीं बदलना चाहिये",
        r"नहीं बदलना चाहिए",
        r"मत बदलना",
        r"मत बदलिए",
        r"मत बदलना है",
        r"बदलना नहीं है",
        r"बदलना नहीं चाहता",
        r"बदलना नहीं चाहती",
        r"बदलना नहीं चाहते",
        r"अपडेट नहीं करना",
        r"अपडेट नहीं करना है",
        r"अपडेट नहीं चाहता",
        r"अपडेट नहीं चाहती",
        r"पिनकोड नहीं बदलना",
        r"पिनकोड नहीं बदलना है",
        r"पिन कोड नहीं बदलना",
        r"पिन कोड नहीं बदलना है",
        r"राज्य नहीं बदलना",
        r"राज्य नहीं बदलना है",
        r"स्टेट नहीं बदलना",
        r"स्टेट नहीं बदलना है",
    ]

    EXPLICIT_ADDRESS_CHANGE_PATTERNS = [
        r"\bchange (?:my|the) address\b",
        r"\bupdate (?:my|the) address\b",
        r"\bmodify (?:my|the) address\b",
        r"\bchange .* address\b",
        r"\bupdate .* address\b",
        r"\bnew address\b",
        r"\baddress change\b",

        r"पता बदलना",
        r"पता बदल",
        r"पता अपडेट",
        r"पते में बदलाव",
        r"नया पता",
        r"address में बदलाव",
        r"address बदल",
        r"address अपडेट",
        r"address .*\b(?:में|to|at)\b.*(?:कर|change|update)",
        r"delivery location changed",
        r"delivery location change",
    ]

    PINCODE_PATTERNS = [
        r"\bpincode\b",
        r"\bpin code\b",
        r"\bpin-code\b",
        r"\bpostal code\b",
        r"\bzip code\b",
        r"\bzipcode\b",
        r"\bzip\b",
        r"पिनकोड",
        r"पिन कोड",
        r"पिन\b",
    ]

    # Explicit state names are included because real transcripts often say
    # "change it to Karnataka" rather than using the word "state".
    STATE_PATTERNS = [
        r"\bstate\b",
        r"राज्य",
        r"स्टेट",

        # Indian state names commonly appearing in delivery addresses.
        r"\bandhra pradesh\b",
        r"\barunachal pradesh\b",
        r"\bassam\b",
        r"\bbihar\b",
        r"\bchhattisgarh\b",
        r"\bgoa\b",
        r"\bgujarat\b",
        r"\bharyana\b",
        r"\bhimachal pradesh\b",
        r"\bjharkhand\b",
        r"\bkarnataka\b",
        r"\bkerala\b",
        r"\bmadhya pradesh\b",
        r"\bmaharashtra\b",
        r"\bmanipur\b",
        r"\bmeghalaya\b",
        r"\bmizoram\b",
        r"\bnagaland\b",
        r"\bodisha\b",
        r"\bpunjab\b",
        r"\brajasthan\b",
        r"\bsikkim\b",
        r"\btamil nadu\b",
        r"\btelangana\b",
        r"\btripura\b",
        r"\buttar pradesh\b",
        r"\buttarkhand\b",
        r"\bwest bengal\b",

        # Hindi spellings commonly produced by ASR.
        r"आंध्र प्रदेश",
        r"अरुणाचल प्रदेश",
        r"असम",
        r"बिहार",
        r"छत्तीसगढ़",
        r"गोवा",
        r"गुजरात",
        r"हरियाणा",
        r"हिमाचल प्रदेश",
        r"झारखंड",
        r"कर्नाटक",
        r"केरल",
        r"मध्य प्रदेश",
        r"महाराष्ट्र",
        r"मणिपुर",
        r"मेघालय",
        r"मिजोरम",
        r"नागालैंड",
        r"ओडिशा",
        r"पंजाब",
        r"राजस्थान",
        r"सिक्किम",
        r"तमिलनाडु",
        r"तेलंगाना",
        r"त्रिपुरा",
        r"उत्तर प्रदेश",
        r"उत्तराखंड",
        r"पश्चिम बंगाल",
    ]

    CITY_LOCATION_PATTERNS = [
        r"\bcity\b",
        r"\blocation\b",
        r"\bdelivery location\b",
        r"\bmove .* to\b",
        r"\bshift .* to\b",
        r"\brelocat(?:e|ion)\b",

        r"शहर",
        r"लोकेशन",
        r"location",
        r"दूसरे शहर",
        r"दूसरी लोकेशन",
        r"दूसरे लोकेशन",
    ]

    STREET_PATTERNS = [
        r"\bstreet\b",
        r"\broad\b",
        r"\blane\b",
        r"सड़क",
        r"गली",
        r"रोड",
        r"लेन",
    ]

    BUILDING_PATTERNS = [
        r"\bbuilding\b",
        r"\bcomplex\b",
        r"\btower\b",
        r"\bblock\b",
        r"बिल्डिंग",
        r"कॉम्प्लेक्स",
        r"टावर",
        r"ब्लॉक",
    ]

    HOUSE_FLAT_PATTERNS = [
        r"\bhouse number\b",
        r"\bhouse no\b",
        r"\bflat number\b",
        r"\bflat no\b",
        r"\broom number\b",
        r"\broom no\b",
        r"\bdoor number\b",
        r"\bapartment number\b",
        r"मकान नंबर",
        r"घर नंबर",
        r"फ्लैट नंबर",
        r"रूम नंबर",
        r"कमरा नंबर",
    ]

    LANDMARK_PATTERNS = [
        r"\blandmark\b",
        r"\bnear\b",
        r"\bnearby\b",
        r"\bopposite\b",
        r"\bnext to\b",
        r"\bbehind\b",
        r"\bclose to\b",
        r"लैंडमार्क",
        r"पास में",
        r"नजदीक",
        r"सामने",
        r"बगल में",
        r"के पास",
    ]

    LOCALITY_PATTERNS = [
        r"\blocality\b",
        r"\barea\b",
        r"\bneighborhood\b",
        r"\bneighbourhood\b",
        r"\bcolony\b",
        r"\bsector\b",
        r"\bdistrict\b",
        r"लोकैलिटी",
        r"इलाका",
        r"क्षेत्र",
        r"कॉलोनी",
        r"सेक्टर",
        r"मोहल्ला",
        r"एरिया",
    ]

    CONFIRMATION_ONLY_PATTERNS = [
        r"\bconfirm (?:my|the) address\b",
        r"\bconfirm .* address\b",
        r"\bverify .* address\b",
        r"\bis .* address (?:correct|right)\b",
        r"\bdoes .* address .* match\b",

        r"पता कन्फर्म",
        r"पता सही है",
        r"पता की पुष्टि",
    ]

    @staticmethod
    def _matches(text: str, patterns: Sequence[str]) -> bool:
        return any(re.search(p, text, re.I) for p in patterns)

    @classmethod
    def has_change_verb(cls, text: str) -> bool:
        return cls._matches(text, cls.CHANGE_VERB_PATTERNS)

    @classmethod
    def has_explicit_address_change(cls, text: str) -> bool:
        return cls._matches(text, cls.EXPLICIT_ADDRESS_CHANGE_PATTERNS)

    @classmethod
    def is_negated_change(cls, text: str) -> bool:
        return cls._matches(text, cls.NEGATED_CHANGE_PATTERNS)

    @classmethod
    def detect_component(cls, text: str) -> AddressComponent:
        """
        Invalid categories have priority.
        This matters when a sentence contains both "address" and "pincode".
        """
        text = TextNormalizer.normalize(text)

        if cls._matches(text, cls.PINCODE_PATTERNS):
            return AddressComponent.PINCODE

        if cls._matches(text, cls.STATE_PATTERNS):
            return AddressComponent.STATE

        if cls._matches(text, cls.CITY_LOCATION_PATTERNS):
            return AddressComponent.CITY_OR_LOCATION

        if cls._matches(text, cls.HOUSE_FLAT_PATTERNS):
            return AddressComponent.VALID_HOUSE_FLAT

        if cls._matches(text, cls.BUILDING_PATTERNS):
            return AddressComponent.VALID_BUILDING

        if cls._matches(text, cls.STREET_PATTERNS):
            return AddressComponent.VALID_STREET

        if cls._matches(text, cls.LANDMARK_PATTERNS):
            return AddressComponent.VALID_LANDMARK

        if cls._matches(text, cls.LOCALITY_PATTERNS):
            return AddressComponent.VALID_LOCALITY

        return AddressComponent.UNKNOWN

    @classmethod
    def detect_intent(cls, text: str) -> AddressIntent:
        text = TextNormalizer.normalize(text)

        # A negated request explicitly says the customer does NOT want the
        # address component changed. Never classify it as CHANGE_REQUEST.
        if cls.is_negated_change(text):
            return AddressIntent.NONE

        if cls.has_explicit_address_change(text):
            return AddressIntent.CHANGE_REQUEST

        component = cls.detect_component(text)

        # KEY FIX:
        # A known address component + explicit change language is itself a
        # change request; "address" does not need to be present.
        if (
            component != AddressComponent.UNKNOWN
            and cls.has_change_verb(text)
        ):
            return AddressIntent.CHANGE_REQUEST

        if cls._matches(text, cls.CONFIRMATION_ONLY_PATTERNS):
            return AddressIntent.CONFIRMATION_ONLY

        return AddressIntent.NONE

    @classmethod
    def detect_from_context(
        cls,
        text: str,
        pending_component: AddressComponent,
    ) -> AddressIntent:
        """
        Used for a follow-up customer turn after they have already said:
            "I want to change my address."
            agent: "Which part?"
            customer: "The pincode."

        It can also recognize a bare component/value follow-up when the
        conversation is already in address-change context.
        """
        text = TextNormalizer.normalize(text)

        # A negated follow-up such as "pincode नहीं बदलना चाहता" cancels the
        # apparent component request rather than confirming it.
        if cls.is_negated_change(text):
            return AddressIntent.NONE

        component = cls.detect_component(text)

        if component != AddressComponent.UNKNOWN:
            return AddressIntent.CHANGE_REQUEST

        if pending_component != AddressComponent.UNKNOWN:
            # A bare value such as "400055" is still ambiguous by itself.
            # We keep the existing component in state, but do not turn arbitrary
            # speech into a component.
            if re.fullmatch(r"[\d\W_]+", text):
                if pending_component == AddressComponent.PINCODE:
                    return AddressIntent.CHANGE_REQUEST

        return AddressIntent.AMBIGUOUS


# ============================================================================
# AGENT ACTION DETECTION
# ============================================================================

class AgentActionDetector:
    """
    Acceptance and refusal are intentionally separate.

    Refusal is checked first so phrases like:
        "We cannot change the pincode"
    are not classified as ACCEPT merely because "change" appears.
    """

    REFUSE_PATTERNS = [
        r"\bcannot\b",
        r"\bcan not\b",
        r"\bcan't\b",
        r"\bnot possible\b",
        r"\bunable to\b",
        r"\bnot allowed\b",
        r"\bnot permitted\b",
        r"\bwe don't allow\b",
        r"\bwe do not allow\b",
        r"\bcannot change\b",
        r"\bcan't change\b",
        r"\bcannot update\b",
        r"\bcan't update\b",
        r"\bcannot be changed\b",
        r"\bonly .*same pincode\b",
        r"\bwithin the same pincode\b",
        r"\bonly allow address changes within\b",
        r"\bonly .* same state\b",

        r"बदलना संभव नहीं",
        r"बदलना मुमकिन नहीं",
        r"बदल नहीं सकते",
        r"अपडेट नहीं कर सकते",
        r"अपडेट संभव नहीं",
        r"पिनकोड बदलना संभव नहीं",
        r"पिनकोड बदल नहीं सकते",
        r"राज्य बदलना संभव नहीं",
        r"राज्य बदल नहीं सकते",
        r"लोकेशन बदलना संभव नहीं",
        r"location बदलना संभव नहीं",
        r"शहर बदलना संभव नहीं",
        r"सिर्फ इसी पिनकोड",
        r"केवल इसी पिनकोड",
        r"उसी पिनकोड",
        r"इस ऑर्डर के लिए .* नहीं बदल सकते",
    ]

    ACCEPT_PATTERNS = [
        r"\b(i'll|i will|we'll|we will|i can|we can)\b.*\b(update|change|modify|edit)\b",
        r"\b(i have|we have) (updated|changed|modified)\b",
        r"\b(done|updated|changed|modified)\b",
        r"\bprocess(?:ed|ing)?\b.*\b(change|update)\b",
        r"\bupdate it\b",
        r"\bchange it\b",
        r"\bI'll update\b",
        r"\bwe can change\b",
        r"\bi can change\b",

        r"कर देता हूँ",
        r"कर देती हूँ",
        r"कर देंगे",
        r"बदल देता हूँ",
        r"बदल देती हूँ",
        r"बदल देंगे",
        r"अपडेट कर देता हूँ",
        r"अपडेट कर देती हूँ",
        r"अपडेट कर देंगे",
        r"बदल दिया",
        r"अपडेट कर दिया",
        r"कर दिया है",
        r"हो गया",
    ]

    CLARIFY_PATTERNS = [
        r"\bwhich part\b",
        r"\bwhat part\b",
        r"\bwhat would you like to change\b",
        r"\bwhich address detail\b",
        r"\bwhat information is incorrect\b",
        r"\bplease provide\b",
        r"\bplease tell me\b",
        r"\bplease confirm\b",
        r"\bcan you confirm\b",
        r"\bwhat would you like\b",
        r"कौन सा हिस्सा",
        r"क्या बदलना",
        r"क्या बदलाव",
        r"कृपया बताएं",
        r"कौन सा address",
    ]

    @staticmethod
    def _matches(text: str, patterns: Sequence[str]) -> bool:
        return any(re.search(p, text, re.I) for p in patterns)

    @classmethod
    def detect(cls, text: str) -> AgentAction:
        text = TextNormalizer.normalize(text)

        if not text:
            return AgentAction.UNKNOWN

        if cls._matches(text, cls.REFUSE_PATTERNS):
            return AgentAction.REFUSE

        if cls._matches(text, cls.ACCEPT_PATTERNS):
            return AgentAction.ACCEPT

        if cls._matches(text, cls.CLARIFY_PATTERNS):
            return AgentAction.CLARIFY

        return AgentAction.UNKNOWN


# ============================================================================
# METRIC
# ============================================================================

class NonCompliantAddressHandlingMetric:
    METRIC_NAME = "Non-Compliant Address Handling"

    INVALID_COMPONENTS = {
        AddressComponent.PINCODE,
        AddressComponent.STATE,
        AddressComponent.CITY_OR_LOCATION,
    }

    def evaluate(self, transcript: Any) -> Dict[str, Any]:
        turns = TranscriptParser.parse(transcript)

        if not turns:
            return self._result(
                MetricValue.FALSE,
                reason=(
                    "No usable transcript turns were found; "
                    "not_applicable maps to false."
                ),
            )

        analyses = self._analyze(turns)

        # A single accepted prohibited change is enough for TRUE.
        accepted_invalid = [
            a
            for a in analyses
            if (
                a.intent == AddressIntent.CHANGE_REQUEST
                and a.component in self.INVALID_COMPONENTS
                and a.agent_action == AgentAction.ACCEPT
            )
        ]

        if accepted_invalid:
            selected = accepted_invalid[-1]
            return self._result(
                MetricValue.TRUE,
                reason=(
                    "The customer requested a prohibited address change "
                    "and the agent accepted or processed it."
                ),
                selected=selected,
                analyses=analyses,
            )

        # Everything else is FALSE by the metric's binary contract.
        return self._result(
            MetricValue.FALSE,
            reason=(
                "No prohibited address change was accepted or processed "
                "by the agent."
            ),
            selected=analyses[-1] if analyses else None,
            analyses=analyses,
        )

    def _analyze(self, turns: List[Turn]) -> List[AddressAnalysis]:
        analyses: List[AddressAnalysis] = []

        # Conversation state.
        address_change_context = False
        pending_component = AddressComponent.UNKNOWN
        pending_customer_index: Optional[int] = None

        for turn in turns:
            if turn.role == TurnRole.CUSTOMER:
                # Negated change requests must clear any pending address-change
                # state so a previously mentioned invalid component cannot leak
                # into a later, separate valid address update.
                if AddressChangeDetector.is_negated_change(turn.text):
                    address_change_context = False
                    pending_component = AddressComponent.UNKNOWN
                    pending_customer_index = None
                    continue

                intent = AddressChangeDetector.detect_intent(turn.text)
                component = AddressChangeDetector.detect_component(turn.text)

                if intent == AddressIntent.CHANGE_REQUEST:
                    address_change_context = True
                    pending_component = component
                    pending_customer_index = turn.index
                    continue

                if address_change_context:
                    follow_intent = AddressChangeDetector.detect_from_context(
                        turn.text,
                        pending_component,
                    )

                    follow_component = AddressChangeDetector.detect_component(
                        turn.text
                    )

                    if follow_component != AddressComponent.UNKNOWN:
                        pending_component = follow_component
                        pending_customer_index = turn.index
                        address_change_context = True
                    elif follow_intent == AddressIntent.CHANGE_REQUEST:
                        address_change_context = True

                continue

            # Agent turn.
            if not address_change_context:
                continue

            action = AgentActionDetector.detect(turn.text)

            # Clarification doesn't resolve the request yet.
            if action == AgentAction.CLARIFY:
                agent_component = AddressChangeDetector.detect_component(
                    turn.text
                )
                if (
                    agent_component != AddressComponent.UNKNOWN
                    and pending_component == AddressComponent.UNKNOWN
                ):
                    pending_component = agent_component
                continue

            agent_component = AddressChangeDetector.detect_component(
                turn.text
            )

            # If customer component was unknown, use an explicit component in
            # the agent's response only when the agent is clearly acting on it.
            if pending_component == AddressComponent.UNKNOWN:
                pending_component = agent_component

            if pending_component == AddressComponent.UNKNOWN:
                intent = AddressIntent.AMBIGUOUS
            else:
                intent = AddressIntent.CHANGE_REQUEST

            analyses.append(
                AddressAnalysis(
                    customer_turn_index=pending_customer_index,
                    agent_turn_index=turn.index,
                    intent=intent,
                    component=pending_component,
                    agent_action=action,
                    result=(
                        MetricValue.TRUE
                        if (
                            pending_component in self.INVALID_COMPONENTS
                            and action == AgentAction.ACCEPT
                        )
                        else MetricValue.FALSE
                    ),
                    customer_text=(
                        turns[pending_customer_index].text
                        if pending_customer_index is not None
                        else None
                    ),
                    agent_text=turn.text,
                    reason=self._reason(
                        pending_component,
                        action,
                    ),
                )
            )

            # A definitive agent refusal/acceptance completes the current
            # address decision. A later customer address request can create
            # a new one.
            if action in {
                AgentAction.ACCEPT,
                AgentAction.REFUSE,
            }:
                address_change_context = False
                pending_component = AddressComponent.UNKNOWN
                pending_customer_index = None

        return analyses

    @staticmethod
    def _reason(
        component: AddressComponent,
        action: AgentAction,
    ) -> str:
        if component in {
            AddressComponent.PINCODE,
            AddressComponent.STATE,
            AddressComponent.CITY_OR_LOCATION,
        }:
            if action == AgentAction.ACCEPT:
                return "Invalid component accepted/processed."

            if action == AgentAction.REFUSE:
                return "Invalid component correctly refused."

            return (
                "Invalid component identified, but agent action "
                "was not definitive."
            )

        if component == AddressComponent.UNKNOWN:
            return "Address-change component is ambiguous."

        return (
            "Valid address component; this does not fail the metric "
            "even if the agent refuses the update."
        )

    @staticmethod
    def _result(
        value: MetricValue,
        *,
        reason: str,
        selected: Optional[AddressAnalysis] = None,
        analyses: Optional[List[AddressAnalysis]] = None,
    ) -> Dict[str, Any]:
        analyses = analyses or []

        return EvaluationResult(
            metric=NonCompliantAddressHandlingMetric.METRIC_NAME,
            value=value.value,
            address_change_detected=any(
                a.intent == AddressIntent.CHANGE_REQUEST
                for a in analyses
            ),
            invalid_component_detected=any(
                a.component
                in NonCompliantAddressHandlingMetric.INVALID_COMPONENTS
                for a in analyses
            ),
            selected_component=(
                selected.component.value
                if selected
                else None
            ),
            selected_customer_turn_index=(
                selected.customer_turn_index
                if selected
                else None
            ),
            selected_agent_turn_index=(
                selected.agent_turn_index
                if selected
                else None
            ),
            reason=reason,
            analyses=[
                {
                    **asdict(a),
                    "intent": a.intent.value,
                    "component": a.component.value,
                    "agent_action": a.agent_action.value,
                    "result": a.result.value,
                }
                for a in analyses
            ],
        ).to_dict()


# ============================================================================
# PUBLIC API
# ============================================================================

def evaluate_non_compliant_address_handling(
    transcript: Any,
) -> Dict[str, Any]:
    """
    Main public API.

    Returns:
        {"value": "true"} or {"value": "false"} plus audit information.
    """
    return NonCompliantAddressHandlingMetric().evaluate(transcript)


def evaluate_batch(
    transcripts: Iterable[Any],
) -> List[Dict[str, Any]]:
    metric = NonCompliantAddressHandlingMetric()
    return [metric.evaluate(t) for t in transcripts]


if __name__ == "__main__":
    examples = [
        {
            "name": "invalid accepted",
            "transcript": {
                "turns": [
                    {
                        "role": "customer",
                        "text": "I want to change my pincode.",
                    },
                    {
                        "role": "agent",
                        "text": "Sure, I'll update the pincode.",
                    },
                ]
            },
        },
        {
            "name": "invalid refused",
            "transcript": {
                "turns": [
                    {
                        "role": "customer",
                        "text": "I want to change my pincode.",
                    },
                    {
                        "role": "agent",
                        "text": "Sorry, pincode changes are not possible.",
                    },
                ]
            },
        },
        {
            "name": "valid update",
            "transcript": {
                "turns": [
                    {
                        "role": "customer",
                        "text": "Please change my house number.",
                    },
                    {
                        "role": "agent",
                        "text": "Sure, please provide the new house number.",
                    },
                ]
            },
        },
    ]

    for example in examples:
        print("=" * 80)
        print(example["name"])
        print(
            json.dumps(
                evaluate_non_compliant_address_handling(
                    example["transcript"]
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
