from __future__ import annotations

import json
import re
from typing import Any

from shared.clients import call_chat_with_fallbacks

_VALID_CATEGORIES = [
    "Administrative",
    "Aggravated Assault",
    "Alarms",
    "Animal Related",
    "Arson",
    "Assistance",
    "Auto Theft",
    "Bomb/Explosives",
    "Burglary",
    "Child Abuse/Neglect",
    "Crashes",
    "DUI/DWI",
    "Deceased/Injured",
    "Disorderly Conduct",
    "Disturbance",
    "Drugs",
    "Evading/Resisting Arrest",
    "Fraud/Forgery",
    "Gambling",
    "Homicide",
    "Intimidation",
    "Lost/Found Property",
    "Missing Persons/Kidnapping",
    "Person Stop/FO",
    "Property Damage/Vandalism",
    "Prostitution",
    "Robbery",
    "Sex Crimes",
    "Shoot/Stab",
    "Simple Assault",
    "Suspicious Things",
    "Theft",
    "Traffic Stop/Hazard",
    "Trespassing",
    "Viol PO/Bond",
    "Warrants/RTA",
    "Weapons/Firearms Violations",
    "Welfare Check",
]

_VALID_CATEGORIES_SET = set(_VALID_CATEGORIES)

_ALL_LABEL_KEYS = {"severity", "category", "dispatch_police", "dispatch_emt", "dispatch_fire"}

_MAX_PARSE_RETRIES = 2


def _resolve_allowed_categories(allowed_categories: list[str]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for category in allowed_categories:
        if category not in _VALID_CATEGORIES_SET:
            raise ValueError(f"unsupported allowed category: {category}")
        if category not in seen:
            resolved.append(category)
            seen.add(category)

    if not resolved:
        raise ValueError("allowed_categories must contain at least one supported category")

    return resolved


def _build_labeling_system_prompt(allowed_categories: list[str]) -> str:
    return f"""\
You are an expert 911 dispatch assistant.
Given a 911 transcript and optional context, predict any missing labels.

You MUST return ONLY valid JSON (no markdown, no commentary, no extra text).

The JSON object must have exactly these keys:
{{
  "severity": <int 0-3 or null if unknown>,
  "category": <string - MUST be one of the allowed categories listed below>,
  "dispatch_police": <true or false>,
  "dispatch_emt": <true or false>,
  "dispatch_fire": <true or false>
}}

Allowed categories (use EXACTLY one of these strings):
{chr(10).join('- ' + c for c in allowed_categories)}

Rules:
- If a label is already provided as known, keep it unchanged.
- Only predict labels that are missing (null / not provided).
- severity: 0=low, 1=moderate, 2=urgent, 3=critical
- category: MUST be exactly one of the allowed categories above. Do NOT invent new categories.
- dispatch_*: whether that unit type should be dispatched
"""


def _compose_user_prompt(
    transcript: str,
    known_labels: dict[str, Any] | None,
    extra_info: str | None,
) -> str:
    parts: list[str] = []

    if known_labels:
        parts.append("## Known Labels (preserve these)")
        parts.append(json.dumps(known_labels, ensure_ascii=False))

    if extra_info:
        parts.append("## Additional Context")
        parts.append(str(extra_info).strip())

    parts.append("## Transcript")
    parts.append(transcript.strip())

    parts.append("\nReturn ONLY the JSON object with all five keys.")
    return "\n\n".join(parts)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from LLM output that may include extra text."""
    text = text.strip()

    # Try to find JSON inside markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))

    # Try to find a bare JSON object
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))

    # Last resort: try the whole string
    return json.loads(text)


# Fuzzy fallback: map common LLM mistakes to valid categories.
_CATEGORY_FALLBACK: dict[str, str] = {
    k.lower(): v
    for k, v in {
        "Medical": "Assistance", "Medical Assist": "Assistance",
        "Medical Assistance": "Assistance", "Medical Aid": "Assistance",
        "Shooting": "Shoot/Stab", "Active Shooter": "Shoot/Stab",
        "Stabbing": "Shoot/Stab", "Shots Fired": "Shoot/Stab",
        "Officer Involved Shooting": "Shoot/Stab",
        "Murder": "Homicide", "Attempted Murder": "Homicide",
        "Motor Vehicle Accident": "Crashes", "Traffic Accident": "Crashes",
        "Accident": "Crashes", "Hit and Run": "Crashes",
        "Bank Robbery": "Robbery", "Armed Robbery": "Robbery",
        "Carjacking": "Robbery", "Assault": "Aggravated Assault",
        "Kidnapping": "Missing Persons/Kidnapping",
        "Missing Person": "Missing Persons/Kidnapping",
        "Structure Fire": "Arson", "Fire": "Arson",
        "Overdose": "Deceased/Injured", "Drowning": "Deceased/Injured",
        "Weapons Violation": "Weapons/Firearms Violations",
        "Person with a Gun": "Weapons/Firearms Violations",
        "Noise Complaint": "Disturbance", "Domestic Violence": "Disturbance",
        "Mental Health": "Welfare Check", "Suicide Attempt": "Welfare Check",
        "Bomb Threat": "Bomb/Explosives", "Explosion": "Bomb/Explosives",
        "Animal Attack": "Animal Related", "DUI": "DUI/DWI",
        "Stolen Vehicle": "Auto Theft", "Hostage Situation": "Intimidation",
        "Sexual Assault": "Sex Crimes", "Vehicle Pursuit": "Evading/Resisting Arrest",
        "Reckless Driving": "Traffic Stop/Hazard",
    }.items()
}


def _validate_labels(parsed: dict[str, Any], allowed_categories: list[str]) -> dict[str, Any]:
    """Normalise and validate the parsed label dict."""
    result: dict[str, Any] = {}
    allowed_categories_set = set(allowed_categories)

    severity = parsed.get("severity")
    if severity is not None:
        severity = int(severity)
        severity = max(0, min(3, severity))
    result["severity"] = severity

    category = parsed.get("category")
    if not category:
        raise ValueError("category is required and must be a supported known class")

    if category not in allowed_categories_set:
        # Try case-insensitive exact match first
        for valid in allowed_categories:
            if valid.lower() == category.lower():
                category = valid
                break
        else:
            # Try fuzzy fallback map. If that still fails, retry instead of
            # silently collapsing into a catch-all label.
            mapped = _CATEGORY_FALLBACK.get(category.lower())
            if mapped is None:
                raise ValueError(f"unsupported category: {category}")
            category = mapped

    if category not in allowed_categories_set:
        raise ValueError(f"unsupported category after normalisation: {category}")
    result["category"] = category

    for key in ("dispatch_police", "dispatch_emt", "dispatch_fire"):
        val = parsed.get(key)
        result[key] = bool(val) if val is not None else False

    return result


def label_transcript_with_context(
    transcript: str,
    known_labels: dict[str, Any] | None = None,
    extra_info: str | None = None,
    *,
    allowed_categories: list[str],
) -> dict[str, Any]:
    """Label a transcript using an LLM, preserving any already-known labels.

    Args:
        transcript: The 911 call transcript text.
        known_labels: Dict of labels already known (will be preserved).
            Keys: ``severity``, ``category``, ``dispatch_police``,
            ``dispatch_emt``, ``dispatch_fire``. Missing keys are predicted.
        extra_info: Optional additional context (text or stringified JSON).
        allowed_categories: Supported category set used to constrain the
            prompt and validation.

    Returns:
        Dict with all five label keys populated.
    """
    if not transcript or not transcript.strip():
        raise ValueError("transcript must be a non-empty string.")

    known = dict(known_labels) if known_labels else {}
    resolved_allowed_categories = _resolve_allowed_categories(allowed_categories)
    if known.get("category") is not None and known["category"] not in set(resolved_allowed_categories):
        raise ValueError(f"known category is not in allowed_categories: {known['category']}")

    user_prompt = _compose_user_prompt(transcript, known or None, extra_info)

    messages = [
        {"role": "system", "content": _build_labeling_system_prompt(resolved_allowed_categories)},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None
    for attempt in range(_MAX_PARSE_RETRIES + 1):
        response = call_chat_with_fallbacks(messages=messages)
        raw = response.choices[0].message.content or ""

        try:
            parsed = _extract_json(raw)
            result = _validate_labels(parsed, resolved_allowed_categories)

            # Preserve known labels
            for key, val in known.items():
                if key in _ALL_LABEL_KEYS and val is not None:
                    result[key] = val

            return result
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            last_error = exc
            if attempt < _MAX_PARSE_RETRIES:
                # Add a retry hint to messages
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": "That was not valid JSON. Please return ONLY a JSON object with the five required keys.",
                })
                continue

    raise RuntimeError(
        f"Failed to parse labeling response after {_MAX_PARSE_RETRIES + 1} attempts. "
        f"Last error: {last_error}"
    )
