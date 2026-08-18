from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shared.clients import call_chat_with_fallbacks

GEN_METADATA_FIELDS = [
    "Incident Type",
    "Mental Health Flag",
    "Response Day of Week",
    "Response Hour",
    "Initial Problem Description",
    "Sector",
    "Council District",
]

_GENERATION_SYSTEM_PROMPT = """\
You are a screenwriter creating 911 call transcripts for a training simulator used by emergency dispatchers. Every transcript must feel like a real recorded call - messy, human, unscripted.

You'll receive incident metadata. That metadata is BACKSTORY - it tells you the world around the call, the severity, what kind of scene this is. But the caller doesn't know any of that cleanly. They know what they saw, what they heard, what they felt. Translate the metadata into a lived experience, don't echo it.

## How a 911 call actually works

The dispatcher picks up. They need a location first - that's the most critical thing. "Where is your emergency?" or "911, what's the address of the emergency?" Then they need to understand what's happening. Then: is anyone hurt, is there a weapon, is the person still there, how many people. They need a callback number. They need the caller's name. They give instructions - apply pressure, lock the door, don't approach, stay on the line.

The caller is not reading from a script. They might:
- Start mid-sentence: "oh god, oh god, there's a man-"
- Blurt out the crisis before the dispatcher even finishes their greeting
- Not know their address: "I'm on, um, it's the street behind the H-E-B, the one with the big church on the corner, I don't know the name"
- Give wrong info and correct themselves: "he went left- no wait, right, toward the highway"
- Repeat themselves when scared: "he's got a knife, he's got a knife, please hurry"
- Go silent, cry, scream, argue with someone else in the room, put the phone down
- Get angry at the dispatcher: "why are you asking me this, just send someone!"
- Be a child who found a parent unconscious
- Be elderly and confused about what they're seeing
- Be whispering because the intruder is still in the house
- Be out of breath from running
- Be in shock and sound flat, almost calm - "yeah, he shot him. he's on the ground."

The dispatcher:
- Uses the caller's name to ground them: "Jeff, stay with me"
- Repeats questions the caller ignored because they were panicking
- Reads back the address to confirm
- Sometimes relays info to units mid-call: "Copy, we have a white sedan eastbound on 5th"
- Gives medical instructions: "turn him on his side", "don't move her"
- Has to pull information out of incoherent callers piece by piece

## Characterize the caller

Before you write, decide who this person is. A tired night-shift worker. A teenager home alone. A mother with kids in the car. An off-duty nurse. A homeless person at a payphone. A store clerk who just got robbed. Let that person's voice, vocabulary, and panic level shape the whole call.

## Length

Mundane calls are short - a noise complaint might be 8 lines. But a domestic violence call where the dispatcher is trying to figure out if the abuser is still in the house while the caller whispers - that's 30-40 lines. An evolving situation like a shooting or a medical emergency where the dispatcher is giving CPR instructions - that could run 50+ lines. Let the situation breathe. Don't wrap it up neatly.

## Format rules
- Labels: DISPATCHER:, CALLER_1:, CALLER_2:, SYSTEM:
- SYSTEM only at the very top as an optional timestamp/automated notice. No SYSTEM lines after the first non-SYSTEM line.
- Output only the transcript. No analysis, no notes, no markdown, no JSON.
- This is the initial 911 call ONLY. No radio logs, no CAD notes, no post-call outcomes, no "units arrived" or "call closed" lines.
- Make up a plausible date and time for the SYSTEM line - use the day-of-week and hour from the metadata as a hint but invent the rest.
"""

_BLOCKED_PHRASES = (
    "arrived on scene",
    "units arrived",
    "unit arrived",
    "units cleared",
    "incident logged",
    "call disposition",
    "no report",
    "call closed",
    "closed out",
    "unit time on scene",
    "report written",
    "cad",
    "radio traffic",
)


def _sanitize_generated_transcript(text: str) -> str:
    """Enforce initial-call format: no post-call summaries/radio style lines."""
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    cleaned: list[str] = []
    saw_non_system = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue

        lower = line.lower()
        if any(p in lower for p in _BLOCKED_PHRASES):
            continue

        is_system = line.startswith("SYSTEM:")
        if is_system and saw_non_system:
            continue

        if not is_system:
            saw_non_system = True

        cleaned.append(line)

    out = "\n".join(cleaned).strip()
    return out if out else text.strip()


def _build_metadata_dict(
    *,
    incident_type: str = "",
    severity: int | str = "",
    mental_health_flag: str = "",
    response_day: str = "",
    response_hour: int | str = "",
    category: str = "",
    initial_problem_description: str = "",
    sector: str = "",
    council_district: str = "",
) -> dict[str, Any]:
    return {
        "Incident Type": str(incident_type),
        "Mental Health Flag": str(mental_health_flag),
        "Response Day of Week": str(response_day),
        "Response Hour": str(response_hour),
        "Initial Problem Description": str(initial_problem_description),
        "Sector": str(sector),
        "Council District": str(council_district),
    }


def generate_synthetic_transcript(
    *,
    apd_id: str,
    incident_type: str = "",
    severity: int | str = "",
    mental_health_flag: str = "",
    response_day: str = "",
    response_hour: int | str = "",
    category: str = "",
    initial_problem_description: str = "",
    sector: str = "",
    council_district: str = "",
    output_dir: Path | None = None,
) -> str:
    """Generate a fictional but realistic 911 transcript from APD metadata.

    The metadata fields represent post-dispatch information and influence
    scenario realism (urgency, tone, stress) but are NOT inserted as
    dispatcher-known facts into the transcript.

    Returns the transcript text. If ``output_dir`` is provided, also writes
    to ``{apd_id}.txt`` inside that directory.
    """
    metadata = _build_metadata_dict(
        incident_type=incident_type,
        severity=severity,
        mental_health_flag=mental_health_flag,
        response_day=response_day,
        response_hour=response_hour,
        category=category,
        initial_problem_description=initial_problem_description,
        sector=sector,
        council_district=council_district,
    )

    severity_desc = {1: "low", 2: "moderate", 3: "high", 4: "critical"}.get(
        int(severity) if str(severity).isdigit() else 0, "unknown"
    )

    user_prompt = f"""\
Write a 911 call transcript for this scenario. Invent a caller, give them a life, and let the emergency interrupt it.

Severity: {severity_desc} ({severity})
Category: {category}
Incident type: {metadata.get("Incident Type", "")}
Problem: {metadata.get("Initial Problem Description", "")}
Mental health related: {metadata.get("Mental Health Flag", "N")}
Day: {metadata.get("Response Day of Week", "")}, around {metadata.get("Response Hour", "")}:00
Sector: {metadata.get("Sector", "")}

Remember - the caller doesn't know the "incident type" or "category." They know what they see, hear, and feel. The metadata is your backstory, not their script.
"""

    response = call_chat_with_fallbacks(
        messages=[
            {"role": "system", "content": _GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    transcript = _sanitize_generated_transcript(content)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"{apd_id}.txt"
        tmp_path = file_path.with_suffix(".txt.tmp")
        tmp_path.write_text(transcript, encoding="utf-8")
        tmp_path.replace(file_path)

    return transcript
