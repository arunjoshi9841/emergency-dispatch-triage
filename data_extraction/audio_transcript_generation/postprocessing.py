from __future__ import annotations

from shared.clients import call_chat_with_fallbacks

POSTPROCESS_SYSTEM_PROMPT = """\
You are a 911 transcript editor.

Goal: produce a clean, faithful transcript suitable for downstream ML labeling.

Rules:
1) Keep meaning faithful. Do not invent details.
2) Correct obvious ASR errors only when correction is very clear.
3) Label each line with one of:
   - SYSTEM: automated announcements
   - DISPATCHER: call taker / operator
   - CALLER_1, CALLER_2, ... for non-dispatch participants
4) If uncertain, keep best-effort text and add [unclear] minimally.
5) Do not output timestamps unless present in input.
6) Return transcript text only, no commentary.
"""


def postprocess_transcript(raw_text: str) -> str:
    """Clean and add speaker labels to a raw Whisper transcript.

    Uses OpenRouter LLM with model fallbacks.
    """
    if not raw_text:
        return ""
    response = call_chat_with_fallbacks(
        messages=[
            {"role": "system", "content": POSTPROCESS_SYSTEM_PROMPT},
            {"role": "user", "content": raw_text},
        ],
    )
    if not response or not response.choices:
        raise RuntimeError(f"OpenRouter returned an empty response (choices={getattr(response, 'choices', None)})")
    return (response.choices[0].message.content or "").strip()
