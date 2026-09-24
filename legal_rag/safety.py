"""Lightweight detection of urgent situations so we can surface helplines first."""

from __future__ import annotations

import re

_EMERGENCY_PATTERNS: dict[str, list[str]] = {
    "self_harm": [
        r"\bsuicid", r"\bkill (my|him|her)self\b", r"\bend (my|this) life\b",
        r"\bwant to die\b", r"आत्महत्या", r"मरना चाहत",
    ],
    "violence": [
        r"\b(beating|beats|beat|hitting|hits|attacking) me\b", r"\bdomestic violence\b",
        r"\bthreaten(ing|ed)? to kill\b", r"\bin danger\b", r"\bkidnapp", r"\bbeing followed\b",
        r"\braped?\b", r"\bsexual(ly)? assault", r"मारपीट", r"मार रहा", r"बलात्कार",
    ],
    "child": [r"\bchild (abuse|marriage|labou?r)\b", r"बाल विवाह"],
    "cyber_fraud": [
        r"\b(money|amount|rs\.?|₹)\s?.{0,20}(debited|deducted|stolen)\b",
        r"\bupi fraud\b", r"\bonline fraud\b", r"\bscam(med)?\b", r"\bcyber ?fraud\b",
        r"\bdigital arrest\b",
    ],
}

_HELPLINES = {
    "self_harm": "If you or someone you know is thinking about self-harm, please call Tele-MANAS at "
    "14416 (free, 24x7) or 112 in an emergency.",
    "violence": "If you are in immediate danger, call 112 (emergency) now. Women can also call the "
    "Women Helpline 181.",
    "child": "For a child in danger or distress, call CHILDLINE 1098 or 112.",
    "cyber_fraud": "For online financial fraud, call 1930 immediately and report at "
    "https://cybercrime.gov.in — acting within the first few hours improves the chance of "
    "freezing the money.",
}


def detect_urgency(text: str) -> list[str]:
    """Return helpline notices that apply to ``text`` (empty if none)."""
    lowered = text.lower()
    notices = []
    for kind, patterns in _EMERGENCY_PATTERNS.items():
        if any(re.search(p, lowered) for p in patterns):
            notices.append(_HELPLINES[kind])
    return notices
