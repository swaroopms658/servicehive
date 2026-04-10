from __future__ import annotations

from pydantic import BaseModel

from app.prompts import INTENT_SYSTEM_PROMPT, SLOT_EXTRACTION_SYSTEM_PROMPT
from app.state import IntentLabel


class IntentResult(BaseModel):
    label: IntentLabel


class LeadDetails(BaseModel):
    name: str = ""
    email: str = ""
    platform: str = ""


HIGH_INTENT_HINTS = (
    "sign up",
    "signup",
    "start trial",
    "free trial",
    "want to try",
    "ready to buy",
    "interested in pro",
    "get started",
    "book a demo",
    "start with pro",
)


def detect_intent(llm, messages) -> IntentLabel:
    latest_user_message = next(
        (message.content for message in reversed(messages) if message.type == "human"),
        "",
    )
    lowered = latest_user_message.lower()
    if any(hint in lowered for hint in HIGH_INTENT_HINTS):
        return "high_intent_lead"

    structured_llm = llm.with_structured_output(IntentResult)
    result = structured_llm.invoke(
        [
            ("system", INTENT_SYSTEM_PROMPT),
            (
                "human",
                f"Conversation:\n{messages}\n\nLatest user message:\n{latest_user_message}",
            ),
        ]
    )
    return result.label


def extract_lead_details(llm, messages) -> LeadDetails:
    latest_user_message = next(
        (message.content for message in reversed(messages) if message.type == "human"),
        "",
    )
    structured_llm = llm.with_structured_output(LeadDetails)
    return structured_llm.invoke(
        [
            ("system", SLOT_EXTRACTION_SYSTEM_PROMPT),
            (
                "human",
                f"Conversation:\n{messages}\n\nLatest user message:\n{latest_user_message}",
            ),
        ]
    )
