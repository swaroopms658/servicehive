from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


IntentLabel = Literal["greeting", "product_pricing_inquiry", "high_intent_lead"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: IntentLabel
    retrieved_context: str
    lead_name: str
    lead_email: str
    creator_platform: str
    lead_capture_done: bool
    capture_result: str
    reply: str
