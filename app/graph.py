from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.intent import extract_lead_details, detect_intent
from app.prompts import RESPONSE_SYSTEM_PROMPT
from app.rag import retrieve_context
from app.state import AgentState
from app.tools import mock_lead_capture


class FallbackStructuredLLM:
    def __init__(self, primary, fallback=None):
        self.primary = primary
        self.fallback = fallback

    def invoke(self, messages):
        try:
            return self.primary.invoke(messages)
        except Exception as primary_error:
            if self.fallback is None:
                raise primary_error
            print(f"Primary LLM failed, falling back to Groq: {primary_error}")
            return self.fallback.invoke(messages)


class FallbackLLM:
    def __init__(self, primary, fallback=None):
        self.primary = primary
        self.fallback = fallback

    def invoke(self, messages):
        try:
            return self.primary.invoke(messages)
        except Exception as primary_error:
            if self.fallback is None:
                raise primary_error
            print(f"Primary LLM failed, falling back to Groq: {primary_error}")
            return self.fallback.invoke(messages)

    def with_structured_output(self, schema):
        primary = self.primary.with_structured_output(schema)
        fallback = None
        if self.fallback is not None:
            fallback = self.fallback.with_structured_output(schema)
        return FallbackStructuredLLM(primary=primary, fallback=fallback)


def build_llm() -> FallbackLLM:
    primary_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    primary = ChatOpenAI(model=primary_model, temperature=0)

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return FallbackLLM(primary=primary)

    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    fallback = ChatGroq(model=groq_model, temperature=0)
    return FallbackLLM(primary=primary, fallback=fallback)


def classify_intent(state: AgentState) -> dict[str, Any]:
    llm = build_llm()
    intent = detect_intent(llm, state["messages"])
    return {"intent": intent}


def retrieve_knowledge(state: AgentState) -> dict[str, Any]:
    latest_user_message = _latest_user_message(state)
    if state.get("intent") == "greeting":
        return {"retrieved_context": ""}
    return {"retrieved_context": retrieve_context(latest_user_message)}


def collect_lead_info(state: AgentState) -> dict[str, Any]:
    if state.get("intent") != "high_intent_lead" and not _has_partial_lead(state):
        return {}

    llm = build_llm()
    details = extract_lead_details(llm, state["messages"])

    updates: dict[str, Any] = {}
    if details.name and not state.get("lead_name"):
        updates["lead_name"] = details.name
    if details.email and not state.get("lead_email"):
        updates["lead_email"] = details.email
    if details.platform and not state.get("creator_platform"):
        updates["creator_platform"] = details.platform
    return updates


def maybe_capture_lead(state: AgentState) -> dict[str, Any]:
    if state.get("lead_capture_done"):
        return {}

    if _missing_fields(state):
        return {}

    result = mock_lead_capture(
        state["lead_name"],
        state["lead_email"],
        state["creator_platform"],
    )
    return {"lead_capture_done": True, "capture_result": result}


def respond(state: AgentState) -> dict[str, Any]:
    llm = build_llm()
    system_message = SystemMessage(content=RESPONSE_SYSTEM_PROMPT)
    latest_user_message = _latest_user_message(state)

    missing = _missing_fields(state)
    if state.get("intent") == "high_intent_lead" or _has_partial_lead(state):
        if state.get("lead_capture_done"):
            reply = (
                f"{state['capture_result']}. You're all set. "
                "A team member can follow up with onboarding details."
            )
            return {"reply": reply, "messages": [AIMessage(content=reply)]}

        if missing:
            readable = _format_missing_fields(missing)
            prompt = (
                "The user is a qualified lead. "
                f"Known values: name={state.get('lead_name', '')}, "
                f"email={state.get('lead_email', '')}, "
                f"platform={state.get('creator_platform', '')}. "
                f"Ask only for the missing fields: {readable}."
            )
            response = llm.invoke(
                [
                    system_message,
                    ("human", prompt),
                ]
            )
            return {"reply": response.content, "messages": [AIMessage(content=response.content)]}

    response = llm.invoke(
        [
            system_message,
            (
                "human",
                f"Retrieved knowledge context:\n{state.get('retrieved_context', '')}\n\n"
                f"Latest user message:\n{latest_user_message}",
            ),
        ]
    )
    return {"reply": response.content, "messages": [AIMessage(content=response.content)]}


def route_after_collection(state: AgentState) -> str:
    if not _missing_fields(state) and not state.get("lead_capture_done"):
        return "capture"
    return "respond"


def _missing_fields(state: AgentState) -> list[str]:
    missing: list[str] = []
    if not state.get("lead_name"):
        missing.append("name")
    if not state.get("lead_email"):
        missing.append("email")
    if not state.get("creator_platform"):
        missing.append("creator platform")
    return missing


def _has_partial_lead(state: AgentState) -> bool:
    return any(
        [
            state.get("lead_name"),
            state.get("lead_email"),
            state.get("creator_platform"),
        ]
    )


def _latest_user_message(state: AgentState) -> str:
    return next(
        (message.content for message in reversed(state["messages"]) if message.type == "human"),
        "",
    )


def _format_missing_fields(fields: list[str]) -> str:
    if len(fields) == 1:
        return fields[0]
    return ", ".join(fields[:-1]) + f", and {fields[-1]}"


def build_graph():
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("classify_intent", classify_intent)
    graph_builder.add_node("retrieve_knowledge", retrieve_knowledge)
    graph_builder.add_node("collect_lead_info", collect_lead_info)
    graph_builder.add_node("capture", maybe_capture_lead)
    graph_builder.add_node("respond", respond)

    graph_builder.add_edge(START, "classify_intent")
    graph_builder.add_edge("classify_intent", "retrieve_knowledge")
    graph_builder.add_edge("retrieve_knowledge", "collect_lead_info")
    graph_builder.add_conditional_edges(
        "collect_lead_info",
        route_after_collection,
        {
            "capture": "capture",
            "respond": "respond",
        },
    )
    graph_builder.add_edge("capture", "respond")
    graph_builder.add_edge("respond", END)

    return graph_builder.compile(checkpointer=MemorySaver())


def invoke_turn(graph, thread_id: str, user_message: str) -> AgentState:
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result
