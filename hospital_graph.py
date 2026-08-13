from state import HospitalState
from nodes import (intake_node, intent_guardrail_extraction_node, document_explainer_node, hospital_policy, other_intent_node, final_response_node, cancel_appointment, reschedule_appointment, book_appointment, route_intent)
from langgraph.graph import (
    StateGraph,
    START,
    END,
)
from pii_masking import pii_masking_node

builder = StateGraph(HospitalState)

# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------
builder.add_node("intake_node", intake_node)
builder.add_node("pii_masking_node", pii_masking_node)
builder.add_node("intent_guardrail_extraction_node", intent_guardrail_extraction_node)
builder.add_node("document_explainer_node", document_explainer_node)
builder.add_node("hospital_policy", hospital_policy)
builder.add_node("other_intent_node", other_intent_node)
builder.add_node("book_appointment", book_appointment)
builder.add_node("cancel_appointment", cancel_appointment)
builder.add_node("reschedule_appointment", reschedule_appointment)
builder.add_node("final_response_node", final_response_node)

# ---------------------------------------------------------------------------
# EDGES
# ---------------------------------------------------------------------------
builder.add_edge(START, "intake_node")
builder.add_edge("intake_node", "pii_masking_node")
builder.add_edge("pii_masking_node", "intent_guardrail_extraction_node")

builder.add_conditional_edges(
    "intent_guardrail_extraction_node",
    route_intent,
    {
        "document_explainer": "document_explainer_node",
        "policy_rag": "hospital_policy",
        "book_appointment": "book_appointment",
        "cancel_appointment": "cancel_appointment",
        "reschedule_appointment": "reschedule_appointment",
        "other": "other_intent_node",
        "emergency": "final_response_node",
        "final_response_node": "final_response_node",
    },
)

builder.add_edge("document_explainer_node", "final_response_node")
builder.add_edge("hospital_policy", "final_response_node")
builder.add_edge("other_intent_node", "final_response_node")
builder.add_edge("book_appointment", "final_response_node")
builder.add_edge("cancel_appointment", "final_response_node")
builder.add_edge("reschedule_appointment", "final_response_node")

builder.add_edge("final_response_node", END)

graph = builder.compile()
