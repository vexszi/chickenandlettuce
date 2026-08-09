from state import HospitalState
from nodes import (intake_node, intent_guardrail_extraction_node, document_explainer_node, hospital_policy, other_intent_node, final_response_node, cancel_appointment, reschedule_appointment, book_appointment)
from langgraph.graph import (
    StateGraph,
    START,
    END,
)

builder = StateGraph(HospitalState)

# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------
builder.add_node("intake_node", intake_node)
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
builder.add_edge("intake_node", "intent_guardrail_extraction_node")

builder.add_conditional_edges(
    "intent_guardrail_extraction_node",
    intent_guardrail_extraction_node,
    {
        "document_explainer_node": "document_explainer_node",
        "hospital_policy": "hospital_policy",
        "book_appointment": "book_appointment",
        "cancel_appointment": "cancel_appointment",
        "reschedule_appointment": "reschedule_appointment",
        "other_intent_node": "other_intent_node",
    },
)

builder.add_edge("patient_info", END)
builder.add_edge("department_doc", "appointment_booking")
builder.add_edge("appointment_booking", END)
builder.add_edge("hospital_policy", END)
builder.add_edge("confirmation", END)
builder.add_edge("emergency_response", END)
builder.add_edge("info_check", END)

graph = builder.compile()
