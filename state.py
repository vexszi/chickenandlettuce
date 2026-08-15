import importlib
import importlib.util
import operator
from typing import TypedDict, Annotated, Literal, Optional

if importlib.util.find_spec("langgraph.graph.message") is not None:
    add_messages = importlib.import_module("langgraph.graph.message").add_messages
else:
    add_messages = object()

class PatientInfo(TypedDict, total=False):
    name: str
    dob: str
    phone: str
    email: str
    address: str
    gender_pref: Optional[str]
    insurance_provider: str
    insurance_id: str
    insurance_card_url: Optional[str]
    

class HospitalState(TypedDict):
    # --- session / identity ---
    is_first_time: Optional[bool]
    is_logged_in: bool
    patient_id: Optional[str]
    auth_method: Optional[Literal["email", "pin", "fingerprint"]]
    thread_id: str 

    # --- conversation ---
    messages: Annotated[list, add_messages]
    # Rolling log of the conversation so far, PII-masked patient turns and
    # translated assistant turns, e.g. "Patient: ...", "Assistant: ...".
    # Every Gemini-calling node in nodes.py reads the tail of this to give
    # the model actual short-term memory -- without it, every prompt only
    # ever saw the current message in isolation (that was the cause of
    # answers feeling hardcoded/repetitive/context-blind). Trimmed to the
    # last N entries in main.py after each turn so it doesn't grow forever.
    conversation_log: Annotated[list[str], operator.add]
    preferred_language: str
    detected_language: Optional[str]

    # --- multimodal tool outputs ---
    translated_text: Optional[str]
    
    # ocr_text ACCUMULATES across multiple uploaded docs in a session,
    # so it uses operator.add as its reducer instead of overwrite-on-write.
    ocr_text: Annotated[list[str], operator.add]
    translated_doc_count: int

    # --- privacy ---
    masked_text: str
    pii_map: dict
    masked_document_text: Optional[str]

    # --- routing ---
    intent: Literal[
        "emergency", "policy_rag", "document_explainer",
        "book_appointment", "cancel_appointment", "reschedule_appointment",
        "other"
    ]
    # ^ NOTE: "emergency" is kept here for schema parity with
    # schemas.IntentAndExtraction, but routing normally short-circuits to
    # final_response_node via status=="blocked" before intent is even
    # checked (see route_intent in nodes.py + emergency_detected in the
    # system prompt). The "emergency" branch in hospital_graph.py's
    # conditional edges is a safety net for the rare case the model
    # returns intent="emergency" without also setting emergency_detected.
    guardrail_triggered: bool
    emergency_detected: bool
    emergency_reason: Optional[str]

    # --- RAG / document explainer ---
    retrieved_chunks: list

    # --- appointment flow ---
    patient_info: PatientInfo
    missing_pii_fields: list[str]
    old_patient_follow_up: bool
    symptoms: Optional[str]
    department: Optional[str]
    requested_doctor_name: Optional[str]   # was doctor_preference -- now ONLY a specifically named doctor request
    booking_checklist: dict[str, bool]

    existing_appointment_id: Optional[str]   #appointment currently being cancelled or rescheduled
    held_appointment_id: Optional[str]     #new slot for rescheduling

    available_slots: list[dict]
    selected_slot: Optional[dict]
    confirmed_appointment: Optional[dict]
    new_account_password: Optional[str]
    available_appointments: list[dict]

    # --- output ---
    output_text: str
    status: Literal["in_progress", "awaiting_input", "complete", "blocked"]