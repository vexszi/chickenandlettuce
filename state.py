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
    conversation_summary: str
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
    candidate_doctors: list[str]
    requested_doctor_name: Optional[str]   # was doctor_preference -- now ONLY a specifically named doctor request
    booking_checklist: dict[str, bool]

    existing_appointment_id: Optional[str]   #appointment currently being cancelled or rescheduled
    held_appointment_id: Optional[str]     #new slot for rescheduling

    available_slots: list[dict]
    selected_slot: Optional[dict]
    confirmed_appointment: Optional[dict]

    # --- output ---
    output_text: str
    status: Literal["in_progress", "awaiting_input", "complete", "blocked"]