import operator
import os
from typing import TypedDict, Annotated, Literal, Optional
from langgraph.graph.message import add_messages
from google import genai

from dotenv import load_dotenv
load_dotenv()
GEMINI_KEY = os.getenv("flash_key")

class PatientInfo(TypedDict, total=False):
    name: str
    dob: str
    phone: str
    email: str
    insurance_provider: str
    insurance_id: str
    insurance_card_url: Optional[str]
    address: str
    gender_pref: Optional[str]


class HospitalState(TypedDict):
    # --- session / identity ---
    thread_id: str
    is_first_time: Optional[bool]
    is_logged_in: bool
    patient_id: Optional[str]
    auth_method: Optional[Literal["email", "pin", "fingerprint"]]

    # --- conversation ---
    messages: Annotated[list, add_messages]
    conversation_summary: str
    preferred_language: str

    # --- multimodal input (file paths, set by whatever collects the input) ---
    raw_input_type: Literal["text", "audio", "image", "pdf"]
    audio_file_path: Optional[str]
    document_file_path: Optional[str]

    # --- multimodal tool outputs ---
    transcribed_text: Optional[str] 
    translated_text: Optional[str]
    
    # ocr_text ACCUMULATES across multiple uploaded docs in a session,
    # so it uses operator.add as its reducer instead of overwrite-on-write.
    ocr_text: Annotated[list[str], operator.add]

    # --- privacy ---
    masked_text: str
    pii_map: dict

    # --- routing ---
    intent: Literal[
        "emergency", "policy_rag", "document_explainer",
        "book_appointment", "cancel_appointment", "reschedule_appointment",
        "other"
    ]
    guardrail_triggered: bool
    emergency_detected: bool

    # --- RAG / document explainer ---
    retrieved_chunks: list
    document_extraction: Optional[dict]
    document_explanation: Optional[str]

    # --- appointment flow ---
    patient_info: PatientInfo
    missing_pii_fields: list[str]
    concern_type: Optional[Literal["new", "follow_up"]]
    symptoms: Optional[str]
    department: Optional[str]
    doctor_preference: Optional[str]
    existing_appointment_id: Optional[str]
    held_appointment_id: Optional[str]
    available_slots: list[dict]
    selected_slot: Optional[dict]
    confirmed_appointment: Optional[dict]

    # --- output ---
    output_text: str
    output_audio_url: Optional[str]
    confirmation_pdf_url: Optional[str]
    status: Literal["in_progress", "awaiting_input", "complete", "blocked"]