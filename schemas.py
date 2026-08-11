from pydantic import BaseModel
from typing import Optional, Literal


class PatientInfoExtraction(BaseModel):
    """Partial PII extraction -- only fields found in THIS message.
    Every field optional since most messages won't mention most of these."""
    name: Optional[str] = None
    dob: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    insurance_provider: Optional[str] = None
    insurance_id: Optional[str] = None
    address: Optional[str] = None
    gender_pref: Optional[str] = None


class IntentAndExtraction(BaseModel):
    intent: Literal[
        "emergency", "policy_rag", "document_explainer",
        "book_appointment", "cancel_appointment", "reschedule_appointment",
        "other",
    ]
    guardrail_triggered: bool
    output_text: Optional[str] = None  # only when guardrail_triggered is True
    emergency_detected: bool
    symptoms: Optional[str] = None
    requested_doctor_name: Optional[str] = None
    patient_info: Optional[PatientInfoExtraction] = None
    emergency_reason: Optional[str] = None