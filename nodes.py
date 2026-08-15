import os
from google import genai
from dotenv import load_dotenv

from state import HospitalState
from schemas import IntentAndExtraction
from tools import translate_incoming_tool, translate_outgoing_tool
from policy import retrieve
from book import (book_appointment_flow,_upcoming_appointments, confirm_booking, find_available_slots,get_appointments)
from auth import delete_appointment


load_dotenv()
GEMINI_KEY = os.getenv("flash_key")
client = genai.Client(api_key=GEMINI_KEY)

# Centralized so a future model swap is a one-line change instead of a
# find-and-replace across every node.
GEMINI_MODEL = "gemini-3.5-flash-lite"

# Shown to the patient (translated like any other output_text) whenever a
# Gemini call fails outright -- quota exhausted, timeout, network error,
# malformed response, etc. Keeps a single bad API call from surfacing as a
# raw 500 / "Failed to fetch" in the browser.
GEMINI_ERROR_TEXT = (
    "Sorry, I'm having trouble processing that right now. Please try again "
    "in a moment, or call our front desk directly if it's urgent."
)


def _safe_generate_content(**kwargs):
    """Wraps client.models.generate_content so a Gemini-side failure (quota,
    timeout, malformed schema response, etc.) never becomes an unhandled
    exception that 500s the whole /api/chat/message request. Returns the
    response object on success, or None on failure -- callers check for
    None and fall back to GEMINI_ERROR_TEXT."""
    try:
        return client.models.generate_content(**kwargs)
    except Exception as e:
        print(f"[Gemini call failed] {e}")
        return None


def _recent_history(state: HospitalState) -> str:
    """Formats the last few turns of conversation_log for inclusion in a
    prompt. Returns "" (not a placeholder string) when there's no history
    yet, so first-turn prompts don't have to special-case an awkward
    "(no history)" line."""
    log = state.get("conversation_log", [])
    if not log:
        return ""
    return "Recent conversation so far:\n" + "\n".join(log) + "\n"


def intake_node(state: HospitalState) -> dict:
    updates = translate_incoming_tool(state)
    updates["status"] = "awaiting_input"
    return updates


# ------------------------------------------------------------------------------
# Intake router + Process patient's message + Guardrail check + Emergency Check
# Extracts and updates state
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the front desk assistant for a hospital -- warm, \
efficient, and human, the way a good in-person receptionist is. You are \
also this turn's information collector and safety monitor. Given the \
recent conversation and the patient's newest message, determine:

1. intent -- what the patient is trying to do (book/cancel/reschedule an \
   appointment, ask about hospital policy, or get a document explained, or other).
2. guardrail_triggered -- true if the message asks for a medical diagnosis, \
   asks something unrelated to the hospital (e.g. general trivia), or \
   otherwise falls outside what a hospital front desk assistant should \
   answer. Also true if the message contains swear words, threats, and/or \
   hate speech. If true, write output_text: a short, warm redirect in your \
   own words -- vary the phrasing turn to turn, don't reuse the same \
   sentence you've used earlier in this conversation. Acknowledge what \
   they actually asked before redirecting, don't just paste a generic \
   disclaimer.
3. emergency_detected -- true if the message suggests a potential medical \
   emergency (severe pain, difficulty breathing, heavy bleeding, chest pain, \
   suicidal ideation, etc). Err on the side of caution.
4. emergency_reason -- if emergency_detected is true, briefly state in one \
   sentence what about the message indicates an emergency (for internal \
   logging). Otherwise leave null.
5. If emergency_detected is true, also write output_text: a calm, \
   supportive message telling the patient this assistant cannot help with \
   emergencies, to call 911 or local emergency services immediately, and \
   to seek immediate medical attention. Do not offer appointment \
   scheduling. Keep it under 80 words.
6. symptoms -- if the patient describes a new medical concern, summarize it \
   briefly while combining it with any existing symptom information from \
   the conversation so far. Otherwise leave null.
7. requested_doctor_name -- any specifically named doctor the patient requests. \
   Otherwise leave null.
8. patient_info -- any personal/insurance details mentioned THIS message \
   only. Leave fields null if not mentioned -- do not guess or fill in \
   placeholder values. Check the recent conversation first: if the patient \
   already gave a field earlier in this session, don't ask for it again \
   and don't flag it as newly mentioned unless they're correcting it.

Use the recent conversation to stay consistent -- don't contradict, repeat \
verbatim, or ask again for something already covered earlier in this \
session.
"""
def intent_guardrail_extraction_node(state: HospitalState) -> dict:
    text = state.get("masked_text", "")
    history = _recent_history(state)

    response = _safe_generate_content(
        model=GEMINI_MODEL,
        contents=f"{SYSTEM_PROMPT}\n\n{history}Patient's newest message:\n{text}",
        config={
            "response_mime_type": "application/json",
            "response_schema": IntentAndExtraction,
        },
    )

    # response.parsed can come back None even on a "successful" call, if
    # Gemini's output didn't cleanly match IntentAndExtraction. Treat that
    # the same as a hard failure rather than letting result.intent etc.
    # below throw an AttributeError.
    result: IntentAndExtraction = response.parsed if response else None
    if result is None:
        return {"output_text": GEMINI_ERROR_TEXT, "status": "blocked"}

    updates: dict = {
        "intent": result.intent,
        "guardrail_triggered": result.guardrail_triggered,
        "emergency_detected": result.emergency_detected,
    }

    if result.emergency_detected:
        updates["emergency_reason"] = result.emergency_reason
        updates["output_text"] = result.output_text
        updates["status"] = "blocked"
    elif result.guardrail_triggered:
        updates["output_text"] = result.output_text
        updates["status"] = "blocked"

    if result.symptoms:
        updates["symptoms"] = result.symptoms

    if result.requested_doctor_name:
        updates["requested_doctor_name"] = result.requested_doctor_name

    if result.patient_info:
        existing = state.get("patient_info", {})
        new_fields = result.patient_info.model_dump(exclude_none=True)
        pii_map = state.get("pii_map", {})

        # if the value Gemini returned is a placeholder we recognize,
        # swap it back to the real value. otherwise leave it as-is.
        unmasked_fields = {
            key: pii_map.get(value, value)
            for key, value in new_fields.items()
        }

        updates["patient_info"] = {**existing, **unmasked_fields}

    return updates


#----------------------------------------------------------------------------
# Langgraph helper function, not a node. Routes to the next node 
# based on the intent extracted.
# ----------------------------------------------------------------------------
def route_intent(state: HospitalState) -> str:
    if state.get("status") == "blocked":
        return "final_response_node"
    return state["intent"]

#----------------------------------------------------------------------------
# Document explainer node
# ----------------------------------------------------------------------------
def document_explainer_node(state: HospitalState) -> dict:
    document_content = state.get("masked_document_text", "")
    patient_question = state.get("masked_text", "")
    history = _recent_history(state)

    prompt = f"""A patient has uploaded a medical document and is asking \
    about it. Explain the relevant contents in clear, simple, patient-friendly \
    language, like a caring front desk staffer walking them through it out \
    loud -- not a form letter. Avoid jargon, and define any medical terms \
    you must use. Do not diagnose or add medical advice beyond what's \
    written in the document. If the recent conversation shows they already \
    asked about part of this document, build on that instead of repeating \
    your earlier explanation.

    {history}
    Document content:
    {document_content}

    Patient's question:
    {patient_question if patient_question else "(No specific question -- give a general explanation of the document.)"}"""

    response = _safe_generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    if response is None:
        return {"output_text": GEMINI_ERROR_TEXT, "status": "complete"}

    return {"output_text": response.text.strip(), "status": "complete"}


# ---------------------------------------------------------------------------
# Hospital policy question node
# ---------------------------------------------------------------------------
def hospital_policy(state: HospitalState) -> dict:
  question = state.get("masked_text", "")
  history = _recent_history(state)
  chunks = retrieve(question, top_k=3)
  context = "\n\n---\n\n".join(chunks)
  prompt = f"""
    You are a hospital front desk assistant answering a policy question.
    Answer the patient's question ONLY using the hospital policy below,
    but say it the way a helpful staffer would explain it out loud --
    plain language, not a copy-pasted policy excerpt. If the recent
    conversation already covered related ground, build on it naturally
    instead of re-explaining from scratch.
    If the answer cannot be found in the policy, say so plainly, in your
    own words -- don't always use the identical sentence every time.

    Rules:
    - Use ONLY the retrieved hospital policy below -- never outside medical
      knowledge, and never invented policy.
    - If the answer is only partially available, answer the supported
      portion and say what you couldn't confirm.
    - DO NOT share anything confidential about the hospital, patients, or
      staff. If the question is about confidential information, decline
      clearly but politely, in your own words.

    {history}
    Hospital Policy: {context}

    Question: {question}

    Answer:
    """

  response = _safe_generate_content(
    model=GEMINI_MODEL,
    contents=prompt,
  )
  if response is None:
    return {
        "output_text": GEMINI_ERROR_TEXT,
        "retrieved_chunks": chunks,
        "status": "complete",
    }

  return {
     "output_text": response.text.strip(),
     "retrieved_chunks": chunks,
     "status": "complete"
 }


# ---------------------------------------------------------------------------
# Catch-all for messages that don't fit a real intent. NOT the same 
# as guardrail_triggered, this is for harmless off-flow stuff, 
# not off-topic/diagnosis requests.
# ---------------------------------------------------------------------------
def other_intent_node(state: HospitalState) -> dict:
    history = _recent_history(state)
    response = _safe_generate_content(
        model=GEMINI_MODEL,
        contents=(
            "You are a hospital front desk assistant. Respond naturally and "
            "briefly to this message, like a real receptionist chatting with "
            "someone at the desk. Only mention that you can help book "
            "appointments, answer policy questions, or explain documents if "
            "it's genuinely relevant right now -- check the recent "
            "conversation first, and don't repeat that reminder if you've "
            "already said it earlier in this session.\n\n"
            f"{history}"
            f"Patient's newest message: {state.get('masked_text', '')}"
        ),
    )
    if response is None:
        return {"output_text": GEMINI_ERROR_TEXT, "status": "complete"}

    return {"output_text": response.text, "status": "complete"}


# ---------------------------------------------------------------------------
# Runs every turn, last. Translates output_text back to the patient's
# detected language (no-ops if already English).
# ---------------------------------------------------------------------------
def final_response_node(state: HospitalState) -> dict:
    return translate_outgoing_tool(state)

# ----------------------------------------------------------------------------
# Book Appointment
# Calls: save_appointment and book.py
# ----------------------------------------------------------------------------
def book_appointment(state: HospitalState) -> dict:
    if not state.get("patient_id") and "is_first_time" not in state:
        return {
            "output_text": "Are you a new patient, or have you visited us before?",
            "status": "needs_patient_type",
        }
    if state.get("patient_id") and not state.get("is_first_time") and "old_patient_follow_up" not in state:
        return {
            "output_text": "Is this a follow-up on an existing concern, or a new visit?",
            "status": "needs_visit_type",
        }
    return book_appointment_flow(state)


# ----------------------------------------------------------------------------
# Cancel Appointment
# Calls: get_appointments and cancel_appointment
# ----------------------------------------------------------------------------
def cancel_appointment(state: HospitalState) -> dict:
    patient_id = state.get("patient_id")
    if not patient_id:
        return {
            "output_text": "You'll need to log in first before cancelling an appointment.",
            "status": "blocked",
        }

    appointment_id = state.get("existing_appointment_id")

    # Step 1: no selection yet -- show them what they have
    if not appointment_id:
        upcoming = _upcoming_appointments(patient_id)
        if not upcoming:
            return {
                "output_text": "You don't have any upcoming appointments to cancel.",
                "status": "complete",
            }
        return {
            "available_appointments": upcoming,   # frontend renders these as buttons
            "status": "awaiting_input",
        }

    # Step 2: they picked one -- cancel it
    deleted = delete_appointment(patient_id, appointment_id)
    if not deleted:
        return {
            "output_text": "I couldn't find that appointment -- it may have already been cancelled.",
            "status": "complete",
        }

    return {
        "output_text": (
            f"Your appointment with {deleted['doctor']} on {deleted['date']} "
            f"has been cancelled."
        ),
        "status": "complete",
    }



# ----------------------------------------------------------------------------
# Reschedule Appointment
# Calls: functions from book_appointment and cancel_appointment
# ----------------------------------------------------------------------------
def reschedule_appointment(state: HospitalState) -> dict:
    patient_id = state.get("patient_id")
    if not patient_id:
        return {
            "output_text": "You'll need to log in first before rescheduling an appointment.",
            "status": "blocked",
        }
    appointment_id = state.get("existing_appointment_id")

    if not appointment_id:
            upcoming = _upcoming_appointments(patient_id)
            if not upcoming:
                return {
                    "output_text": "You don't have any upcoming appointments to reschedule.",
                    "status": "complete",
                }
            return {
                "available_appointments": upcoming,   # frontend renders these as buttons
                "status": "awaiting_input",
            }

    old_department = state.get("department")
    old_doctor = state.get("requested_doctor_name")
    if not old_department or not old_doctor:
        matching = next(
            (a for a in get_appointments(patient_id)
            if a["appointment_id"] == appointment_id),
            None,
        )
        if matching:
            old_department = matching["department"]
            old_doctor = matching["doctor"]

    state = {**state, "department": old_department, "requested_doctor_name": old_doctor}

    available = state.get("available_slots")
    if not available:
        available = find_available_slots(state)
        if not available:
            return {
                "output_text": "Sorry, your doctor has no available slots right now. Please call our front desk directly.",
                "department": old_department,
                "requested_doctor_name": old_doctor,
                "status": "complete",
            }
        return {
            "available_slots": available,
            "department": old_department,
            "requested_doctor_name": old_doctor,
            "status": "awaiting_input",
        }

    selected_slot = state.get("selected_slot")
    if not selected_slot:
        return {
                    "output_text": "Please select a slot.",
                    "status": "awaiting_input",
                }
    confirmation = confirm_booking(state)
    new_appt = confirmation["confirmed_appointment"]["appointment_id"]
    
    deleted = delete_appointment(patient_id, appointment_id)
    if not deleted:
        return {
            "output_text": "I couldn't find the appointment to cancel--it may already have been cancelled.",
            "status": "complete",
        }
    
    return {
        "output_text": (
            f"Your appointment with {deleted['doctor']} on {deleted['date']} "
            f"has been rescheduled."
        ),
        "status": "complete",
        "held_appointment_id": new_appt
    }