import os
from google import genai
from dotenv import load_dotenv

from state import HospitalState
from schemas import IntentAndExtraction
from tools import translate_incoming_tool, translate_outgoing_tool
from pii_masking import pii_masking_node
from policy import retrieve


load_dotenv()
GEMINI_KEY = os.getenv("flash_key")
client = genai.Client(api_key=GEMINI_KEY)


def intake_node(state: HospitalState) -> dict:
    updates = translate_incoming_tool(state)
    updates["status"] = "awaiting_input"
    return updates


# ------------------------------------------------------------------------------
# Intake router + Process patient's message + Guardrail check + Emergency Check
# Extracts and updates state
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an information collector and safety monitor for a \
hospital front desk assistant. Given the patient's message, determine:
 
1. intent -- what the patient is trying to do (book/cancel/reschedule an \
   appointment, ask about hospital policy, or get a document explained, or other).
2. guardrail_triggered -- true if the message asks for a medical diagnosis, \
   asks something unrelated to the hospital (e.g. general trivia), or \
   otherwise falls outside what a hospital front desk assistant should \
   answer. If true, also write a short, polite output_text redirecting \
   the patient (e.g. "I can't provide a diagnosis, but I can help you book \
   an appointment with a doctor who can."). Also make true if the message \
   contains any swear words, threats, and/or hate speech.
3. 3. emergency_detected -- true if the message suggests a potential medical \
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
   briefly while combining it with any existing symptom information. \
   Otherwise leave null.
7. requested_doctor_name -- any specifically named doctor the patient requests. \
   Otherwise leave null.
8. patient_info -- any personal/insurance details mentioned THIS message \
   only. Leave fields null if not mentioned -- do not guess or fill in \
   placeholder values.
"""
def intent_guardrail_extraction_node(state: HospitalState) -> dict:
    text = state.get("masked_text", "")

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=f"{SYSTEM_PROMPT}\n\nPatient message:\n{text}",
        config={
            "response_mime_type": "application/json",
            "response_schema": IntentAndExtraction,
        },
    )

    result: IntentAndExtraction = response.parsed
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

    prompt = f"""A patient has uploaded a medical document and is asking \
    about it. Explain the relevant contents in clear, simple, patient-friendly \
    language. Avoid jargon, and define any medical terms you must use. Do not \
    diagnose or add medical advice beyond what's written in the document.

    Document content:
    {document_content}

    Patient's question:
    {patient_question if patient_question else "(No specific question -- give a general explanation of the document.)"}"""

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
    )

    return {"output_text": response.text.strip(), "status": "complete"}


# ---------------------------------------------------------------------------
# Hospital policy question node
# ---------------------------------------------------------------------------
def hospital_policy(state: HospitalState) -> dict:
  question = state.get("masked_text", "")
  chunks = retrieve(question, top_k=3)
  context = "\n\n---\n\n".join(chunks)
  prompt = f"""
    You are a hospital policy assistant.
    Answer the user's question ONLY using the hospital policy below.
    If the answer cannot be found in the policy, respond:
      "I couldn't find that information in our hospital policy."

    Rules:
    - Use ONLY the retrieved hospital policy.
    - Do not use outside medical knowledge.
    - If the answer is partially available, answer only the supported portion.
    - If the answer isn't in the retrieved policy, explicitly say you couldn't find it.
    - DO NOT share any information that can be confidential for hospital, patient, or staff information.
      Reply with "Sorry, I cannot provide that information." if the question is about confidential information.
    - Never invent hospital policies.

    Hospital Policy: {context}

    Question: {question}

    Answer:
    """

  response = client.models.generate_content(
    model="gemini-flash-latest",
    contents=prompt,
)
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
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=(
            "You are a hospital front desk assistant. Respond helpfully and "
            "briefly to this message, and if relevant, remind the patient "
            "you can help them book appointments, answer policy questions, "
            f"or explain documents.\n\nMessage: {state.get('masked_text', '')}"
        ),
    )
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
    # Placeholder for booking logic
    return {"status": "complete", "output_text": "Appointment booked successfully."}



# ----------------------------------------------------------------------------
# Cancel Appointment
# Calls: get_appointments and cancel_appointment
# ----------------------------------------------------------------------------
def cancel_appointment(state: HospitalState) -> dict:
    # Placeholder for cancellation logic
    return {"status": "complete", "output_text": "Appointment cancelled successfully."}



# ----------------------------------------------------------------------------
# Reschedule Appointment
# Calls: book_appointment and cancel_appointment
# ----------------------------------------------------------------------------
def reschedule_appointment(state: HospitalState) -> dict:

    return {"status": "complete", "output_text": "Appointment cancelled successfully."}

