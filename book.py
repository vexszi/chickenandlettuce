# book.py
import os
import json
from google import genai
from dotenv import load_dotenv

from state import HospitalState
from policy import retrieve

load_dotenv()
GEMINI_KEY = os.getenv("flash_key")
client = genai.Client(api_key=GEMINI_KEY)


# ---------------------------------------------------------------------------
# Figures out which of your 3 flows we're in.
# ---------------------------------------------------------------------------
def _patient_type(state: HospitalState) -> str:
    if state.get("is_first_time"):
        return "new_patient"
    if state.get("old_patient_follow_up"):
        return "old_followup"
    return "old_new_visit"


# ---------------------------------------------------------------------------
# Builds the checklist ONCE per booking. If it already exists in state,
# just reuse it -- don't wipe out progress from earlier turns.
# ---------------------------------------------------------------------------
def _init_checklist(state: HospitalState) -> dict:
    if state.get("booking_checklist"):
        return state["booking_checklist"]

    patient_type = _patient_type(state)
    if patient_type == "new_patient":
        return {"pii_collection": False, "symptoms_and_gender": False,
                "department": False, "doctor": False}
    elif patient_type == "old_new_visit":
        return {"symptoms_and_gender": False, "department": False, "doctor": False}
    else:  # old_followup
        return {"which_appointment": False, "new_symptoms": False}


# ---------------------------------------------------------------------------
# Same LLM prompt as your pasted patient_info() -- turns a list of missing
# things into one natural question, asking for up to 2 at a time.
# ---------------------------------------------------------------------------
def _ask_for(missing: list[str]) -> str:
    to_ask = missing[:2]
    prompt = f"""
    You are a hospital receptionist.
    Generate ONE natural message asking the patient for the following missing information ONLY:
    {to_ask}

    Rules
    - Ask naturally for all items above, in one short friendly message.
    - Do NOT mention or ask about anything not in the list above.
    - Return ONLY the message.
    """
    response = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
    return response.text.strip()


# ---------------------------------------------------------------------------
# Handles the "old patient, follow-up" flow -- just 2 things to check.
# ---------------------------------------------------------------------------
def _handle_followup(state: HospitalState, checklist: dict) -> dict:
    missing = []

    if state.get("existing_appointment_id"):
        checklist["which_appointment"] = True
    else:
        checklist["which_appointment"] = False
        missing.append("which past appointment you're following up on")

    if state.get("symptoms") is not None:
        checklist["new_symptoms"] = True
    else:
        checklist["new_symptoms"] = False
        missing.append("whether you have any new symptoms since your last visit")

    if missing:
        return {
            "booking_checklist": checklist,
            "output_text": _ask_for(missing),
            "status": "awaiting_input",
        }

    return {"booking_checklist": checklist, "status": "complete"}


# ---------------------------------------------------------------------------
# Handles "new patient" and "old patient, new visit" -- these share every
# step except PII (new patients only).
# ---------------------------------------------------------------------------
def _handle_new_or_returning(state: HospitalState, checklist: dict, patient_type: str) -> dict:
    missing = []

    # Step 1: PII -- only for brand new patients
    if patient_type == "new_patient":
        info = state.get("patient_info", {})
        pii_missing = []
        if not info.get("name"):
            pii_missing.append("name")
        if not info.get("dob"):
            pii_missing.append("date of birth")
        if not info.get("phone"):
            pii_missing.append("phone number")
        if not info.get("email"):
            pii_missing.append("email address")
        if not info.get("insurance_provider") or not info.get("insurance_id"):
            pii_missing.append("insurance information")

        checklist["pii_collection"] = len(pii_missing) == 0
        missing += pii_missing

    # Step 2: symptoms + doctor gender preference
    if not state.get("symptoms"):
        missing.append("main symptoms")
    if not state.get("patient_info", {}).get("gender_pref"):
        missing.append("doctor gender preference (male, female, or no preference)")
    checklist["symptoms_and_gender"] = (
        bool(state.get("symptoms")) and bool(state.get("patient_info", {}).get("gender_pref"))
    )

    # If anything above is still missing, ask and stop here for this turn
    if missing:
        return {
            "booking_checklist": checklist,
            "output_text": _ask_for(missing),
            "status": "awaiting_input",
        }

    # Step 3: department + candidate doctors (automatic, no human input)
    if not checklist["department"] or not checklist["doctor"]:
        decision = _decide_department_and_doctors(state)
        checklist["department"] = True
        checklist["doctor"] = True
        return {
            **decision,
            "booking_checklist": checklist,
            "status": "in_progress",
        }

    # Step 4: find open slots across the candidate doctors
    if not state.get("available_slots"):
        slots = find_available_slots(state)
        return {
            "available_slots": slots,
            "booking_checklist": checklist,
            "status": "awaiting_input",  # waiting for patient to pick one
        }

    # Step 5: waiting on patient to actually pick a slot
    if not state.get("selected_slot"):
        return {"booking_checklist": checklist, "status": "awaiting_input"}

    # Step 6: confirm + save the booking
    if not state.get("confirmed_appointment"):
        confirmation = confirm_booking(state)
        return {
            "confirmed_appointment": confirmation,
            "booking_checklist": checklist,
            "status": "complete",
        }

    return {"booking_checklist": checklist, "status": "complete"}


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT -- this is what nodes.py's book_appointment node calls.
# ---------------------------------------------------------------------------
def book_appointment_flow(state: HospitalState) -> dict:
    patient_type = _patient_type(state)
    checklist = _init_checklist(state)

    if patient_type == "old_followup":
        return _handle_followup(state, checklist)

    return _handle_new_or_returning(state, checklist, patient_type)


# ---------------------------------------------------------------------------
# Automatic step, no human input: picks a department from symptoms, then
# lists doctors in that department matching gender preference.
# NEEDS a doctor-directory doc in hospital_docs/ to return real results.
# ---------------------------------------------------------------------------
def _decide_department_and_doctors(state: HospitalState) -> dict:
    symptoms = state.get("symptoms", "")
    requested_doctor = state.get("requested_doctor_name")
    gender_pref = state.get("patient_info", {}).get("gender_pref")

    query = (
        f"Symptoms: {symptoms}\n"
        f"Requested doctor: {requested_doctor or 'None'}\n"
        f"Gender preference: {gender_pref or 'None'}"
    )
    docs = retrieve(query, top_k=3)
    context = "\n\n".join(docs)

    prompt = f"""
    Hospital Directory:
    {context}

    Patient symptoms: {symptoms}
    Requested doctor: {requested_doctor or "None"}
    Gender preference: {gender_pref or "No preference"}

    Instructions:
    1. Choose the single best-fitting department for these symptoms.
    2. List every doctor in that department who matches the gender
       preference (if any). If a specific doctor was requested and works
       in this department, always include them regardless of gender.

    Return ONLY JSON in this shape:
    {{"department": "", "candidate_doctors": ["", ""]}}
    """
    response = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
    result = json.loads(response.text.strip())

    return {
        "department": result.get("department"),
        "candidate_doctors": result.get("candidate_doctors", []),
    }


# ---------------------------------------------------------------------------
# PLACEHOLDERS -- called from _handle_new_or_returning above, but not
# built yet. Will raise an error if the flow actually reaches them.
# ---------------------------------------------------------------------------
def find_available_slots(state: HospitalState) -> list[dict]:
    """Will check each name in candidate_doctors' schedules and return the
    earliest 3 slots across all of them."""
    raise NotImplementedError


def confirm_booking(state: HospitalState) -> dict:
    """Will call auth.save_appointment() using state['selected_slot'],
    then trigger the email + PDF confirmation."""
    raise NotImplementedError