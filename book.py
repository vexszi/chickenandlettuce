import os
import json
from typing import Optional
from google import genai
from datetime import datetime, timedelta

from state import HospitalState
from auth import save_appointment
from auth import get_doctor_appointments
from mcp_server.server import send_confirmation_email
from auth import register_patient
from auth import get_appointments

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

_doctors_cache: Optional[dict] = None


def _load_doctors() -> dict:
    """doctors.json is static reference data -- previously every call to
    _decide_department() and find_available_slots() re-read and re-parsed
    it from disk. Cached in memory now; restart the process (or call
    _load_doctors.cache_clear-equivalent by resetting _doctors_cache)
    if doctors.json is edited while the server is running."""
    global _doctors_cache
    if _doctors_cache is None:
        with open("doctors.json") as f:
            _doctors_cache = json.load(f)
    return _doctors_cache

from dotenv import load_dotenv
load_dotenv()
GEMINI_KEY = os.getenv("flash_key")
client = genai.Client(api_key=GEMINI_KEY)

# Same model + safety net as nodes.py. Kept as a local copy rather than
# importing from nodes.py to avoid a circular import (nodes.py imports
# book_appointment_flow from this file).
GEMINI_MODEL = "gemini-3.5-flash-lite"


def _safe_generate_content(**kwargs):
    """See nodes.py -- catches Gemini failures instead of letting them
    bubble up as an unhandled exception that kills the whole booking
    request (this was previously unguarded here, unlike every Gemini call
    in nodes.py)."""
    try:
        return client.models.generate_content(**kwargs)
    except Exception as e:
        print(f"[Gemini call failed in book.py] {e}")
        return None


def _recent_history(state: HospitalState) -> str:
    log = state.get("conversation_log", [])
    if not log:
        return ""
    return "Recent conversation so far:\n" + "\n".join(log) + "\n"


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
                "department": False}
    elif patient_type == "old_new_visit":
        return {"symptoms_and_gender": False, "department": False}
    else:  # old_followup
        return {"which_appointment": False, "new_symptoms": False}


# ---------------------------------------------------------------------------
# Same LLM prompt as your pasted patient_info() -- turns a list of missing
# things into one natural question, asking for up to 2 at a time.
# ---------------------------------------------------------------------------
def _ask_for(missing: list[str], state: HospitalState) -> str:
    to_ask = missing[:2]
    history = _recent_history(state)
    prompt = f"""
    You are a hospital receptionist mid-conversation with a patient you're
    already talking to -- not greeting them for the first time.
    Generate ONE natural message asking for the following missing
    information ONLY: {to_ask}

    Rules
    - Ask naturally for all items above, in one short friendly message.
    - Do NOT mention or ask about anything not in the list above.
    - Do NOT open with "Hello" or re-introduce yourself -- check the recent
      conversation below and continue naturally, the way a person mid-chat
      would, not the way you'd open a brand new conversation.
    - Return ONLY the message.

    {history}
    """
    response = _safe_generate_content(model=GEMINI_MODEL, contents=prompt)
    if response is None:
        # Deterministic fallback so a Gemini hiccup doesn't crash the whole
        # booking flow -- plain but functional.
        return f"Could you also share your {' and '.join(to_ask)}?"
    return response.text.strip()


# ---------------------------------------------------------------------------
# Handles the "old patient, follow-up" flow -- just 2 things to check.
# ---------------------------------------------------------------------------
def _handle_followup(state: HospitalState, checklist: dict) -> dict:
    if not state.get("existing_appointment_id"):
        checklist["which_appointment"] = False
        upcoming = _upcoming_appointments(state.get("patient_id"))
        if not upcoming:
            return {
                "output_text": "I don't see any past appointments to follow up on.",
                "booking_checklist": checklist,
                "status": "complete",
            }
        return {
            "available_appointments": upcoming,
            "booking_checklist": checklist,
            "status": "awaiting_input",
        }
    checklist["which_appointment"] = True

    missing = []
    if state.get("symptoms") is not None:
        checklist["new_symptoms"] = True
    else:
        checklist["new_symptoms"] = False
        missing.append("whether you have any new symptoms since your last visit")

    if missing:
        return {
            "booking_checklist": checklist,
            "output_text": _ask_for(missing, state),
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

    # Step 2: symptoms (gender preference is optional -- mentioned once, never required)
    if not state.get("symptoms"):
        missing.append("main symptoms")
    checklist["symptoms_and_gender"] = bool(state.get("symptoms"))

    if missing:
        message = _ask_for(missing, state)
        if not checklist.get("gender_mentioned"):
            message += " If you have a preference for a male or female doctor, feel free to mention it -- totally optional."
            checklist["gender_mentioned"] = True
        return {
            "booking_checklist": checklist,
            "output_text": message,
            "status": "awaiting_input",
        }

    # Step 3: department (automatic, no human input)
    if not checklist["department"]:
        decision = _decide_department(state)
        checklist["department"] = True
        state = {**state, **decision}   # so Step 4 below can see the new department

    # Step 4: find open slots across the candidate doctors
    if not state.get("available_slots"):
        slots = find_available_slots(state)
        if not slots:
            return {
                "output_text": "Sorry, no doctors are available for your criteria right now. Please call our front desk directly.",
                "department": state.get("department"),
                "booking_checklist": checklist,
                "status": "complete",
            }
        return {
            "available_slots": slots,
            "department": state.get("department"),
            "booking_checklist": checklist,
            "status": "awaiting_input",
        }
    
        # Step 5: waiting on patient to actually pick a slot
    if not state.get("selected_slot"):
        return {
            "booking_checklist": checklist,
            "output_text": "Please choose one of the appointment times above.",
            "status": "awaiting_input",
        }

    # Step 5.5: a brand-new patient (no patient_id yet) needs a password
    # before we can create their account.
    if not state.get("patient_id") and not state.get("new_account_password"):
        return {
            "booking_checklist": checklist,
            "output_text": "Almost done! Since this will be your first visit with us, please create a password to set up your account.",
            "status": "awaiting_input",
            "needs_password": True,
        }

    # Step 6: confirm + save the booking
    if not state.get("confirmed_appointment"):
        confirmation = confirm_booking(state)
        return {**confirmation, "booking_checklist": checklist}

    return {"booking_checklist": checklist, "status": "complete"}



# ---------------------------------------------------------------------------
# Automatic step, no human input: picks a department from symptoms.
# ---------------------------------------------------------------------------
def _decide_department(state: HospitalState) -> dict:
    symptoms = state.get("symptoms", "")
    requested_doctor = state.get("requested_doctor_name")

    doctors = _load_doctors()

    # If a specific doctor was named and actually exists, skip guessing --
    # just use their department directly.
    if requested_doctor and requested_doctor in doctors:
        return {"department": doctors[requested_doctor]["department"]}

    departments = sorted({info["department"] for info in doctors.values()})

    prompt = f"""
    Available hospital departments: {departments}

    Patient symptoms: {symptoms}

    Instructions:
    Choose the single best-fitting department from the list above for these symptoms.
    Return ONLY JSON in this shape:
    {{"department": ""}}
    """
    response = _safe_generate_content(model=GEMINI_MODEL, contents=prompt)
    if response is None:
        # Gemini call failed outright -- fall back to the first department
        # alphabetically rather than crashing the booking flow.
        return {"department": departments[0] if departments else None}

    raw = response.text.strip()
    # Gemini sometimes wraps JSON in ```json ... ``` fences despite being
    # asked for raw JSON -- strip those before parsing so this doesn't
    # throw and take the whole request down with it.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
        department = result.get("department")
    except (json.JSONDecodeError, AttributeError):
        department = None

    if department not in departments:
        # Model returned something outside the known list (or failed to
        # parse) -- fall back rather than storing a bogus department.
        department = departments[0] if departments else None

    return {"department": department}


def find_available_slots(state: HospitalState) -> list[dict]:
    doctors = _load_doctors()

    department = state.get("department")
    gender_pref = state.get("patient_info", {}).get("gender_pref")
    requested_doctor = state.get("requested_doctor_name")

    filtered = {name: info for name, info in doctors.items()
                if info["department"] == department}

    if gender_pref and gender_pref.lower() not in ("no preference", "none"):
        filtered = {name: info for name, info in filtered.items()
                    if info["gender"] == gender_pref.lower()}

    if requested_doctor and requested_doctor in filtered:
        filtered = {requested_doctor: filtered[requested_doctor]}

    if not filtered:
        return []

    # Precompute each doctor's booked times once, instead of re-querying
    # get_doctor_appointments() inside the day loop like before.
    booked_by_doctor = {
        name: [
            datetime.strptime(a["date"], "%Y-%m-%d %H:%M")
            for a in get_doctor_appointments(name)
        ]
        for name in filtered
    }

    current_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=2)
    results = []
    days_checked = 0
    MAX_DAYS = 90
    NUM_SLOTS = 3

    while len(results) < NUM_SLOTS and days_checked < MAX_DAYS:
        day_name = DAY_NAMES[current_day.weekday()]
        best_for_day = None  # (slot_start, doctor_name) -- earliest across all qualifying doctors THIS day

        for doctor_name, info in filtered.items():
            availability = info["availability"]
            if day_name not in availability:
                continue

            work_start = datetime.strptime(availability[day_name]["start"], "%H:%M").time()
            work_end = datetime.strptime(availability[day_name]["end"], "%H:%M").time()
            slot_start = datetime.combine(current_day, work_start)
            day_end = datetime.combine(current_day, work_end)
            booked_times = booked_by_doctor[doctor_name]

            while slot_start + timedelta(minutes=45) <= day_end:
                slot_end = slot_start + timedelta(minutes=45)
                overlap = any(
                    slot_start < (appt_start + timedelta(minutes=45)) and appt_start < slot_end
                    for appt_start in booked_times
                )
                if not overlap:
                    if best_for_day is None or slot_start < best_for_day[0]:
                        best_for_day = (slot_start, doctor_name)
                    break  # earliest opening for this doctor today -- move to next doctor
                slot_start += timedelta(minutes=45)

        if best_for_day:
            slot_start, doctor_name = best_for_day
            results.append({
                "doctor": doctor_name,
                "date": slot_start.strftime("%Y-%m-%d"),
                "time": slot_start.strftime("%H:%M"),
            })

        current_day += timedelta(days=1)
        days_checked += 1

    return results



def confirm_booking(state: HospitalState) -> dict:
    slot = state["selected_slot"]  # {"doctor": ..., "date": ..., "time": ...}
    patient_info = state.get("patient_info", {})

    patient_id = state.get("patient_id")
    if not patient_id:
        patient_id = register_patient(
            email=patient_info.get("email"),
            password=state.get("new_account_password"),
            name=patient_info.get("name"),
        )

    full_datetime = f"{slot['date']} {slot['time']}"

    appointment_id = save_appointment(
        patient_id=patient_id,
        date=full_datetime,
        doctor=slot["doctor"],
        department=state.get("department"),
    )

    confirmation = {
        "patient_id": patient_id,
        "appointment_id": appointment_id,
        "patient_name": patient_info.get("name"),
        "doctor": slot["doctor"],
        "date": slot["date"],
        "time": slot["time"],
        "department": state.get("department"),
    }

    message = (
            f"You're all set, {patient_info.get('name', 'there')}! Your appointment "
            f"with {slot['doctor']} in {state.get('department')} is confirmed for "
            f"{slot['date']} at {slot['time']}. A confirmation email is on its way to you."
        )
    
    try:
        send_confirmation_email(
            patient_email=patient_info.get("email"),
            patient_name=patient_info.get("name"),
            doctor_name=slot["doctor"],
            appointment_date=slot["date"],
            appointment_time=slot["time"],
        )
        email_sent = True
    except Exception as e:
        print(f"[confirm_booking] email send failed: {e}")
        email_sent = False

    if not email_sent:
        message += " (We couldn't send the confirmation email right now, but your appointment is booked.)"

    return {
    "patient_id": patient_id,
    "confirmed_appointment": confirmation,
    "output_text": message,
    "status": "complete",
    "available_slots": None,
    "available_appointments": None,
}

def book_appointment_flow(state: HospitalState) -> dict:
    patient_type = _patient_type(state)
    checklist = _init_checklist(state)

    if patient_type == "old_followup":
        return _handle_followup(state, checklist)

    return _handle_new_or_returning(state, checklist, patient_type)

#-------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------CANCEL--------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------
def _upcoming_appointments(patient_id: str) -> list[dict]:
    """Returns this patient's appointments that haven't happened yet."""
    all_appts = get_appointments(patient_id)
    now = datetime.now()
    return [
        appt for appt in all_appts
        if datetime.strptime(appt["date"], "%Y-%m-%d %H:%M") > now
    ]