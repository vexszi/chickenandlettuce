import json
import os
import hashlib
import secrets
from typing import Optional
from cryptography.fernet import Fernet


PATIENTS_FILE = "patients.json"
APPOINTMENTS_FILE = "appointments.json"

fernet = Fernet(os.getenv("fernet_key"))

def _load(filepath: str) -> dict:              #opens the file and loads data from it
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "rb") as f:
        encrypted = f.read()
    if not encrypted:
        return {}
    return json.loads(fernet.decrypt(encrypted))


def _save(filepath: str, data: dict) -> None:
    encrypted = fernet.encrypt(json.dumps(data).encode())
    with open(filepath, "wb") as f:
        f.write(encrypted)


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 100_000
    ).hex()

def register_patient(
    email: str,
    password: str,
    name: str,
    dob: Optional[str] = None,
    phone: Optional[str] = None,
    patient_id: Optional[str] = None,
) -> str:
    """Creates a new patient record. Returns the new patient_id.
    Raises ValueError if the email is already registered."""
    patients = _load(PATIENTS_FILE)

    if email in patients:
        raise ValueError("An account with this email already exists.")

    salt = secrets.token_hex(8)
    if patient_id is None:
        patient_id = f"p_{secrets.token_hex(4)}"

    patients[email] = {
        "patient_id": patient_id,
        "password_hash": _hash_password(password, salt),
        "salt": salt,
        "name": name,
        "dob": dob,
        "phone": phone,
        "email": email,
        "insurance_provider": None,
        "insurance_id": None,
        "address": None,
        "gender_pref": None,
    }
    _save(PATIENTS_FILE, patients)
    return patient_id


def login(email: str, password: str) -> Optional[dict]:
    """Returns the patient's full record if credentials match, else None."""
    patients = _load(PATIENTS_FILE)
    record = patients.get(email)

    if not record:
        return None

    expected_hash = _hash_password(password, record["salt"])
    if expected_hash != record["password_hash"]:
        return None

    recover = _update_state(record)  # Preload PII into state on login
    return recover


def _update_state(patient: dict) -> dict:
    """Given a patient record, returns their stored PII fields to pre-populate
    state["patient_info"] on login. Strips auth fields -- password data
    should never end up in graph state."""
    return {
        k: v for k, v in patient.items()
        if k not in ("password_hash", "salt")
        }



def get_appointments(patient_id: str) -> list:         #the appointments the user has made/been to
    appointments = _load(APPOINTMENTS_FILE)            #use during cancel to show user which appointment to cancel.
    return appointments.get(patient_id, [])            


def save_appointment(patient_id: str, date: str, doctor: str, department: str) -> str:
    """Creates a new appointment record for a patient. Generates the
    appointment_id itself. Returns the new appointment_id."""
    appointments = _load(APPOINTMENTS_FILE)

    appointment_id = f"a_{secrets.token_hex(4)}"

    appointment = {
        "appointment_id": appointment_id,
        "date": date,          # format: "YYYY-MM-DD HH:MM"
        "doctor": doctor,
        "department": department,
    }

    appointments.setdefault(patient_id, []).append(appointment)
    _save(APPOINTMENTS_FILE, appointments)

    return appointment_id


def delete_appointment(patient_id: str, appointment_id: str) -> Optional[dict]:
    """Deletes an appointment. Returns the deleted appointment's info if
    successful (so the node can confirm details to the patient), or None
    if not found."""
    appointments = _load(APPOINTMENTS_FILE)
    patient_appointments = appointments.get(patient_id, [])

    for i, appt in enumerate(patient_appointments):
        if appt.get("appointment_id") == appointment_id:
            deleted = patient_appointments.pop(i)
            appointments[patient_id] = patient_appointments
            _save(APPOINTMENTS_FILE, appointments)
            return deleted

    return None

def get_doctor_appointments(doctor_name: str) -> list:
    """Returns every booked appointment for a given doctor, across all
    patients. Used to check a doctor's schedule when finding open slots."""
    appointments = _load(APPOINTMENTS_FILE)
    doctor_appts = []

    for patient_id, appts in appointments.items():
        for appt in appts:
            if appt.get("doctor") == doctor_name:
                doctor_appts.append(appt)

    return doctor_appts

def update_patient_insurance(email: str, insurance_provider: Optional[str], insurance_id: Optional[str]) -> bool:
    """Patches insurance fields onto an existing patient record. Used by
    the signup flow, which uploads the insurance card photo AFTER the
    account already exists and doesn't go through the graph at all --
    so intent_guardrail_extraction_node never runs for it. Returns True
    if anything was actually written."""
    patients = _load(PATIENTS_FILE)
    if email not in patients:
        return False

    wrote_something = False
    if insurance_provider:
        patients[email]["insurance_provider"] = insurance_provider
        wrote_something = True
    if insurance_id:
        patients[email]["insurance_id"] = insurance_id
        wrote_something = True

    if wrote_something:
        _save(PATIENTS_FILE, patients)
    return wrote_something
