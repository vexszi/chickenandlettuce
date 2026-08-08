import json
import os
import hashlib
import secrets
from typing import Optional

PATIENTS_FILE = "patients.json"
APPOINTMENTS_FILE = "appointments.json"


def _load(filepath: str) -> dict:
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r") as f:
        return json.load(f)


def _save(filepath: str, data: dict) -> None:
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode()).hexdigest()


def register_patient(email: str, password: str, name: str) -> str:
    """Creates a new patient record. Returns the new patient_id.
    Raises ValueError if the email is already registered."""
    patients = _load(PATIENTS_FILE)

    if email in patients:
        raise ValueError("An account with this email already exists.")

    salt = secrets.token_hex(8)
    patient_id = f"p_{secrets.token_hex(4)}"

    patients[email] = {
        "patient_id": patient_id,
        "password_hash": _hash_password(password, salt),
        "salt": salt,
        "name": name,
        "dob": None,
        "phone": None,
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

    return record


def recover_saved_pii(patient_id: str) -> dict:
    """Given a patient_id, returns their stored PII fields to pre-populate
    state["patient_info"] on login. Strips auth fields -- password data
    should never end up in graph state."""
    patients = _load(PATIENTS_FILE)

    for record in patients.values():
        if record["patient_id"] == patient_id:
            return {
                k: v for k, v in record.items()
                if k not in ("password_hash", "salt")
            }

    return {}


def get_appointments(patient_id: str) -> list:
    appointments = _load(APPOINTMENTS_FILE)
    return appointments.get(patient_id, [])


def save_appointment(patient_id: str, appointment: dict) -> None:
    """Appends a new appointment record for a patient."""
    appointments = _load(APPOINTMENTS_FILE)
    appointments.setdefault(patient_id, []).append(appointment)
    _save(APPOINTMENTS_FILE, appointments)