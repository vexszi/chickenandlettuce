import os
import pickle
import tempfile

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import HumanMessage

import threading
from collections import defaultdict

from auth import register_patient, login
from hospital_graph import graph
from tools import ocr_tool, translate_outgoing_tool, transcription_tool, tts_tool

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

# CORS: the browser blocks fetch() calls to a different origin/port by
# default. This tells the browser "it's fine, let shore.html talk to me."
# allow_origins=["*"] is fine for local testing -- you'd lock this down
# to your actual frontend's URL before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# tools.tts_tool writes mp3s to this directory -- mount it so the frontend
# can just <audio src="..."> / fetch the returned URL directly, instead of
# the backend having to read the file back and stream bytes itself.
os.makedirs("audio_output", exist_ok=True)
app.mount("/audio", StaticFiles(directory="audio_output"), name="audio")

# ---------------------------------------------------------------------------
# Session store: holds onto state between messages instead of the graph
# doing it automatically. Persisted to disk (pickle, not JSON -- session
# state holds LangChain message objects, not just plain JSON-safe data) so
# a server restart or an uvicorn --reload cycle mid-conversation doesn't
# silently wipe every active patient's session (that was the cause of the
# "greeting repeats" / "keeps re-asking for insurance" bugs -- the graph
# was starting fresh every time the process restarted).
# ---------------------------------------------------------------------------
SESSIONS_FILE = "sessions.pkl"


def _load_sessions() -> dict:
    if not os.path.exists(SESSIONS_FILE):
        return {}
    try:
        with open(SESSIONS_FILE, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        # Corrupt/unreadable session file shouldn't take the whole server
        # down -- just start fresh, same as if it never existed.
        print(f"[sessions] couldn't load {SESSIONS_FILE}, starting fresh: {e}")
        return {}


def _save_sessions() -> None:
    try:
        with open(SESSIONS_FILE, "wb") as f:
            pickle.dump(SESSIONS, f)
    except Exception as e:
        print(f"[sessions] couldn't save {SESSIONS_FILE}: {e}")


SESSIONS: dict = _load_sessions()


# ---------------------------------------------------------------------------
# Request shapes -- Pydantic models describing exactly what JSON each
# endpoint expects the frontend to send. FastAPI uses these to validate
# incoming requests automatically.
# ---------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    thread_id: str
    message: str
    is_first_time: Optional[bool] = None
    old_patient_follow_up: Optional[bool] = None
    patient_id: Optional[str] = None
    patient_info: Optional[dict] = None   # <-- add this
    new_account_password: Optional[str] = None
    selected_slot: Optional[dict] = None
    existing_appointment_id: Optional[str] = None
    preferred_language: Optional[str] = None


class GreetingRequest(BaseModel):
    preferred_language: str


# The exact string initAssistantChat() used to hardcode client-side in
# shore.html. Kept here so the welcome-screen language picker can get it
# translated before the chat even starts, instead of the first message
# always showing up in English regardless of what the patient selected.
DEFAULT_GREETING = (
    "Hello! I'm your Cove Assistant. How can I help you today? "
    "You can book, reschedule, or ask questions."
)


# ---------------------------------------------------------------------------
# Registration -- calls your existing auth.py directly, no graph
# ---------------------------------------------------------------------------
@app.post("/api/patients/register")
def api_register(req: RegisterRequest):
    try:
        patient_id = register_patient(email=req.email, password=req.password, name=req.name)
        return {"success": True, "patientId": patient_id}
    except ValueError as e:
        return {"success": False, "reason": str(e)}


# ---------------------------------------------------------------------------
# Login -- straight to auth.py.
# ---------------------------------------------------------------------------
@app.post("/api/patients/login")
def api_login(req: LoginRequest):
    record = login(req.email, req.password)
    if record is None:
        return {"success": False, "reason": "Invalid email or password."}
    return {
        "success": True,
        "patientId": record["patient_id"],
        "patientInfo": record,  # name, email, phone, dob, insurance, etc.
    }


# ---------------------------------------------------------------------------
# The main conversation endpoint -- this is what replaces
# processIncomingUserIntent()'s fake regex logic in shore.html.
# ---------------------------------------------------------------------------
@app.post("/api/chat/message")
def api_chat(req: ChatRequest):
    # Pull up this patient's ongoing state, or start a fresh one.
    state = SESSIONS.get(req.thread_id, {"thread_id": req.thread_id, "patient_info": {}})

    # Direct fields the frontend sets itself (never typed in chat) --
    # e.g. new_account_password, since that should never pass through
    # the LLM/PII pipeline.
    if req.is_first_time is not None:
        state["is_first_time"] = req.is_first_time
    if req.old_patient_follow_up is not None:
        state["old_patient_follow_up"] = req.old_patient_follow_up
    if req.patient_id is not None:
        state["patient_id"] = req.patient_id
    if req.patient_info is not None:
        state["patient_info"] = {**req.patient_info, **state.get("patient_info", {})}
    if req.new_account_password is not None:
        state["new_account_password"] = req.new_account_password
    if req.selected_slot is not None:
        state["selected_slot"] = req.selected_slot
    if req.existing_appointment_id is not None:
        state["existing_appointment_id"] = req.existing_appointment_id
    # Set once, on the first turn, from the welcome-screen language picker.
    # translate_outgoing_tool falls back to this only when detected_language
    # isn't set yet (i.e. before the patient's own message has been through
    # translate_incoming_tool for the first time) -- see tools.py.
    if req.preferred_language is not None and "preferred_language" not in state:
        state["preferred_language"] = req.preferred_language

    # The patient's typed message becomes a real LangChain message object --
    # translate_incoming_tool needs .content, not a plain dict. APPEND,
    # don't overwrite: state["messages"] used to be reset to a single-item
    # list every turn, which silently threw away the whole conversation on
    # every request -- that's why nothing the model generated ever
    # reflected earlier turns.
    state.setdefault("messages", [])
    state["messages"].append(HumanMessage(content=req.message))
    state["messages"] = state["messages"][-20:]  # keep the session lean

    result = graph.invoke(state)

    # Merge this turn's updates into the saved session for next time.
    state.update(result)

    # Log this turn (PII-masked patient side, translated assistant side) so
    # the Gemini-calling nodes in nodes.py can see recent conversation
    # context on the *next* turn, instead of only ever seeing the current
    # message in isolation. Kept short (last ~6 exchanges) so it doesn't
    # balloon the prompt or the token bill.
    patient_line = state.get("masked_text") or req.message
    assistant_line = result.get("output_text")
    log_additions = [f"Patient: {patient_line}"]
    if assistant_line:
        log_additions.append(f"Assistant: {assistant_line}")
    state["conversation_log"] = (state.get("conversation_log", []) + log_additions)[-12:]
    if result.get("status") == "complete":
        for key in (
            "selected_slot", "available_slots", "available_appointments",
            "booking_checklist", "department", "requested_doctor_name",
            "existing_appointment_id", "held_appointment_id",
            "confirmed_appointment", "symptoms",
            "new_account_password",
        ):
            state.pop(key, None)

    SESSIONS[req.thread_id] = state
    _save_sessions()

    # Only send back what the frontend actually needs to render.
    return {
        "output_text": result.get("output_text"),
        "status": result.get("status"),
        "available_slots": result.get("available_slots"),
        "available_appointments": result.get("available_appointments"),
        "needs_password": result.get("needs_password", False),
        "emergency_detected": result.get("emergency_detected", False),
        "emergency_reason": result.get("emergency_reason"),
        "confirmed_appointment": result.get("confirmed_appointment"),
    }


# ---------------------------------------------------------------------------
# Greeting translation -- called once from the welcome screen after the
# patient picks a language, so the very first message they see is already
# in their preferred language instead of always starting in English.
# Deliberately NOT run through the full graph: there's no patient message
# yet to translate/mask/route, just a static string to translate outward.
# ---------------------------------------------------------------------------
@app.post("/api/chat/greeting")
def api_greeting(req: GreetingRequest):
    translated = translate_outgoing_tool({
        "output_text": DEFAULT_GREETING,
        "detected_language": req.preferred_language,
    })
    return {"output_text": translated.get("output_text", DEFAULT_GREETING)}


# ---------------------------------------------------------------------------
# Document upload -- OCRs the file (Azure Document Intelligence, via
# tools.ocr_tool) and appends the extracted text onto this thread's
# ocr_text list. Runs OUTSIDE the graph, same as tools.py documents:
# the *next* chat turn is what actually translates + masks it (in
# translate_incoming_tool / pii_masking_node) and feeds it to
# document_explainer_node once the patient asks about it.
# ---------------------------------------------------------------------------
@app.post("/api/documents/upload")
async def api_upload_document(thread_id: str = Form(...), file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        extracted_text = ocr_tool(tmp_path)
    except Exception as e:
        return {"success": False, "reason": f"Couldn't read that document: {e}"}
    finally:
        os.remove(tmp_path)

    state = SESSIONS.get(thread_id, {"thread_id": thread_id, "patient_info": {}})
    state.setdefault("ocr_text", [])
    state["ocr_text"].append(extracted_text)
    SESSIONS[thread_id] = state
    _save_sessions()

    return {"success": True, "doc_count": len(state["ocr_text"])}

# ---------------------------------------------------------------------------
# Reset conversation session
# ---------------------------------------------------------------------------
@app.post("/api/session/reset")
async def reset_session(thread_id: str):
    SESSIONS.pop(thread_id, None)
    _save_sessions()

    return {
        "success": True
    }


class TTSRequest(BaseModel):
    text: str
    language: Optional[str] = "en"


# ---------------------------------------------------------------------------
# Voice input -- transcribes a recorded audio clip (Deepgram Nova 3, via
# tools.transcription_tool). Runs OUTSIDE the graph, triggered directly by
# the mic button in shore.html. DOES NOT touch session state itself -- the
# transcript comes back to the frontend, which sends it into
# /api/chat/message exactly like a typed message.
# ---------------------------------------------------------------------------
@app.post("/api/audio/transcribe")
async def api_transcribe_audio(audio: UploadFile = File(...)):
    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        transcript = transcription_tool(tmp_path, content_type=audio.content_type or "audio/webm")
    except Exception as e:
        return {"success": False, "reason": f"Couldn't transcribe that: {e}"}
    finally:
        os.remove(tmp_path)

    return {"success": True, "transcript": transcript}


# ---------------------------------------------------------------------------
# Text-to-speech -- synthesizes a chat bubble's text (Google Cloud TTS, via
# tools.tts_tool). Runs OUTSIDE the graph, triggered by the 🔊 Listen button
# on any bubble. tts_tool expects state-shaped kwargs, so it's called with a
# minimal dict rather than the full session state.
# ---------------------------------------------------------------------------
@app.post("/api/audio/tts")
def api_tts(req: TTSRequest):
    if not req.text.strip():
        return {"success": False, "reason": "No text to speak."}

    try:
        result = tts_tool({"output_text": req.text, "detected_language": req.language})
    except Exception as e:
        return {"success": False, "reason": f"Couldn't generate audio: {e}"}

    audio_path = result.get("output_audio_url")
    if not audio_path:
        return {"success": False, "reason": "Couldn't generate audio."}

    return {"success": True, "audio_url": f"/audio/{os.path.basename(audio_path)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)