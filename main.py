from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import HumanMessage

from auth import register_patient, login
from hospital_graph import graph

app = FastAPI()

# CORS: the browser blocks fetch() calls to a different origin/port by
# default. This tells the browser "it's fine, let shore.html talk to me."
# allow_origins=["*"] is fine for local testing -- you'd lock this down
# to your actual frontend's URL before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session store: hold onto state between messages instead of the 
# graph doing it automatically. Resets every time the server restarts.
# ---------------------------------------------------------------------------
SESSIONS: dict = {}


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
    new_account_password: Optional[str] = None
    selected_slot: Optional[dict] = None
    existing_appointment_id: Optional[str] = None


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
    return {"success": True, "patientId": record["patient_id"]}


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
    if req.new_account_password is not None:
        state["new_account_password"] = req.new_account_password
    if req.selected_slot is not None:
        state["selected_slot"] = req.selected_slot
    if req.existing_appointment_id is not None:
        state["existing_appointment_id"] = req.existing_appointment_id

    # The patient's typed message becomes a real LangChain message object --
    # translate_incoming_tool needs .content, not a plain dict.
    state["messages"] = [HumanMessage(content=req.message)]

    result = graph.invoke(state)

    # Merge this turn's updates into the saved session for next time.
    state.update(result)
    SESSIONS[req.thread_id] = state

    # Only send back what the frontend actually needs to render.
    return {
        "output_text": result.get("output_text"),
        "status": result.get("status"),
        "available_slots": result.get("available_slots"),
        "available_appointments": result.get("available_appointments"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)