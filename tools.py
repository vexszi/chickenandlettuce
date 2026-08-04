import os
import time
import base64
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from state import HospitalState

load_dotenv()

NOVA3_KEY = os.getenv("nova3_key")
AZURE_KEY1 = os.getenv("azure_key1")
AZURE_ENDPOINT = os.getenv("azure_endpoint")
GOOGLE_CLOUD_KEY = os.getenv("cloud_key")

AUDIO_OUTPUT_DIR = "audio_output"


# ---------------------------------------------------------------------------
# 1. Translation (Google Cloud Translation)
#    Input: state["masked_text"] (falls back to transcribed_text if empty)
#    Output: state["translated_text"]
# ---------------------------------------------------------------------------
def translation_tool(state: HospitalState) -> dict:
    text = state.get("masked_text") or state.get("transcribed_text") or ""
    target_lang = state.get("preferred_language", "en")

    url = "https://translation.googleapis.com/language/translate/v2"
    params = {"key": GOOGLE_CLOUD_KEY}
    payload = {"q": text, "target": target_lang, "format": "text"}

    resp = requests.post(url, params=params, json=payload, timeout=15)
    resp.raise_for_status()

    translated = resp.json()["data"]["translations"][0]["translatedText"]
    return {"translated_text": translated}


# ---------------------------------------------------------------------------
# 2. Transcription (Deepgram Nova 3)
#    Input: state["audio_file_path"]
#    Output: state["transcribed_text"]
# ---------------------------------------------------------------------------
def transcription_tool(state: HospitalState) -> dict:
    audio_path = state.get("audio_file_path")
    if not audio_path:
        return {}

    url = "https://api.deepgram.com/v1/listen"
    headers = {
        "Authorization": f"Token {NOVA3_KEY}",
        "Content-Type": "audio/wav",  # swap to audio/mpeg, audio/webm, etc. as needed
    }
    params = {"model": "nova-3", "smart_format": "true"}

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    resp = requests.post(url, headers=headers, params=params, data=audio_bytes, timeout=30)
    resp.raise_for_status()

    result = resp.json()
    transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
    return {"transcribed_text": transcript}


# ---------------------------------------------------------------------------
# 3. OCR (Azure Document Intelligence, prebuilt-read model)
#    Input: state["document_file_path"]
#    Output: appends to state["ocr_text"] list (reducer handles the append,
#    so we just return the single new item wrapped in a list)
# ---------------------------------------------------------------------------
def ocr_tool(state: HospitalState) -> dict:
    file_path = state.get("document_file_path")
    if not file_path:
        return {}

    analyze_url = (
        f"{AZURE_ENDPOINT.rstrip('/')}/documentintelligence/documentModels/"
        f"prebuilt-read:analyze?api-version=2024-11-30"
    )
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_KEY1,
        "Content-Type": "application/octet-stream",
    }

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    resp = requests.post(analyze_url, headers=headers, data=file_bytes, timeout=30)
    resp.raise_for_status()
    operation_url = resp.headers["Operation-Location"]

    # Azure Document Intelligence is async -- poll until done
    poll_headers = {"Ocp-Apim-Subscription-Key": AZURE_KEY1}
    for _ in range(30):  # ~30s max wait
        poll_resp = requests.get(operation_url, headers=poll_headers, timeout=15)
        poll_resp.raise_for_status()
        result = poll_resp.json()

        if result["status"] == "succeeded":
            extracted_text = result["analyzeResult"]["content"]
            return {"ocr_text": [extracted_text]}  # reducer appends this to the list
        if result["status"] == "failed":
            raise RuntimeError(f"Azure OCR failed: {result}")

        time.sleep(1)

    raise TimeoutError("Azure OCR polling timed out after 30s")


# ---------------------------------------------------------------------------
# 4. Text-to-Speech (Google Cloud TTS)
#    Input: state["output_text"]
#    Output: state["output_audio_url"] (local file path to the saved mp3)
# ---------------------------------------------------------------------------
def tts_tool(state: HospitalState) -> dict:
    text = state.get("output_text", "")
    if not text:
        return {}

    lang_code = state.get("preferred_language", "en-US")
    # Google TTS wants full locale codes like "en-US", not just "en" --
    # fall back to en-US if you're only storing short codes elsewhere.
    if len(lang_code) == 2:
        lang_code = f"{lang_code}-US" if lang_code == "en" else lang_code

    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_CLOUD_KEY}"
    payload = {
        "input": {"text": text},
        "voice": {"languageCode": lang_code, "ssmlGender": "NEUTRAL"},
        "audioConfig": {"audioEncoding": "MP3"},
    }

    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()

    audio_content_b64 = resp.json()["audioContent"]

    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    filename = os.path.join(AUDIO_OUTPUT_DIR, f"response_{timestamp}.mp3")

    with open(filename, "wb") as f:
        f.write(base64.b64decode(audio_content_b64))

    return {"output_audio_url": filename}