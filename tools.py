import os
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from langdetect import detect_langs, LangDetectException
from state import HospitalState

from google.cloud import translate_v2 as translate
from google.cloud import texttospeech
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
load_dotenv()

NOVA3_KEY = os.getenv("nova3_key")
AZURE_KEY1 = os.getenv("azure_key1")
AZURE_ENDPOINT = os.getenv("azure_endpoint")
GOOGLE_CLOUD_KEY = os.getenv("cloud_key")

translate_client = translate.Client()
tts_client = texttospeech.TextToSpeechClient()
azure_client = DocumentIntelligenceClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_KEY1),
)

# ---------------------------------------------------------------------------
# Cheap local pre-filter so we don't pay a Google Translate API round-trip
# on every single message. langdetect is a lightweight, offline, pure-python
# library -- no network call, no model download. We only trust it enough to
# SKIP the real API when it's confident the text is English; anything it's
# unsure about (or errors on, e.g. very short messages) still goes through
# the real translate call so accuracy never regresses.
# ---------------------------------------------------------------------------
EN_CONFIDENCE_THRESHOLD = 0.90

def _confidently_english(text: str) -> bool:
    try:
        guesses = detect_langs(text)
    except LangDetectException:
        # e.g. text with no detectable letters (numbers/punctuation only) --
        # don't guess, let the real API decide.
        return False
    return bool(guesses) and guesses[0].lang == "en" and guesses[0].prob >= EN_CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# 1a. Translate INCOMING text to English (normalizes whatever language
#     the patient just used, and captures what that language was).
#     Runs INSIDE the graph, every turn. Audio is transcribed
#     and user-edited on the FRONTEND.
# ---------------------------------------------------------------------------
def translate_incoming_tool(state: HospitalState) -> dict:
    updates = {}

    message_text = state["messages"][-1].content if state.get("messages") else None
    if message_text:
        if _confidently_english(message_text):
            updates["translated_text"] = message_text
            updates["detected_language"] = "en"
        else:
            try:
                result = translate_client.translate(message_text, target_language="en")
                updates["translated_text"] = result["translatedText"]
                updates["detected_language"] = result["detectedSourceLanguage"]
            except Exception as e:
                # Translate API failure (quota/auth/network) shouldn't crash
                # the whole turn -- fall back to using the raw text as-is.
                # Downstream will treat it as English; not ideal for a
                # non-English patient, but far better than a hard failure.
                print(f"[translate_incoming_tool] message translation failed: {e}")
                updates["translated_text"] = message_text
                updates["detected_language"] = "en"

    doc_list = state.get("ocr_text", [])
    already_translated = state.get("translated_doc_count", 0)

    if len(doc_list) > already_translated:
        new_doc = doc_list[-1]
        if _confidently_english(new_doc):
            updates["translated_document_text"] = new_doc
            updates.setdefault("detected_language", "en")
        else:
            try:
                result = translate_client.translate(new_doc, target_language="en")
                updates["translated_document_text"] = result["translatedText"]
                updates.setdefault("detected_language", result["detectedSourceLanguage"])
            except Exception as e:
                print(f"[translate_incoming_tool] document translation failed: {e}")
                updates["translated_document_text"] = new_doc
                updates.setdefault("detected_language", "en")
        updates["translated_doc_count"] = len(doc_list)

    return updates


# ---------------------------------------------------------------------------
# 1b. Translate OUTGOING response back into whatever language was
#     detected this turn (falls back to preferred_language on turn 1,
#     skips the API call entirely if it's already English).
#     Runs INSIDE the graph, every turn.
# ---------------------------------------------------------------------------
def translate_outgoing_tool(state: HospitalState) -> dict:

    target_lang = state.get("detected_language") or state.get("preferred_language", "en")

    if target_lang == "en":
        return {}  # already English, skip the API call

    text = state.get("output_text", "")
    if not text:
        return {}

    try:
        result = translate_client.translate(text, target_language=target_lang)
        return {"output_text": result["translatedText"]}
    except Exception as e:
        # Better to show the patient the English response than to crash
        # the request and show them nothing at all.
        print(f"[translate_outgoing_tool] translation failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# 2. Transcription (Deepgram Nova 3)
#    Runs OUTSIDE the graph now when frontend detecrs audio upload.
#    DOES NOT UPDATE STATE
# ---------------------------------------------------------------------------
def transcription_tool(audio_path: str, content_type: str = "audio/wav") -> str:
    url = "https://api.deepgram.com/v1/listen"
    headers = {
        "Authorization": f"Token {NOVA3_KEY}",
        "Content-Type": content_type,
    }
    params = {"model": "nova-3", "smart_format": "true", "language": "multi" }

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    resp = requests.post(url, headers=headers, params=params, data=audio_bytes, timeout=30)
    resp.raise_for_status()

    result = resp.json()
    transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
    return transcript


# ---------------------------------------------------------------------------
# 3. OCR (Azure Document Intelligence, prebuilt-read model)
#    Runs OUTSIDE the graph now -- called by a standalone endpoint the
#    instant a file is uploaded on the frontend.
#    DOES NOT UPDATE STATE
# ---------------------------------------------------------------------------
def ocr_tool(file_path: str) -> str:
    with open(file_path, "rb") as f:
        poller = azure_client.begin_analyze_document(
            "prebuilt-read",
            body=f,
            content_type="application/octet-stream",
        )

    result = poller.result()
    return result.content


# ---------------------------------------------------------------------------
# 4. Text-to-Speech (Google Cloud TTS)
#    Input: state["output_text"], state["detected_language"]
#    Output: state["output_audio_url"]
#    Runs OUTSIDE the graph, when triggered by the frontend.
# ---------------------------------------------------------------------------
AUDIO_OUTPUT_DIR = "audio_output"
def tts_tool(state: HospitalState) -> dict:

    text = state.get("output_text", "")
    if not text:
        return {}

    LANG_CODE_MAP = {
        "en": "en-US",
        "es": "es-ES",
        "fr": "fr-FR",
        "de": "de-DE",
        "it": "it-IT",
        "pt": "pt-BR",
        "nl": "nl-NL",
        "da": "da-DK",
        "sv": "sv-SE",
        "fi": "fi-FI",
        "no": "no-NO",
        "pl": "pl-PL",
        "cs": "cs-CZ",
        "ro": "ro-RO",
        "hu": "hu-HU",
        "bg": "bg-BG",
        "el": "el-GR",
        "uk": "uk-UA",
        "ru": "ru-RU",
        "tr": "tr-TR",
        "id": "id-ID",
        "vi": "vi-VN",
        "th": "th-TH",
        "ja": "ja-JP",
        "ko": "ko-KR",
        "zh": "zh-CN",
        "hi": "hi-IN",
    }

    lang_code = state.get("detected_language") or state.get("preferred_language", "en")
    lang_code = LANG_CODE_MAP.get(lang_code, "en-US")  # fallback to English if unknown

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=lang_code,
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    filename = os.path.join(AUDIO_OUTPUT_DIR, f"response_{timestamp}.mp3")

    with open(filename, "wb") as f:
        f.write(response.audio_content)

    return {"output_audio_url": filename}