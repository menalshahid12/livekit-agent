import logging
import os
import shutil
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pyttsx3
import soundfile as sf
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq

# Local module in the same src/ directory
from ist_knowledge import ISTDocument, load_ist_corpus, search, is_yes_no_question


logger = logging.getLogger("gradio_voice_agent")

ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_AUDIO_DIR = ROOT_DIR / "logs" / "audio"
LOG_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env.local (including OPENAI_API_KEY)
load_dotenv(ROOT_DIR / ".env.local")

# Initialize Groq client (expects GROQ_API_KEY in env)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize local TTS engine (Windows SAPI, no cloud)
tts_engine = pyttsx3.init()

# Global flags and messages
GROQ_AVAILABLE = True
HUMAN_ESCALATION_MESSAGE = (
    "This question is outside the information I have for IST admissions. "
    "I will schedule your call with a human admissions counselor for detailed guidance."
)

import uvicorn

# Load IST knowledge base once on startup
IST_DOCS: list[ISTDocument] = load_ist_corpus()

# FastAPI app
app = FastAPI(title="IST Admissions Voice Agent")
app.mount("/logs/audio", StaticFiles(directory=LOG_AUDIO_DIR), name="audio")


def apply_simple_vad(in_path: str, threshold: float = 0.01) -> str:
    """Very basic VAD: trim leading/trailing low-energy segments and re-save as WAV."""
    try:
        audio, sr = sf.read(in_path)
    except Exception as e:
        logger.warning("Failed to read audio for VAD: %s", e)
        return in_path

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    energy = np.abs(audio)
    mask = energy > threshold

    if not mask.any():
        # All silence, just keep original
        return in_path

    start = int(np.argmax(mask))
    end = int(len(mask) - np.argmax(mask[::-1]))
    trimmed = audio[start:end]

    out_path = LOG_AUDIO_DIR / f"vad_{int(time.time() * 1000)}.wav"
    sf.write(out_path, trimmed, sr)
    return str(out_path)


def transcribe_audio(audio_path: str) -> str:
    """Send audio to Groq Whisper for transcription (no OpenAI)."""
    global GROQ_AVAILABLE

    if not GROQ_AVAILABLE:
        # Groq already failed earlier in this run
        return ""

    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        # Use Groq's Whisper model; EN-only, fast and cheap
        resp = groq_client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), audio_bytes),
            model="distil-whisper-large-v3-en",
        )
        return (resp.text or "").strip()
    except Exception as e:
        logging.exception("Groq STT error: %s", e)
        GROQ_AVAILABLE = False
        return ""


def build_ist_context(query: str, max_chars: int = 1500) -> str:
    """Retrieve relevant IST snippets to ground the LLM."""
    docs = search(query, IST_DOCS, top_k=5)
    if not docs:
        return "No highly relevant IST website content was found for this question."

    snippets: list[str] = []
    for d in docs:
        snippet = d.text[:400]
        snippets.append(
            f"TITLE: {d.title or 'N/A'}\nURL: {d.url}\nCONTENT: {snippet}"
        )
        joined = "\n\n".join(snippets)
        if len(joined) >= max_chars:
            break
    return "\n\n".join(snippets)


def counselor_llm_response(user_text: str) -> str:
    """Use an LLM as a calling agent for IST admissions, grounded strictly in IST knowledge."""
    global GROQ_AVAILABLE

    ist_context = build_ist_context(user_text)

    # If we couldn't find relevant IST content, escalate to a human counselor
    if ist_context.startswith("No highly relevant IST website content was found"):
        # Try a lightweight yes/no heuristic before escalating
        if is_yes_no_question(user_text):
            hits = search(user_text, IST_DOCS, top_k=3)
            if hits:
                snippet = hits[0].text[:300].strip()
                return f"Yes. Based on IST content: {snippet}"
            else:
                return "I don't have enough information in the IST documents to answer that with confidence. " + HUMAN_ESCALATION_MESSAGE
        return HUMAN_ESCALATION_MESSAGE

    if not GROQ_AVAILABLE:
        # LLM not available (e.g. quota or network issue)
        return HUMAN_ESCALATION_MESSAGE

    system_prompt = (
        "You are a polite, confident calling agent who handles IST (Institute of Space Technology) "
        "university admission queries. You speak as if you are on a phone call with the student, "
        "confirming key details and guiding them step by step.\n\n"
        "You MUST follow these rules strictly:\n"
        "1) You are only allowed to use the information provided in the IST WEBSITE CONTEXT below.\n"
        "2) If the answer the student wants is not clearly supported by that context, or if the question is "
        "   not about IST admissions, you MUST reply with exactly this sentence and nothing else:\n"
        f"   \"{HUMAN_ESCALATION_MESSAGE}\"\n"
        "3) Do not invent program details, fees, or admission policies if they are not in the context.\n"
        "4) Keep answers concise, friendly, and easy to understand."
    )

    user_prompt = (
        f"Student message:\n{user_text}\n\n"
        "Here is context extracted from the official IST website. Use ONLY this context. "
        "If the context does not clearly contain the answer, or the question is outside IST admissions, "
        "then respond with the escalation sentence exactly as described in the rules.\n\n"
        f"IST WEBSITE CONTEXT:\n{ist_context}"
    )

    try:
        resp = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logging.exception("Groq LLM error: %s", e)
        GROQ_AVAILABLE = False
        return HUMAN_ESCALATION_MESSAGE


def synthesize_with_tts(text: str) -> str | None:
    """Generate an audio file for the counselor reply using local Windows TTS (pyttsx3)."""
    if not text.strip():
        return None

    out_path = LOG_AUDIO_DIR / f"reply_{int(time.time() * 1000)}.mp3"

    # Use the system TTS voice; fully local, no quotas
    tts_engine.save_to_file(text, str(out_path))
    tts_engine.runAndWait()
    return str(out_path)


def pipeline_fn(audio_path: str | None):
    """Full pipeline:

    1. Take mic audio.
    2. Save + apply simple VAD (trim silence).
    3. STT via Groq Whisper.
    4. LLM counselor answer (grounded in IST).
    5. TTS for the answer.
    """
    if not audio_path:
        return "No audio received.", "", None

    # Save original in logs for debugging / QA
    ts = int(time.time() * 1000)
    original_copy = LOG_AUDIO_DIR / f"raw_{ts}.wav"
    try:
        shutil.copy(audio_path, original_copy)
    except Exception as e:
        logger.warning("Failed to copy raw audio: %s", e)

    # 1) VAD / trimming
    vad_path = apply_simple_vad(audio_path)

    # 2) STT
    try:
        transcript = transcribe_audio(vad_path)
    except Exception as e:
        logger.exception("STT failed: %s", e)
        return "Transcription failed.", "", None

    if not transcript:
        return "Could not understand the audio.", "", None

    # 3) LLM 'counselor brain'
    try:
        reply_text = counselor_llm_response(transcript)
    except Exception as e:
        logger.exception("LLM failed: %s", e)
        return transcript, "LLM failed to generate a response.", None

    # 4) TTS for the reply
    try:
        tts_path = synthesize_with_tts(reply_text)
    except Exception as e:
        logger.exception("TTS failed: %s", e)
        tts_path = None

    return transcript, reply_text, tts_path


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Simple HTML UI: upload or record audio and see IST counselor response."""
    return """
    <html>
      <head>
        <title>IST Admissions Voice Agent</title>
      </head>
      <body>
        <h1>IST Admissions Voice Agent</h1>
        <p>
          Record a short question about IST admissions and upload the audio.
          The agent will answer using only information scraped from the IST website.
          If your question is outside that knowledge, it will schedule your call
          with a human counselor.
        </p>
        <form action="/chat" method="post" enctype="multipart/form-data">
          <label>Audio file (you can record with your mic):</label><br/>
          <input type="file" name="audio" accept="audio/*" capture="microphone" required><br/><br/>
          <button type="submit">Ask IST admissions agent</button>
        </form>
      </body>
    </html>
    """


@app.post("/chat", response_class=HTMLResponse)
async def chat(audio: UploadFile = File(...)) -> str:
    """Handle uploaded audio, run the pipeline, and render HTML with results."""
    # Save uploaded audio to a temp file
    upload_dir = ROOT_DIR / "logs" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time() * 1000)
    upload_path = upload_dir / f"{ts}_{audio.filename}"
    data = await audio.read()
    with upload_path.open("wb") as f:
        f.write(data)

    transcript, reply_text, tts_path = pipeline_fn(str(upload_path))

    audio_tag = ""
    if tts_path:
        # Expose the reply audio via the /logs/audio mount
        audio_url = "/logs/audio/" + os.path.basename(tts_path)
        audio_tag = f'<audio controls src="{audio_url}"></audio>'

    return f"""
    <html>
      <head>
        <title>IST Admissions Voice Agent - Result</title>
      </head>
      <body>
        <h1>IST Admissions Voice Agent</h1>
        <p><strong>Transcript:</strong> {transcript or "N/A"}</p>
        <p><strong>Agent reply:</strong> {reply_text or "N/A"}</p>
        <p><strong>Agent voice reply:</strong><br/>{audio_tag or "No audio generated."}</p>
        <p><a href="/">Ask another question</a></p>
      </body>
    </html>
    """


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

