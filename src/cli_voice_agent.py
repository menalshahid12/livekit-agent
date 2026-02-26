import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from groq import Groq

from ist_knowledge import ISTDocument, load_ist_corpus, search, build_vector_index


logger = logging.getLogger("cli_voice_agent")
logging.basicConfig(level=logging.INFO)

ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_AUDIO_DIR = ROOT_DIR / "logs" / "audio"
LOG_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
CALL_LOG_PATH = ROOT_DIR / "logs" / "call_log.json"
CALL_RECORDS_PATH = ROOT_DIR / "logs" / "call_records.json"
LEAD_LOGS_PATH = ROOT_DIR / "logs" / "lead_logs.txt"

# TTS timeout so we never block forever (pyttsx3 can hang on Windows)
TTS_TIMEOUT_S = 20.0

load_dotenv(ROOT_DIR / ".env.local")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# When query is complex and cannot be answered from KB: forward to admin and ask for phone
HUMAN_ESCALATION_MESSAGE = (
    "We will forward this query to our admissions team. Please tell me your phone number so we can call you back."
)
GREETING_TEXT = (
    "Hello, this is Institute of Space Technology. How can I help you today?"
)

# Minimal embedded context when data folder is missing on deploy (e.g. Render). So we still answer programs/fees/contact.
EMBEDDED_FALLBACK_CONTEXT = """
TITLE: IST Departments and Programs
CONTENT: IST has 9 departments. BS programs: Aerospace, Avionics, Electrical, Computer, Mechanical, Materials Science, Biotechnology, Computer Science, Software Engineering, Data Science, AI, Space Science, Physics, Mathematics. MS and PhD in Aerospace, Electrical, Materials, Mechanical, CS, Mathematics, Physics, Astronomy. Admissions: 051 9075100, ist.edu.pk/admission.

TITLE: IST Fee Structure
CONTENT: BS Aerospace/Electrical/Avionics/Mechanical: about 1 lakh 48 thousand per semester. Materials: 1 lakh 42 thousand. Computing (CS, Software Eng, Data Science, AI): 1 lakh 26 thousand per semester. Other BS (Space Science, Mathematics, Physics, Biotechnology): 1 lakh 2 thousand per semester. One-time charges 49 thousand. MS about 1 lakh 20 thousand per year. PhD about 1 lakh 30 thousand per year. Admissions 051 9075100.
""".strip()

IST_DOCS: list[ISTDocument] = load_ist_corpus()
# Build vector index for semantic search (cost/fees, etc.) if chromadb + sentence-transformers available.
# Skip on Render/free tier to avoid slow OOM-prone startup (keyword search still works).
if not os.getenv("SKIP_VECTOR_INDEX"):
    build_vector_index(IST_DOCS)

# In-memory call log; each entry has call_start, call_end, stt_latency_s, llm_latency_s, tts_latency_s, e2e_s, transcript, escalated
CALL_LOG: list[dict[str, Any]] = []


def record_from_mic(seconds: int = 10, sample_rate: int = 16000) -> str:
    """Record from the default microphone for a fixed number of seconds."""
    import sounddevice as sd
    logger.info("Recording for %s seconds... Speak now.", seconds)
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    ts = int(time.time() * 1000)
    path = LOG_AUDIO_DIR / f"mic_{ts}.wav"
    sf.write(path, audio, sample_rate)
    logger.info("Saved recording to %s", path)
    return str(path)


def apply_simple_vad(in_path: str, threshold: float = 0.005) -> str:
    """Very basic VAD: trim leading/trailing low-energy segments and re-save as WAV."""
    try:
        audio, sr = sf.read(in_path)
    except Exception as e:
        logger.warning("Failed to read audio for VAD: %s", e)
        return in_path

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Simple energy-based VAD; threshold is intentionally low so we don't
    # accidentally trim away normal speech on quieter microphones.
    energy = np.abs(audio)
    mask = energy > threshold

    if not mask.any():
        # All below threshold, keep original to give STT a chance anyway
        logger.info("VAD found only silence, using original audio.")
        return in_path

    start = int(np.argmax(mask))
    end = int(len(mask) - np.argmax(mask[::-1]))
    trimmed = audio[start:end]

    # If trimming removed almost everything, fall back to original
    if len(trimmed) < int(0.5 * sr):
        logger.info("VAD trimmed too aggressively, using original audio.")
        return in_path

    out_path = LOG_AUDIO_DIR / f"vad_{int(time.time() * 1000)}.wav"
    sf.write(out_path, trimmed, sr)
    return str(out_path)


def transcribe_audio(audio_path: str, language: str | None = None) -> str:
    """STT using Groq's Whisper. Each call is independent — a transient error does NOT block future calls."""
    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        whisper_model = os.getenv("WHISPER_MODEL", "whisper-large-v3")
        if whisper_model not in ("whisper-large-v3", "whisper-large-v3-turbo"):
            whisper_model = "whisper-large-v3"
        kw: dict = {"file": (os.path.basename(audio_path), audio_bytes), "model": whisper_model}
        if language == "urdu":
            kw["language"] = "ur"
        elif language == "english":
            kw["language"] = "en"

        resp = groq_client.audio.transcriptions.create(**kw)
        text = (resp.text or "").strip()
        text = fix_common_stt_errors(text)
        logger.info("Transcript: %s", text)
        return text
    except Exception as e:
        logger.exception("Groq STT error (will retry next call): %s", e)
        return ""


def fix_common_stt_errors(transcript: str) -> str:
    """Correct common STT mishearings so fee/structure/cost queries are understood."""
    if not transcript:
        return transcript
    t = transcript
    # Fees often heard as "ees" or "ee"
    t = re.sub(r"\bees\b", "fees", t, flags=re.IGNORECASE)
    t = re.sub(r"\bee\b", "fee", t, flags=re.IGNORECASE)
    t = t.replace(" the ees ", " the fees ")
    t = t.replace(" what ees ", " what fees ")
    t = t.replace(" about ees ", " about fees ")
    # Fee vs free (common mishearing)
    t = t.replace("free structure", "fee structure")
    t = t.replace("Free structure", "Fee structure")
    t = t.replace("we structure", "fee structure")
    t = t.replace("We structure", "Fee structure")
    t = t.replace("be structure", "fee structure")
    t = t.replace("the structure", "fee structure")
    t = t.replace("free of", "fee of")
    t = t.replace("the free", "the fee")
    t = t.replace("what is the free", "what is the fee")
    t = t.replace("tell me free", "tell me fee")
    t = t.replace("free for", "fee for")
    t = t.replace(" free ", " fee ")
    t = t.replace("about free", "about fee")
    t = t.replace("about the free", "about the fee")
    # "cost" sometimes heard as "course" when asking about fees
    t = t.replace("course structure", "fee structure")
    t = t.replace("what is the course", "what is the fee")
    t = t.replace("tell me course", "tell me fee")
    # "programs" often misheard as "my name" or "pronouns" when asking what programs are offered
    t = t.replace("what's my name", "what programs")
    t = t.replace("what is my name", "what programs")
    t = t.replace("what my name", "what programs")
    t = t.replace("my name are offered", "programs are offered")
    t = t.replace("my name is offered", "programs are offered")
    t = t.replace("my name offered", "programs offered")
    t = t.replace("my name are", "programs are")
    t = t.replace("pronouns are offered", "programs are offered")
    t = t.replace("about pronouns", "about programs")
    t = t.replace("what pronouns", "what programs")
    t = re.sub(r"\bpronouns\b", "programs", t, flags=re.IGNORECASE)
    # IST often misheard as ISD
    t = t.replace(" in ISD", " in IST")
    t = t.replace(" in ISD.", " in IST.")
    # Incomplete query: "are offered in IST" -> "what programs are offered in IST"
    t = t.strip()
    if t.startswith("are offered") or t == "offered in IST." or t == "offered in IST":
        t = "what programs " + t
    if t.startswith("programs are offered in ISD"):
        t = t.replace("ISD", "IST")
    # Program names: STT often hears "electrical" as "musical", etc.
    t = t.replace("musical engineering", "electrical engineering")
    t = t.replace("musical department", "electrical department")
    t = t.replace("musical program", "electrical program")
    t = t.replace(" musical ", " electrical ")
    t = t.replace(" mechanism ", " mechanical ")
    t = t.replace("mechanic engineering", "mechanical engineering")
    t = t.replace("aviation engineering", "avionics engineering")
    t = t.replace("computer since", "computer science")
    t = t.replace("data since", "data science")
    return t.strip()


def build_ist_context(query: str, max_chars: int = 3500) -> str:
    docs = search(query, IST_DOCS, top_k=8)
    # If main query matched nothing, try a broad fallback so we don't escalate when knowledge base has content
    if not docs and IST_DOCS:
        docs = search("IST Institute of Space Technology admission programs fee merit eligibility", IST_DOCS, top_k=8)
    if not docs:
        return "No highly relevant IST website content was found for this question."

    snippets = []
    for d in docs:
        snippet = d.text[:800]
        snippets.append(
            f"TITLE: {d.title or 'N/A'}\nURL: {d.url}\nCONTENT: {snippet}"
        )
        joined = "\n\n".join(snippets)
        if len(joined) >= max_chars:
            break
    return "\n\n".join(snippets)


def counselor_llm_response(
    user_text: str,
    recent_turns: list[tuple[str, str]] | None = None,
    language: str = "english",
) -> str:
    """Groq LLM calling agent, strictly grounded in IST content. English only.
    recent_turns: optional [(user_msg, agent_reply), ...] from same call.
    language: kept for API compatibility; always answer in English.
    Each call is independent — a transient error does NOT block future calls.
    """
    recent_turns = recent_turns or []
    q_lower = user_text.lower().strip()
    # For short or referential follow-ups, augment search with previous query so retrieval stays strong for question 5, 6, 7...
    search_query = user_text
    if recent_turns and len(user_text.strip()) < 60:
        prev_user = recent_turns[-1][0].strip()
        ref_words = ("that", "it", "same", "what about", "and", "also", "for that", "about that", "there", "those", "this", "they")
        if prev_user and (len(user_text.split()) <= 5 or any(w in q_lower for w in ref_words)):
            search_query = f"{prev_user} {user_text}".strip()
    # Build main context from (possibly augmented) query
    ist_context = build_ist_context(search_query)
    # If still no context but we have docs, force-include a broad slice so we answer from KB instead of escalating
    if ist_context.startswith("No highly relevant") and IST_DOCS:
        fallback = build_ist_context("admission programs fee merit eligibility IST", max_chars=2500)
        if not fallback.startswith("No highly relevant"):
            ist_context = fallback
    # When we augmented for follow-up, also pull in context from previous query only so we have both topics
    if recent_turns and search_query != user_text:
        prev_ctx = build_ist_context(recent_turns[-1][0].strip(), max_chars=1800)
        if not prev_ctx.startswith("No highly relevant") and prev_ctx not in ist_context:
            ist_context = (prev_ctx + "\n\n---\n\n" + ist_context)[:5000]
    # When user asks about fees/cost/structure, ensure we pull in fee-related docs (handles synonyms and mishearings)
    q = user_text.lower()
    if any(w in q for w in ("fee", "fees", "free", "ees", "structure", "cost", "tuition", "charges", "course")):
        fee_ctx = build_ist_context("fee fees tuition cost structure")
        if not fee_ctx.startswith("No highly relevant") and fee_ctx not in ist_context:
            ist_context = fee_ctx + "\n\n" + ist_context
    # Fee for BS Physics, Space Science, Mathematics, Biotechnology (Other BS programs) — ensure we get the 1 lakh 2 thousand line
    if any(w in q for w in ("fee", "fees", "cost", "tuition")) and any(w in q for w in ("physics", "space science", "mathematics", "math ", "biotechnology", "humanities", "other bs")):
        other_bs_fee_ctx = build_ist_context("Other BS programs Space Science Mathematics Physics Biotechnology fee 1 lakh 2 thousand")
        if not other_bs_fee_ctx.startswith("No highly relevant") and other_bs_fee_ctx not in ist_context:
            ist_context = other_bs_fee_ctx + "\n\n" + ist_context
    # When user asks about programs/degrees offered, ensure we pull in program-list content
    if any(w in q for w in ("program", "programs", "offered", "degrees", "courses", "software engineering", "computer engineering")):
        prog_ctx = build_ist_context("programs offered degrees list software engineering computer engineering")
        if not prog_ctx.startswith("No highly relevant") and prog_ctx not in ist_context:
            ist_context = prog_ctx + "\n\n" + ist_context
    # For follow-up messages (e.g. "matric 800 fsc 900 test 70") also search merit so we have formula
    if recent_turns and ("merit" in q or any(c.isdigit() for c in user_text)):
        merit_ctx = build_ist_context("merit aggregate formula")
        if merit_ctx not in ist_context and not merit_ctx.startswith("No highly relevant"):
            ist_context = merit_ctx + "\n\n" + ist_context
    # When user asks about closing merit, last year merit, or will merit increase/decrease, include closing merit history
    if any(w in q for w in ("closing", "last year", "last year merit", "merit trend", "merit increase", "merit decrease", "merit up", "merit down", "cutoff", "cut off")):
        closing_ctx = build_ist_context("closing merit history last 6 years trend")
        if not closing_ctx.startswith("No highly relevant") and closing_ctx not in ist_context:
            ist_context = closing_ctx + "\n\n" + ist_context
    # When user asks about transport, hostel, scholarship, application rules, test optional, interview, laundry
    if any(w in q for w in (
        "transport", "bus", "buses", "hostel", "hostels", "scholarship", "financial", "financial assistance",
        "more than one program", "multiple program", "preference", "change major", "edit application", "test optional",
        "interview", "laundry", "dorm", "high school", "pre medical", "pre engineering", "ics"
    )):
        faq_ctx = build_ist_context("transport hostel scholarship application interview laundry")
        if not faq_ctx.startswith("No highly relevant") and faq_ctx not in ist_context:
            ist_context = faq_ctx + "\n\n" + ist_context
    # When user asks about admissions open, deadline, eligibility, entry test, NAT, documents, merit list, career
    if any(w in q for w in (
        "admission open", "admissions open", "deadline", "last date", "apply", "application", "portal", "arn",
        "eligibility", "fsc", "matric", "dae", "a-level", "equivalence", "ibcc", "merit list", "closing merit",
        "nat", "entry test", "hat", "net", "ecat", "etea", "sat", "documents", "cnic", "domicile", "attested",
        "career", "job", "orientation", "classes start", "fee refund", "installment", "challan", "hbl"
    )):
        full_faq_ctx = build_ist_context("admission FAQs deadlines eligibility entry test documents merit list")
        if not full_faq_ctx.startswith("No highly relevant") and full_faq_ctx not in ist_context:
            ist_context = full_faq_ctx + "\n\n" + ist_context
    # When user asks about foreign students, Al Khwarizmi, lunar mission, ICUBE, NCGSA, placements, journal, innovation
    if any(w in q for w in (
        "foreign", "scholarship", "al khwarizmi", "khwarizmi", "international student", "lunar", "moon", "icube",
        "cube sat", "cubesat", "chang", "ncgsa", "gis", "placement", "internship", "job fair", "journal of space",
        "jst", "innovation", "commercialization", "startup", "quality enhancement", "qec", "sports", "student committee"
    )):
        manual_ctx = build_ist_context(" ".join(q.split()[:8]))
        if not manual_ctx.startswith("No highly relevant") and manual_ctx not in ist_context:
            ist_context = manual_ctx + "\n\n" + ist_context

    if ist_context.startswith("No highly relevant IST website content was found"):
        # Last resort: if we have any docs loaded, use first few so we NEVER escalate when KB has content
        if IST_DOCS:
            snippets = []
            for d in IST_DOCS[:5]:
                snippet = d.text[:600]
                snippets.append(f"TITLE: {d.title or 'N/A'}\nCONTENT: {snippet}")
            ist_context = "\n\n".join(snippets)[:4000]
        if ist_context.startswith("No highly relevant IST website content was found"):
            # When data folder missing on deploy (e.g. Render): use embedded fallback so we answer programs/fees/contact
            logger.warning("Using embedded fallback context (data folder not loaded). Check Render: ensure 'data' is in repo.")
            ist_context = EMBEDDED_FALLBACK_CONTEXT
        if ist_context.startswith("No highly relevant IST website content was found"):
            return HUMAN_ESCALATION_MESSAGE

    # Build recent-conversation block ONCE (was wrongly referenced as recent_context in system_prompt before it was defined)
    recent_block = ""
    if recent_turns:
        recent_block = "Previous exchange in this call (for reference only — do NOT answer this again):\n"
        for u, a in recent_turns[-1:]:  # only last one exchange
            recent_block += f"Caller: {u}\nAgent: {a}\n"
        recent_block += (
            "\nCURRENT caller message (this is the ONLY question to answer): "
            "Answer ONLY this. If they asked about something new (e.g. fees, then programs), answer the NEW topic only. "
            "Do NOT say 'since you are asking about' or repeat their words. Do NOT link to the previous message unless they said 'that', 'same program', 'for it', 'what about that'.\n\n"
        )

    system_prompt = (
    "You are a polite, confident calling agent who handles IST (Institute of Space Technology) "
    "university admission queries. You speak as if you are on a phone call with the student, "
    "confirming key details and guiding them step by step.\n\n"
    "You MUST follow these rules strictly:\n"
    "1) You are only allowed to use the information provided in the IST WEBSITE CONTEXT below. "
    "DO NOT invent, guess, or make up ANY information, figures, or details.\n"
    "2) Be confident: Always present information as factual from IST sources. If the user says 'you are wrong' or disagrees, "
    "politely restate the facts from the context without changing them (e.g., 'According to IST's official information, ...').\n"
    "3) If the context is 'No highly relevant...', check if the question is a simple yes/no that can be answered accurately "
    "without specifics (e.g., 'Does IST offer BS programs?' → Yes, based on general KB). If yes, answer briefly. "
    "If not (needs details or complex), reply with exactly this sentence and nothing else: "
    f"\"{HUMAN_ESCALATION_MESSAGE}\"\n"
    "4) Keep responses SHORT: 1-2 sentences max. No filler, no pleasantries unless asked.\n"
    "5) NEVER repeat or paraphrase the user's question (e.g., no 'You asked about...'). Start directly with the answer.\n"
    "6) Handle long calls (10-12 questions): Use recent conversation for continuity, but stick to context.\n"
    "7) If escalated, ask for phone number ONLY once.\n"
    "8) ALWAYS try to give a direct, useful answer from the context — even if the context is fallback/basic information.\n"
    "9) NEVER say \"I am facing technical issue\" or \"I don't have information\" — always answer from the context or use the escalation sentence. For any IST-related question, you HAVE context; use it.\n"
    "10) If the question is about IST admissions, programs, fees, merit, dates, eligibility — answer using the knowledge even if it's basic/fallback.\n"
    "11) Only escalate (ask for phone number) when:\n"
    "    - The question is clearly personal / not about IST (e.g., \"what's my future\", \"who will win PSL\")\n"
    "    - OR the question requires very specific up-to-the-minute info that is NOT in knowledge (e.g., \"what is today's merit list position\")\n"
    "12) Be concise on the phone. 1–4 sentences max per answer.\n"
    "13) Do NOT repeat the user's question.\n"
    "14) Speak naturally like a phone agent.\n\n"
    f"IST WEBSITE CONTEXT:\n{ist_context}\n\n"
    f"{recent_block}"
)

    user_prompt = (
        f"{recent_block}{user_text}\n\n"
        "INSTRUCTIONS: Answer ONLY the question above in 1-2 sentences. Do NOT repeat or paraphrase the question. Do NOT add extra sentences or pleasantries. "
        "Start directly with the answer. If providing marks in reply to a previous request, compute aggregate and state only the total. "
        "If the caller asks to end the call, say only 'Thank you for calling IST. Goodbye.'\n\n"
        f"IST WEBSITE CONTEXT:\n{ist_context}"
    )

    llm_model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    try:
        resp = groq_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = (resp.choices[0].message.content or "").strip()
        logger.info("LLM reply: %s", answer)
        # Never return "technical issue" to the user — treat as bad output and escalate properly
        if answer and ("technical issue" in answer.lower() or "facing technical" in answer.lower() or "technical difficulties" in answer.lower()):
            logger.warning("LLM returned technical-issue phrasing; replacing with escalation message.")
            return HUMAN_ESCALATION_MESSAGE
        return answer
    except Exception as e:
        logger.exception("Groq LLM error (will retry next call): %s", e)
        return HUMAN_ESCALATION_MESSAGE


def _tts_worker(text: str, out_path: Path) -> bool:
    """Run TTS in a thread: fresh engine per call to avoid Windows hang. Only used on Windows."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.save_to_file(text, str(out_path))
        engine.runAndWait()
        del engine
        return True
    except Exception as e:
        logger.warning("TTS worker error: %s", e)
        return False


def _tts_edge(text: str, out_path: Path, language: str = "english") -> bool:
    """Natural TTS using Edge (Microsoft) neural voices: Pakistani Urdu + clear English, low delay."""
    try:
        import asyncio
        import edge_tts
        # Pakistani Urdu: ur-PK-UzmaNeural; English: en-US-JennyNeural (natural, clear)
        voice = "ur-PK-UzmaNeural" if language == "urdu" else "en-US-JennyNeural"
        save_path = str(out_path)
        async def _run():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(save_path)
        asyncio.run(_run())
        return Path(save_path).exists()
    except Exception as e:
        logger.warning("Edge TTS error: %s", e)
        return False


def _tts_gtts(text: str, out_path: Path, language: str = "english") -> bool:
    """Fallback TTS using gTTS (when Edge fails)."""
    try:
        from gtts import gTTS
        lang = "ur" if language == "urdu" else "en"
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(str(out_path))
        return True
    except Exception as e:
        logger.warning("gTTS error: %s", e)
        return False


def synthesize_with_tts(text: str, language: str = "english", session_id: str | None = None) -> str | None:
    """Natural voice: Edge TTS (Pakistani Urdu + US English). Fallback: pyttsx3 (Windows) or gTTS.
    session_id: optional, used in filename so concurrent calls do not overwrite each other's audio.
    """
    if not text.strip():
        return None

    ts = int(time.time() * 1000)
    prefix = f"{session_id}_" if session_id else ""
    wav_path = LOG_AUDIO_DIR / f"reply_{prefix}{ts}.wav"
    mp3_path = LOG_AUDIO_DIR / f"reply_{prefix}{ts}.mp3"

    # Prefer Edge TTS (natural, fast, Pakistani Urdu + clear English)
    if _tts_edge(text, mp3_path, language=language) and mp3_path.exists():
        logger.info("Saved TTS reply (Edge) to %s", mp3_path)
        return str(mp3_path)

    # On Linux/Render without Edge: gTTS fallback
    if os.name != "nt":
        if _tts_gtts(text, mp3_path, language=language) and mp3_path.exists():
            logger.info("Saved TTS reply (gTTS) to %s", mp3_path)
            return str(mp3_path)
        return None

    # Windows: try pyttsx3 then gTTS
    done = threading.Event()
    ok = [False]

    def run():
        ok[0] = _tts_worker(text, wav_path)
        done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    if done.wait(timeout=TTS_TIMEOUT_S) and ok[0] and wav_path.exists():
        logger.info("Saved TTS reply (pyttsx3) to %s", wav_path)
        return str(wav_path)
    if _tts_gtts(text, mp3_path, language=language) and mp3_path.exists():
        logger.info("Saved TTS reply (gTTS) to %s", mp3_path)
        return str(mp3_path)
    return None


def play_audio_file_blocking(file_path: str) -> None:
    """Play WAV/MP3 (or other format supported by soundfile) and block until done."""
    try:
        import sounddevice as sd
        data, sr = sf.read(file_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        sd.play(data, sr)
        sd.wait()
    except Exception as e:
        logger.warning("Could not play %s: %s. Open manually if needed.", file_path, e)
        if os.name == "nt" and os.path.isfile(file_path):
            os.startfile(file_path)  # type: ignore[attr-defined]


def save_call_log() -> None:
    """Persist CALL_LOG to logs/call_log.json."""
    try:
        CALL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CALL_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(CALL_LOG, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Could not save call log: %s", e)


def save_call_record(
    call_id: str,
    start_time: str,
    end_time: str,
    turns: list[tuple[str, str]],
    escalated: bool,
    phone_number: str | None = None,
) -> None:
    """Append one call record to logs/call_records.json (unique call_id, start/end, full Q&A, escalated, phone if any)."""
    try:
        CALL_RECORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        if CALL_RECORDS_PATH.exists():
            with open(CALL_RECORDS_PATH, encoding="utf-8") as f:
                records = json.load(f)
        record = {
            "call_id": call_id,
            "start_time": start_time,
            "end_time": end_time,
            "turns": [{"user": u, "agent": a} for u, a in turns],
            "escalated": escalated,
            "phone_number": phone_number,
        }
        records.append(record)
        with open(CALL_RECORDS_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        # When escalated and caller gave phone number, append to lead_logs for admin follow-up
        if escalated and phone_number:
            last_query = turns[-1][0].strip() if turns else ""
            try:
                LEAD_LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(LEAD_LOGS_PATH, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.now().isoformat()} | phone={phone_number} | call_id={call_id} | query={last_query}\n")
                logger.info("Lead log appended: phone=%s", phone_number[:6] + "***")
            except Exception as ex:
                logger.warning("Could not append lead log: %s", ex)
    except Exception as e:
        logger.warning("Could not save call record: %s", e)


def looks_like_phone_number(text: str) -> bool:
    """True if transcript looks like a Pakistani phone number (for post-escalation capture)."""
    if not text or len(text.strip()) < 10:
        return False
    digits = "".join(c for c in text if c.isdigit())
    return len(digits) >= 10 and (digits.startswith("03") or digits.startswith("92") or digits.startswith("3"))


def is_meaningful_transcript(transcript: str) -> bool:
    """False if user effectively said nothing (silence, filler, or too short). Avoids agent replying to noise."""
    if not transcript or not transcript.strip():
        return False
    t = transcript.strip()
    if len(t) < 2:
        return False
    t_lower = t.lower()
    # Filler / silence often transcribed as these (do not include "yes"/"no" - they can be language choice)
    filler = (
        "uh", "um", "hmm", "ah", "oh", "...", "na", "haan",
        "mm", "mhm", "err", "eh", "uh huh", "ok", "okay",
    )
    if t_lower in filler or t_lower.replace(".", "").replace("?", "") in filler:
        return False
    return True


def user_asked_to_end_call(transcript: str) -> bool:
    """True when the caller says they want to end the call (English or Urdu)."""
    if not transcript or len(transcript.strip()) < 3:
        return False
    t = transcript.strip().lower()
    end_phrases = (
        "end call", "end the call", "end call please", "goodbye", "good bye",
        "that's all", "that is all", "no more questions", "no more query",
        "have no query", "i have no query", "no query", "nothing else",
        "that's all thank you", "thank you goodbye", "bye", "bye bye",
        "bas", "khatam", "call khatam", "call end", "khatam karo", "call khatam karo",
        "aur nahi", "koi sawal nahi", "sawal nahi", "no more", "phone rakh do",
    )
    return any(p in t for p in end_phrases)


def print_call_log_entry(entry: dict[str, Any], index: int) -> None:
    """Print one call log entry to the console."""
    print("\n" + "=" * 60)
    print(f"CALL #{index} LOG")
    print("=" * 60)
    print(f"  Call started:  {entry['call_start']}")
    print(f"  Call ended:    {entry['call_end']}")
    print(f"  STT latency:   {entry['stt_latency_s']:.2f} s")
    print(f"  LLM latency:   {entry['llm_latency_s']:.2f} s")
    print(f"  TTS latency:   {entry['tts_latency_s']:.2f} s")
    print(f"  E2E round-trip: {entry['e2e_s']:.2f} s")
    print(f"  Transcript:    {entry.get('transcript', '')[:80]}...")
    print(f"  Escalated:     {entry.get('escalated', False)}")
    print("=" * 60 + "\n")


def print_average_delays() -> None:
    """Print average STT/LLM/TTS/E2E over the last 5+ calls."""
    if len(CALL_LOG) < 5:
        return
    recent = CALL_LOG[-5:]
    n = len(recent)
    avg_stt = sum(e["stt_latency_s"] for e in recent) / n
    avg_llm = sum(e["llm_latency_s"] for e in recent) / n
    avg_tts = sum(e["tts_latency_s"] for e in recent) / n
    avg_e2e = sum(e["e2e_s"] for e in recent) / n
    print("\n" + "=" * 60)
    print("AVERAGE RESPONSE DELAY (last 5 calls)")
    print("=" * 60)
    print(f"  Avg STT latency:   {avg_stt:.2f} s")
    print(f"  Avg LLM latency:   {avg_llm:.2f} s")
    print(f"  Avg TTS latency:   {avg_tts:.2f} s")
    print(f"  Avg E2E round-trip: {avg_e2e:.2f} s")
    print("=" * 60 + "\n")


def main() -> None:
    global CALL_LOG
    # Load existing call log if present (so "after 5 calls" can span runs)
    if CALL_LOG_PATH.exists():
        try:
            with open(CALL_LOG_PATH, encoding="utf-8") as f:
                CALL_LOG = json.load(f)
        except Exception:
            CALL_LOG = []

    print("IST Admissions Voice Agent — Admission queries on call")
    print("GROQ_API_KEY must be set in .env.local.")
    print("Press Enter to start a call. Say 'end call' or 'no more query' when done. Type 'q' to quit.")

    while True:
        cmd = input("> ").strip().lower()
        if cmd == "q":
            break

        call_start_iso = datetime.now().isoformat()
        call_start_wall = time.perf_counter()

        # 1) AI greeting
        print("\nPlaying greeting...")
        greeting_path = synthesize_with_tts(GREETING_TEXT)
        if greeting_path:
            play_audio_file_blocking(greeting_path)

        # 2) Call loop: keep taking questions until user says to end (no Enter between turns)
        call_id = str(uuid.uuid4())
        call_turns: list[tuple[str, str]] = []
        call_escalated = False
        while True:
            print("Recording your query (5 s)... Speak now.")
            audio_path = record_from_mic(seconds=5)
            vad_path = apply_simple_vad(audio_path)

            t_stt_start = time.perf_counter()
            transcript = transcribe_audio(vad_path, language="english")
            t_stt_end = time.perf_counter()
            stt_latency_s = t_stt_end - t_stt_start

            if not transcript:
                print("Could not understand the audio.")
                print("Press Enter to try again, or type 'q' to end call and quit.")
                if input("> ").strip().lower() == "q":
                    break
                continue
            if not is_meaningful_transcript(transcript):
                print("Could not understand (no clear speech). Listening again...")
                time.sleep(1)
                continue

            t_llm_start = time.perf_counter()
            reply = counselor_llm_response(transcript, recent_turns=call_turns, language="english")
            t_llm_end = time.perf_counter()
            llm_latency_s = t_llm_end - t_llm_start
            escalated = (
                "we will forward" in reply.lower() or "phone number" in reply.lower()
            )
            call_escalated = call_escalated or escalated
            call_turns.append((transcript, reply))

            t_tts_start = time.perf_counter()
            tts_path = synthesize_with_tts(reply, language="english")
            t_tts_end = time.perf_counter()
            tts_latency_s = t_tts_end - t_tts_start if tts_path else 0.0

            e2e_s = t_tts_end - t_stt_start
            call_end_iso = datetime.now().isoformat()

            # Play reply immediately so user hears the answer
            if tts_path:
                print("Playing reply...")
                play_audio_file_blocking(tts_path)
            else:
                print("(TTS failed or timed out; reply text below)")

            print(f"\nTranscript: {transcript}")
            print(f"Agent reply: {reply}\n")

            # Log this turn
            entry = {
                "call_start": call_start_iso,
                "call_end": call_end_iso,
                "stt_latency_s": round(stt_latency_s, 3),
                "llm_latency_s": round(llm_latency_s, 3),
                "tts_latency_s": round(tts_latency_s, 3),
                "e2e_s": round(e2e_s, 3),
                "transcript": transcript,
                "escalated": escalated,
            }
            CALL_LOG.append(entry)
            save_call_log()
            print_call_log_entry(entry, len(CALL_LOG))
            print_average_delays()

            # End call only when user clearly says they have no more questions
            if user_asked_to_end_call(transcript):
                # Print call summary: time and latencies for this call
                n_turns = len(call_turns)
                call_entries = CALL_LOG[-n_turns:] if n_turns else []
                if call_entries:
                    avg_stt = sum(e["stt_latency_s"] for e in call_entries) / len(call_entries)
                    avg_llm = sum(e["llm_latency_s"] for e in call_entries) / len(call_entries)
                    avg_tts = sum(e["tts_latency_s"] for e in call_entries) / len(call_entries)
                    avg_e2e = sum(e["e2e_s"] for e in call_entries) / len(call_entries)
                    print("=" * 60)
                    print("CALL ENDED — SUMMARY")
                    print("=" * 60)
                    print(f"  Call started:   {call_entries[0]['call_start']}")
                    print(f"  Call ended:     {call_entries[-1]['call_end']}")
                    print(f"  Turns in call: {n_turns}")
                    print(f"  Avg STT:       {avg_stt:.2f} s")
                    print(f"  Avg LLM:       {avg_llm:.2f} s")
                    print(f"  Avg TTS:       {avg_tts:.2f} s")
                    print(f"  Avg E2E:       {avg_e2e:.2f} s")
                    print("=" * 60 + "\n")
                save_call_record(
                    call_id, call_start_iso, datetime.now().isoformat(),
                    call_turns, call_escalated, None,
                )
                print("Call ended.\n")
                break

            # Same call continues: short pause then record next question (no Enter needed)
            print("Next question in 2 seconds... (say 'end call' or 'no more query' when you're done)")
            time.sleep(2)

        print("Press Enter to start a new call, or type 'q' to quit.")


if __name__ == "__main__":
    main()

