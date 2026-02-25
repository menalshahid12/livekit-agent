import logging
import os
from livekit.agents import JobContext, WorkerOptions, JobProcess, voice_assistant
from livekit.plugins import openai, silero, rag, groq, xai
from livekit_chroma import ChromaVectorStore # Ensure this is installed

logger = logging.getLogger("my-agent")

async def entrypoint(ctx: JobContext):
    # 1. SETUP KNOWLEDGE BASE (RAG)
    # We use a pre-built chroma_db folder pushed to GitHub
    store = ChromaVectorStore(persist_directory="./chroma_db")
    
    # 2. DEFINE SYSTEM INSTRUCTIONS
    # Stay confident, don't repeat, handle fallback
    instructions = (
        "You are a professional assistant. "
        "Use ONLY the provided context to answer questions. "
        "If someone says you are wrong, stay polite but remain confident in your facts. "
        "If you don't know the answer: check if it's a simple Yes/No question. "
        "If it's not a Yes/No question, say: 'I will forward your query to the admin office. "
        "Please provide your phone number so they can call you back.'"
    )

    # 3. CONFIGURE AGENT (Interruption & Noise)
    # min_endpointing_delay prevents stopping on small noises
    assistant = voice_assistant.VoiceAssistant(
        vad=silero.VAD.load(),
        stt=groq.STT(), # Fast & reliable
        llm=xai.LLM(model="grok-beta"), # Intelligent reasoning
        tts=openai.TTS(),
        chat_ctx=openai.ChatContext().append(role="system", text=instructions),
    )

    # 4. HANDLE LOGGING (Phone numbers & Logs)
    @assistant.on("user_speech_committed")
    def on_speech(msg):
        # Check if user provided a phone number to save it
        import re
        # crude phone number regex: sequences of 8+ digits possibly separated by spaces or dashes
        m = re.search(r"(\+?\d[\d\s\-]{7,}\d)", msg.content)
        if m:
            phone = m.group(1).strip()
            with open("lead_logs.txt", "a", encoding="utf-8") as f:
                f.write(f"Lead Found: {phone} -- source: {msg.content}\n")

    @assistant.on("user_speech_start")
    def on_user_start(msg):
        """When the user begins speaking, interrupt any ongoing assistant TTS.

        This allows the assistant to stop speaking and listen to the user mid-reply.
        """
        try:
            # VoiceAssistant implementations may expose a stop/resume API for playback
            if hasattr(assistant, "stop_speaking"):
                assistant.stop_speaking()
        except Exception:
            logger.debug("Assistant stop_speaking() not available or failed")

    await ctx.connect()
    assistant.start(ctx.room)
    await assistant.say("Hello! How can I help you today?", allow_interruptions=True)

if __name__ == "__main__":
    WorkerOptions(entrypoint_fnc=entrypoint).run()