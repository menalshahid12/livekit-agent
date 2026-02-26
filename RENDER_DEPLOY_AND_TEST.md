# Deploy IST AI Calling Agent on Render

This guide covers deploying the **IST Voice Agent** (web call UI + RAG-based Q&A) on Render so it greets callers, answers from the knowledge base, and escalates only when needed (with lead logs and phone capture).

---

## What Was Fixed

- **"Technical issue" on every query**: The agent was hitting an error because `recent_context` was used in the system prompt but never defined. That is fixed: the conversation block is now built before the prompt and used correctly. The LLM reply is also filtered so any response containing "technical issue" or "technical difficulties" is replaced with the proper escalation message.
- **RAG and escalation**: The agent now answers from the IST knowledge base (Chroma/keyword search). It only escalates when the question is not in the KB and cannot be answered with a simple yes/no; then it asks for a phone number and saves it to **lead logs**.
- **Lead logs**: When a call is escalated and the caller provides a phone number, the app appends a line to `logs/lead_logs.txt` with timestamp, phone number, call ID, and the last query. Call records (with full Q&A and phone) are also saved in `logs/call_records.json`.
- **Barge-in vs noise**: The web UI stops playing the AI reply when you start speaking (barge-in). The threshold was increased so background noise is less likely to stop playback; only clearer speech stops it.
- **Concurrent users**: The app uses one Gunicorn worker with 6 threads, so multiple callers (e.g. 6–8) can be handled. Each call has its own session; logs are written with a lock to avoid corruption.

---

## Prerequisites

- **GitHub repo** with this project (including `data/` and optionally `chroma_db/`).
- **Render** account.
- **GROQ API key** for STT and LLM (create at [console.groq.com](https://console.groq.com)).

---

## Step 1: Prepare the Repo

1. Ensure these exist and are committed:
   - `data/` – folder with IST knowledge files (e.g. `IST_FULL_WEBSITE_MANUAL.txt`, `FEE_STRUCTURE.txt`, etc.). Required for RAG.
   - `chroma_db/` – optional. If you use Chroma for vector search, include a pre-built `chroma_db` folder (or an empty one with a `.gitkeep` so `COPY chroma_db/` in the Dockerfile does not fail).
2. If `chroma_db` is missing and you want to avoid Docker build errors, create it and add a placeholder:
   ```bash
   mkdir -p chroma_db
   echo "" > chroma_db/.gitkeep
   git add chroma_db/.gitkeep && git commit -m "Add chroma_db for Docker build"
   ```

---

## Step 2: Create a Web Service on Render

1. Log in to [Render](https://render.com) and open the dashboard.
2. **New → Web Service**.
3. Connect your GitHub account and select the repository that contains this project.
4. Configure:
   - **Name**: e.g. `ist-voice-agent`
   - **Region**: Choose the one closest to your users.
   - **Branch**: `main` (or your default branch).
   - **Runtime**: **Docker**.
   - **Dockerfile path**: `./Dockerfile` (or leave default if it’s in the root).
   - **Instance type**: Starter (or higher for more traffic).

---

## Step 3: Environment Variables

In the Render service → **Environment** tab, add:

| Key            | Value                    | Notes                          |
|----------------|--------------------------|--------------------------------|
| `GROQ_API_KEY` | `your-groq-api-key`      | **Secret**; required for STT/LLM. |
| `SKIP_VECTOR_INDEX` | `1`               | Optional; keeps startup fast; keyword search still works. |

Do **not** commit `GROQ_API_KEY` in the repo; set it only in Render.

---

## Step 4: Deploy and Health Check

1. Click **Create Web Service**. Render will build the Docker image and start the app.
2. The Dockerfile uses `PORT` set by Render: `gunicorn ... -b 0.0.0.0:${PORT:-8000} ...`.
3. **Health check**: In **Settings**, set:
   - **Health Check Path**: `/health`
   - The app returns `{"status":"ok"}` when running.

Build can take a few minutes. After deploy, the service URL will be like:

`https://ist-voice-agent-xxxx.onrender.com`

---

## Step 5: Testing After Deploy

### 5.1 Health and debug

- Open: `https://<your-service>.onrender.com/health`  
  Expected: `{"status":"ok"}`.

- Open: `https://<your-service>.onrender.com/api/debug`  
  Expected: JSON with `docs_loaded` (number of IST documents), `data_dir`, `fee_structure_exists`, `manual_exists`. If `docs_loaded` is 0, the `data/` folder may be missing from the build.

### 5.2 Full call flow

1. Open `https://<your-service>.onrender.com/` in a browser (Chrome/Edge recommended; allow microphone).
2. Click **Call** (or “Start AI Call”). Allow microphone when prompted.
3. You should hear the greeting: *“Hello, this is Institute of Space Technology. How can I help you today?”*
4. Speak or type (if the UI supports it). Ask e.g.:
   - “What programs do you offer?”
   - “What is the fee for BS Computer Science?”
   - “When do admissions open?”
5. The agent should answer from the knowledge base in 1–2 sentences, without saying “technical issue” or “I don’t have information” for IST-related questions.
6. Ask something **not** in the KB (e.g. “What will be my merit position tomorrow?”). The agent should respond with the escalation message and ask for your phone number.
7. Say a phone number (e.g. “03001234567”). After you end the call, check:
   - **Backend**: `logs/lead_logs.txt` should have a new line with timestamp, phone, call ID, and last query.
   - **Backend**: `logs/call_records.json` should have the call with `escalated: true` and `phone_number` set.

### 5.3 Multiple questions in one call

- Ask 5–10 questions in the same call (programs, fees, merit, hostels, deadlines). The agent should answer each from the KB without repeating your question and without escalating unless the question is out of scope.

### 5.4 Barge-in and noise

- While the agent is speaking, start speaking clearly: playback should stop and the app should start listening again.
- Background noise (e.g. fan, distant talk) should **not** stop playback; only clearer speech should (higher barge-in threshold).

### 5.5 Concurrent users

- Open the same URL in 2–3 devices or incognito windows. Start a call on each. All should get the greeting and be able to ask questions without interfering with each other (each has its own session).

---

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| “Technical issue” or “forward to admin” on every query | Ensure you have the latest code (fix for `recent_context` and LLM reply filter). Redeploy and clear browser cache. |
| No greeting / TTS fails | Check Render logs for errors. On Linux, TTS uses gTTS if Edge TTS is not available; ensure `GROQ_API_KEY` is set. |
| “No audio” / “Could not understand” | Allow microphone for the site; use HTTPS; check `/api/debug` for `docs_loaded` > 0. |
| `docs_loaded` is 0 | Ensure `data/` is in the repo and not in `.dockerignore`. Rebuild on Render. |
| Build fails on `COPY chroma_db/` | Add an empty `chroma_db/` directory with a `.gitkeep` file and commit it. |
| Health check fails | Confirm Health Check Path is `/health` and the service is listening on `PORT`. |

---

## Files Reference

- **App entry**: `src/web_call_app.py` (Flask; run with Gunicorn).
- **LLM + RAG logic**: `src/cli_voice_agent.py` (`counselor_llm_response`, `build_ist_context`).
- **Knowledge load/search**: `src/ist_knowledge.py`.
- **Lead log path**: `logs/lead_logs.txt` (created at runtime; do not commit PII).
- **Call records**: `logs/call_records.json`.

---

## Optional: More concurrent users

Default is 1 worker and 6 threads. For more simultaneous callers you can increase workers (and possibly instance size) by changing the Dockerfile CMD, for example:

```dockerfile
CMD ["sh", "-c", "exec gunicorn -w 2 --threads 4 -b 0.0.0.0:${PORT:-8000} --timeout 120 --chdir src web_call_app:app"]
```

Adjust `-w` and `--threads` based on Render plan and memory.
