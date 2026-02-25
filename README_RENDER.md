# Deploying IST Voice Agent to Render

Quick steps to deploy the `my-agent` service to Render with a persistent knowledge base.

1) Ensure `data/` (all txt files) and `chroma_db/` (if prebuilt) are present in the repository root under `my-agent/data` and `my-agent/data/chroma_db`.

2) Configure environment variables in Render:
- `GROQ_API_KEY` — required for Groq STT/LLM calls (if used).
- `OPENAI_API_KEY` — optional if OpenAI TTS/LLM used.

3) Use the provided `render.yaml` or create a new Web service pointing to the repo and use the start command in `Procfile`.

4) Start with `plan: free` for testing, but note: installing `chromadb` and `sentence-transformers` may exceed build limits on free plans. For best reliability, prebuild `chroma_db` locally and commit it to the repo.

5) Test the deployed service by visiting the Render URL and uploading a sample audio file.

Notes:
- If `chroma_db` is missing, the app will serve using keyword fallback and will build the vector index in background (if `chromadb` is available). This allows faster startup on constrained hosts.
- For concurrency, the Procfile uses 4 Gunicorn workers; adjust based on CPU/memory.
