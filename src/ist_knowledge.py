"""
IST Knowledge Base Loading & Retrieval
Handles loading documents from data/ folder and provides search functionality.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────
# PATH RESOLUTION - important for Render / different environments
# ────────────────────────────────────────────────
def _data_dir() -> Path:
    possible_roots = [
        Path(__file__).resolve().parents[1],      # src/../ → project root
        Path.cwd(),                               # current working dir
        Path.cwd().parent,                        # sometimes Render does cd src/
    ]
    for root in possible_roots:
        d = root / "data"
        if d.exists() and (d / "FEE_STRUCTURE.txt").exists():
            logger.info(f"Found data directory at: {d}")
            return d
    # Last resort
    fallback = Path(__file__).resolve().parents[1] / "data"
    logger.warning(f"No valid data/ folder found. Using fallback: {fallback}")
    return fallback


DATA_DIR = _data_dir()


def get_data_dir_status() -> dict:
    """Return diagnostic info about the data directory (used by web_call_app debug endpoint)."""
    fee_path = DATA_DIR / "FEE_STRUCTURE.txt"
    manual_path = DATA_DIR / "IST_FULL_WEBSITE_MANUAL.txt"
    return {
        "data_dir": str(DATA_DIR),
        "data_dir_exists": DATA_DIR.exists(),
        "fee_structure_exists": fee_path.exists(),
        "manual_exists": manual_path.exists(),
    }


# Main files
MASTER_JSON_PATH          = DATA_DIR / "99_MASTER_JSON.json"
FEE_STRUCTURE_PATH        = DATA_DIR / "FEE_STRUCTURE.txt"
ADMISSION_DATES_PATH      = DATA_DIR / "ADMISSION_DATES_AND_STATUS.txt"
ADMISSION_FAQS_PATH       = DATA_DIR / "ADMISSION_FAQS_COMPLETE.txt"
CLOSING_MERIT_PATH        = DATA_DIR / "CLOSING_MERIT_HISTORY.txt"
DEPARTMENTS_PATH          = DATA_DIR / "IST_DEPARTMENTS_AND_PROGRAMS_SUMMARY.txt"
MERIT_CRITERIA_PATH       = DATA_DIR / "MERIT_CRITERIA_AND_AGGREGATE.txt"
TRANSPORT_HOSTEL_PATH     = DATA_DIR / "TRANSPORT_HOSTEL_FAQS.txt"
PROGRAMS_EXTRA_PATH       = DATA_DIR / "PROGRAMS_FEES_MERIT_EXTRA.txt"
ADMISSION_INFO_PATH       = DATA_DIR / "ADMISSION_INFO.txt"
ANNOUNCEMENTS_PATH        = DATA_DIR / "ANNOUNCEMENTS.txt"
FACILITIES_PATH           = DATA_DIR / "06_FACILITIES.txt"
FACULTY_PATH              = DATA_DIR / "07_FACULTY.txt"
DEPARTMENTS_EXTRA_PATH    = DATA_DIR / "05_DEPARTMENTS.txt"
RESEARCH_PATH             = DATA_DIR / "11_RESEARCH.txt"
FULL_WEBSITE_MANUAL_PATH  = DATA_DIR / "IST_FULL_WEBSITE_MANUAL.txt"

CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"

# Global vector store (populated only if chromadb is available)
_vector_collection = None
_docs_list: List["ISTDocument"] = []
_VECTOR_AVAILABLE = False

try:
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    _VECTOR_AVAILABLE = True
    logger.info("ChromaDB + sentence-transformers available → vector search enabled")
except ImportError as e:
    logger.info(f"Vector search disabled (chromadb or sentence-transformers not installed): {e}")


@dataclass
class ISTDocument:
    url: str
    title: str
    text: str

    def __post_init__(self):
        if not self.title:
            self.title = "Untitled Document"


def load_ist_corpus() -> List[ISTDocument]:
    """
    Load all available IST documents.
    Priority: master JSON → individual .txt files → hardcoded fallback
    """
    docs: List[ISTDocument] = []

    # 1. Try master JSON
    if MASTER_JSON_PATH.exists():
        try:
            with open(MASTER_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Adjust this parsing depending on your actual JSON structure
            documents = data.get("documents", data.get("categories", {}))
            if isinstance(documents, dict):
                for cat, items in documents.items():
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, str):
                                docs.append(ISTDocument(url=item, title=cat, text=""))
                            elif isinstance(item, dict):
                                docs.append(ISTDocument(
                                    url=item.get("url", ""),
                                    title=item.get("title", cat),
                                    text=item.get("text", item.get("content", ""))
                                ))
            logger.info(f"Loaded {len(docs)} entries from master JSON")
        except Exception as e:
            logger.error(f"Failed to parse {MASTER_JSON_PATH}: {e}")

    # 2. Load important individual .txt files
    txt_files = [
        (FEE_STRUCTURE_PATH,        "Fee Structure"),
        (ADMISSION_DATES_PATH,      "Admission Dates & Cycle"),
        (ADMISSION_FAQS_PATH,       "Admission FAQs Complete"),
        (CLOSING_MERIT_PATH,        "Closing Merit History"),
        (DEPARTMENTS_PATH,          "Departments & Programs Summary"),
        (MERIT_CRITERIA_PATH,       "Merit Criteria & Aggregate"),
        (TRANSPORT_HOSTEL_PATH,     "Transport, Hostel & FAQs"),
        (PROGRAMS_EXTRA_PATH,       "Programs, Fees & Merit Extra"),
        (ADMISSION_INFO_PATH,       "Admission Key Information"),
        (ANNOUNCEMENTS_PATH,        "Current Announcements"),
        (FACILITIES_PATH,           "Campus Facilities"),
        (FACULTY_PATH,              "Faculty Information"),
        (DEPARTMENTS_EXTRA_PATH,    "Departments Detailed"),
        (RESEARCH_PATH,             "Research Overview"),
        (FULL_WEBSITE_MANUAL_PATH,  "Full Website Manual Reference"),
    ]

    for path, default_title in txt_files:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    docs.append(ISTDocument(
                        url=str(path),
                        title=default_title,
                        text=content
                    ))
            except Exception as e:
                logger.warning(f"Could not read {path}: {e}")

    # 3. Strong embedded fallback if literally nothing was loaded
    if not docs:
        logger.warning("NO DOCUMENTS LOADED FROM DISK → USING HARDCODED FALLBACK")
        fallback_text = """
Institute of Space Technology (IST) - Key Information (Fallback)

Programs (BS):
• Aerospace Engineering
• Avionics Engineering
• Electrical Engineering
• Mechanical Engineering
• Materials Science and Engineering
• Computer Science
• Software Engineering
• Artificial Intelligence
• Data Science
• Space Science
• Mathematics
• Physics
• Biotechnology

Approximate BS fee per semester:
• Aerospace / Avionics / Electrical / Mechanical: ~1 lakh 48 thousand PKR
• Materials Science: ~1 lakh 42 thousand PKR
• Computing programs (CS, SE, AI, DS): ~1 lakh 26 thousand PKR
• Space Science / Math / Physics / Biotech: ~1 lakh 2 thousand PKR

One-time charges (all BS): ~49 thousand PKR

Admissions:
• Usually open February–March
• Close end of June (Fall intake only – no spring for BS)
• Merit list around August
• Classes start September

Contact: 051-9075100 | admissions@ist.edu.pk

Merit (engineering): typically 10% Matric + 40% FSC + 50% Entry Test
        """.strip()

        docs.append(ISTDocument(
            url="embedded:fallback",
            title="IST Admissions Fallback Information",
            text=fallback_text
        ))

    global _docs_list
    _docs_list = docs
    logger.info(f"Total documents loaded: {len(docs)}")
    return docs


def build_vector_index(docs: List[ISTDocument]) -> None:
    """Build Chroma vector index if possible"""
    global _vector_collection, _VECTOR_AVAILABLE

    if not _VECTOR_AVAILABLE:
        logger.info("Skipping vector index – ChromaDB not available")
        return

    if not docs:
        logger.warning("No documents to index")
        return

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

        collection = client.get_or_create_collection(
            name="ist_knowledge",
            embedding_function=embedding_fn
        )

        ids = [f"doc_{i}" for i in range(len(docs))]
        texts = [d.text for d in docs]
        metadatas = [{"title": d.title, "url": d.url} for d in docs]

        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )

        _vector_collection = collection
        logger.info(f"Vector index built with {len(docs)} documents")
    except Exception as e:
        logger.error(f"Failed to build vector index: {e}")
        _vector_collection = None


def simple_keyword_search(query: str, docs: List[ISTDocument], top_k: int = 5) -> List[ISTDocument]:
    """Fallback keyword-based search when vector is not available"""
    if not docs or not query:
        return []

    query_lower = query.lower()
    scored = []

    for doc in docs:
        text_lower = doc.text.lower()
        score = text_lower.count(query_lower)
        if score > 0:
            scored.append((score, doc))

    scored.sort(reverse=True)
    return [doc for _, doc in scored[:top_k]]


def vector_search(query: str, top_k: int = 5) -> List[ISTDocument]:
    """Semantic search using Chroma"""
    global _vector_collection

    if _vector_collection is None:
        return []

    try:
        results = _vector_collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        hits = []
        for doc_text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            hits.append(ISTDocument(
                url=meta.get("url", ""),
                title=meta.get("title", "Untitled"),
                text=doc_text
            ))
        return hits
    except Exception as e:
        logger.warning(f"Vector search failed: {e}")
        return []


def search(query: str, docs: Optional[List[ISTDocument]] = None, top_k: int = 5) -> List[ISTDocument]:
    """Main search entry point: vector if available, else keyword"""
    if docs is None:
        docs = _docs_list

    if not docs:
        return []

    if _vector_collection is not None:
        hits = vector_search(query, top_k)
        if hits:
            return hits

    # fallback
    return simple_keyword_search(query, docs, top_k)


def build_ist_context(
    query: str,
    docs: Optional[List[ISTDocument]] = None,
    max_chars: int = 3200
) -> str:
    """
    Build LLM context string from top relevant documents
    """
    if docs is None:
        docs = _docs_list

    if not docs:
        return "No IST knowledge documents are currently available."

    results = search(query, docs, top_k=8)

    # If nothing relevant → try broad fallback search
    if not results and docs:
        broad_query = "IST admission programs fees merit eligibility contact"
        results = search(broad_query, docs, top_k=6)

    if not results:
        return "No highly relevant IST information found for this question."

    snippets = []
    current_length = 0

    for doc in results:
        snippet = f"TITLE: {doc.title}\nURL: {doc.url}\nCONTENT: {doc.text[:900]}"
        if current_length + len(snippet) > max_chars:
            break
        snippets.append(snippet)
        current_length += len(snippet)

    return "\n\n".join(snippets)


def is_yes_no_question(text: str) -> bool:
    """Heuristic to detect simple yes/no questions.

    This is intentionally lightweight: checks if the first token is a
    common auxiliary verb used in yes/no questions.
    """
    if not text:
        return False
    text = text.strip().lower()
    tokens = text.split()
    if not tokens:
        return False
    starters = {
        "is", "are", "was", "were", "do", "does", "did",
        "can", "could", "would", "will", "shall", "should",
        "has", "have", "had"
    }
    return tokens[0] in starters


def init_knowledge(background_build: bool = True) -> List[ISTDocument]:
    """Load corpus and (optionally) ensure a vector index is available.

    If ChromaDB is available we will try to build the vector index. If a
    persistent `chroma_db` directory already exists we build synchronously;
    otherwise we optionally spawn a background thread to build so startup is
    fast on constrained hosts (e.g. Render free tier).
    """
    docs = load_ist_corpus()

    if not _VECTOR_AVAILABLE:
        logger.info("Vector search not available; skipping index build")
        return docs

    try:
        # Ensure persist dir exists
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        # If there's already content in the chroma folder, build now (blocking)
        if any(CHROMA_PERSIST_DIR.iterdir()):
            logger.info("Existing Chroma DB detected; building vector index now")
            build_vector_index(docs)
        elif background_build:
            logger.info("No Chroma DB found; starting background vector index builder")
            import threading

            t = threading.Thread(target=build_vector_index, args=(docs,), daemon=True)
            t.start()
        else:
            logger.info("Skipping vector build at startup (background_build=False)")
    except Exception as e:
        logger.warning(f"Failed to initialize vector index: {e}")

    return docs


# Initialize documents and attempt vector index build on import so web/agent
# code has a predictable fallback even when Chroma isn't present on deploy.
try:
    _docs_list = load_ist_corpus()
    # start background indexer if possible
    init_knowledge(background_build=True)
except Exception as e:
    logger.warning(f"IST knowledge initialization failed: {e}")