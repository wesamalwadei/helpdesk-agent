"""Tools for the IT helpdesk agent: search the IT docs and create support tickets.

Two ways to search the guides:
- semantic_search: by MEANING, with embeddings (the default, see smart_search)
- search_docs: by KEYWORDS (used as a fallback if embeddings are not available)
"""
import hashlib
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

DOCS_PATH = Path(__file__).parent / "it_docs.md"
TICKETS_PATH = Path(__file__).parent / "tickets.json"

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "to", "of", "and", "or", "in", "on",
    "for", "my", "i", "me", "do", "does", "how", "what", "who", "can", "it",
    "with", "if", "your", "you", "be", "not", "from", "at", "this", "that",
    "should", "will", "get", "need", "help", "please", "our", "we", "by",
}


def words(text):
    """Lowercase keywords without stopwords."""
    text = text.lower().replace("wi-fi", "wifi")
    return {w for w in re.findall(r"[a-z0-9-]+", text) if w not in STOPWORDS}


def load_sections(path=DOCS_PATH):
    """Split the docs into (title, body) pairs using '## ' headings."""
    text = Path(path).read_text(encoding="utf-8")
    sections = []
    for part in text.split("\n## ")[1:]:
        title, _, body = part.partition("\n")
        sections.append((title.strip(), body.strip()))
    return sections


def search_docs(query, top_k=2, min_score=2):
    """Return the best matching sections, or an empty list if nothing matches."""
    query_words = words(query)
    results = []
    for title, body in load_sections():
        title_hits = len(query_words & words(title))
        body_hits = len(query_words & words(body))
        score = title_hits * 2 + body_hits
        if score >= min_score:
            results.append({"section": title, "text": body, "score": score})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


# ---------- Semantic search (embeddings) ----------
EMBED_CACHE = Path(__file__).parent / "embeddings.json"


# Read from .env when used (not at import), so values set in .env always apply.
def embed_model():
    return os.getenv("EMBED_MODEL", "openai/text-embedding-3-small")


def min_similarity_setting():
    return float(os.getenv("MIN_SIMILARITY", "0.35"))


def openrouter_embed(texts):
    """Turn texts into vectors with an embedding model on OpenRouter."""
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
    response = client.embeddings.create(model=embed_model(), input=texts)
    return [item.embedding for item in response.data]


EMBED = openrouter_embed  # tests replace this with a fake embedder


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def section_vectors(cache=None):
    """Vectors for every section, computed once and saved to embeddings.json.

    They are computed again only when the guides (or the embedding model) change.
    """
    cache = Path(cache or EMBED_CACHE)
    sections = load_sections()
    texts = [f"{title}\n{body}" for title, body in sections]
    key = hashlib.sha256((embed_model() + "\n".join(texts)).encode("utf-8")).hexdigest()
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved.get("key") == key:
            return sections, saved["vectors"]
    vectors = EMBED(texts)
    cache.write_text(json.dumps({"key": key, "model": embed_model(), "vectors": vectors}), encoding="utf-8")
    return sections, vectors


def semantic_search(query, top_k=2, min_similarity=None, cache=None):
    """Return the sections closest in MEANING to the query, or [] if none is close enough."""
    min_similarity = min_similarity_setting() if min_similarity is None else min_similarity
    sections, vectors = section_vectors(cache)
    query_vector = EMBED([query])[0]
    scored = sorted(
        ((cosine(query_vector, v), title, body) for (title, body), v in zip(sections, vectors)),
        reverse=True,
    )
    return [{"section": t, "text": b, "score": round(sim, 3)}
            for sim, t, b in scored[:top_k] if sim >= min_similarity]


def smart_search(query):
    """Semantic search, falling back to keyword search if embeddings fail.

    Returns (results, mode) where mode is "semantic" or "keyword".
    """
    try:
        return semantic_search(query), "semantic"
    except Exception:
        return search_docs(query), "keyword"


def create_ticket(summary, details, path=TICKETS_PATH, category="other", priority="medium"):
    """Save a support ticket to tickets.json and return it."""
    path = Path(path)
    tickets = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    ticket = {
        "id": f"IT-{len(tickets) + 1:04d}",
        "summary": summary,
        "details": details,
        "category": category,
        "priority": priority,
        "status": "open",
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    tickets.append(ticket)
    path.write_text(json.dumps(tickets, indent=2, ensure_ascii=False), encoding="utf-8")
    return ticket
