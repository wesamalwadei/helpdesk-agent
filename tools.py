"""Tools for the IT helpdesk agent: search the IT docs and create support tickets."""
import json
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
