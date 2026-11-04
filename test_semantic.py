"""Test semantic search with a tiny fake embedder (no API key, no credit).

The fake embedder gives words with the same MEANING the same direction,
so we can check that search follows meaning, not exact keywords.
"""
import tools

MEANINGS = [
    {"vpn", "remote", "remotely", "home", "house", "outside", "reach"},  # working from outside
    {"slow", "laggy", "sluggish", "freezing", "performance"},          # slow computer
    {"password", "locked", "lockout", "unlock", "attempts"},           # accounts
    {"printer", "print", "printing", "paper", "scan"},                 # printers
]


def fake_embed(texts):
    vectors = []
    for text in texts:
        words = text.lower().replace(",", " ").replace(".", " ").split()
        vectors.append([sum(w in group for w in words) for group in MEANINGS])
    return vectors


def use_fake(monkeypatch, calls=None):
    def embed(texts):
        if calls is not None:
            calls.append(len(texts))
        return fake_embed(texts)
    monkeypatch.setattr(tools, "EMBED", embed)


def test_finds_vpn_by_meaning_where_keywords_fail(monkeypatch):
    use_fake(monkeypatch)
    question = "I can't reach work stuff from my house"
    assert tools.search_docs(question) == []  # no shared keywords with the VPN guide
    results = tools.semantic_search(question, min_similarity=0.5)
    assert results[0]["section"] == "VPN connection"


def test_finds_slow_computer_by_meaning(monkeypatch):
    use_fake(monkeypatch)
    results = tools.semantic_search("my laptop is super laggy", min_similarity=0.5)
    assert results[0]["section"] == "Slow computer"


def test_unrelated_question_returns_nothing(monkeypatch):
    use_fake(monkeypatch)
    assert tools.semantic_search("what is the weather today", min_similarity=0.5) == []


def test_section_vectors_are_cached(monkeypatch):
    calls = []
    use_fake(monkeypatch, calls)
    tools.semantic_search("printer jam", min_similarity=0.5)
    tools.semantic_search("printer jam again", min_similarity=0.5)
    assert calls.count(len(tools.load_sections())) == 1  # the 25 sections were embedded only once
    assert tools.EMBED_CACHE.exists()


def test_falls_back_to_keywords_when_embeddings_fail():
    # conftest.py makes the embedder fail, like a missing key or no internet
    results, mode = tools.smart_search("My VPN is not connecting")
    assert mode == "keyword"
    assert results[0]["section"] == "VPN connection"
