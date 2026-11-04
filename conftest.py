"""Shared test setup: tests never call the real embeddings API (no key, no credit)."""
import pytest

import tools


def _no_embeddings(texts):
    raise RuntimeError("no real embeddings in tests")


@pytest.fixture(autouse=True)
def offline_embeddings(monkeypatch, tmp_path):
    # Without embeddings, smart_search falls back to keyword search.
    monkeypatch.setattr(tools, "EMBED", _no_embeddings)
    monkeypatch.setattr(tools, "EMBED_CACHE", tmp_path / "embeddings.json")
