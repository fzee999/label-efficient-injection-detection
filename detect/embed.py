"""Text embeddings from the local model server, cached per split."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import requests

from .config import CACHE, EMBED_MODEL, EMBED_PREFIX, OLLAMA_URL
from .splits import load_split

BATCH = 64


def embed_texts(texts: list[str]) -> np.ndarray:
    """Unit-length float32 vectors, one row per text."""
    out = []
    for i in range(0, len(texts), BATCH):
        batch = [EMBED_PREFIX + t for t in texts[i:i + BATCH]]
        r = requests.post(f"{OLLAMA_URL}/api/embed", timeout=600,
                          json={"model": EMBED_MODEL, "input": batch, "truncate": True})
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
    x = np.asarray(out, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def fingerprint(texts: list[str]) -> str:
    """Changes if the model, the prefix or any text changes."""
    h = hashlib.sha256(f"{EMBED_MODEL}\n{EMBED_PREFIX}\n".encode())
    for t in texts:
        h.update(t.encode())
        h.update(b"\0")
    return h.hexdigest()


def embeddings(name: str) -> np.ndarray:
    """Embeddings for a split, computed once and reused while the fingerprint matches."""
    texts = load_split(name)["text"].tolist()
    vec_path, meta_path = CACHE / f"emb_{name}.npy", CACHE / f"emb_{name}.json"
    fp = fingerprint(texts)
    if vec_path.exists() and meta_path.exists() and json.loads(meta_path.read_text())["fingerprint"] == fp:
        return np.load(vec_path)
    CACHE.mkdir(parents=True, exist_ok=True)
    x = embed_texts(texts)
    np.save(vec_path, x)
    meta_path.write_text(json.dumps({"model": EMBED_MODEL, "prefix": EMBED_PREFIX,
                                     "rows": len(texts), "dim": int(x.shape[1]), "fingerprint": fp}))
    return x
