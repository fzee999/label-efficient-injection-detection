"""Download, verify and load the datasets."""
from __future__ import annotations

import hashlib
import json
import re

import pandas as pd
import requests

from .config import DATA, RAW

PARQUET_URL = ("https://huggingface.co/datasets/{repo}/resolve/"
               "refs%2Fconvert%2Fparquet/default/{split}/0000.parquet")

SOURCES = {
    "safeguard": {
        "repo": "xTRam1/safe-guard-prompt-injection",
        "revision": "a3a877d608f37b7d20d9945671902df895ecdb46",
        "license": "not stated",
        "splits": ["train", "test"],
    },
    "deepset": {
        "repo": "deepset/prompt-injections",
        "revision": "4f61ecb038e9c3fb77e21034b22511b523772cdd",
        "license": "apache-2.0",
        "splits": ["train", "test"],
    },
}

MANIFEST = DATA / "sources.json"


def raw_path(name: str, split: str):
    return RAW / f"{name}_{split}.parquet"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(force: bool = False) -> dict:
    """Fetch every split; record url, checksum and row count in the manifest."""
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    for name, src in SOURCES.items():
        for split in src["splits"]:
            path = raw_path(name, split)
            url = PARQUET_URL.format(repo=src["repo"], split=split)
            if force or not path.exists():
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                path.write_bytes(r.content)
            digest = sha256(path)
            key = f"{name}/{split}"
            expected = manifest.get(key, {}).get("sha256")
            if expected and expected != digest:
                raise RuntimeError(f"{key}: file changed since it was first recorded")
            manifest[key] = {
                "repo": src["repo"], "revision": src["revision"], "license": src["license"],
                "url": url, "sha256": digest, "rows": len(pd.read_parquet(path)),
            }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load(name: str, split: str) -> pd.DataFrame:
    """Columns: text (str), label (int, 1 = injection)."""
    df = pd.read_parquet(raw_path(name, split))
    df = df.rename(columns={"prompt": "text"})[["text", "label"]]
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


_ws = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lower-case, collapse whitespace; used to find near-duplicates."""
    return _ws.sub(" ", text.lower()).strip()
