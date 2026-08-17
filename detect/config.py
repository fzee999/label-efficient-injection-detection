"""Paths and settings in one place. Override any of them with environment
variables or a .env file in the project root."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
RAW = DATA / "raw"          # downloaded source data (git-ignored)
CACHE = DATA / "cache"      # embeddings and other derived arrays (git-ignored)
SPLITS = DATA / "splits"    # the fixed pool / test / OOD id lists (committed)
RESULTS = ROOT / "results"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
# nomic-embed-text was trained with task prefixes; "classification: " is the
# one meant for features that feed a classifier.
EMBED_PREFIX = os.environ.get("EMBED_PREFIX", "classification: ")
LABELER_MODEL = os.environ.get("LABELER_MODEL", "llama3.1:8b-8k")

SEEDS = list(range(10))
