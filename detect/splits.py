"""The fixed pool, test and out-of-distribution (OOD) sets."""
from __future__ import annotations

import re

import pandas as pd

from .config import DATA, SPLITS
from .data import load, normalize

LAB_FILE = DATA / "ood" / "lab.csv"

_german = re.compile(r"\b(und|der|das|ich|nicht|ist|mit|eine|für|wie|sind|auch|nur|alle)\b", re.I)
_spanish = re.compile(r"\b(que|los|las|del|una|por|para|todo|dices|eres)\b", re.I)
_roleplay = re.compile(r"\b(act as|acting as|you are now|pretend|role[- ]?play|from now on you"
                       r"|agieren|fungieren|so tun, als|actúa como)\b", re.I)


def language(text: str) -> str:
    """Rough tag: 'de', 'es' or 'en'. A heuristic, not a language detector."""
    if re.search(r"[äöüß]", text, re.I) or len(_german.findall(text)) >= 2:
        return "de"
    if re.search(r"[ñ¿¡]", text) or len(_spanish.findall(text)) >= 2:
        return "es"
    return "en"


def is_roleplay(text: str) -> bool:
    return bool(_roleplay.search(text))


def build_pool_and_test() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train, test = load("safeguard", "train"), load("safeguard", "test")
    train["row"], test["row"] = train.index, test.index
    train["n"] = train["text"].map(normalize)
    log = {"train_rows": len(train)}

    leaked = train["n"].isin(set(test["text"].map(normalize)))
    log["removed_in_test"] = int(leaked.sum())
    pool = train[~leaked]

    conflicted = pool.groupby("n")["label"].transform("nunique") > 1
    log["removed_conflicting_labels"] = int(conflicted.sum())
    pool = pool[~conflicted]

    dup = pool["n"].duplicated()
    log["removed_duplicates"] = int(dup.sum())
    pool = pool[~dup]

    log["pool_rows"], log["test_rows"] = len(pool), len(test)
    return pool[["row", "label"]], test[["row", "label"]], log


def build_ood_deepset() -> tuple[pd.DataFrame, dict]:
    d = load("deepset", "test")
    d["row"] = d.index
    seen = set(load("safeguard", "train")["text"].map(normalize))
    overlap = d["text"].map(normalize).isin(seen)
    d = d[~overlap].copy()
    d["lang"] = d["text"].map(language)
    d["roleplay"] = d["text"].map(is_roleplay)
    log = {"removed_in_pool": int(overlap.sum()), "rows": len(d),
           "lang": d["lang"].value_counts().to_dict(),
           "roleplay": int(d["roleplay"].sum())}
    return d[["row", "label", "lang", "roleplay"]], log


def load_split(name: str) -> pd.DataFrame:
    """pool | test | ood_deepset | ood_lab -> DataFrame with text, label (+ tags)."""
    if name == "ood_lab":
        return pd.read_csv(LAB_FILE)
    ids = pd.read_csv(SPLITS / f"{name}.csv")
    source = "deepset" if name == "ood_deepset" else "safeguard"
    split = "train" if name == "pool" else "test"
    texts = load(source, split)["text"]
    ids.insert(1, "text", texts.loc[ids["row"]].to_numpy())
    return ids
