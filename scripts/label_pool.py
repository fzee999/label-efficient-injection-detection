"""Label every pool text once with the LLM; resumable. Writes results/labeler/pool_labels.csv."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detect.config import LABELER_MODEL, RESULTS  # noqa: E402
from detect.labeler import INSTRUCTION, SYSTEM, label_text  # noqa: E402
from detect.splits import load_split  # noqa: E402

OUT = RESULTS / "labeler" / "pool_labels.csv"
META = RESULTS / "labeler" / "pool_labels.json"
FIELDS = ["pool_index", "p_injection", "label", "ok", "letter_mass", "raw", "seconds"]


def prompt_fingerprint() -> str:
    return hashlib.sha256(f"{LABELER_MODEL}\n{SYSTEM}\n{INSTRUCTION}".encode()).hexdigest()


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fp = prompt_fingerprint()
    if META.exists() and json.loads(META.read_text())["prompt_fingerprint"] != fp:
        raise SystemExit("labels on disk came from a different prompt or model; move them aside first")
    META.write_text(json.dumps({"model": LABELER_MODEL, "prompt_fingerprint": fp}, indent=2) + "\n")

    pool = load_split("pool")
    done = set()
    if OUT.exists():
        with OUT.open(newline="", encoding="utf-8") as f:
            done = {int(r["pool_index"]) for r in csv.DictReader(f)}
    todo = [i for i in range(len(pool)) if i not in done]
    print(f"{len(done)} already labeled, {len(todo)} to go", flush=True)

    new = not OUT.exists()
    t0 = time.perf_counter()
    with OUT.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for k, i in enumerate(todo, 1):
            r = label_text(pool.text.iloc[i])
            w.writerow({"pool_index": i, **{key: r[key] for key in FIELDS[1:]}})
            if k % 250 == 0:
                f.flush()
                rate = k / (time.perf_counter() - t0)
                print(f"{len(done) + k}/{len(pool)}  {rate:.1f}/s  ~{(len(todo) - k) / rate / 60:.0f} min left", flush=True)
    print(f"finished in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
