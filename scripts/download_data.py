"""Download both datasets into data/raw and write data/sources.json."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detect.data import download  # noqa: E402

if __name__ == "__main__":
    force = "--force" in sys.argv
    for key, m in download(force=force).items():
        print(f"{key:18} {m['rows']:6} rows  sha256 {m['sha256'][:16]}…  license: {m['license']}")
