#!/usr/bin/env python3
"""Cache passive protocol-check outputs for one Sherlock scenario."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    out = E.save_protocol(sys.argv[1])
    print(f"{sys.argv[1]}: {out} ({time.time() - t0:.0f} s)")
