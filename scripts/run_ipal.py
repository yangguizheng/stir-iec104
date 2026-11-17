#!/usr/bin/env python3
"""Run SIMPLE using a user-supplied checkout of the IPAL IDS Framework."""
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_ipal.py FRAMEWORK_DIR [ipal-iids arguments]")
    framework = Path(sys.argv[1]).resolve()
    if not (framework / "ipal_iids" / "iids.py").is_file():
        raise SystemExit("FRAMEWORK_DIR must point to an IPAL IDS Framework checkout")
    sys.path.insert(0, str(framework))
    sys.argv = ["ipal-iids", *sys.argv[2:]]
    from ipal_iids import iids

    iids.main()


if __name__ == "__main__":
    main()
