#!/usr/bin/env python3
"""Entry point — ensures `src/` is on path when run without installation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from facemood.app.application import main

if __name__ == '__main__':
    main()

