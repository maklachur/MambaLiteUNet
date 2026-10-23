#!/usr/bin/env python3
"""Write 5%/10%/20% labeled-fraction splits for a local train id list.

Does not download ISIC, HAM10000, or PH2 and does not train a model.
See docs/QAM_SEG_SSL.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qamseg.splits import main


if __name__ == "__main__":
    raise SystemExit(main())
