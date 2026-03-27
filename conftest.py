"""Root conftest — ensures ``src`` is importable when running ``pytest``."""

import sys
from pathlib import Path

# Add project root to sys.path so ``from src.xxx import ...`` works
# regardless of how pytest is invoked.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
