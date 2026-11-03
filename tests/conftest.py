"""Loaded by pytest before the tests: lets them `import verifier` etc. from src/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
