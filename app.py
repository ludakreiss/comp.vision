"""
Hugging Face Spaces Entry Point
Deepfake Detection Robustness Demo
"""

import sys
from pathlib import Path

# Ensure repository root and demo directory are in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEMO_DIR = ROOT_DIR / "demo"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

try:
    import spaces
except ImportError:
    pass

from demo.app import demo

if __name__ == "__main__":
    demo.launch()
