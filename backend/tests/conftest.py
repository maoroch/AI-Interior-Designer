import sys
from pathlib import Path

# Позволяет запускать `pytest` из корня backend/ без установки пакета.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
