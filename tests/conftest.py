"""Put experiments/ on the path: branches A and B are experiment code and live there, but their
samplers are subclasses of the published Steerer and so are worth testing like any other."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
