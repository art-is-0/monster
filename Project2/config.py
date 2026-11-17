from pathlib import Path

SEED = 67

ROOT = Path(__file__).resolve().parent  # Repo root

FIGURES_PATH = ROOT / "figures"
RESULTS_PATH = ROOT / "results"
PICTURE_PATH = ROOT / "pictures"

FIGURES_PATH.mkdir(exist_ok=True)
RESULTS_PATH.mkdir(exist_ok=True)