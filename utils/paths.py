from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATASETS_DIR: Final[Path] = PROJECT_ROOT / "datasets"
DATASETS_RAW_DIR: Final[Path] = DATASETS_DIR / "raw"
DATASETS_CLEANED_DIR: Final[Path] = DATASETS_DIR / "cleaned"
DATASETS_PREPROCESSED_DIR: Final[Path] = DATASETS_DIR / "preprocessed"
NOTEBOOKS_DIR: Final[Path] = PROJECT_ROOT / "notebooks"
