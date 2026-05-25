"""Project paths.

Centralised so notebooks and helper modules never hardcode filesystem layout.
The folder names are deliberately Spanish (project artefacts the director will
see); identifiers in this file stay English (code).
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Data
DATOS_BRUTOS = PROJECT_ROOT / "datos" / "brutos"
DATOS_PROCESADOS = PROJECT_ROOT / "datos" / "procesados"

CSV_RAW = DATOS_BRUTOS / "healthcare_noshows.csv"
CSV_FULL_CLEAN = DATOS_PROCESADOS / "full_clean_v1.csv"
CSV_TRAIN = DATOS_PROCESADOS / "train_v1.csv"
CSV_TEST = DATOS_PROCESADOS / "test_v1.csv"

# Outputs
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURAS = OUTPUTS / "figuras"
MODELOS = OUTPUTS / "modelos"
REPORTES = OUTPUTS / "reportes"
BOOTSTRAP = OUTPUTS / "bootstrap"

# Convention: random seed shared across all notebooks for reproducibility
SEED = 42
