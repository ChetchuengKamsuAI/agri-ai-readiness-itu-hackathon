"""Configuration for the Agri AI Readiness prototype."""
from __future__ import annotations
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMADB_DIR = DATA_DIR / "chromadb"
LOG_DIR = DATA_DIR / "logs"
MODEL_DIR = DATA_DIR / "models"

_DEFAULT_DATASET = BASE_DIR.parent.parent / "repos" / "Sandbox_Training" / "Dataset"
DATASET_DIR = Path(os.environ.get("AGRIDS_DATASET", str(_DEFAULT_DATASET)))
UAE_POLICY_DIR = DATASET_DIR / "UAE" / "PolicyData"
UAE_USECASE_CSV = DATASET_DIR / "UAE" / "UsecaseData" / "Crop_recommendation.csv"

POLICY_FOLDERS = ["07_Food_Security_and_Agriculture", "02_Data_Protection", "01_AI_Strategy_and_Governance"]

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("AGRIDS_LLM_MODEL", "qwen2.5:14b")
EMBED_MODEL = os.environ.get("AGRIDS_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL_SMALL = "llama3.1:8b"

CHUNK_SIZE = 256
CHUNK_OVERLAP = 30
MAX_CHARS = 2000

KB_MAIN = "kb_agri_main"
KB_FRAMEWORK = "kb_framework"
KB_DECISIONS = "kb_decisions"

TOP_K = 5
SIMILARITY_THRESHOLD = 0.3
RF_N_ESTIMATORS = 100
RF_RANDOM_STATE = 42
TEST_SIZE = 0.2
SCENARIO_LOG_FILE = LOG_DIR / "scenario_audit.jsonl"

for d in (DATA_DIR, CHROMADB_DIR, LOG_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
