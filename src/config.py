import os
from pathlib import Path

# Get the project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

CACHE_DIR = str(PROJECT_ROOT / 'data' / 'cache')
METADATA_DIR = str(PROJECT_ROOT / 'data' / 'metadata')
DATA_DIR = str(PROJECT_ROOT / 'data')