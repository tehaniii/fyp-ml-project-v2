import json
import re

METADATA_FILE = "coin_metadata.json"

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    coin_metadata = json.load(f)

def normalize_label(label):
    # Normalize label by removing hyphens, underscores, and extra spaces
    label = re.sub(r"[_\-–]+", " ", label)  # replaces _ - – with space
    label = re.sub(r"\s+", " ", label)  # collapse multiple spaces
    return label.strip().lower()

# Pre-process metadata keys for fuzzy matching
normalized_map = {normalize_label(k): v for k, v in coin_metadata.items()}

def get_metadata(label):
    key = normalize_label(label)
    return normalized_map.get(key)
