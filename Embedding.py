import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
INPUT_FILE = "blocks.json"
OUTPUT_FILE = "embedded_blocks.json"
MODEL_NAME = "all-MiniLM-L6-v2"
# ----------------------------

def main():
    # Load blocks
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        blocks = json.load(f)

    print(f"Loaded {len(blocks)} blocks")

    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded: {MODEL_NAME}")

    embedded_blocks = []

    for idx, text in enumerate(blocks):
        text = text.strip()
        if not text:
            continue

        vector = model.encode(text)

        embedded_blocks.append({
            "id": idx,
            "text": text,
            "vector": vector.tolist()
        })


    # Save embeddings
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(embedded_blocks, f)

    print(f"Saved {len(embedded_blocks)} embedded blocks to {OUTPUT_FILE}")

    # Checks
    dims = set(len(b["vector"]) for b in embedded_blocks)
    print("Embedding dimensions:", dims)

    sample_vec = np.array(embedded_blocks[0]["vector"])
    print("Contains NaN:", np.isnan(sample_vec).any())

if __name__ == "__main__":
    main()
