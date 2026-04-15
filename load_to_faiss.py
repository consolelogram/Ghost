import json
import numpy as np
import faiss

# ---------- CONFIG ----------
INPUT_FILE = "embedded_blocks.json"
FAISS_INDEX_FILE = "ghost.index"
TEXT_STORE_FILE = "ghost_texts.json"
DIM = 384
# ----------------------------

# Load embedded blocks
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    blocks = json.load(f)

print(f"Loaded {len(blocks)} blocks")

# Build FAISS index (cosine similarity via inner product on normalized vectors)
index = faiss.IndexFlatIP(DIM)

vectors = []
texts = {}

for block in blocks:
    vec = np.array(block["vector"], dtype=np.float32)
    # Normalize for cosine similarity
    vec /= np.linalg.norm(vec) + 1e-10
    vectors.append(vec)
    texts[block["id"]] = block["text"]

matrix = np.stack(vectors)
index.add(matrix)

# Save index and text store
faiss.write_index(index, FAISS_INDEX_FILE)

with open(TEXT_STORE_FILE, "w", encoding="utf-8") as f:
    json.dump(texts, f)

print(f"FAISS index saved to {FAISS_INDEX_FILE}")
print(f"Text store saved to {TEXT_STORE_FILE}")
print(f"Total vectors indexed: {index.ntotal}")
