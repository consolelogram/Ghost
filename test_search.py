import json
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
FAISS_INDEX_FILE = "ghost.index"
TEXT_STORE_FILE = "ghost_texts.json"
TOP_K = 30
# ----------------------------

# -------- setup --------
index = faiss.read_index(FAISS_INDEX_FILE)

with open(TEXT_STORE_FILE, "r", encoding="utf-8") as f:
    raw = json.load(f)
    # JSON keys are always strings, convert back to int
    texts = {int(k): v for k, v in raw.items()}

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def energy_score(text):
    return (
        len(text.split())
        + text.count("!") * 2
        + text.count("?")
        + sum(1 for c in text if c.isupper())
    )

def call_llm(prompt):
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3:mini",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 40
                }
            }
        )
        return res.json()["response"].strip()
    except Exception as e:
        return f"[LLM Error: {e}]"

print("Watashi wa Ghost. Ghost wa okiteiru. Empty input to exit.\n")

# -------- chat loop --------
while True:
    query = input("You: ").strip()
    if not query:
        break

    q_energy = energy_score(query)

    # Embed and normalize query
    q_vec = embed_model.encode(query).astype(np.float32)
    q_vec /= np.linalg.norm(q_vec) + 1e-10
    q_vec = q_vec.reshape(1, -1)

    # ---- FAISS retrieval ----
    scores, indices = index.search(q_vec, TOP_K)

    # FAISS IndexFlatIP returns cosine similarity (higher = better)
    # Convert to distance (lower = better) to keep the same logic as before
    scored = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        dist = 1.0 - float(score)  # cosine distance
        text = texts.get(idx, "")
        scored.append((dist, text))

    # ---- energy filter ----
    filtered = [
        (d, t) for d, t in scored
        if abs(energy_score(t) - q_energy) < 10
    ]
    if not filtered:
        filtered = scored

    # ---- selector ----
    # Sort ASCENDING (low distance = better match)
    filtered.sort(key=lambda x: x[0])

    best_dist = filtered[0][0]

    if best_dist > 0.8:
        print(f"Ghost: ... (No relevant memories found, closest dist: {best_dist:.2f})")
        continue

    # Select results within a tight margin of the best match
    final = [
        t for d, t in filtered
        if d < best_dist + 0.05
    ][:5]

    if not final:
        print("Ghost: ... (No memories found)")
        continue

    # ---- LLM layer ----
    anchor = final[0]

    prompt = f"""
    Below is a real message previously sent by the same person.

    Original message:
    "{anchor}"

    The user now said:
    "{query}"

    Rewrite the original message so it works as a reply.
    Keep the same structure, bluntness, and length.
    Change as little as possible.
    Do not add new ideas.
    Do not explain.
    This is a casual WhatsApp-style reply.
    """

    reply = call_llm(prompt)
    print("\nGhost:", reply)
