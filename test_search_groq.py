import json
import numpy as np
import faiss
import requests
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------- CONFIG ----------
FAISS_INDEX_FILE = "ghost.index"
TEXT_STORE_FILE = "ghost_texts.json"
TOP_K = 30
GROQ_MODEL = "llama3-8b-8192"
# ----------------------------

# -------- setup --------
index = faiss.read_index(FAISS_INDEX_FILE)

with open(TEXT_STORE_FILE, "r", encoding="utf-8") as f:
    raw = json.load(f)
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
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 40
            }
        )
        return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[LLM Error: {e}]"

print("Watashi wa Ghost. Ghost wa okiteiru. Empty input to exit.\n")

# -------- chat loop --------
while True:
    query = input("You: ").strip()
    if not query:
        break

    q_energy = energy_score(query)

    q_vec = embed_model.encode(query).astype(np.float32)
    q_vec /= np.linalg.norm(q_vec) + 1e-10
    q_vec = q_vec.reshape(1, -1)

    # ---- FAISS retrieval ----
    scores, indices = index.search(q_vec, TOP_K)

    scored = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        dist = 1.0 - float(score)
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
    filtered.sort(key=lambda x: x[0])

    best_dist = filtered[0][0]

    if best_dist > 0.8:
        print(f"Ghost: ... (No relevant memories found, closest dist: {best_dist:.2f})")
        continue

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
