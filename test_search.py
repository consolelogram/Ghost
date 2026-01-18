import redis
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# -------- setup --------
r = redis.Redis(host="127.0.0.1", port=6379, decode_responses=False)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def energy_score(text):
    if isinstance(text, bytes):
        text = text.decode("utf-8")
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

print("Ghost (Redis-backed) is awake. Empty input to exit.\n")

# -------- chat loop --------
while True:
    query = input("You: ").strip()
    if not query:
        break

    q_energy = energy_score(query)
    q_vec_np = embed_model.encode(query)
    q_vec_bytes = q_vec_np.astype(np.float32).tobytes()

    # ---- Redis retrieval ----
    try:
        # We ask for the vector_score. 
        # Note: Even with DIALECT 2, we will parse as a flat list to be safe.
        res = r.execute_command(
            "FT.SEARCH", "ghost_idx",
            "*=>[KNN 30 @embedding $vec AS vector_score]",
            "PARAMS", 2, "vec", q_vec_bytes,
            "RETURN", 2, "text", "vector_score",
            "DIALECT", 2
        )
    except redis.exceptions.ResponseError as e:
        print(f"Redis Error: {e}")
        continue

    # ---- parse Redis results (Fixed for Flat List) ----
    scored = []
    
    # res structure: [total_count, key1, [fields1], key2, [fields2], ...]
    # We iterate starting at index 1, stepping by 2 to hit the keys
    for i in range(1, len(res), 2):
        # res[i] is the key (we ignore it)
        payload = res[i+1] # This is the list of fields: [b'text', b'...', b'vector_score', b'0.123']
        
        # Convert the fields list into a dictionary
        data = {payload[j]: payload[j+1] for j in range(0, len(payload), 2)}
        
        text_content = data.get(b"text", b"").decode("utf-8")
        # Redis returns distance as a string, need to cast to float
        dist = float(data.get(b"vector_score", 1.0))
        
        scored.append((dist, text_content))

    # ---- energy filter ----
    filtered = [
        (d, t) for d, t in scored
        if abs(energy_score(t) - q_energy) < 10
    ]
    if not filtered:
        filtered = scored

# ---- selector ----
    # Sort ASCENDING (Low distance = Better match)
    filtered.sort(key=lambda x: x[0], reverse=False)

    best_dist = filtered[0][0]
    
    # [ADDED] Absolute Threshold Check
    # If the closest match is still "far away" (distance > 0.5), ignore it.
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