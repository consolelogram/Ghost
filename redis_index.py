import redis

r = redis.Redis(host="localhost", port=6379, decode_responses=False)

try:
    r.execute_command("""
    FT.CREATE ghost_idx
    ON HASH
    PREFIX 1 ghost:
    SCHEMA
        text TEXT
        embedding VECTOR HNSW 6
        TYPE FLOAT32
        DIM 384
        DISTANCE_METRIC COSINE
    """)
    print("Index created")
except Exception as e:
    print("Index probably already exists:", e)
