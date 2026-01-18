import json
import redis
import numpy as np

# Connect to Redis
r = redis.Redis(host="localhost", port=6379, decode_responses=False)

# Load data
with open("embedded_blocks.json", "r", encoding="utf-8") as f:
    blocks = json.load(f)

# Configure batching
BATCH_SIZE = 1000
pipe = r.pipeline()

count = 0
for block in blocks:
    key = f"ghost:{block['id']}"
    # Convert vector list to bytes
    vector = np.array(block["vector"], dtype=np.float32).tobytes()

    # Queue the command in the pipeline
    pipe.hset(key, mapping={
        "text": block["text"],
        "embedding": vector
    })
    
    count += 1

    # Execute batch when full
    if count % BATCH_SIZE == 0:
        pipe.execute()
        print(f"Committed batch of {BATCH_SIZE}...")

# Execute any remaining commands in the pipeline
pipe.execute()

print(f"Successfully inserted {len(blocks)} blocks into Redis.")