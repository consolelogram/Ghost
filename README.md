# Ghost: Style-Matching RAG System

**Ghost** is a Retrieval-Augmented Generation (RAG) engine designed for personality cloning and style transfer. Unlike traditional RAG systems that prioritize factual retrieval, Ghost optimizes for "vibe consistency" by retrieving historical messages based on semantic meaning and emotional intensity ("energy").

The system indexes a user's chat history, performs high-dimensional vector search to find contextually appropriate past responses, and uses a local Large Language Model (Phi-3) to rewrite those responses to fit new conversational contexts.

## System Architecture

The pipeline consists of four distinct stages: Vectorization, Indexing, Retrieval, and Generation.

### 1. Vectorization & Embedding

* **Model:** `all-MiniLM-L6-v2` (SentenceTransformer)
* **Dimensions:** 384
* **Process:** Raw text blocks are converted into dense vector embeddings. This maps semantic concepts to coordinates in a 384-dimensional space, allowing the system to understand that "Hello" and "Hi" are mathematically close, while "Hello" and "Apple" are distant.

### 2. High-Performance Indexing (Redis)

* **Engine:** Redis Stack (RediSearch module)
* **Algorithm:** Hierarchical Navigable Small World (HNSW)
* **Metric:** Cosine Similarity
* **Structure:** Data is stored as Redis Hashes containing both the raw text payload and the binary vector blob.
* **Configuration:** The HNSW graph is configured with `M=6` (edges per node), optimizing for low memory usage and fast insertion speeds at the cost of slight recall degradation.

### 3. Heuristic Retrieval Strategy

The search process is not purely semantic. It employs a two-layer filtering mechanism to ensure the "tone" matches the user's input.

* **Layer 1: Vector Search (KNN)**
The system queries the Redis HNSW index to find the 30 "Nearest Neighbors" (K=30) to the user's input vector. This filters the database down to messages that are *semantically* relevant.
* **Layer 2: Energy Filtering**
A custom "Energy Score" is calculated for the input query and the retrieved candidates.
* *Formula:* `Score = Length + (Exclamations * 2) + QuestionMarks + UppercaseCount`
* *Logic:* Candidates with an energy score deviation > 10 are discarded. This prevents the system from responding to a calm "hello" with a high-intensity, angry paragraph, even if the semantic meaning is similar.


* **Layer 3: Re-Ranking**
The remaining candidates are re-scored using a precise Dot Product calculation in Python to identify the single best "Anchor" message.

### 4. Generative Style Transfer

* **Model:** Microsoft Phi-3 (via Ollama)
* **Prompt Strategy:** The system does not ask the LLM to answer the user's question directly. Instead, it provides the "Anchor" message (the historical reply) and instructs the LLM to rewrite it.
* **Goal:** This forces the LLM to adopt the syntax, slang, and sentence structure of the original user, effectively functioning as a style-transfer engine rather than a chatbot.

## Repository Structure

* **`Embedding.py`**: Batch processor that loads `blocks.json`, computes embeddings using the CPU/GPU, and saves the output as a JSON object containing text/vector pairs.
* **`redis_index.py`**: Database schema definition. Executes `FT.CREATE` to initialize the vector search index with specific schema constraints (Float32 type, 384 dimensions).
* **`load_to_redis.py`**: Ingestion script. Converts standard Python lists into C-compatible binary blobs (`numpy.tobytes()`) and pipelines them into Redis memory.
* **`test_search.py`**: The runtime application. Handles the chat loop, computes real-time energy scores, executes the `FT.SEARCH` command, and manages the API calls to the local LLM inference server.
