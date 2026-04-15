# Ghost

Ghost is a local AI that replies the way you do — it retrieves messages from your own chat history and rewrites them as contextual replies using an LLM.

---

## How it works

1. Your messages are chunked into blocks and embedded using `all-MiniLM-L6-v2`
2. Embeddings are stored in a local FAISS index
3. At query time, Ghost retrieves the closest matching blocks by cosine similarity
4. An energy score filter narrows results by linguistic style
5. An LLM rewrites the best match as a reply to what you just said

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

Copy `.env` and fill in your Groq key:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get one free at [console.groq.com](https://console.groq.com)

---

## Usage

Run the pipeline once to build the index:

```bash
python Embedding.py        # embed blocks.json → embedded_blocks.json
python load_to_faiss.py    # build FAISS index → ghost.index + ghost_texts.json
```

Then start Ghost:

```bash
python test_search_groq.py
```

Empty input exits.

---

## Files

| File | Purpose |
|------|---------|
| `blocks.json` | Raw text blocks (your messages) |
| `Embedding.py` | Embeds blocks using sentence-transformers |
| `load_to_faiss.py` | Builds and saves the FAISS index |
| `test_search_groq.py` | Main chat loop (Groq) |
| `test_search.py` | Alternate chat loop (local Ollama) |
| `ghost.index` | FAISS vector index (generated) |
| `ghost_texts.json` | ID → text lookup (generated) |
| `.env` | API keys |

---

## Config

At the top of `test_search_groq.py`:

```python
GROQ_MODEL = "llama3-8b-8192"   # swap to llama3-70b-8192 for better quality
TOP_K = 30                        # number of candidates to retrieve
```

---

## Local LLM (no API)

If you'd rather run fully offline with Ollama:

```bash
# install ollama from ollama.com, then:
ollama pull phi3:mini
python test_search.py
```

---

## Requirements

- Python 3.9+
- `faiss-cpu`, `sentence-transformers`, `requests`, `numpy`, `python-dotenv`
