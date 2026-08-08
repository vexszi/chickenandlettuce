import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

DOCS_DIR = "hospital_docs"
INDEX_FILE = "rag_index.npz"       # stores the embedding vectors
CHUNKS_FILE = "rag_chunks.json"    # stores the text each vector corresponds to

# bge-m3 loaded once at module level -- same "build once, reuse" pattern
# as your API clients in tools.py. Loading this model is real, slow work.
embedding_model = SentenceTransformer("BAAI/bge-m3")


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits text into overlapping character chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_index() -> None:
    """Run this once (as a script, not inside the graph) to (re)build the
    embedding index from everything in hospital_docs/."""
    all_chunks = []       # the actual text of every chunk, across all docs
    all_chunk_sources = []  # which filename each chunk came from

    for filename in os.listdir(DOCS_DIR):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(DOCS_DIR, filename)
        with open(filepath, "r") as f:
            text = f.read()

        for chunk in _chunk_text(text):
            all_chunks.append(chunk)
            all_chunk_sources.append(filename)

    print(f"Embedding {len(all_chunks)} chunks from {DOCS_DIR}/ ...")
    # encode() turns each chunk of text into a vector -- a list of numbers
    # capturing its meaning, such that semantically similar text ends up
    # as numerically similar vectors.
    embeddings = embedding_model.encode(all_chunks, show_progress_bar=True)

    np.savez(INDEX_FILE, embeddings=embeddings)
    with open(CHUNKS_FILE, "w") as f:
        json.dump({"chunks": all_chunks, "sources": all_chunk_sources}, f, indent=2)

    print(f"Saved index to {INDEX_FILE} and {CHUNKS_FILE}")


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Measures how similar two vectors' directions are"""
    a_norm = a / np.linalg.norm(a)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True))
    return b_norm @ a_norm


def retrieve(query: str, top_k: int = 3) -> list[str]:
    """Given a patient's question, returns the top_k most relevant chunks
    of hospital policy text. Call this from inside the graph."""
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(
            "No RAG index found -- run build_index() first (see the "
            "bottom of this file for how to do that from the command line)."
        )

    data = np.load(INDEX_FILE)
    embeddings = data["embeddings"]

    with open(CHUNKS_FILE, "r") as f:
        chunk_data = json.load(f)
    chunks = chunk_data["chunks"]

    query_embedding = embedding_model.encode([query])[0]
    similarities = _cosine_similarity(query_embedding, embeddings)

    # argsort gives indices that would sort ascending; [::-1] reverses to
    # descending (most similar first), then take the top_k.
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [chunks[i] for i in top_indices]


if __name__ == "__main__":
    # Run this file directly (`python rag.py`) to build/rebuild the index.
    build_index()