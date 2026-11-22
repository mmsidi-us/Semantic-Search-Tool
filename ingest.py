"""
Module 7 Project — Semantic Search Engine
==========================================
ingest.py — document loading, chunking, and ChromaDB storage

Run with:
    python ingest.py
    python ingest.py --chunk-size 200 --overlap 50
"""

import argparse
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# ── Configuration ─────────────────────────────────────────────────────────────
DOCS_DIR = Path("docs")
CHROMA_PATH = Path("chroma_data")
COLLECTION_NAME = "semantic_search"
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100

_EMBEDDER: SentenceTransformer | None = None


def _get_embedder(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load and cache the SentenceTransformer embedding model."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(model_name)
    return _EMBEDDER


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into fixed-size chunks with overlap.

    Args:
        text:       Full document text.
        chunk_size: Maximum characters per chunk.
        overlap:    Characters of overlap between consecutive chunks.

    Returns:
        List of non-empty chunk strings.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and strictly less than chunk_size.")

    cleaned = text.strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    text_len = len(cleaned)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start += step

    return chunks


def load_documents(docs_dir: Path) -> list[dict]:
    """
    Read all .txt and .md files from docs_dir.

    Returns:
        List of dicts: {"filename": str, "text": str}
    """
    if not docs_dir.exists() or not docs_dir.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    documents: list[dict] = []
    files = sorted(
        [p for p in docs_dir.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".md"}]
    )

    for file_path in files:
        content = file_path.read_text(encoding="utf-8").strip()
        if content:
            documents.append({"filename": file_path.name, "text": content})

    return documents


def get_collection(chroma_path: Path, collection_name: str):
    """Create (or retrieve) a persistent ChromaDB collection."""
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def ingest(chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
    """
    Full ingestion pipeline: load → chunk → embed → upsert.

    Each chunk is stored with metadata: source filename, chunk index,
    and the chunk size used — so experiments with different sizes can
    be compared without ambiguity.
    """
    documents = load_documents(DOCS_DIR)
    if not documents:
        print(f"No .txt or .md files found in '{DOCS_DIR}'.")
        return

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    # Recreate collection cleanly so old chunks from prior chunk sizes don't linger
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    for doc in documents:
        filename = doc["filename"]
        chunks = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{filename}::c{chunk_size}::o{overlap}::{idx}"
            ids.append(chunk_id)
            texts.append(chunk)
            metadatas.append(
                {
                    "source": filename,
                    "chunk_index": idx,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                }
            )

    if not texts:
        print("No chunks generated from documents.")
        return

    embedder = _get_embedder(MODEL_NAME)
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(
        f"Indexed {len(texts)} chunks from {len(documents)} documents "
        f"(chunk_size={chunk_size}, overlap={overlap}) into '{COLLECTION_NAME}'."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index docs/ into ChromaDB")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    args = parser.parse_args()
    ingest(chunk_size=args.chunk_size, overlap=args.overlap)