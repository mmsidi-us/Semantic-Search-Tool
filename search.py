"""
Module 7 Project — Semantic Search Engine
==========================================
search.py — query ChromaDB and return ranked results

Import this module into app.py:
    from search import search, get_collection_stats
"""

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# ── Configuration (must match ingest.py) ─────────────────────────────────────
CHROMA_PATH = Path("chroma_data")
COLLECTION_NAME = "semantic_search"
MODEL_NAME = "all-MiniLM-L6-v2"

_EMBEDDER: SentenceTransformer | None = None


def _get_embedder(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load and cache the SentenceTransformer embedding model."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(model_name)
    return _EMBEDDER


def get_collection():
    """Return the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def search(
    query: str,
    n_results: int = 5,
    sources: list[str] = None,
    distance_threshold: float = None,
) -> list[dict]:
    """
    Search the ChromaDB collection and return ranked results.

    Args:
        query:              Natural language search query.
        n_results:          Maximum number of results to return.
        sources:            If provided, only return chunks from these filenames.
        distance_threshold: If provided, exclude results with distance above this
                            value (lower = more similar).

    Returns:
        List of result dicts sorted by distance ascending (best first):
            {
                "text":        str,
                "source":      str,
                "chunk_index": int,
                "distance":    float,
                "score":       float,  # 1 - distance
            }
        Returns [] for empty queries or if the collection has no documents.
    """
    if not query or not query.strip() or n_results <= 0:
        return []

    collection = get_collection()
    total_count = collection.count()
    if total_count == 0:
        return []

    where_filter = None
    if sources:
        if len(sources) == 1:
            where_filter = {"source": sources[0]}
        else:
            where_filter = {"source": {"$in": sources}}

    query_embedding = _get_embedder(MODEL_NAME).encode(
        [query.strip()], show_progress_bar=False
    ).tolist()

    query_kwargs = {
        "query_embeddings": query_embedding,
        "n_results": min(n_results, total_count),
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter is not None:
        query_kwargs["where"] = where_filter

    try:
        raw = collection.query(**query_kwargs)
    except Exception:
        return []

    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    results: list[dict] = []
    for text, meta, dist in zip(docs, metas, distances):
        dist_val = float(dist)
        if distance_threshold is not None and dist_val > distance_threshold:
            continue

        score_val = round(1.0 - dist_val, 4)
        results.append(
            {
                "text": text,
                "source": meta.get("source", "unknown"),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "distance": round(dist_val, 4),
                "score": score_val,
            }
        )

    results.sort(key=lambda item: item["distance"])
    return results


def get_collection_stats() -> dict:
    """
    Return basic stats about the indexed collection.

    Returns:
        {
            "total_chunks":   int,
            "unique_sources": int,
            "source_names":   list[str],
        }
    """
    try:
        collection = get_collection()
        total_chunks = collection.count()
        if total_chunks == 0:
            return {"total_chunks": 0, "unique_sources": 0, "source_names": []}

        data = collection.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
        source_set = {
            m.get("source") for m in metadatas if isinstance(m, dict) and m.get("source")
        }
        source_names = sorted(source_set)

        return {
            "total_chunks": total_chunks,
            "unique_sources": len(source_names),
            "source_names": source_names,
        }
    except Exception:
        return {"total_chunks": 0, "unique_sources": 0, "source_names": []}