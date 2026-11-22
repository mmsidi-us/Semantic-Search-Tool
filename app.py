"""
Module 7 Project — Semantic Search Engine
==========================================
app.py — Streamlit search interface

Run with:
    streamlit run app.py

Make sure you've indexed documents first:
    python ingest.py
"""

import streamlit as st
from search import search, get_collection_stats
from ingest import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, ingest

st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔎",
    layout="wide",
)


def _relevance_badge(score: float) -> str:
    """Return a visual relevance indicator based on similarity score."""
    if score >= 0.65:
        return "🟢 High Relevance"
    if score >= 0.40:
        return "🟡 Moderate Relevance"
    return "🟠 Low Relevance"


def main():
    st.title("🔎 Semantic Search Engine")
    st.caption("Search across indexed technical documents using embeddings and ChromaDB.")

    stats = get_collection_stats()

    # ── Sidebar Controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Collection & Search Settings")

        st.metric(
            label="Indexed Chunks",
            value=stats["total_chunks"],
            delta=f"{stats['unique_sources']} source files",
        )

        st.divider()

        n_results = st.slider(
            label="Number of Results (Top-N)",
            min_value=1,
            max_value=15,
            value=5,
            step=1,
        )

        use_threshold = st.checkbox("Apply Distance Threshold", value=False)
        distance_threshold = None
        if use_threshold:
            distance_threshold = st.slider(
                label="Max Distance (lower = more similar)",
                min_value=0.05,
                max_value=1.50,
                value=0.75,
                step=0.05,
            )

        selected_sources = st.multiselect(
            label="Filter by Source File",
            options=stats["source_names"],
            default=[],
            help="Leave empty to search across all indexed files.",
        )

        st.divider()
        st.subheader("Re-Index Documents")
        chunk_size = st.number_input(
            "Chunk Size (chars)",
            min_value=50,
            max_value=2000,
            value=DEFAULT_CHUNK_SIZE,
            step=50,
        )
        overlap = st.number_input(
            "Overlap (chars)",
            min_value=0,
            max_value=500,
            value=DEFAULT_OVERLAP,
            step=25,
        )

        if st.button("🔄 Re-Index Collection", use_container_width=True):
            if overlap >= chunk_size:
                st.error("Overlap must be smaller than chunk size.")
            else:
                with st.spinner("Chunking and embedding documents..."):
                    ingest(chunk_size=int(chunk_size), overlap=int(overlap))
                st.success("Re-indexing complete!")
                st.rerun()

    # ── Main Search Area ──────────────────────────────────────────────────────
    if stats["total_chunks"] == 0:
        st.warning(
            "No documents are currently indexed in ChromaDB. "
            "Run `python ingest.py` in your terminal or click **Re-Index Collection** in the sidebar."
        )

    query = st.text_input(
        label="Enter your search question or query",
        placeholder="e.g., How do vector embeddings measure similarity?",
    )

    if query is not None and query.strip() != "":
        results = search(
            query=query,
            n_results=n_results,
            sources=selected_sources if selected_sources else None,
            distance_threshold=distance_threshold,
        )

        st.subheader(f"Results ({len(results)})")

        if not results:
            st.info(
                "No matching chunks found above the current threshold or source filter. "
                "Try broadening your query, clearing source filters, or adjusting the distance threshold."
            )
        else:
            for rank, item in enumerate(results, start=1):
                with st.container(border=True):
                    top_col1, top_col2, top_col3 = st.columns([3, 2, 2])
                    with top_col1:
                        st.markdown(f"**#{rank} — `{item['source']}`** (Chunk `{item['chunk_index']}`)")
                    with top_col2:
                        st.markdown(f"**Score:** `{item['score']:.4f}` | **Dist:** `{item['distance']:.4f}`")
                    with top_col3:
                        st.markdown(f"**{_relevance_badge(item['score'])}**")

                    st.write(item["text"])
    elif query is not None and query.strip() == "" and st.session_state.get("searched_empty"):
        st.info("Please enter a non-empty query.")


if __name__ == "__main__":
    main()