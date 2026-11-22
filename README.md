# Semantic Search Engine (Module 7)

A local semantic search engine built with **ChromaDB**, **SentenceTransformers (`all-MiniLM-L6-v2`)**, and **Streamlit**. Users can index technical documents, perform natural-language semantic queries, filter by source document, adjust similarity thresholds, and evaluate retrieval quality across different chunking strategies.

---

## Project Structure

```text
semantic-search/
├── app.py               # Streamlit web interface
├── ingest.py            # Document loading, configurable chunking, and ChromaDB indexing
├── search.py            # Semantic query execution, filtering, and ranking
├── evaluate.py          # Source-level Precision and Recall evaluation framework
├── docs/                # Technical corpus (8 .txt documents)
├── chroma_data/         # Persistent ChromaDB vector store
├── requirements.txt     # Project dependencies
└── README.md            # Setup instructions, architecture, and experiment findings

Setup & Usage Instructions
1. Create & Activate Virtual Environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

2. Install Dependencies
python ingest.py --chunk-size 500 --overlap 100

3. Ingest Documents into ChromaDB
python ingest.py --chunk-size 500 --overlap 100

4. Launch the Streamlit Application
streamlit run app.py

5. Run Evaluation
python evaluate.py --n-results 5

