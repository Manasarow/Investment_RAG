INVESTMENT RAG SYSTEM — USER GUIDE

--------------------------------------------------
1. OVERVIEW
--------------------------------------------------

This project implements a production-grade Retrieval-Augmented Generation (RAG) system for financial documents such as SEC filings, earnings reports, and transcripts.

Key capabilities:
- Multimodal document ingestion (PDF, HTML, DOCX, XLSX)
- Table-aware parsing and extraction
- Hybrid retrieval (dense + sparse + reranking)
- Grounded LLM answers with citations


--------------------------------------------------
2. REQUIREMENTS
--------------------------------------------------

- Python 3.11+
- Docker (for Qdrant)
- OpenAI API key (GPT-4o)

Install dependencies:

    python -m venv .venv
    .venv\Scripts\activate   (Windows)
    pip install -r requirements.txt


--------------------------------------------------
3. ENVIRONMENT VARIABLES
--------------------------------------------------

Create a .env file:

    OPENAI_API_KEY=your_gpt4o_api_key
    QDRANT_HOST=localhost
    QDRANT_PORT=6333
    SEC_USER_AGENT=YourName your@email.com

IMPORTANT:
- You must use your own GPT-4o API key
- The system will not run without it


--------------------------------------------------
4. START QDRANT (VECTOR DATABASE)
--------------------------------------------------

Run Qdrant locally using Docker:

    docker run -d -p 6333:6333 -p 6334:6334 \
      -v %cd%/qdrant_storage:/qdrant/storage \
      qdrant/qdrant

Verify:

    curl http://localhost:6333/healthz


--------------------------------------------------
5. RUNNING THE PIPELINE
--------------------------------------------------

Step 1 — Download SEC filings:

    python src/ingest/sec_downloader.py

Step 2 — Parse documents:

    python src/parse/docling_parser.py

Step 3 — Chunk documents:

    python src/chunk/hierarchical_chunker.py

Step 4 — Setup Qdrant collection:

    python src/index/qdrant_setup.py --recreate

Step 5 — Index data:

    python src/index/indexer.py

Step 6 — Query the system:

    python scripts/query.py --q "What was Microsoft's revenue in FY2024?"

Interactive mode:

    python scripts/query.py


--------------------------------------------------
6. REPRODUCING RESULTS
--------------------------------------------------

Dataset:
- SEC filings (2023–2025)
- Top US companies (subset)

IMPORTANT NOTE:
- Only ~10 companies were fully indexed and tested due to time constraints
- The pipeline supports 50+ companies, but indexing:
  - Takes significant time
  - Depends on hardware (CPU/GPU)

Example test queries:

    What was Microsoft's total revenue in FY2024?
    Break down Amazon's revenue by segment for the last reported fiscal year.
    How has Apple's gross margin trended over the past 3 fiscal years?
    Give me an overview on the recent performance of Nvidia.
    Compare R&D spending as a percentage of revenue between Google and Microsoft.

Expected output:
- Final answer
- Citations (document + page)
- Context chunks used


--------------------------------------------------
7. USING CUSTOM DATASET
--------------------------------------------------

Step 1 — Add your documents:

    data/raw/{TICKER}/{FORM_TYPE}/

Example:

    data/raw/TSLA/10-K/tsla_10k_2024.pdf

Step 2 — Add manifest entry:

    {
      "ticker": "TSLA",
      "form_type": "10-K",
      "fiscal_year": 2024,
      "filing_date": "2024-02-01",
      "raw_path": "data/raw/TSLA/10-K/tsla_10k_2024.pdf"
    }

Step 3 — Run pipeline:

    python src/parse/docling_parser.py
    python src/chunk/hierarchical_chunker.py
    python src/index/indexer.py

Step 4 — Query:

    python scripts/query.py --q "What is Tesla's debt?"

Supported data:
- Financial reports
- Earnings transcripts
- Table-heavy PDFs

Not optimized for:
- Pure images without text
- Highly unstructured documents


--------------------------------------------------
8. SYSTEM ARCHITECTURE
--------------------------------------------------

Pipeline:

    Ingestion → Parsing → Chunking → Embedding → Indexing → Retrieval → Generation

Components:
- Docling parser (OCR + layout + tables)
- Hierarchical chunker
- BGE-M3 embeddings (dense + sparse)
- Qdrant vector database
- Hybrid retrieval + reranking
- GPT-4o generation


--------------------------------------------------
9. SHARING QDRANT INDEX (DOCKER)
--------------------------------------------------

Option 1 — Share storage folder:

    zip -r qdrant_storage.zip qdrant_storage

Receiver runs:

    docker run -d -p 6333:6333 \
      -v $(pwd)/qdrant_storage:/qdrant/storage \
      qdrant/qdrant


Option 2 — Snapshot:

    curl -X POST "http://localhost:6333/collections/rag_investment/snapshots"


--------------------------------------------------
10. LIMITATIONS
--------------------------------------------------

- Only ~10 companies indexed for testing
- Full indexing (50+ companies) can take hours
- Performance depends on:
  - Hardware
  - Dataset size
  - GPU availability


--------------------------------------------------
11. SUMMARY
--------------------------------------------------

This system provides:

- Financial-document-aware parsing
- Table-level reasoning
- Hybrid retrieval with reranking
- Grounded, citation-based answers

It is designed to scale to large financial corpora and handle complex investor queries with high accuracy.