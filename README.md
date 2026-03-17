# 📊 Investment RAG System

## User Guide

---

## 1. Overview

This project implements a production-grade Retrieval-Augmented Generation (RAG) system for financial documents such as SEC filings, earnings reports, and transcripts.

### Key Capabilities

- Multimodal document ingestion (PDF, HTML, DOCX, XLSX)
- Table-aware parsing and extraction
- Hybrid retrieval (dense + sparse + reranking)
- Grounded LLM answers with citations

---

## 2. Requirements

- Python 3.11+
- Docker (for Qdrant)
- OpenAI API key (GPT-4o required)

### Installation

- conda create --name rag-investment python=3.11
- conda activate rag-investment
- pip install -r requirements.txt

---

## 3. Environment Variables

Create a `.env` file:

- OPENAI_API_KEY=your_gpt4o_api_key
- QDRANT_HOST=localhost
- QDRANT_PORT=6333
- SEC_USER_AGENT=YourName your@email.com

IMPORTANT:
- Please you the appropriate API key
- The system will not run without it

---

## 4. Start Qdrant

docker run -d -p 6333:6333 -p 6334:6334 \
  -v %cd%/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

view at: http://localhost:6333/dashboard#/collections

---

## 5. Running the Pipeline

- python src/ingest/sec_downloader.py
- python src/parse/docling_parser.py
- python src/chunk/hierarchical_chunker.py
- python src/index/qdrant_setup.py --recreate
- python src/index/indexer.py

Query:
- python scripts/query.py --q "What was Microsoft's revenue in FY2024?"

---

## 6. Reproducing Results

- SEC filings (2023–2025)
- Only ~10 companies indexed due to time constraints
- Full pipeline supports 50+ companies

Example queries:
- What was Microsoft's total revenue in FY2024?
- Break down Amazon's revenue by segment

---

## 7. Custom Dataset

Place files in:
data/raw/{TICKER}/{FORM_TYPE}/

Create Manifest Entries:
{
  "ticker": "TSLA",
  "form_type": "10-K",
  "fiscal_year": 2024,
  "filing_date": "2024-02-01",
  "raw_path": "data/raw/TSLA/10-K/tsla_10k_2024.pdf"
}

Add manifest entry, then run:
- python src/parse/docling_parser.py
- python src/chunk/hierarchical_chunker.py
- python src/index/indexer.py

---

## 8. Architecture

Ingestion → Parsing → Chunking → Embedding → Indexing → Retrieval → Generation

---

## 9. Limitations

- Due to indexing contraints only ~10 companies were fully indexed and tested
- Full pipeline supports 50+ companies, but indexing large datasets takes time

