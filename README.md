# Policy Document QA & Extraction Engine
## Evaluated Retrieval-Augmented Generation System for Enterprise Documents

A domain-agnostic Retrieval-Augmented Generation (RAG) system designed for high-accuracy question answering and information extraction from enterprise legal contracts, corporate policies, compliance documents, and other structured PDF-based document collections.

The system combines structure-aware document processing, dense semantic retrieval, sparse lexical retrieval, Reciprocal Rank Fusion (RRF), BGE cross-encoder reranking, grounded LLM generation, source attribution, and quantitative RAG evaluation using the CUAD dataset and RAGAS.

---

## 1. Project Overview

Large enterprise documents such as legal contracts, corporate policies, and compliance frameworks contain information that is distributed across sections, subsections, clauses, definitions, and page ranges.

Traditional keyword search can miss semantically related information, while unrestricted semantic retrieval can return passages from irrelevant sections or even unrelated documents.

This project addresses these challenges through a multi-stage RAG architecture that combines:

- Structure-aware PDF parsing
- Section-aware document chunking
- Dense semantic retrieval
- FAISS vector search
- BM25 sparse retrieval
- Reciprocal Rank Fusion
- BGE cross-encoder reranking
- Document-level filtering
- Context-aware prompt construction
- Groq-hosted LLM generation
- Source and page attribution
- CUAD-based evaluation
- RAGAS-based quality measurement
- Checkpointed and resumable evaluation

The result is an end-to-end document question-answering platform capable of retrieving relevant contractual evidence and generating grounded answers with traceable sources.

---

# 2. Key Objectives

The primary objectives of the project are:

1. Build a domain-agnostic RAG system for enterprise documents.
2. Preserve structural information during document processing.
3. Combine semantic and lexical retrieval approaches.
4. Improve retrieval precision using cross-encoder reranking.
5. Restrict evaluation retrieval to the correct source contract.
6. Generate answers grounded in retrieved evidence.
7. Provide document, section, and page-level source information.
8. Quantitatively evaluate the RAG pipeline using RAGAS.
9. Use CUAD as a benchmark for contract understanding.
10. Provide a modular architecture suitable for future production deployment.

---

# 3. Complete System Architecture

The following diagram represents the complete architecture of the implemented system, including document ingestion, indexing, hybrid retrieval, reranking, answer generation, and evaluation.
![Complete System Architecture](architecture.png)

The architecture contains two major workflows:

### Online Question Answering

```text
User Question
      |
      v
RAG Pipeline
      |
      v
Hybrid Retrieval
      |
      +--------------------+
      |                    |
      v                    v
Dense Retrieval        Sparse Retrieval
FAISS                  BM25
      |                    |
      +---------+----------+
                |
                v
       Reciprocal Rank Fusion
                |
                v
       BGE Cross-Encoder
           Reranking
                |
                v
        Top-K Context
                |
                v
        Context Builder
                |
                v
             Groq LLM
                |
                v
       Grounded Answer
                |
                v
       Source Attribution
```

### Offline Evaluation

```text
CUAD Question
      +
Target Contract
      |
      v
Document-Level Filtering
      |
      v
Hybrid Retrieval
      |
      v
RRF + BGE Reranking
      |
      v
Grounded Answer
      |
      v
RAGAS Evaluation
      |
      +-----------------------------+
      |             |               |
      v             v               v
Faithfulness   Answer Relevancy   Context Metrics
                                  |
                                  +-- Precision
                                  +-- Recall
```

---

# 4. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.10+ |
| PDF Processing | PyMuPDF |
| Embedding Model | `BAAI/bge-base-en-v1.5` |
| Dense Vector Search | FAISS |
| Sparse Retrieval | BM25 |
| Retrieval Fusion | Reciprocal Rank Fusion |
| Reranker | `BAAI/bge-reranker-large` |
| Generation Model | Groq `openai/gpt-oss-120b` |
| RAG Framework | Custom RAG Pipeline with LlamaIndex-compatible components |
| Evaluation Framework | RAGAS 0.4.x |
| Evaluation Dataset | CUAD |
| User Interface | Streamlit |
| API Layer | FastAPI |
| Configuration | `.env` + `config/settings.py` |
| Index Storage | FAISS + BM25 + JSON metadata |
| Evaluation Storage | JSON |

---

# 5. High-Level Workflow

The complete system operates through the following stages:

```text
Enterprise / Legal PDF Documents
              |
              v
       PDF Document Parser
              |
              v
     Structural Processing
              |
              v
      Structure-Aware Chunks
              |
        +-----+-----+
        |           |
        v           v
   BGE Embeddings  BM25
        |           |
        v           v
      FAISS     Sparse Index
        |           |
        +-----+-----+
              |
              v
     Hybrid Retrieval
              |
              v
     Reciprocal Rank Fusion
              |
              v
      BGE Cross-Encoder
          Reranking
              |
              v
       Top-K Context
              |
              v
        Context Builder
              |
              v
          Groq LLM
              |
              v
      Grounded Answer
              |
              v
     Source Attribution
```

The evaluation pipeline extends this architecture with:

```text
CUAD Question
      +
Target Contract
      |
      v
Document-Level Retrieval Filter
      |
      v
RAG Pipeline
      |
      v
Generated Answer + Retrieved Context
      |
      +
CUAD Ground Truth
      |
      v
RAGAS
      |
      v
Evaluation Metrics
```

---

# 6. Document Ingestion Pipeline

The first stage of the system converts raw PDF documents into searchable structured information.

The ingestion process is:

```text
PDF
 |
 v
PyMuPDF
 |
 v
Text Extraction
 |
 v
Structural Analysis
 |
 v
Section-Aware Chunking
 |
 v
Chunk Metadata
```

The system recursively discovers PDF files from:

```text
data/raw/full_contract_pdf/
```

The CUAD corpus used during implementation contained:

- 510 PDF documents
- 14,739 generated chunks

Each chunk preserves useful structural metadata.

---

# 7. PDF Processing

PyMuPDF is used to extract textual content from PDF documents.

The parser captures:

- Document text
- Page numbers
- Document boundaries
- Section-related information
- Page ranges associated with chunks

The objective is not simply to convert PDFs into plain text, but to preserve information required for downstream retrieval and source attribution.

---

# 8. Structure-Aware Chunking

A major design feature of the system is structure-aware chunking.

Instead of treating the document as an unstructured sequence of characters, the system attempts to preserve document hierarchy.

Conceptually:

```text
Document
 |
 +-- Section
      |
      +-- Subsection
           |
           +-- Chunk
                |
                +-- Text
                +-- Section ID
                +-- Section Title
                +-- Parent Section
                +-- Page Range
```

Each chunk therefore contains both:

1. The actual textual content.
2. Metadata describing where that content came from.

This is particularly useful for legal and policy documents because a clause often needs to be interpreted in relation to its section and surrounding structure.

---

# 9. Dense Semantic Retrieval

Dense retrieval uses:

```text
BAAI/bge-base-en-v1.5
```

to convert documents and user queries into vector representations.

The process is:

```text
User Question
      |
      v
BGE Query Encoder
      |
      v
Query Vector
      |
      v
FAISS Similarity Search
      |
      v
Dense Candidate Chunks
```

Dense retrieval is useful when the wording of the query differs from the wording used in the source document.

---

# 10. Sparse Retrieval Using BM25

The system also implements BM25-based sparse retrieval.

The process is:

```text
User Question
      |
      v
Tokenization
      |
      v
BM25 Search
      |
      v
Sparse Candidate Chunks
```

BM25 provides lexical relevance and is particularly useful for:

- Legal terminology
- Contract names
- Defined terms
- Clause terminology
- Exact phrases
- Named entities
- Specific obligations

---

# 11. Hybrid Retrieval

The system combines dense and sparse retrieval.

```text
                User Question
                     |
          +----------+----------+
          |                     |
          v                     v
   Dense Retrieval        Sparse Retrieval
       FAISS                    BM25
          |                     |
          +----------+----------+
                     |
                     v
               RRF Fusion
                     |
                     v
             Candidate Ranking
```

This provides two complementary retrieval signals:

- Dense retrieval for semantic similarity.
- BM25 for lexical similarity and exact terminology.

---

# 12. Reciprocal Rank Fusion

The outputs from FAISS and BM25 are combined using Reciprocal Rank Fusion.

The purpose of RRF is to combine rankings generated by different retrieval systems without requiring their raw scores to be directly comparable.

```text
FAISS Ranking
     |
     +----------------+
                      |
                      v
                 RRF Fusion
                      ^
                      |
     +----------------+
     |
BM25 Ranking
```

The resulting fused ranking is passed to the reranker.

---

# 13. BGE Cross-Encoder Reranking

The RRF candidates are passed to:

```text
BAAI/bge-reranker-large
```

The reranker evaluates the relationship between the query and each candidate passage.

```text
RRF Candidates
      |
      v
BGE Cross-Encoder
      |
      v
Relevance Scores
      |
      v
Final Top-K Context
```

This two-stage architecture separates high-recall retrieval from higher-precision final ranking.

---

# 14. Document-Level Filtering

The CUAD evaluation pipeline receives both a question and its target contract.

The target document is passed into the retrieval layer itself:

```text
CUAD Question
      +
Target Contract
      |
      v
Document Filter
      |
      +----------+
      |          |
      v          v
    FAISS      BM25
      |          |
      +----+-----+
           |
           v
          RRF
           |
           v
       Reranking
```

This ensures that evaluation retrieval remains within the specified source contract rather than relying only on an instruction to the LLM.

Document-name normalization also accounts for differences such as:

```text
ContractName
```

versus:

```text
ContractName.pdf
```

---

# 15. Context Construction

After reranking, the selected chunks are converted into structured context for the LLM.

Each source contains:

- Document
- Section
- Section title
- Parent section
- Pages
- Content

A conceptual source block looks like:

```text
[SOURCE 1]

Document: ExampleContract.pdf
Section: 8.2 Confidentiality
Parent section: Confidentiality
Pages: 14-15

Content:
...
```

This provides the LLM with both the evidence and its structural location.

---

# 16. Grounded Answer Generation

The final retrieved context is supplied to:

```text
Groq
openai/gpt-oss-120b
```

The generation process is:

```text
User Question
      +
Retrieved Context
      +
Structural Metadata
      |
      v
Grounded Prompt
      |
      v
Groq LLM
      |
      v
Grounded Answer
```

The system instructs the LLM to answer using retrieved evidence and cite relevant source sections and pages.

---

# 17. Source Attribution

Every retrieved source returned by the pipeline contains:

- Document name
- Section identifier
- Section title
- Page range
- Retrieval score
- Reranking score
- Source text

This creates an evidence trail between:

```text
Question
   ↓
Retrieved Evidence
   ↓
Generated Answer
```

Source attribution is particularly important for legal and compliance-oriented use cases where users need to verify the basis of an answer.

---

# 18. Application Layer

The project provides two application interfaces.

## Streamlit Interface

The Streamlit application provides an interactive interface for:

- Entering questions
- Running the RAG pipeline
- Viewing generated answers
- Viewing source documents
- Inspecting section and page information

Run:

```bash
streamlit run app.py
```

## FastAPI Interface

The FastAPI application exposes the system through an API interface.

Typical development command:

```bash
uvicorn api:app --reload
```

---

# 19. Evaluation Framework

The project uses the Contract Understanding Atticus Dataset (CUAD) for evaluation.

CUAD provides contract-related questions covering concepts such as:

- Termination
- Confidentiality
- Liability
- Licensing
- Commitments
- Contract names
- Contractual obligations
- Other legal clauses

The prepared CUAD source contains:

```text
6,702 answerable questions
```

A controlled validation set containing:

```text
30 questions
```

was created for the initial evaluation stage.

---

# 20. RAGAS Evaluation

The project uses RAGAS 0.4.x.

Four metrics are evaluated.

## Faithfulness

Measures whether the generated answer is supported by retrieved context.

## Answer Relevancy

Measures whether the generated answer addresses the original question.

## Context Precision

Measures the relevance of retrieved context to the question and reference.

## Context Recall

Measures whether the retrieved context contains the information necessary to cover the reference answer.

The evaluation flow is:

```text
CUAD Question
      +
Target Contract
      |
      v
Document-Filtered RAG
      |
      v
Generated Answer
      +
Retrieved Context
      +
CUAD Ground Truth
      |
      v
RAGAS
      |
      +------------------+
      |        |         |
      v        v         v
Faithfulness  Answer    Context
              Relevancy Precision / Recall
```

---

# 21. Evaluation Results

The implemented evaluation pipeline successfully demonstrated the complete retrieval, generation, document filtering, and RAGAS evaluation workflow.

The first completed validation cases produced the following results.

## Evaluation Case 1

**Category:** Termination For Convenience

**Target contract:**

```text
PrudentialBancorpInc_20170606_8-K_EX-10.4_10474434_EX-10.4_Endorsement Agreement
```

**Retrieved source:**

```text
PrudentialBancorpInc_20170606_8-K_EX-10.4_10474434_EX-10.4_Endorsement Agreement.pdf
```

| Metric | Score |
|---|---:|
| Faithfulness | 0.8000 |
| Answer Relevancy | 0.7030 |
| Context Precision | 1.0000 |
| Context Recall | 1.0000 |

This case demonstrated correct target-document retrieval and perfect Context Precision and Context Recall.

---

## Evaluation Case 2

**Category:** Document Name

**Target contract:**

```text
PelicanDeliversInc_20200211_S-1_EX-10.3_11975895_EX-10.3_Development Agreement1
```

**Retrieved source:**

```text
PelicanDeliversInc_20200211_S-1_EX-10.3_11975895_EX-10.3_Development Agreement1.pdf
```

| Metric | Score |
|---|---:|
| Faithfulness | 0.0000 |
| Answer Relevancy | 0.7380 |
| Context Precision | 1.0000 |
| Context Recall | 1.0000 |

The retrieval system again identified the correct target contract and achieved perfect Context Precision and Context Recall.

The Faithfulness score of 0.0000 identifies an individual answer-generation case that should be investigated as part of continued evaluation. This demonstrates the value of using multiple RAGAS metrics rather than relying on retrieval performance alone.

---

## Evaluation Status

The 30-question evaluation pipeline is implemented and operational.

The validation process demonstrated:

- Correct target-contract filtering
- Successful FAISS retrieval
- Successful BM25 retrieval
- Successful RRF fusion
- Successful BGE reranking
- Successful Groq answer generation
- Successful RAGAS metric execution
- Checkpointed evaluation
- Resumable evaluation after API interruptions

The complete 30-question evaluation is constrained by the available Groq token quota because answer generation and RAGAS evaluation require LLM calls.

The reported scores above are therefore the scores from completed validation cases and are not presented as an artificial 30-question aggregate.

---

# 22. Checkpointed Evaluation

Long-running evaluations can be interrupted by API limits or other runtime conditions.

The evaluator maintains:

```text
evals/latest_results.json
```

The checkpoint records:

- Question
- Target document
- Generated answer
- Ground truth
- RAGAS scores
- Retrieved sources

If an evaluation stops because of a rate limit, previously completed questions remain stored and the pending question is not incorrectly marked as completed.

A subsequent run can resume from the first pending question.

---

# 23. Project Structure

```text
POLICY-RAG/
│
├── app.py
├── api.py
├── requirements.txt
├── README.md
├── architecture.png
├── Dockerfile
├── docker-compose.yml
├── setup_windows.bat
├── setup_linux.sh
├── .env
├── .env.example
├── .gitignore
│
├── config/
│   └── settings.py
│
├── data/
│   ├── raw/
│   │   └── full_contract_pdf/
│   ├── processed/
│   └── evaluation/
│       ├── questions.example.jsonl
│       └── questions.jsonl
│
├── src/
│   └── policy_qa/
│       ├── models.py
│       ├── pdf_parser.py
│       ├── chunker.py
│       ├── embeddings.py
│       ├── index.py
│       ├── reranker.py
│       ├── llm.py
│       ├── pipeline.py
│       ├── ingestion.py
│       ├── evaluation.py
│       ├── ui.py
│       └── llamaindex_bridge.py
│
├── scripts/
│   ├── ingest.py
│   ├── query.py
│   ├── evaluate.py
│   └── retrieval_experiment.py
│
├── storage/
│   └── index/
│       ├── dense.faiss
│       ├── bm25.pkl
│       └── chunks.json
│
├── evals/
│   └── latest_results.json
│
├── logs/
└── tests/
```

---

# 24. Important Modules

### `pdf_parser.py`

Extracts document text and page information from PDFs.

### `chunker.py`

Creates structure-aware chunks and preserves document hierarchy.

### `embeddings.py`

Loads the BGE embedding model and generates document/query embeddings.

### `index.py`

Implements:

- FAISS dense retrieval
- BM25 sparse retrieval
- Target-document filtering
- Reciprocal Rank Fusion

### `reranker.py`

Loads and applies the BGE cross-encoder reranker.

### `llm.py`

Initializes the LLM used for grounded answer generation.

### `pipeline.py`

Coordinates:

```text
Retrieval
    ↓
Reranking
    ↓
Context Construction
    ↓
Prompt Construction
    ↓
LLM Generation
```

### `evaluation.py`

Coordinates:

```text
CUAD Questions
    ↓
Document-Filtered RAG
    ↓
RAGAS Metrics
    ↓
Checkpointed Results
```

---

# 25. Installation

## Prerequisites

The system requires:

- Python 3.10+
- Virtual environment
- Internet access for model/API access
- Groq API key
- Sufficient memory for the embedding and reranking models

### Windows

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

If required by the environment:

```bash
python -m pip install groq openai
```

---

# 26. Environment Configuration

Create `.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Never commit the actual API key.

The `.env` file should remain excluded through `.gitignore`.

---

# 27. Dataset Preparation

Place the contract PDFs under:

```text
data/raw/full_contract_pdf/
```

The ingestion system recursively searches this directory for PDF files.

Evaluation questions are stored in:

```text
data/evaluation/questions.jsonl
```

A typical evaluation record contains:

```json
{
  "qa_id": 1,
  "question": "Contract-related question",
  "document": "TargetContract",
  "ground_truth": "Reference answer"
}
```

---

# 28. Building the Search Index

After adding documents, execute:

```bash
python -m scripts.ingest
```

The ingestion process:

1. Discovers PDFs.
2. Parses documents.
3. Creates structure-aware chunks.
4. Generates BGE embeddings.
5. Builds the FAISS index.
6. Builds the BM25 index.
7. Saves chunk metadata.

The generated files are stored under:

```text
storage/index/
```

Expected files:

```text
dense.faiss
bm25.pkl
chunks.json
```

---

# 29. Running a Query

Run:

```bash
python -m scripts.query "What are the confidentiality obligations?"
```

The query executes:

```text
Question
   ↓
FAISS + BM25
   ↓
RRF
   ↓
BGE Reranker
   ↓
Top-K Context
   ↓
Groq LLM
   ↓
Grounded Answer
```

---

# 30. Running Evaluation

Run:

```bash
python -m scripts.evaluate
```

The evaluator loads:

```text
data/evaluation/questions.jsonl
```

and calculates:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

Results are written to:

```text
evals/latest_results.json
```

---

# 31. Handling API Rate Limits

The evaluation pipeline uses hosted LLM calls for answer generation and RAGAS evaluation.

Large evaluations can therefore consume substantial API tokens.

The checkpoint/resume mechanism prevents completed questions from being lost when a rate limit occurs.

The following components remain independent of the LLM quota:

- PDF ingestion
- Chunking
- FAISS indexing
- BM25 indexing
- RRF retrieval
- BGE reranking

LLM-dependent operations require the external API quota.

---

# 32. Configuration

Application configuration is maintained in:

```text
config/settings.py
```

Configuration includes:

- Embedding model
- Reranker model
- LLM model
- FAISS index directory
- Evaluation directory
- Dense retrieval top-K
- BM25 top-K
- Reranking top-K
- RRF configuration
- Data paths

---

# 33. Reproducibility

For reproducible experiments, record:

- Python version
- Dependency versions
- CUAD dataset version
- Number of documents
- Number of chunks
- Embedding model
- Reranker model
- LLM model
- Retrieval parameters
- Evaluation questions
- RAGAS version

The persistent FAISS, BM25, and chunk metadata files prevent re-indexing for every query.

---

# 34. Security and Data Handling

The system may process legal, corporate, or confidential documents.

Therefore:

- API keys must not be hard-coded.
- `.env` must not be committed.
- Sensitive documents should not be published to public repositories.
- Production deployments should follow organizational security and privacy requirements.
- Access control should be implemented before deploying confidential collections.

---

# 35. Limitations

The current implementation has several practical limitations:

1. LLM generation and RAGAS evaluation depend on an external hosted API.
2. Large-scale evaluation is constrained by API token quotas.
3. Answer quality depends on document quality and retrieval quality.
4. The current reported evaluation results are from completed validation cases rather than an artificially calculated full 30-question aggregate.
5. The system is primarily designed around PDF document collections.
6. Local FAISS and BM25 storage would require additional infrastructure for large distributed deployments.
7. Generated legal answers should be treated as information-retrieval and decision-support outputs and should not replace professional legal review.

---

# 36. Future Enhancements

Potential improvements include:

- Query rewriting
- Multi-query retrieval
- Parent-child retrieval
- Metadata-aware retrieval
- Citation verification
- Hallucination detection
- Retrieval ablation studies
- Model comparison experiments
- Larger CUAD evaluations
- Evaluation dashboards
- Experiment tracking
- Cloud vector database integration
- Distributed indexing
- Production authentication
- Role-based access control
- Document upload and automatic indexing
- Enterprise-scale deployment

---

# 37. Project Outcome

The project successfully implements a complete end-to-end evaluated RAG architecture for enterprise document question answering.

The implemented system demonstrates:

- PDF document ingestion
- Structure-aware chunking
- Dense semantic retrieval
- FAISS vector search
- BM25 lexical retrieval
- Hybrid retrieval
- Reciprocal Rank Fusion
- BGE cross-encoder reranking
- Target-document filtering
- Grounded LLM generation
- Source attribution
- Streamlit application support
- FastAPI application support
- CUAD-based evaluation
- RAGAS-based evaluation
- Checkpointed evaluation
- Resumable evaluation

The indexing stage successfully processed:

```text
510 PDF documents
14,739 document chunks
```

The completed validation cases confirmed that the retrieval system can correctly restrict retrieval to the requested CUAD contract.

For the reported validation cases, Context Precision and Context Recall both reached:

```text
1.0000
```

The first validation case achieved:

```text
Faithfulness       = 0.8000
Answer Relevancy   = 0.7030
Context Precision  = 1.0000
Context Recall     = 1.0000
```

The second validation case achieved:

```text
Faithfulness       = 0.0000
Answer Relevancy   = 0.7380
Context Precision  = 1.0000
Context Recall     = 1.0000
```

The second case also demonstrates the value of multi-dimensional RAG evaluation: retrieval quality was strong while the faithfulness metric identified an answer-generation case requiring further investigation.

Overall, the project establishes a functional, modular, traceable, and quantitatively evaluable RAG architecture for enterprise legal and policy documents.

---

# 38. Final Architecture Summary

```text
                    DOCUMENT INGESTION
                           |
                           v
                  PDF / Contract Files
                           |
                           v
                    PyMuPDF Parser
                           |
                           v
              Structure-Aware Chunking
                           |
                           v
                    Document Chunks
                           |
             +-------------+-------------+
             |                           |
             v                           v
       BGE Embeddings                  BM25
             |                           |
             v                           v
           FAISS                  Sparse Index
             |                           |
             +-------------+-------------+
                           |
                           v
                    Hybrid Retrieval
                           |
                           v
                         RRF
                           |
                           v
                  BGE Cross-Encoder
                      Reranking
                           |
                           v
                    Top-K Context
                           |
                           v
                  Context Construction
                           |
                           v
                  Groq GPT-OSS-120B
                           |
                           v
                 Grounded Answer
                           |
                           v
              Source / Page Attribution


                    EVALUATION PIPELINE
                           |
                           v
                    CUAD Questions
                           |
                           v
               Target Contract Filter
                           |
                           v
                    RAG Pipeline
                           |
                           v
                 Generated Answer
                           |
                           v
                         RAGAS
                           |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
   Faithfulness      Answer Relevancy   Context Metrics
                                           |
                                  +--------+--------+
                                  |                 |
                                  v                 v
                             Precision           Recall
                                  |
                                  v
                       latest_results.json
                                  |
                                  v
                         Resume Evaluation
```

---

## Conclusion

The Policy Document QA & Extraction Engine provides a complete RAG-based architecture for enterprise document intelligence.

By combining structure-aware document processing, hybrid dense and sparse retrieval, Reciprocal Rank Fusion, cross-encoder reranking, grounded LLM generation, source attribution, document-level filtering, and RAGAS-based evaluation, the system provides a strong foundation for reliable question answering over complex legal and policy documents.

The modular architecture allows individual components to be independently improved, benchmarked, and replaced as the system evolves toward larger-scale enterprise deployment.
