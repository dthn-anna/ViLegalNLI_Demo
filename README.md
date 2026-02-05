# ViLegalNLI Demo  
A Vietnamese Legal Retrieval and Natural Language Inference System

## 1. Introduction
ViLegalNLI is a research-oriented demo system developed for an undergraduate
thesis in the field of **Vietnamese Legal Artificial Intelligence**.

The system aims to automatically verify whether a given **legal hypothesis**
is supported by existing Vietnamese legal provisions, based on textual evidence.

The proposed approach integrates:
- Dense semantic retrieval using sentence embeddings
- Vector similarity search
- Legal Natural Language Inference (NLI)

This project is intended solely for **research, experimentation, and academic
demonstration purposes**.

---

## 2. System Overview
Given a legal hypothesis expressed in natural language, the system performs the
following steps:
1. Encodes the hypothesis into a dense vector representation
2. Retrieves semantically relevant legal premises using vector similarity
3. Applies a Legal NLI model to evaluate entailment
4. Aggregates inference scores to produce a final decision
5. Presents supporting legal citations for interpretation

---

## 3. Architecture
The overall architecture of the system is illustrated below:

Presents supporting legal citations for interpretation
```text
User Hypothesis
↓
Sentence Embedding
↓
Vector Retrieval (Cosine Similarity)
↓
Legal NLI Inference
↓
Score Aggregation
↓
Final Decision and Legal Evidence
```

---

## 4. Project Structure
```text
ViLegalNLI_Demo/
├── Code/
│ ├── embeded_index.py # Generate embeddings for legal premises
│ ├── vector_search.py # Vector-based retrieval (cosine similarity)
│ ├── dashboard.py # Streamlit-based demo interface
│ ├── convert_data.py # Convert legal data from CSV to Parquet format
│ └── .gitignore
│
├── requirements.txt # Python dependencies
└── README.md
```

## 5. Academic Disclaimer
This demo system is developed for academic research purposes only and does not
constitute legal advice or authoritative legal interpretation.

