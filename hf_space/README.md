---
title: Bioasq Rag
emoji: 🌖
colorFrom: purple
colorTo: yellow
sdk: gradio
sdk_version: 6.31.0
python_version: '3.12'
app_file: app.py
pinned: false
short_description: BM25 + MedCPT retrieval and reranking over PubMed abstracts
---

API backend for the BioASQ biomedical-literature RAG showcase: hybrid BM25 + MedCPT dense
retrieval over ~58k PubMed abstracts, MedCPT cross-encoder reranking (on ZeroGPU), and
optional cited answers from Gemini.

Source, benchmarks and methodology: https://github.com/thataintme/BioASQ-challenge
