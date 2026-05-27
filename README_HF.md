---
title: Symptom Specialist Mapper
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app/app.py
pinned: false
---

# 🏥 Doctor Symptom to Specialist Mapper

NLP-based system that maps symptoms to medical specialists using semantic similarity.

## How it works
1. Describe your symptoms in plain language
2. Sentence embeddings + FAISS similarity search
3. Returns top-3 specialist recommendations with confidence scores

## Model
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- Index: FAISS (384-dimensional vectors)
- Knowledge base: 3,831 symptom patterns across 15 specialties