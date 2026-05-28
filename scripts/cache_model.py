from sentence_transformers import SentenceTransformer
print("Pre-caching sentence transformer model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Model cached successfully!")