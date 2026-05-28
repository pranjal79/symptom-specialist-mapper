import os

files = [
    'models/embeddings/faiss_index.bin',
    'models/embeddings/metadata.json',
    'models/embeddings/kb_with_index.csv',
    'models/specialist_mapper.pkl',
    'data/knowledge_base/knowledge_base.csv',
]

all_good = True
for f in files:
    exists = os.path.exists(f)
    size   = os.path.getsize(f) / 1024 if exists else 0
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {f} ({size:.1f} KB)")
    if not exists:
        all_good = False

if all_good:
    print("\nAll model files verified!")
else:
    print("\nWARNING: Some files are missing!")