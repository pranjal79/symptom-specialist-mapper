import os
import json
import numpy as np
import pandas as pd
import yaml
import logging
import time
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

def load_knowledge_base(config):
    kb_path = os.path.join(config["data"]["knowledge_base"], "knowledge_base.csv")
    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"❌ Knowledge base not found at {kb_path}. Run preprocessing first.")
    df = pd.read_csv(kb_path)
    logger.info(f"✅ Loaded knowledge base: {df.shape[0]} rows")
    return df

def generate_embeddings(texts, model_name, batch_size=64):
    """Generate sentence embeddings for a list of texts"""
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={batch_size})...")
    start_time = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Normalize for cosine similarity
    )

    elapsed = time.time() - start_time
    logger.info(f"✅ Embeddings generated in {elapsed:.2f}s | Shape: {embeddings.shape}")
    return embeddings, model

def build_faiss_index(embeddings):
    """Build a FAISS index for fast similarity search"""
    logger.info("Building FAISS index...")

    dimension = embeddings.shape[1]

    # Use Inner Product (cosine similarity since embeddings are normalized)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))

    logger.info(f"✅ FAISS index built | Dimension: {dimension} | Total vectors: {index.ntotal}")
    return index

def save_artifacts(index, embeddings, knowledge_base, model_name, config):
    """Save FAISS index, embeddings, and metadata"""
    output_dir = config["models"]["embeddings"]
    os.makedirs(output_dir, exist_ok=True)

    # Save FAISS index
    faiss_path = os.path.join(output_dir, "faiss_index.bin")
    faiss.write_index(index, faiss_path)
    logger.info(f"✅ FAISS index saved to {faiss_path}")

    # Save raw embeddings
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings)
    logger.info(f"✅ Embeddings saved to {embeddings_path}")

    # Save knowledge base with index mapping
    kb_path = os.path.join(output_dir, "kb_with_index.csv")
    knowledge_base.to_csv(kb_path, index=True)
    logger.info(f"✅ Knowledge base index saved to {kb_path}")

    # Save metadata
    metadata = {
        "model_name": model_name,
        "embedding_dimension": int(embeddings.shape[1]),
        "total_vectors": int(embeddings.shape[0]),
        "specialists": sorted(knowledge_base["specialist"].unique().tolist()),
        "specialist_counts": knowledge_base["specialist"].value_counts().to_dict()
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Metadata saved to {metadata_path}")

    return {
        "faiss_path": faiss_path,
        "embeddings_path": embeddings_path,
        "kb_path": kb_path,
        "metadata_path": metadata_path
    }

def run_embedding_pipeline():
    config = load_config()
    params = load_params()

    model_name  = params["embedding_model"]
    batch_size  = params["batch_size"]

    # Load knowledge base
    kb = load_knowledge_base(config)

    # Prepare texts — combine disease name + symptom text for richer embeddings
    kb["combined_text"] = kb["disease"] + ": " + kb["symptom_text"]
    texts = kb["combined_text"].tolist()

    logger.info(f"Sample text: {texts[0]}")

    # Generate embeddings
    embeddings, model = generate_embeddings(texts, model_name, batch_size)

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Save everything
    paths = save_artifacts(index, embeddings, kb, model_name, config)

    logger.info("=" * 60)
    logger.info("✅ EMBEDDING PIPELINE COMPLETE!")
    logger.info(f"   Model       : {model_name}")
    logger.info(f"   Vectors     : {embeddings.shape[0]}")
    logger.info(f"   Dimensions  : {embeddings.shape[1]}")
    logger.info(f"   FAISS index : {paths['faiss_path']}")
    logger.info("=" * 60)

    return index, embeddings, kb, model

if __name__ == "__main__":
    run_embedding_pipeline()