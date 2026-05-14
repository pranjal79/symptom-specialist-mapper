import os
import json
import time
import faiss
import pickle
import numpy as np
import pandas as pd
import yaml
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ================================================================
# SPECIALIST DESCRIPTIONS
# Shown to user in the final output card
# ================================================================
SPECIALIST_DESCRIPTIONS = {
    "Neurologist": (
        "Specializes in disorders of the brain, spinal cord, and nervous system. "
        "Consult for headaches, seizures, memory issues, numbness, or movement disorders."
    ),
    "Cardiologist": (
        "Specializes in heart and cardiovascular conditions. "
        "Consult for chest pain, palpitations, high blood pressure, or shortness of breath."
    ),
    "Dermatologist": (
        "Specializes in skin, hair, and nail conditions. "
        "Consult for rashes, acne, itching, skin lesions, or hair loss."
    ),
    "Gastroenterologist": (
        "Specializes in digestive system disorders. "
        "Consult for abdominal pain, nausea, vomiting, diarrhea, or liver issues."
    ),
    "Pulmonologist": (
        "Specializes in lung and respiratory conditions. "
        "Consult for persistent cough, breathing difficulty, wheezing, or chest tightness."
    ),
    "Orthopedist": (
        "Specializes in bones, joints, and musculoskeletal conditions. "
        "Consult for joint pain, fractures, back pain, or sports injuries."
    ),
    "Endocrinologist": (
        "Specializes in hormonal and metabolic disorders. "
        "Consult for diabetes, thyroid issues, weight problems, or hormonal imbalances."
    ),
    "Psychiatrist": (
        "Specializes in mental health disorders. "
        "Consult for depression, anxiety, mood swings, sleep disorders, or behavioral changes."
    ),
    "Urologist": (
        "Specializes in urinary tract and kidney conditions. "
        "Consult for painful urination, kidney stones, frequent urination, or urinary infections."
    ),
    "Ophthalmologist": (
        "Specializes in eye and vision conditions. "
        "Consult for vision changes, eye pain, redness, or cataracts."
    ),
    "ENT Specialist": (
        "Specializes in ear, nose, and throat conditions. "
        "Consult for sinus problems, hearing loss, sore throat, or nasal congestion."
    ),
    "Oncologist": (
        "Specializes in cancer diagnosis and treatment. "
        "Consult for unexplained lumps, weight loss, fatigue, or abnormal test results."
    ),
    "Gynecologist": (
        "Specializes in female reproductive health. "
        "Consult for menstrual issues, pelvic pain, hormonal problems, or pregnancy care."
    ),
    "Pediatrician": (
        "Specializes in health care for children and adolescents. "
        "Consult for child illnesses, growth concerns, vaccinations, or developmental issues."
    ),
    "General Physician": (
        "A primary care doctor who handles a wide range of conditions. "
        "Consult for general illness, fever, infections, or initial evaluation of any symptom."
    ),
}

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

class SpecialistMapper:
    """
    Core model class for mapping symptoms to specialists.
    Loads FAISS index + knowledge base, runs semantic search.
    """

    def __init__(self, config_path="config/config.yaml", params_path="params.yaml"):
        self.config = load_config(config_path)
        self.params = load_params(params_path)
        self.top_k  = self.params["top_k"]

        self.model      = None
        self.index      = None
        self.kb         = None
        self.metadata   = None

        self._load_artifacts()

    def _load_artifacts(self):
        """Load all required artifacts"""
        embed_dir = self.config["models"]["embeddings"]

        # Load metadata
        metadata_path = os.path.join(embed_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        logger.info(f"✅ Metadata loaded | Model: {self.metadata['model_name']}")

        # Load sentence transformer model
        logger.info(f"Loading embedding model: {self.metadata['model_name']}")
        self.model = SentenceTransformer(self.metadata["model_name"])
        logger.info("✅ Embedding model loaded")

        # Load FAISS index
        faiss_path = os.path.join(embed_dir, "faiss_index.bin")
        self.index = faiss.read_index(faiss_path)
        logger.info(f"✅ FAISS index loaded | Vectors: {self.index.ntotal}")

        # Load knowledge base
        kb_path = os.path.join(embed_dir, "kb_with_index.csv")
        self.kb = pd.read_csv(kb_path, index_col=0)
        logger.info(f"✅ Knowledge base loaded | Rows: {len(self.kb)}")

    def _embed_query(self, query_text):
        """Embed a single query text"""
        embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.astype(np.float32)

    def _aggregate_results(self, indices, scores):
        """
        Aggregate raw FAISS results into top-K specialist recommendations.
        Groups by specialist, averages scores, picks best matching symptoms.
        """
        specialist_data = {}

        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue

            row        = self.kb.iloc[idx]
            specialist = row["specialist"]
            disease    = row["disease"]
            symptoms   = row["symptom_text"]

            if specialist not in specialist_data:
                specialist_data[specialist] = {
                    "specialist":      specialist,
                    "scores":          [],
                    "diseases":        [],
                    "symptom_samples": [],
                }

            specialist_data[specialist]["scores"].append(float(score))
            specialist_data[specialist]["diseases"].append(str(disease))
            specialist_data[specialist]["symptom_samples"].append(str(symptoms))

        # Build final results
        results = []
        for specialist, data in specialist_data.items():
            avg_score   = np.mean(data["scores"])
            top_disease = data["diseases"][np.argmax(data["scores"])]
            top_symptom = data["symptom_samples"][np.argmax(data["scores"])]

            results.append({
                "specialist":          specialist,
                "confidence":          round(float(avg_score) * 100, 1),
                "top_matched_disease": top_disease,
                "matched_symptoms":    top_symptom,
                "description":         SPECIALIST_DESCRIPTIONS.get(
                                           specialist,
                                           f"Specialist in {specialist}-related conditions."
                                       ),
            })

        # Sort by confidence descending
        results = sorted(results, key=lambda x: x["confidence"], reverse=True)
        return results[:self.top_k]

    def predict(self, symptom_text):
        """
        Main prediction method.
        Input : plain English symptom description
        Output: list of top-3 specialist recommendations
        """
        start_time = time.time()

        if not symptom_text or not symptom_text.strip():
            raise ValueError("Symptom text cannot be empty.")

        logger.info(f"Query: '{symptom_text}'")

        # Embed the query
        query_embedding = self._embed_query(symptom_text)

        # Search FAISS — retrieve top 50 candidates then aggregate
        search_k = min(50, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, search_k)

        # Aggregate into top-3 specialists
        results = self._aggregate_results(indices, scores)

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Inference time: {elapsed_ms:.1f}ms")

        return {
            "query":          symptom_text,
            "results":        results,
            "inference_ms":   round(elapsed_ms, 1),
        }

    def save(self):
        """Serialize the mapper config as a lightweight pkl"""
        artifact_path = self.config["models"]["artifact"]
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)

        save_data = {
            "model_name":    self.metadata["model_name"],
            "top_k":         self.top_k,
            "embed_dir":     self.config["models"]["embeddings"],
            "specialists":   self.metadata["specialists"],
        }
        with open(artifact_path, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"✅ Model artifact saved to {artifact_path}")
        return artifact_path


def run_model_registration():
    """Register the model and run a quick smoke test"""
    logger.info("Initializing SpecialistMapper...")
    mapper = SpecialistMapper()

    # Save artifact
    mapper.save()

    # Smoke tests
    test_queries = [
        "I have been experiencing severe headaches and blurred vision for the past week",
        "My chest hurts when I breathe and I feel short of breath",
        "I have itchy red patches on my skin that won't go away",
        "I feel very sad, hopeless, and have lost interest in everything",
        "I have sharp pain in my lower abdomen and burning when I urinate",
    ]

    logger.info("\n" + "="*60)
    logger.info("🧪 SMOKE TEST RESULTS")
    logger.info("="*60)

    for query in test_queries:
        result = mapper.predict(query)
        print(f"\n📝 Query: {query}")
        print(f"⏱  Inference: {result['inference_ms']}ms")
        for i, r in enumerate(result["results"], 1):
            print(f"  #{i} {r['specialist']:<22} | Confidence: {r['confidence']}%")
            print(f"      Matched: {r['matched_symptoms'][:60]}...")

    logger.info("\n✅ Model registration and smoke test complete!")
    return mapper


if __name__ == "__main__":
    run_model_registration()