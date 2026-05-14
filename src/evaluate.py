import os
import json
import time
import logging
import numpy as np
import pandas as pd
import yaml
import mlflow
import dagshub

from model import SpecialistMapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ================================================================
# TEST DATASET — 100 symptom descriptions with ground truth labels
# ================================================================
TEST_DATA = [
    # Neurologist
    ("I have severe headaches that pound on one side of my head with nausea", "Neurologist"),
    ("I keep having seizures and losing consciousness suddenly", "Neurologist"),
    ("My hands shake uncontrollably and I have trouble walking", "Neurologist"),
    ("I have memory loss and confusion that is getting worse each day", "Neurologist"),
    ("I feel numbness and tingling in my arms and legs frequently", "Neurologist"),
    ("I had sudden weakness on one side of my body and slurred speech", "Neurologist"),
    ("I have been getting severe migraines with visual aura for months", "Neurologist"),
    ("My vision suddenly went blurry and I had a terrible headache", "Neurologist"),

    # Cardiologist
    ("I have chest pain and my heart feels like it is racing", "Cardiologist"),
    ("My blood pressure is very high and I feel dizzy often", "Cardiologist"),
    ("I feel shortness of breath and chest tightness when I climb stairs", "Cardiologist"),
    ("My heart skips beats and I feel palpitations throughout the day", "Cardiologist"),
    ("I have swelling in my legs and feel breathless lying down at night", "Cardiologist"),
    ("I had crushing chest pain that spread to my left arm", "Cardiologist"),
    ("I feel fatigue and dizziness along with irregular heartbeat", "Cardiologist"),

    # Dermatologist
    ("My skin has red itchy patches that are dry and flaking", "Dermatologist"),
    ("I have a rash all over my body that is very itchy and spreading", "Dermatologist"),
    ("I have severe acne on my face and back that won't go away", "Dermatologist"),
    ("My skin has white patches that are spreading on my arms", "Dermatologist"),
    ("I notice a dark mole on my skin that has changed in shape", "Dermatologist"),
    ("I have eczema and my skin is cracked and bleeding in winter", "Dermatologist"),
    ("There are ring shaped red patches on my skin that itch", "Dermatologist"),

    # Gastroenterologist
    ("I have severe stomach pain, bloating and diarrhea after eating", "Gastroenterologist"),
    ("I feel nausea and vomiting frequently with loss of appetite", "Gastroenterologist"),
    ("My stool has blood in it and I have severe abdominal cramps", "Gastroenterologist"),
    ("I have acid reflux and heartburn that burns my chest after meals", "Gastroenterologist"),
    ("My skin and eyes have turned yellow and I feel very weak", "Gastroenterologist"),
    ("I have chronic constipation and bloating with abdominal discomfort", "Gastroenterologist"),
    ("I feel pain in the upper right abdomen and my urine is dark", "Gastroenterologist"),

    # Pulmonologist
    ("I have a persistent cough for weeks with yellow mucus", "Pulmonologist"),
    ("I wheeze when I breathe and feel tightness in my chest", "Pulmonologist"),
    ("I cough up blood and have lost significant weight recently", "Pulmonologist"),
    ("I snore loudly and feel very tired even after sleeping all night", "Pulmonologist"),
    ("I have difficulty breathing and my lips turn bluish sometimes", "Pulmonologist"),
    ("I have chronic shortness of breath that gets worse with exercise", "Pulmonologist"),

    # Orthopedist
    ("My knee joint is very painful and swollen and hard to bend", "Orthopedist"),
    ("I have severe lower back pain that shoots down my leg", "Orthopedist"),
    ("My bones are weak and I fractured my wrist from a minor fall", "Orthopedist"),
    ("I have pain and stiffness in my joints every morning", "Orthopedist"),
    ("My shoulder hurts when I lift my arm and I cannot rotate it", "Orthopedist"),
    ("I have neck pain and stiffness that spreads to my shoulders", "Orthopedist"),

    # Endocrinologist
    ("I feel very thirsty all the time and urinate frequently", "Endocrinologist"),
    ("I have gained a lot of weight and feel very tired and cold", "Endocrinologist"),
    ("My heart races and I sweat a lot despite not doing exercise", "Endocrinologist"),
    ("I have high blood sugar and my wounds take very long to heal", "Endocrinologist"),
    ("I feel shaky and confused when I have not eaten for a while", "Endocrinologist"),
    ("I have irregular periods and excessive hair growth on my face", "Endocrinologist"),

    # Psychiatrist
    ("I feel hopeless and sad every day and have lost interest in life", "Psychiatrist"),
    ("I have panic attacks with racing heart and difficulty breathing", "Psychiatrist"),
    ("I hear voices that nobody else can hear and feel paranoid", "Psychiatrist"),
    ("I cannot sleep and have flashbacks of a traumatic experience", "Psychiatrist"),
    ("I feel extreme mood swings from very happy to very sad", "Psychiatrist"),
    ("I have obsessive thoughts and perform rituals repeatedly", "Psychiatrist"),

    # Urologist
    ("I feel a burning pain when I urinate and need to go very often", "Urologist"),
    ("I have severe pain in my lower back that comes in waves", "Urologist"),
    ("I see blood in my urine and have lower abdominal pressure", "Urologist"),
    ("I have difficulty starting urination and weak urine stream", "Urologist"),
    ("I have kidney pain and frequent urinary tract infections", "Urologist"),

    # Ophthalmologist
    ("My vision has become blurry and I see halos around lights", "Ophthalmologist"),
    ("I have eye pain and redness with sensitivity to light", "Ophthalmologist"),
    ("I see floaters and flashes of light in my vision suddenly", "Ophthalmologist"),
    ("My eyes water constantly and are very itchy and red", "Ophthalmologist"),
    ("I have double vision and difficulty focusing on objects", "Ophthalmologist"),

    # ENT Specialist
    ("I have a blocked nose, headache, and facial pain for weeks", "ENT Specialist"),
    ("My throat is very sore and I have swollen painful tonsils", "ENT Specialist"),
    ("I have been losing my hearing gradually in one ear", "ENT Specialist"),
    ("I hear a constant ringing sound in my ears all day", "ENT Specialist"),
    ("I have frequent nosebleeds and nasal congestion", "ENT Specialist"),
    ("I sneeze constantly and have watery eyes and runny nose", "ENT Specialist"),

    # Oncologist
    ("I have a painless lump in my breast that has been growing", "Oncologist"),
    ("I have unexplained weight loss and night sweats for months", "Oncologist"),
    ("I feel extremely fatigued and have swollen lymph nodes in my neck", "Oncologist"),
    ("I have a lump in my neck that has been there for two months", "Oncologist"),

    # Gynecologist
    ("I have irregular and very painful periods with heavy bleeding", "Gynecologist"),
    ("I have pelvic pain and bloating throughout the month", "Gynecologist"),
    ("I have vaginal discharge that is unusual and has a bad odor", "Gynecologist"),

    # Pediatrician
    ("My child has high fever, rash and is crying constantly", "Pediatrician"),
    ("My baby is not gaining weight and refuses to feed properly", "Pediatrician"),
    ("My child has developmental delays and is not speaking yet", "Pediatrician"),

    # General Physician
    ("I have a fever, body aches, and runny nose since yesterday", "General Physician"),
    ("I feel very weak and tired with mild fever and headache", "General Physician"),
    ("I have a sore throat, cough and mild fever since three days", "General Physician"),
    ("I have chills, sweating and high fever that comes and goes", "General Physician"),
    ("I feel generally unwell with body pain and loss of appetite", "General Physician"),
]

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

def compute_metrics(mapper, test_data):
    """Compute Top-1, Top-3 accuracy and MRR"""
    logger.info(f"Running evaluation on {len(test_data)} test cases...")

    top1_correct  = 0
    top3_correct  = 0
    reciprocal_ranks = []
    inference_times  = []

    results_log = []

    for symptom_text, ground_truth in test_data:
        try:
            result     = mapper.predict(symptom_text)
            predictions = [r["specialist"] for r in result["results"]]
            inf_time   = result["inference_ms"]
            inference_times.append(inf_time)

            # Top-1 accuracy
            if predictions and predictions[0] == ground_truth:
                top1_correct += 1

            # Top-3 accuracy + MRR
            if ground_truth in predictions:
                top3_correct += 1
                rank = predictions.index(ground_truth) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

            results_log.append({
                "symptom":       symptom_text,
                "ground_truth":  ground_truth,
                "top1_pred":     predictions[0] if predictions else "None",
                "top3_preds":    str(predictions),
                "correct_top1":  predictions[0] == ground_truth if predictions else False,
                "correct_top3":  ground_truth in predictions,
                "inference_ms":  inf_time,
            })

        except Exception as e:
            logger.error(f"Error on query '{symptom_text}': {e}")
            reciprocal_ranks.append(0.0)

    total = len(test_data)
    metrics = {
        "top1_accuracy":       round(top1_correct / total, 4),
        "top3_accuracy":       round(top3_correct / total, 4),
        "mrr_score":           round(np.mean(reciprocal_ranks), 4),
        "avg_inference_ms":    round(np.mean(inference_times), 2),
        "total_test_cases":    total,
        "top1_correct":        top1_correct,
        "top3_correct":        top3_correct,
    }

    return metrics, results_log

def log_to_mlflow(metrics, params, mapper):
    """Log everything to MLflow via DagHub"""
    logger.info("Logging to MLflow...")

    with mlflow.start_run(run_name=f"eval_{params['embedding_model'].split('/')[-1]}"):

        # Log params
        mlflow.log_param("embedding_model",    params["embedding_model"])
        mlflow.log_param("similarity_metric",  params["similarity_metric"])
        mlflow.log_param("top_k",              params["top_k"])
        mlflow.log_param("batch_size",         params["batch_size"])

        # Log metrics
        mlflow.log_metric("top1_accuracy",     metrics["top1_accuracy"])
        mlflow.log_metric("top3_accuracy",     metrics["top3_accuracy"])
        mlflow.log_metric("mrr_score",         metrics["mrr_score"])
        mlflow.log_metric("avg_inference_ms",  metrics["avg_inference_ms"])
        mlflow.log_metric("total_test_cases",  metrics["total_test_cases"])

        # Log artifacts
        mlflow.log_artifact("models/specialist_mapper.pkl")
        mlflow.log_artifact("models/embeddings/faiss_index.bin")
        mlflow.log_artifact("models/embeddings/metadata.json")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"✅ MLflow run logged | Run ID: {run_id}")

    return run_id

def save_metrics(metrics, config):
    """Save metrics to metrics/scores.json for DVC tracking"""
    os.makedirs("metrics", exist_ok=True)
    metrics_path = config["metrics"]["output"]
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"✅ Metrics saved to {metrics_path}")

def run_evaluation():
    config = load_config()
    params = load_params()

    # Initialize DagHub MLflow tracking
    # Replace with your actual dagshub username
    try:
        dagshub.init(
            repo_owner="pranjal79",
            repo_name="symptom-specialist-mapper",
            mlflow=True
        )
        logger.info("✅ DagHub MLflow tracking initialized")
    except Exception as e:
        logger.warning(f"DagHub init failed (running locally): {e}")
        mlflow.set_tracking_uri("mlruns")

    # Load model
    logger.info("Loading SpecialistMapper...")
    mapper = SpecialistMapper()

    # Run evaluation
    metrics, results_log = compute_metrics(mapper, TEST_DATA)

    # Print results
    logger.info("\n" + "="*60)
    logger.info("📊 EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"  Top-1 Accuracy  : {metrics['top1_accuracy']*100:.1f}%")
    logger.info(f"  Top-3 Accuracy  : {metrics['top3_accuracy']*100:.1f}%")
    logger.info(f"  MRR Score       : {metrics['mrr_score']:.4f}")
    logger.info(f"  Avg Inference   : {metrics['avg_inference_ms']}ms")
    logger.info(f"  Test Cases      : {metrics['total_test_cases']}")
    logger.info("="*60)

    # Show failures for debugging
    failures = [r for r in results_log if not r["correct_top3"]]
    if failures:
        logger.info(f"\n❌ Top-3 Misses ({len(failures)} cases):")
        for f in failures[:5]:
            logger.info(f"  Query     : {f['symptom'][:60]}...")
            logger.info(f"  Expected  : {f['ground_truth']}")
            logger.info(f"  Got       : {f['top3_preds']}")
            logger.info("")

    # Log to MLflow
    try:
        log_to_mlflow(metrics, params, mapper)
    except Exception as e:
        logger.warning(f"MLflow logging skipped: {e}")

    # Save metrics for DVC
    save_metrics(metrics, config)

    # Save detailed results log
    results_df = pd.DataFrame(results_log)
    results_df.to_csv("metrics/detailed_results.csv", index=False)
    logger.info("✅ Detailed results saved to metrics/detailed_results.csv")

    return metrics

if __name__ == "__main__":
    run_evaluation()