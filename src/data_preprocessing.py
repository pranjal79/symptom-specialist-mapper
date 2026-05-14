import os
import pandas as pd
import numpy as np
import yaml
import logging
import json
from data_ingestion import run_ingestion

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ================================================================
# SPECIALIST MAPPING DICTIONARY
# Disease → Medical Specialist
# ================================================================
DISEASE_TO_SPECIALIST = {
    # Neurological
    "migraine": "Neurologist",
    "epilepsy": "Neurologist",
    "alzheimer's disease": "Neurologist",
    "parkinson's disease": "Neurologist",
    "multiple sclerosis": "Neurologist",
    "stroke": "Neurologist",
    "brain tumor": "Neurologist",
    "meningitis": "Neurologist",
    "vertigo": "Neurologist",
    "paralysis (brain hemorrhage)": "Neurologist",
    "cervical spondylosis": "Neurologist",

    # Cardiovascular
    "heart attack": "Cardiologist",
    "hypertension": "Cardiologist",
    "heart failure": "Cardiologist",
    "arrhythmia": "Cardiologist",
    "coronary artery disease": "Cardiologist",
    "angina": "Cardiologist",
    "atherosclerosis": "Cardiologist",

    # Dermatological
    "eczema": "Dermatologist",
    "psoriasis": "Dermatologist",
    "acne": "Dermatologist",
    "rosacea": "Dermatologist",
    "melanoma": "Dermatologist",
    "skin cancer": "Dermatologist",
    "dermatitis": "Dermatologist",
    "urticaria": "Dermatologist",
    "vitiligo": "Dermatologist",
    "fungal infection": "Dermatologist",
    "ringworm": "Dermatologist",
    "chickenpox": "Dermatologist",

    # Gastroenterological
    "gastroenteritis": "Gastroenterologist",
    "irritable bowel syndrome": "Gastroenterologist",
    "crohn's disease": "Gastroenterologist",
    "ulcerative colitis": "Gastroenterologist",
    "gerd": "Gastroenterologist",
    "peptic ulcer disease": "Gastroenterologist",
    "liver cirrhosis": "Gastroenterologist",
    "hepatitis a": "Gastroenterologist",
    "hepatitis b": "Gastroenterologist",
    "hepatitis c": "Gastroenterologist",
    "hepatitis d": "Gastroenterologist",
    "hepatitis e": "Gastroenterologist",
    "appendicitis": "Gastroenterologist",
    "jaundice": "Gastroenterologist",
    "chronic cholestasis": "Gastroenterologist",
    "alcoholic hepatitis": "Gastroenterologist",
    "gastroesophageal reflux disease": "Gastroenterologist",

    # Pulmonological
    "asthma": "Pulmonologist",
    "pneumonia": "Pulmonologist",
    "tuberculosis": "Pulmonologist",
    "copd": "Pulmonologist",
    "bronchial asthma": "Pulmonologist",
    "bronchitis": "Pulmonologist",
    "lung cancer": "Oncologist",
    "pulmonary embolism": "Pulmonologist",
    "sleep apnea": "Pulmonologist",

    # Orthopedic
    "arthritis": "Orthopedist",
    "osteoporosis": "Orthopedist",
    "osteoarthristis": "Orthopedist",
    "fracture": "Orthopedist",
    "scoliosis": "Orthopedist",
    "back pain": "Orthopedist",
    "carpal tunnel syndrome": "Orthopedist",
    "tendinitis": "Orthopedist",
    "sports injury": "Orthopedist",

    # Endocrinological
    "diabetes": "Endocrinologist",
    "hypothyroidism": "Endocrinologist",
    "hyperthyroidism": "Endocrinologist",
    "hypoglycemia": "Endocrinologist",
    "obesity": "Endocrinologist",
    "cushing's syndrome": "Endocrinologist",
    "addison's disease": "Endocrinologist",
    "polycystic ovary syndrome": "Endocrinologist",
    "thyroid": "Endocrinologist",

    # Psychiatric
    "depression": "Psychiatrist",
    "anxiety": "Psychiatrist",
    "schizophrenia": "Psychiatrist",
    "bipolar disorder": "Psychiatrist",
    "ocd": "Psychiatrist",
    "ptsd": "Psychiatrist",
    "eating disorder": "Psychiatrist",
    "adhd": "Psychiatrist",

    # Urological
    "urinary tract infection": "Urologist",
    "kidney stones": "Urologist",
    "prostate cancer": "Oncologist",
    "bladder cancer": "Oncologist",
    "chronic kidney disease": "Urologist",
    "kidney failure": "Urologist",

    # Ophthalmological
    "glaucoma": "Ophthalmologist",
    "cataract": "Ophthalmologist",
    "macular degeneration": "Ophthalmologist",
    "diabetic retinopathy": "Ophthalmologist",
    "conjunctivitis": "Ophthalmologist",

    # ENT
    "sinusitis": "ENT Specialist",
    "tonsillitis": "ENT Specialist",
    "hearing loss": "ENT Specialist",
    "otitis media": "ENT Specialist",
    "laryngitis": "ENT Specialist",
    "tinnitus": "ENT Specialist",
    "allergy": "ENT Specialist",

    # Oncological
    "breast cancer": "Oncologist",
    "colon cancer": "Oncologist",
    "leukemia": "Oncologist",
    "lymphoma": "Oncologist",
    "aids": "Oncologist",

    # General/Infectious
    "common cold": "General Physician",
    "influenza": "General Physician",
    "fever": "General Physician",
    "malaria": "General Physician",
    "typhoid": "General Physician",
    "dengue": "General Physician",
    "covid-19": "General Physician",
    "drug reaction": "General Physician",
    "dimorphic hemmorhoids(piles)": "General Physician",
    "impetigo": "General Physician",
    "pneumonia": "General Physician",
    "varicose veins": "General Physician",
    "hypothyroidism": "Endocrinologist",
}

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def map_specialist(disease_name):
    """Map a disease name to its specialist"""
    disease_lower = str(disease_name).lower().strip()
    if disease_lower in DISEASE_TO_SPECIALIST:
        return DISEASE_TO_SPECIALIST[disease_lower]
    for disease, specialist in DISEASE_TO_SPECIALIST.items():
        if disease in disease_lower or disease_lower in disease:
            return specialist
    return "General Physician"

def clean_text(text):
    """Basic text cleaning"""
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = " ".join(text.split())
    return text

# ================================================================
# DATASET 1: Disease + Symptom_1 to Symptom_17
# ================================================================
def preprocess_disease_symptom(df):
    logger.info("Preprocessing Dataset 1: disease-symptom...")
    df = df.copy()

    symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]

    def merge_symptoms(row):
        symptoms = [clean_text(row[c]) for c in symptom_cols if pd.notna(row[c]) and str(row[c]).strip() != ""]
        return ", ".join(symptoms)

    df["symptom_text"] = df.apply(merge_symptoms, axis=1)
    df["disease"]      = df["Disease"].apply(clean_text)
    df["specialist"]   = df["disease"].apply(map_specialist)

    result = df[["disease", "symptom_text", "specialist"]].dropna()
    result = result[result["symptom_text"] != ""]
    result.drop_duplicates(inplace=True)

    logger.info(f"Dataset 1 processed: {result.shape[0]} rows")
    return result

# ================================================================
# DATASET 2: label (disease) + text (symptom description)
# ================================================================
def preprocess_symptom2disease(df):
    logger.info("Preprocessing Dataset 2: symptom2disease...")
    df = df.copy()

    df["disease"]      = df["label"].apply(clean_text)
    df["symptom_text"] = df["text"].apply(clean_text)
    df["specialist"]   = df["disease"].apply(map_specialist)

    result = df[["disease", "symptom_text", "specialist"]].dropna()
    result = result[result["symptom_text"] != ""]
    result.drop_duplicates(inplace=True)

    logger.info(f"Dataset 2 processed: {result.shape[0]} rows")
    return result

# ================================================================
# DATASET 3: medical_specialty + description + transcription
# ================================================================

# Map medical transcription specialties → our specialist labels
SPECIALTY_MAP = {
    "neurology":                  "Neurologist",
    "neurosurgery":               "Neurologist",
    "cardiology":                 "Cardiologist",
    "cardiovascular / pulmonary": "Cardiologist",
    "dermatology":                "Dermatologist",
    "gastroenterology":           "Gastroenterologist",
    "pulmonology":                "Pulmonologist",
    "orthopedic":                 "Orthopedist",
    "endocrinology":              "Endocrinologist",
    "psychiatry / psychology":    "Psychiatrist",
    "urology":                    "Urologist",
    "ophthalmology":              "Ophthalmologist",
    "ENT - Otolaryngology":       "ENT Specialist",
    "oncology":                   "Oncologist",
    "general medicine":           "General Physician",
    "surgery":                    "General Physician",
    "obstetrics / gynecology":    "Gynecologist",
    "pediatrics":                 "Pediatrician",
    "radiology":                  "General Physician",
    "hematology - oncology":      "Oncologist",
    "nephrology":                 "Urologist",
    "rheumatology":               "Orthopedist",
    "allergy / immunology":       "ENT Specialist",
    "emergency room reports":     "General Physician",
    "physical medicine - rehab":  "Orthopedist",
}

def map_specialty(specialty):
    if pd.isna(specialty):
        return None
    s = str(specialty).lower().strip()
    for key, val in SPECIALTY_MAP.items():
        if key.lower() in s:
            return val
    return "General Physician"

def preprocess_medical_transcriptions(df):
    logger.info("Preprocessing Dataset 3: medical transcriptions...")
    df = df.copy()

    # Combine description + transcription as symptom_text
    df["symptom_text"] = (
        df["description"].apply(clean_text) + " " +
        df["transcription"].apply(clean_text)
    ).str.strip()

    df["disease"]    = df["medical_specialty"].apply(clean_text)
    df["specialist"] = df["medical_specialty"].apply(map_specialty)

    result = df[["disease", "symptom_text", "specialist"]].dropna()
    result = result[result["symptom_text"].str.len() > 20]
    result = result[result["specialist"].notna()]
    result.drop_duplicates(inplace=True)

    logger.info(f"Dataset 3 processed: {result.shape[0]} rows")
    return result

# ================================================================
# MERGE + BUILD KNOWLEDGE BASE
# ================================================================
def build_knowledge_base(df1, df2, df3, config):
    logger.info("Building unified knowledge base...")

    combined = pd.concat([df1, df2, df3], ignore_index=True)
    combined.drop_duplicates(subset=["symptom_text"], inplace=True)
    combined.dropna(subset=["symptom_text", "specialist"], inplace=True)
    combined = combined[combined["symptom_text"].str.strip() != ""]
    combined.reset_index(drop=True, inplace=True)

    logger.info(f"Total knowledge base size: {combined.shape[0]} rows")
    logger.info(f"Specialist distribution:\n{combined['specialist'].value_counts()}")

    # Save processed data
    os.makedirs(config["data"]["processed"], exist_ok=True)
    processed_path = os.path.join(config["data"]["processed"], "combined_dataset.csv")
    combined.to_csv(processed_path, index=False)
    logger.info(f"✅ Saved processed data to {processed_path}")

    # Save knowledge base
    os.makedirs(config["data"]["knowledge_base"], exist_ok=True)
    kb_path = os.path.join(config["data"]["knowledge_base"], "knowledge_base.csv")
    combined.to_csv(kb_path, index=False)
    logger.info(f"✅ Saved knowledge base to {kb_path}")

    # Save specialist list as JSON
    specialist_list = sorted(combined["specialist"].unique().tolist())
    specialist_path = os.path.join(config["data"]["knowledge_base"], "specialists.json")
    with open(specialist_path, "w") as f:
        json.dump(specialist_list, f, indent=2)
    logger.info(f"✅ Saved specialist list: {specialist_list}")

    return combined

def run_preprocessing():
    config = load_config()
    df1, df2, df3 = run_ingestion()

    df1_clean = preprocess_disease_symptom(df1)
    df2_clean = preprocess_symptom2disease(df2)
    df3_clean = preprocess_medical_transcriptions(df3)

    knowledge_base = build_knowledge_base(df1_clean, df2_clean, df3_clean, config)

    logger.info("✅ Full preprocessing pipeline complete!")
    return knowledge_base

if __name__ == "__main__":
    kb = run_preprocessing()
    print("\n📊 Sample Knowledge Base:")
    print(kb.head(10).to_string())