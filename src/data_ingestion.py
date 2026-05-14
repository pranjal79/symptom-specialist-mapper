import os
import pandas as pd
import yaml
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_file_exists(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ {name} not found at: {path}")

    logger.info(f"✅ Found: {name} at {path}")


# =========================================================
# DATASET 1: Disease Symptom Dataset
# =========================================================
def load_disease_symptom(folder_path):
    """
    Load primary disease symptom dataset
    """

    path = os.path.join(folder_path, "dataset.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"dataset.csv not found in {folder_path}")

    logger.info(f"Loading disease-symptom dataset from {path}")

    df = pd.read_csv(path)

    logger.info(f"Shape: {df.shape} | Columns: {list(df.columns)}")

    return df


# =========================================================
# DATASET 2: Symptom2Disease
# =========================================================
def load_symptom2disease(folder_path):
    """
    Load Symptom2Disease dataset
    """

    path = os.path.join(folder_path, "Symptom2Disease.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Symptom2Disease.csv not found in {folder_path}")

    logger.info(f"Loading symptom2disease dataset from {path}")

    df = pd.read_csv(path)

    logger.info(f"Shape: {df.shape} | Columns: {list(df.columns)}")

    return df


# =========================================================
# DATASET 3: Medical Transcriptions
# =========================================================
def load_medical_transcriptions(folder_path):
    """
    Load medical transcription dataset
    """

    path = os.path.join(folder_path, "mtsamples.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"mtsamples.csv not found in {folder_path}")

    logger.info(f"Loading medical transcriptions from {path}")

    df = pd.read_csv(path)

    logger.info(f"Shape: {df.shape} | Columns: {list(df.columns)}")

    return df


# =========================================================
# MAIN INGESTION PIPELINE
# =========================================================
def run_ingestion():

    config = load_config()

    # Validate folders exist
    validate_file_exists(
        config["data"]["raw"]["disease_symptom"],
        "Disease-Symptom folder"
    )

    validate_file_exists(
        config["data"]["raw"]["symptom2disease"],
        "Symptom2Disease folder"
    )

    validate_file_exists(
        config["data"]["raw"]["medical_transcriptions"],
        "Medical Transcriptions folder"
    )

    # Load datasets
    df1 = load_disease_symptom(
        config["data"]["raw"]["disease_symptom"]
    )

    df2 = load_symptom2disease(
        config["data"]["raw"]["symptom2disease"]
    )

    df3 = load_medical_transcriptions(
        config["data"]["raw"]["medical_transcriptions"]
    )

    logger.info("✅ All datasets loaded successfully!")

    logger.info(f"Dataset 1 columns: {list(df1.columns)}")
    logger.info(f"Dataset 2 columns: {list(df2.columns)}")
    logger.info(f"Dataset 3 columns: {list(df3.columns)}")

    return df1, df2, df3


# =========================================================
# STANDALONE EXECUTION
# =========================================================
if __name__ == "__main__":

    logger.info("Running data ingestion validation...")

    df1, df2, df3 = run_ingestion()

    logger.info("✅ Data ingestion complete — all files validated.")