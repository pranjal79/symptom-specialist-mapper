from setuptools import setup, find_packages

setup(
    name="symptom-specialist-mapper",
    version="1.0.0",
    description="NLP-based symptom to medical specialist mapper using sentence embeddings",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "sentence-transformers==2.7.0",
        "faiss-cpu==1.8.0",
        "pandas==2.2.2",
        "numpy==1.26.4",
        "scikit-learn==1.5.0",
        "streamlit==1.35.0",
        "pyyaml==6.0.1",
        "torch==2.3.0",
        "transformers==4.41.2",
        "mlflow==2.13.0",
        "dagshub==0.3.25",
        "dvc==3.51.2",
        "tqdm==4.66.4",
    ],
)