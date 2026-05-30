<div align="center">

<img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/DVC-945DD6?style=for-the-badge&logo=dvc&logoColor=white"/>
<img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
<img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white"/>

# 🏥 Doctor Symptom to Specialist Mapper

### *Describe your symptoms in plain language. Get matched to the right doctor.*

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Click_Here-46E3B7?style=for-the-badge)](https://symptom-specialist-mapper.onrender.com/)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Spaces-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/Pranjal79/symptom-specialist-mapper)
[![MLflow](https://img.shields.io/badge/📊_MLflow-DagHub-0194E2?style=for-the-badge)](https://dagshub.com/pranjal79/symptom-specialist-mapper.mlflow)
[![GitHub](https://img.shields.io/badge/💻_GitHub-Repo-181717?style=for-the-badge)](https://github.com/pranjal79/symptom-specialist-mapper)

![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge)
![Top-3 Accuracy](https://img.shields.io/badge/Top--3_Accuracy-84.5%25-blue?style=for-the-badge)
![Inference](https://img.shields.io/badge/Inference-23ms-orange?style=for-the-badge)
![Specialists](https://img.shields.io/badge/Specialists-15-purple?style=for-the-badge)

</div>

---

## 🎯 Problem Statement

Patients often don't know which doctor to consult. They describe vague symptoms like:

> *"I feel a sharp pain in my chest when I breathe"*
> *"my skin has been itchy and flaking for weeks"*
> *"I feel hopeless and can't sleep at night"*

This system takes raw natural language input and returns the **top-3 most relevant medical specialists** with confidence scores — using **semantic similarity, not keyword search**.

---

## 🚀 Live Demo

| Platform | Link |
|----------|------|
| 🌐 Render | https://symptom-specialist-mapper.onrender.com/ |
| 🤗 HuggingFace | https://huggingface.co/spaces/Pranjal79/symptom-specialist-mapper |

> ⚠️ Render free tier sleeps after 15 min inactivity. First load may take 30-60 seconds.

---

## 💡 How It Works
User types symptoms in plain English
│
▼
Sentence Transformer (all-MiniLM-L6-v2)
→ 384-dimensional embedding vector
│
▼
FAISS Index Search (3,831 vectors)
→ Top-50 most similar symptom patterns
│
▼
Specialist Aggregator
→ Group + rank by confidence score
│
▼
Top-3 Specialists with confidence %
**Example Predictions:**

| Input | Top Specialist | Confidence |
|-------|---------------|------------|
| *"chest pain radiating to left arm"* | Cardiologist | 72.3% |
| *"itchy red patches on skin"* | Dermatologist | 68.1% |
| *"burning when urinating"* | Urologist | 65.4% |
| *"feeling hopeless and sad"* | Psychiatrist | 61.2% |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| NLP Model | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS |
| Frontend | Streamlit |
| Experiment Tracking | MLflow + DagHub |
| Data Versioning | DVC + DagHub |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deployment | Render + HuggingFace |
| Language | Python 3.10 |

---

## 📊 Evaluation Results

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Top-1 Accuracy | 53.6% | > 50% | ✅ Pass |
| **Top-3 Accuracy** | **84.5%** | **> 80%** | ✅ Pass |
| MRR Score | 0.6726 | > 0.65 | ✅ Pass |
| Avg Inference Time | 23ms | < 100ms | ✅ Pass |

---

## 🏥 Supported Specialists (15)

| Icon | Specialist | Conditions |
|------|-----------|------------|
| 🧠 | Neurologist | Migraine, Epilepsy, Parkinson's |
| ❤️ | Cardiologist | Heart Attack, Hypertension |
| 🩹 | Dermatologist | Eczema, Psoriasis, Acne |
| 🫁 | Gastroenterologist | IBS, Hepatitis, GERD |
| 🌬️ | Pulmonologist | Asthma, COPD, Tuberculosis |
| 🦴 | Orthopedist | Arthritis, Fractures, Back Pain |
| ⚗️ | Endocrinologist | Diabetes, Thyroid, PCOS |
| 🧘 | Psychiatrist | Depression, Anxiety, PTSD |
| 💧 | Urologist | UTI, Kidney Stones |
| 👁️ | Ophthalmologist | Glaucoma, Cataract |
| 👂 | ENT Specialist | Sinusitis, Tinnitus |
| 🔬 | Oncologist | Cancer, Lymphoma |
| 🌸 | Gynecologist | PCOS, Endometriosis |
| 👶 | Pediatrician | Child Illnesses |
| 🩺 | General Physician | Fever, Flu, Infections |

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/pranjal79/symptom-specialist-mapper.git
cd symptom-specialist-mapper

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Pull data and models
dvc pull

# Run the app
streamlit run app/app.py
```

---

## ⚙️ ML Pipeline (DVC)

```bash
# Run full pipeline end to end
dvc repro

# Check pipeline status
dvc status

# Visualize pipeline
dvc dag
```

Pipeline stages:
data_ingestion → data_preprocessing → generate_embeddings → register_model → evaluate
---

## 🐳 Docker

```bash
docker build -t symptom-specialist-mapper .
docker run -p 8501:8501 symptom-specialist-mapper
```

---

## 🔗 All Links

| Resource | Link |
|----------|------|
| 🚀 Live App | https://symptom-specialist-mapper.onrender.com/ |
| 🤗 HuggingFace | https://huggingface.co/spaces/Pranjal79/symptom-specialist-mapper |
| 📊 MLflow | https://dagshub.com/pranjal79/symptom-specialist-mapper.mlflow |
| 📦 DagHub DVC | https://dagshub.com/pranjal79/symptom-specialist-mapper |
| 💻 GitHub | https://github.com/pranjal79/symptom-specialist-mapper |
| ⚙️ CI/CD | https://github.com/pranjal79/symptom-specialist-mapper/actions |

---

## ⚠️ Disclaimer

This tool is for **informational and educational purposes only**.
It does **NOT** constitute medical advice or diagnosis.
Always consult a qualified licensed physician for proper evaluation.

---

## 👤 Author

**Pranjal**

- 🐙 GitHub: [@pranjal79](https://github.com/pranjal79)
- 🤗 HuggingFace: [Pranjal79](https://huggingface.co/Pranjal79)
- 📦 DagHub: [pranjal79](https://dagshub.com/pranjal79)

---

<div align="center">

⭐ If you found this useful, please star the repo! ⭐

**Built with ❤️ | Python 🐍 | NLP 🧠 | MLOps 🚀**

</div>