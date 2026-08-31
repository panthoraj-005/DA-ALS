# EMG ALS Multi-Modal AI Screening Platform

An end-to-end clinical neurophysiology and artificial intelligence platform for **Amyotrophic Lateral Sclerosis (ALS)** screening using electromyographic (EMG) signals.

The system combines **1-D CNN time-series analysis**, **Florence-2 Vision-Language Model (VLM)** image representation, **XGBoost Meta-Learner feature fusion**, **Grad-CAM & Tree SHAP explainability**, and a **Clinical AI Assistant** powered by Google Gemini.

---

## 🌟 Key Features

- **Multi-Modal AI Fusion Architecture:**
  - **1-D Deep CNN:** Operates directly on bandpass-filtered and z-score normalized raw EMG waveforms to detect motor unit action potential (MUAP) morphology and instability.
  - **Florence-2 Vision-Language Model:** Evaluates visual raster plots and spectrogram interference patterns to capture global density and recruitment anomalies.
  - **XGBoost Meta-Learner:** Synthesizes temporal probabilities, visual embeddings, segment energy dynamics, and entropy features into an aggregate calibrated verdict.
- **Explainable AI (XAI):**
  - **1-D Grad-CAM:** Pinpoints specific high-energy, pathological anomaly windows across the recording.
  - **Tree SHAP:** Computes exact feature contribution bars showing what drove the classification.
- **Clinical AI Assistant (Gemini LLM):**
  - Context-aware neurophysiology Q&A strictly scoped to EMG analysis and patient results.
  - Generates dynamic, clickable follow-up inquiries.
  - Zero-emoji academic markdown formatting with strict clinical guardrails.
  - Interactive on/off toggle for both API and frontend analysis.
- **Modern Minimalist Clinical Web UI:**
  - Single-page application built with React 18, Vite, and TypeScript.
  - Drag-and-drop signal intake supporting `.npy` and `.csv` files.
  - Interactive waveform canvas, segment energy heatmaps, and downloadable clinical reports.

---

## 🏗️ System Architecture

```text
                     +---------------------------------------+
                     |         Raw EMG Signal (CSV / NPY)     |
                     +-------------------+-------------------+
                                         |
                       +-----------------+-----------------+
                       | Preprocessing & Z-Score Norm      |
                       +-----------------+-----------------+
                                         |
                 +-----------------------+-----------------------+
                 |                                               |
                 v                                               v
     +-----------------------+                       +-----------------------+
     |   1-D Waveform CNN    |                       | Florence-2 Vision VLM |
     |  (Temporal Patterns)  |                       |  (Visual Interference)|
     +-----------+-----------+                       +-----------+-----------+
                 |                                               |
                 +-----------------------+-----------------------+
                                         |
                                         v
                         +-------------------------------+
                         |     XGBoost Meta-Learner      |
                         |   (Multi-Modal Feature Fusion)|
                         +---------------+---------------+
                                         |
                                         v
                         +-------------------------------+
                         |   Prediction: ALS vs Normal   |
                         |     Confidence & Severity     |
                         +---------------+---------------+
                                         |
                         +---------------+---------------+
                         |                               |
                         v                               v
             +-----------------------+       +-----------------------+
             | Grad-CAM & Tree SHAP  |       | Clinical AI Assistant |
             |   Explainability      |       | (Gemini 3.1 Flash)    |
             +-----------------------+       +-----------------------+
```

---

## 📁 Repository Structure

```text
├── backend/                  # FastAPI Python backend
│   ├── main.py               # API routes and server initialization
│   ├── chat.py               # Clinical AI Assistant & Gemini orchestration
│   ├── explain.py            # Grad-CAM, SHAP, and summary generation
│   ├── models.py             # PyTorch & XGBoost model loading routines
│   ├── pipeline.py           # Multi-modal inference pipeline
│   ├── config.py             # Environment configuration parser
│   ├── signal_io.py          # EMG file parsing and normalization
│   ├── report.py             # Diagnostic report generator
│   ├── middleware.py         # Rate limiting, security headers, CORS
│   └── requirements.txt      # Backend Python dependencies
├── frontend/                 # React + TypeScript + Vite UI
│   ├── src/
│   │   ├── components/       # UI components (Chatbot, Intake, Verdict, etc.)
│   │   ├── api.ts            # Typed REST API client
│   │   ├── types.ts          # TypeScript interfaces
│   │   └── styles.css        # Warm minimalist design system
│   ├── index.html            # Application entry HTML
│   └── package.json          # Node dependencies and scripts
├── models/                   # Pretrained model checkpoints
│   ├── cnn_best.pt           # 1-D CNN model weights
│   ├── florence_als_head.pt  # Fine-tuned Florence-2 projection head
│   ├── meta_learner.pkl      # Trained XGBoost meta-classifier
│   └── README.md             # Model documentation
├── sample_signals/           # Anonymized sample EMG recordings (.npy, .csv)
├── training_notebook/        # Jupyter notebook for full model training
│   └── ALS_MultiModal_Training_Pipeline.ipynb
├── outputs/                  # Runtime generated charts & Grad-CAM images
├── .env.example              # Configuration environment template
├── .gitignore                # Git ignore rules
├── Dockerfile                # Multi-stage production container
├── docker-compose.yml        # Docker composition configuration
├── setup.bat / setup.sh      # Automated installation scripts
└── run.bat / run.sh          # Quick-start execution scripts
```

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.10+**
- **Node.js 18+** and **npm**
- (Optional) **Google Gemini API Key** for LLM clinical summaries and assistant

---

### Method 1: Automatic Setup (Windows)

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/als-screening-platform.git
   cd als-screening-platform
   ```

2. Run the automated setup script:
   ```cmd
   setup.bat
   ```

3. Configure your `.env` file with your Gemini API key (optional):
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ENABLE_LLM=1
   ```

4. Launch both Backend & Frontend:
   ```cmd
   run.bat
   ```
   Open **`http://localhost:5173`** in your browser.

---

### Method 2: Manual Installation

#### 1. Backend Setup
```bash
cd backend
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
cp ../.env.example ../.env
```

Start the FastAPI backend server:
```bash
python -m uvicorn main:app --port 8000 --host 127.0.0.1 --reload
```
The API documentation is available at `http://127.0.0.1:8000/docs`.

#### 2. Frontend Setup
In a new terminal:
```bash
cd frontend
npm install
npm run dev
```
The frontend is available at `http://localhost:5173`.

---

### Method 3: Running with Docker Compose

```bash
docker-compose up --build
```
Access the application at `http://localhost:8000`.

---

## 🧪 Training & Model Development

The complete dataset preprocessing, 1-D CNN training, Florence-2 VLM fine-tuning, Grad-CAM attribution, and XGBoost Meta-Learner training pipeline is documented in the Jupyter notebook:

```text
training_notebook/ALS_MultiModal_Training_Pipeline.ipynb
```

You can open and execute this notebook in **Google Colab** or locally with GPU acceleration.

---

## 📡 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Upload an EMG signal (`.npy` / `.csv`) or choose a sample to run screening. Accepts `use_llm` toggle. |
| `POST` | `/api/chat` | Send inquiries to the Clinical AI Assistant with patient context. |
| `GET` | `/api/health` | Inspect loaded models, device status, and API capabilities. |
| `GET` | `/api/samples` | Retrieve list of available bundled sample EMG signals. |
| `GET` | `/api/report/{id}` | Export a comprehensive diagnostic summary for a screened record. |

---

## 👥 Authors

This project was designed, developed, and maintained by:

| Author | GitHub |
|---|---|
| Panthoraj | [@panthoraj-005](https://github.com/panthoraj-005) |
| PBS | [@pbs002-s](https://github.com/pbs002-s) |

Contributions covering the multi-modal model architecture, training pipeline, FastAPI backend, and React clinical interface.

---

## ⚖️ Clinical & Research Disclaimer

This software is developed strictly as an **experimental research and screening aid**. It does **not** provide definitive clinical diagnoses. All predictions, probabilities, and AI-generated interpretations must be reviewed and verified by a licensed neurologist or healthcare professional in conjunction with clinical electromyography, patient history, and standard diagnostic criteria (such as the revised El Escorial or Gold Coast criteria).

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.
