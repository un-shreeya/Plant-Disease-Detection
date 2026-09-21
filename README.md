# Leaf AI — Intelligent Plant Disease & Pest Advisory

A hybrid deep learning platform that uses Groq Vision AI (LLaMA-3.2 Vision) as an upstream guardrail to detect insect pests and non-plant uploads, routing valid leaf scans to a custom-trained PyTorch CNN (EfficientNet-B3) for 38-class disease classification, Grad-CAM explainability, and HSV severity estimation. Built with FastAPI and a responsive vanilla JS frontend.

---

## Architecture

```
Image Upload
     │
     ▼
Groq Vision Pre-Check
     ├─ insect_pest  ──► Pest advice (no CNN needed)
     ├─ not_a_plant  ──► Rejected with guidance
     └─ leaf_disease ──► EfficientNet-B3 CNN
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              Grad-CAM           Severity Estimator
              Heatmap            (HSV color analysis)
                    │                   │
                    └─────────┬─────────┘
                              ▼
                    Recommendation Engine
                    + AI Agronomist Chat
```

## Screenshots:
![Insect Attack1](https://github.com/un-shreeya/Plant-Disease-Detection/blob/main/Screenshot%202026-09-21%20210622.png)
![Insect Attack1 AI chatbot Response](https://github.com/un-shreeya/Plant-Disease-Detection/blob/main/Screenshot%202026-09-21%20210633.png)
![Insect Attack1 AI chatbot Response contd.](https://github.com/un-shreeya/Plant-Disease-Detection/blob/main/Screenshot%202026-09-21%20210651.png)
![Insect Attack2](https://github.com/un-shreeya/Plant-Disease-Detection/blob/main/Screenshot%202026-09-21%20210907.png)
![Leaf Disease Analysis + AI chatbot Response](https://github.com/un-shreeya/Plant-Disease-Detection/blob/main/Screenshot%202026-09-21%20211149.png)

---

## After cloning

### 1. Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com/keys)

### 2. Install dependencies
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 3. Set up your `.env` file
Create a file named `.env` in the project root (this is gitignored for security):
```
GROQ_API_KEY=your_api_key_here
MODEL_PATH=models/saved/best_model.pth
```

### 4. Download the dataset (for training)
```bash
python data/download_dataset.py
```

### 5. Train the model
```bash
# Full training (approx 6 hours on CPU, approx 30 min on GPU)
python models/train.py

# Quick test run (5% of data, 2 epochs)
python models/train.py --quick
```
The trained model weights (`best_model.pth`) are not included in this repo due to file size. The model has to be trained first.

### 6. Run the app
```bash
python -m uvicorn backend.app:app --reload
```
Then open [http://localhost:8000](http://localhost:8000)

---

## Features

| Feature | Technology |
|---|---|
| Leaf disease classification (38 classes) | EfficientNet-B3 CNN (PlantVillage dataset) |
| Insect pest detection | Groq Qwen 3.8-27b |
| Explainability heatmap | Grad-CAM |
| Disease severity estimate | HSV color segmentation |
| Treatment recommendations | JSON disease database |
| AI chat assistant | Groq Qwen 3.8-27b |

---

## Supported Diseases (38 classes)
Tomato, Potato, Pepper, Apple, Grape, Corn, Cherry, Peach, Strawberry, Squash — covering diseases including Early Blight, Late Blight, Leaf Spot, Rust, Powdery Mildew, and healthy class.

---

## Project Structure
```
plant-disease-detection/
├── backend/          # FastAPI routes and app entry point
├── chatbot/          # Groq Vision pre-check + AI chatbot
├── data/             # Dataset download script + class mapping
├── frontend/         # HTML, CSS, JS (served as static files)
├── models/           # EfficientNet model, Grad-CAM, severity, training
├── recommendation/   # Disease database + recommendation engine
├── requirements.txt
└── .env              
```
