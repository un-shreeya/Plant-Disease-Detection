# Leaf AI: Intelligent Plant Disease Advisory System

A complete, end-to-end deep learning project for plant disease detection using PyTorch, FastAPI, and Gemini AI.

## Features
1. **Classification**: 38 classes using EfficientNet-B0.
2. **Explainable AI**: Grad-CAM heatmaps showing where the model is looking.
3. **Severity Estimation**: HSV color-based thresholding to estimate disease severity %.
4. **Advisory System**: Retrieval-based chemical/organic treatments.
5. **LLM Chatbot**: Gemini 1.5 Flash integrated for conversational Q&A.
6. **Beautiful Frontend**: Glassmorphism UI built with vanilla JS/HTML/CSS.

## How to Run the Project

### 1. Setup Environment
```bash
cd plant-disease-detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys
Copy `.env.example` to `.env` and add your Google Gemini API key:
```bash
cp .env.example .env
# Edit .env and insert GEMINI_API_KEY=your_key_here
```

### 3. Run the Full Pipeline
If you haven't trained a model yet, you'll need data:
```bash
# 1. Download and process dataset
python data/download_dataset.py

# 2. Train the model (will save to models/saved/best_model.pth)
python models/train.py

# 3. Evaluate and generate reports/charts
python models/evaluate.py
python evaluation/generate_report.py
```

### 4. Run the Web App (Backend + Frontend)
Once you have a trained model (`best_model.pth`), start the FastAPI server:
```bash
uvicorn backend.app:app --reload
```

Then, open your browser and go to:
**http://127.0.0.1:8000**

You can upload leaf images, see the Grad-CAM heatmap, get treatment advice, and chat with the AI!
