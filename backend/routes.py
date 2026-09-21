import os
import io
import json
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from PIL import Image

# Use absolute imports relative to project root
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.efficientnet_model import create_model
from models.gradcam import GradCAMExplainer
from models.severity import SeverityEstimator
from recommendation.engine import RecommendationEngine
from chatbot.gemini_chatbot import GeminiChatbot

router = APIRouter()

# Global variables for models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
class_names = []
gradcam_explainer = None
severity_estimator = SeverityEstimator()
rec_engine = RecommendationEngine()
chatbot = GeminiChatbot()

def load_model_if_needed():
    global model, class_names, gradcam_explainer
    if model is None:
        model_path = os.getenv("MODEL_PATH", "models/saved/best_model.pth")
        
        # Load class mapping
        try:
            with open("data/class_mapping.json", "r") as f:
                mapping = json.load(f)
                # Sort by key (which is string integer)
                class_names = [mapping[str(i)] for i in range(len(mapping))]
        except Exception:
            class_names = [f"Class_{i}" for i in range(38)] # Fallback

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model = create_model(num_classes=38, pretrained=False)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("WARNING: Using untrained model for inference. Train model first!")
            model = create_model(num_classes=38, pretrained=False)
        
        model = model.to(device)
        model.eval()
        gradcam_explainer = GradCAMExplainer(model, device)

class ChatMessage(BaseModel):
    message: str
    context: dict = None

@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    load_model_if_needed()
    
    # Read and process image
    image_bytes = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_img_np = np.array(pil_image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # --- Gemini Vision Pre-Check Gate ---
    # Ask Gemini to classify the image BEFORE running the CNN.
    # This catches insect pests and non-plant images that the CNN cannot handle.
    vision_result = chatbot.vision_checker.check_image(image_bytes)
    vision_category = vision_result.get('category', 'leaf_disease')

    if vision_category == 'insect_pest':
        pest_name = vision_result.get('pest_name') or 'Unknown Pest'
        advice = vision_result.get('advice', 'Please consult an agricultural expert for pest control options.')
        return {
            "detection_type": "insect_pest",
            "class_index": -1,
            "class_name": f"Insect Pest: {pest_name}",
            "confidence": 1.0,
            "gradcam_overlay": None,
            "severity": {"severity": "N/A", "percentage": 0.0},
            "recommendation": {
                "crop": "Unknown",
                "disease": f"Insect Pest: {pest_name}",
                "symptoms": ["Insect pest detected — this is not a leaf disease."],
                "cause": f"The image shows {pest_name}, which are insects damaging the plant.",
                "chemical_treatment": advice,
                "organic_remedy": "Spray neem oil or insecticidal soap on affected areas. Remove heavily infested leaves by hand.",
                "prevention": [
                    "Inspect plants regularly for early signs of infestation.",
                    "Introduce natural predators like ladybugs for aphid control.",
                    "Avoid over-fertilizing with nitrogen, which attracts soft-bodied insects."
                ],
                "weather_precautions": "Pests thrive in warm, humid weather. Increase inspection frequency during these conditions."
            }
        }

    if vision_category == 'not_a_plant':
        return {
            "detection_type": "not_a_plant",
            "class_index": -1,
            "class_name": "Not a Plant",
            "confidence": 1.0,
            "gradcam_overlay": None,
            "severity": {"severity": "N/A", "percentage": 0.0},
            "recommendation": {
                "crop": "Unknown",
                "disease": "Not a Plant",
                "symptoms": ["The uploaded image does not appear to be a plant leaf."],
                "cause": "Please upload a clear photo of a plant leaf for diagnosis.",
                "chemical_treatment": "N/A",
                "organic_remedy": "N/A",
                "prevention": ["Upload a well-lit, close-up photo of a single plant leaf."],
                "weather_precautions": "N/A"
            }
        }

    # Prepare tensor
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(pil_image).unsqueeze(0)

    # 1. Grad-CAM & Prediction
    cam_result = gradcam_explainer.generate(input_tensor, original_image=original_img_np)
    pred_class_idx = cam_result['predicted_class']
    confidence = cam_result['confidence']
    class_name = class_names[pred_class_idx]
    
    # Check Confidence Threshold
    if confidence < 0.50:
        class_name = "Unrecognized / Low Confidence"
        severity_result = {"severity": "Unknown", "percentage": 0.0}
        recommendation = {
            "crop": "Unknown",
            "disease": "Unrecognized",
            "symptoms": ["Confidence is too low to identify symptoms."],
            "cause": "Image may be blurry, not a leaf, or an unknown disease.",
            "chemical_treatment": "N/A",
            "organic_remedy": "N/A",
            "prevention": ["Please upload a clear, focused image of a plant leaf."],
            "weather_precautions": "N/A"
        }
    else:
        # 2. Severity Estimation
        severity_result = severity_estimator.estimate(original_img_np)
        
        # 3. Recommendation
        recommendation = rec_engine.get_recommendation(class_name)

    return {
        "detection_type": "leaf_disease",
        "class_index": pred_class_idx if confidence >= 0.50 else -1,
        "class_name": class_name,
        "confidence": confidence,
        "gradcam_overlay": cam_result['overlay_base64'],
        "severity": severity_result,
        "recommendation": recommendation
    }

@router.post("/chatbot")
async def chat(chat_req: ChatMessage):
    response = chatbot.get_response(chat_req.message, chat_req.context)
    return {"reply": response}

@router.get("/health")
def health_check():
    return {"status": "healthy"}
