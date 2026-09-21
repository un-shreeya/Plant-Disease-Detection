import json
import os

class RecommendationEngine:
    def __init__(self, db_path='recommendation/disease_database.json'):
        self.db_path = db_path
        self.database = self._load_database()

    def _load_database(self):
        try:
            with open(self.db_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load disease database: {e}")
            return {}

    def get_recommendation(self, class_name):
        if class_name in self.database:
            return self.database[class_name]
        
        return {
            "crop": class_name.split("___")[0].replace("_", " "),
            "disease": class_name.split("___")[-1].replace("_", " "),
            "symptoms": ["Information currently unavailable in database."],
            "cause": "Unknown",
            "chemical_treatment": "Consult local agricultural extension.",
            "organic_remedy": "Ensure proper watering and air circulation.",
            "prevention": ["Maintain good field hygiene.", "Rotate crops."],
            "weather_precautions": "Monitor for extreme weather."
        }
