import os
import base64
import re
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


class GeminiVisionChecker:
    def __init__(self, client):
        self.client = client

    def check_image(self, image_bytes: bytes) -> dict:
        if not self.client:
            return {'category': 'leaf_disease', 'pest_name': None, 'advice': ''}

        prompt = (
            'You are an expert botanist and entomologist. Look at this image carefully.\n\n'
            'Classify it into EXACTLY ONE of these categories:\n'
            '1. leaf_disease -- the image shows a plant leaf with a fungal/bacterial disease\n'
            '2. insect_pest -- the image shows insects, pests, or insect damage on a plant (e.g. aphids, whiteflies, caterpillars)\n'
            '3. not_a_plant -- the image is not a plant or leaf at all\n\n'
            'Respond in this exact JSON format (no markdown, no explanation):\n'
            '{"category": "...", "pest_name": "...", "advice": "..."}\n\n'
            'For pest_name: common name of pest if insect_pest, otherwise null.\n'
            'For advice: if insect_pest, write 2-3 plain-English sentences with advice for the farmer. Otherwise empty string.'
        )

        # Detect mime type from magic bytes
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            mime_type = 'image/png'
        elif image_bytes[:3] == b'\xff\xd8\xff':
            mime_type = 'image/jpeg'
        elif image_bytes[:6] in (b'GIF87a', b'GIF89a'):
            mime_type = 'image/gif'
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            mime_type = 'image/webp'
        else:
            mime_type = 'image/jpeg'  # fallback

        try:
            print(f'[VisionChecker] Sending image ({mime_type}, {len(image_bytes)} bytes) to Gemini...')
            response = self.client.models.generate_content(
                model='gemini-3.6-flash',
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    types.Part.from_text(text=prompt),
                ],
            )
            text = response.text.strip()
            text = re.sub(r'^```(?:json)?\n?', '', text)
            text = re.sub(r'\n?```$', '', text)
            result = json.loads(text)
            print(f'[VisionChecker] Result: {result}')
            return result
        except Exception as e:
            print(f'[VisionChecker] ERROR: {e}')
            return {'category': 'leaf_disease', 'pest_name': None, 'advice': ''}


class GeminiChatbot:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.chat_session = None
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            self.vision_checker = GeminiVisionChecker(self.client)
        else:
            self.client = None
            self.vision_checker = GeminiVisionChecker(None)
            print('Warning: GEMINI_API_KEY not found. Chatbot will run in fallback mode.')

    def get_response(self, user_message, context=None):
        if not self.client:
            return 'I am currently running in offline mode because the Gemini API key is not configured. Please add it to the .env file to enable AI responses.'

        system_instruction = (
            'You are a friendly agricultural assistant chatting with a farmer. '
            'Speak in a warm, conversational tone like a knowledgeable friend, not a report. '
            'CRITICAL FORMATTING RULES: Never use markdown. No asterisks, no bold, no bullet points with *, no headers. '
            'If you need to list things, use plain numbered sentences like "1. Do this. 2. Do that." '
            'Keep your response under 120 words and always sound natural.'
        )
        if context:
            system_instruction += (
                f'\n\nThe farmer plant scan shows: Disease: {context.get("disease", "Unknown")}, '
                f'Severity: {context.get("severity", "Unknown")}, '
                f'Symptoms: {", ".join(context.get("symptoms", []))}. '
                f'Suggested treatment: {context.get("chemical_treatment", "N/A")}.'
            )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7
        )

        try:
            if self.chat_session is None:
                self.chat_session = self.client.chats.create(
                    model='gemini-3.6-flash',
                    config=config
                )
            response = self.chat_session.send_message(user_message)
            return response.text
        except Exception as e:
            print(f'Gemini API Error: {e}')
            return "I'm having trouble connecting to my AI brain right now. Please try again in a moment!"
