import os
import base64
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class GroqVisionChecker:
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

        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            mime_type = 'image/png'
        elif image_bytes[:3] == b'\xff\xd8\xff':
            mime_type = 'image/jpeg'
        elif image_bytes[:6] in (b'GIF87a', b'GIF89a'):
            mime_type = 'image/gif'
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            mime_type = 'image/webp'
        else:
            mime_type = 'image/jpeg'

        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        try:
            import threading
            result_holder = [None]
            error_holder = [None]

            def call_groq():
                try:
                    print(f'[VisionChecker] Sending image to Groq Qwen...')
                    response = self.client.chat.completions.create(
                        model="qwen/qwen3.8-27b",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime_type};base64,{base64_image}",
                                        }
                                    }
                                ]
                            }
                        ],
                        temperature=0,
                        max_tokens=256
                    )
                    text = response.choices[0].message.content.strip()
                    text = re.sub(r'^```(?:json)?\n?', '', text)
                    text = re.sub(r'\n?```$', '', text)
                    result_holder[0] = json.loads(text)
                    print(f'[VisionChecker] Result: {result_holder[0]}')
                except Exception as e:
                    error_holder[0] = e

            thread = threading.Thread(target=call_groq, daemon=True)
            thread.start()
            thread.join(timeout=15)

            if thread.is_alive():
                print('[VisionChecker] Timeout — falling back to CNN pipeline')
                return {'category': 'leaf_disease', 'pest_name': None, 'advice': ''}
            if error_holder[0]:
                print(f'[VisionChecker] ERROR: {error_holder[0]}')
                return {'category': 'leaf_disease', 'pest_name': None, 'advice': ''}
            return result_holder[0] or {'category': 'leaf_disease', 'pest_name': None, 'advice': ''}
        except Exception as e:
            print(f'[VisionChecker] Unexpected error: {e}')
            return {'category': 'leaf_disease', 'pest_name': None, 'advice': ''}

class GroqChatbot:
    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY')
        self.chat_history = []
        if self.api_key and self.api_key != 'your_groq_api_key_here':
            self.client = Groq(api_key=self.api_key)
            self.vision_checker = GroqVisionChecker(self.client)
        else:
            self.client = None
            self.vision_checker = GroqVisionChecker(None)
            print('Warning: GROQ_API_KEY not found. Chatbot will run in fallback mode.')

    def get_response(self, user_message, context=None):
        if not self.client:
            return 'I am currently running in offline mode because the Groq API key is not configured. Please add it to the .env file to enable AI responses.'

        system_instruction = (
            'You are a friendly agricultural assistant chatting with a farmer. '
            'Speak in a warm, conversational tone like a knowledgeable friend, not a report. '
            'CRITICAL FORMATTING RULES: Never use markdown. No asterisks, no bold, no bullet points with *, no headers. '
            'If you need to list things, use plain numbered sentences like "1. Do this. 2. Do that." '
            'Keep your response under 120 words and always sound natural.'
        )
        
        current_disease = context.get('disease') if context else None
        if not hasattr(self, 'last_disease'):
            self.last_disease = current_disease
        elif self.last_disease != current_disease:
            self.chat_history = []
            self.last_disease = current_disease
            
        if context:
            system_instruction += (
                f'\n\nThe farmer plant scan shows: Disease: {context.get("disease", "Unknown")}, '
                f'Severity: {context.get("severity", "Unknown")}, '
                f'Symptoms: {", ".join(context.get("symptoms", []))}. '
                f'Suggested treatment: {context.get("chemical_treatment", "N/A")}.'
            )

        # Only store a short history to prevent context bloat
        if len(self.chat_history) == 0:
            self.chat_history.append({"role": "system", "content": system_instruction})
        else:
            # Update system prompt if context changed
            self.chat_history[0] = {"role": "system", "content": system_instruction}

        self.chat_history.append({"role": "user", "content": user_message})

        try:
            import threading
            result_holder = [None]
            error_holder = [None]
            
            def call_chat():
                try:
                    response = self.client.chat.completions.create(
                        model="qwen/qwen3.8-27b",
                        messages=self.chat_history,
                        temperature=0.7,
                        max_tokens=256
                    )
                    text = response.choices[0].message.content
                    self.chat_history.append({"role": "assistant", "content": text})
                    result_holder[0] = text
                except Exception as e:
                    error_holder[0] = e
                    
            thread = threading.Thread(target=call_chat, daemon=True)
            thread.start()
            thread.join(timeout=10)
            
            if thread.is_alive():
                print('[Chatbot] Timeout waiting for Groq API.')
                self.chat_history.pop() # Remove user message since it failed
                return "The AI is currently overloaded with requests and taking too long to respond. Please try again in a minute!"
                
            if error_holder[0]:
                print(f'[Chatbot] API Error: {error_holder[0]}')
                self.chat_history.pop()
                return "I'm having trouble connecting to my AI brain right now. Please try again in a moment!"
                
            return result_holder[0]
        except Exception as e:
            print(f'[Chatbot] Unexpected Error: {e}')
            return "I encountered an unexpected error. Please try again."
