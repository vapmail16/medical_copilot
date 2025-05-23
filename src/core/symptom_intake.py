import os
from typing import Optional, Union, Dict, Any
from pathlib import Path
import base64
from PIL import Image
import io

import openai
from openai import OpenAI
import whisper
from deepgram import Deepgram

class SymptomIntake:
    def __init__(self):
        self.openai_client = OpenAI()
        self.whisper_model = whisper.load_model("base")
        self.deepgram = Deepgram(os.getenv("DEEPGRAM_API_KEY"))

    async def process_text_input(self, text: str) -> Dict[str, Any]:
        """Process text input for symptoms."""
        try:
            # Basic validation
            if not text or len(text.strip()) == 0:
                raise ValueError("Empty text input")
            
            return {
                "type": "text",
                "content": text,
                "status": "success"
            }
        except Exception as e:
            return {
                "type": "text",
                "error": str(e),
                "status": "error"
            }

    async def process_voice_input(self, audio_file_path: Union[str, Path], use_whisper: bool = True) -> Dict[str, Any]:
        """Process voice input using either Whisper or Deepgram."""
        try:
            if use_whisper:
                # Use OpenAI Whisper
                result = self.whisper_model.transcribe(str(audio_file_path))
                text = result["text"]
            else:
                # Use Deepgram
                with open(audio_file_path, "rb") as audio:
                    source = {"buffer": audio, "mimetype": "audio/wav"}
                    response = await self.deepgram.transcription.prerecorded(source, {
                        "smart_format": True,
                        "model": "nova",
                    })
                    text = response["results"]["channels"][0]["alternatives"][0]["transcript"]

            return {
                "type": "voice",
                "content": text,
                "status": "success"
            }
        except Exception as e:
            return {
                "type": "voice",
                "error": str(e),
                "status": "error"
            }

    async def process_image_input(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Process image input using OpenAI Vision API."""
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Use OpenAI Vision API
            response = self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this medical image and describe any visible symptoms or conditions. Focus on medical details."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )

            return {
                "type": "image",
                "content": response.choices[0].message.content,
                "status": "success"
            }
        except Exception as e:
            return {
                "type": "image",
                "error": str(e),
                "status": "error"
            }

    async def process_multi_modal_input(
        self,
        text: Optional[str] = None,
        audio_file: Optional[Union[str, Path]] = None,
        image_file: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Process multiple input types and combine their results."""
        results = []
        
        if text:
            text_result = await self.process_text_input(text)
            results.append(text_result)
            
        if audio_file:
            voice_result = await self.process_voice_input(audio_file)
            results.append(voice_result)
            
        if image_file:
            image_result = await self.process_image_input(image_file)
            results.append(image_result)
            
        return {
            "results": results,
            "status": "success" if all(r["status"] == "success" for r in results) else "partial_success"
        } 