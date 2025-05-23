from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from typing import Optional

from ..main import MedicalCopilot

app = FastAPI(title="Medical Copilot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the copilot
copilot = MedicalCopilot()

@app.post("/analyze")
async def analyze_symptoms(
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Analyze symptoms from text, audio, or image input.
    """
    try:
        # Handle file uploads
        audio_path = None
        image_path = None
        
        if audio:
            # Save audio file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                content = await audio.read()
                temp_audio.write(content)
                audio_path = temp_audio.name
        
        if image:
            # Save image file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
                content = await image.read()
                temp_image.write(content)
                image_path = temp_image.name
        
        # Process the input
        result = await copilot.process_input(
            text=text,
            audio_file=audio_path,
            image_file=image_path
        )
        
        # Cleanup temporary files
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
        if image_path and os.path.exists(image_path):
            os.unlink(image_path)
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    copilot.close() 