import os
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv

from core.symptom_intake import SymptomIntake
from agents.symptom_extractor import SymptomExtractor
from utils.neo4j_client import Neo4jClient

class MedicalCopilot:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize components
        self.symptom_intake = SymptomIntake()
        self.symptom_extractor = SymptomExtractor()
        self.neo4j_client = Neo4jClient()
        
        # Connect to Neo4j
        if not self.neo4j_client.connect():
            print("Warning: Failed to connect to Neo4j. Some features may be limited.")

    async def process_input(
        self,
        text: str = None,
        audio_file: str = None,
        image_file: str = None
    ) -> Dict[str, Any]:
        """Process multi-modal input and generate diagnosis."""
        try:
            # 1. Process multi-modal input
            input_results = await self.symptom_intake.process_multi_modal_input(
                text=text,
                audio_file=audio_file,
                image_file=image_file
            )
            
            if input_results["status"] == "error":
                return {
                    "status": "error",
                    "message": "Failed to process input",
                    "details": input_results
                }
            
            # 2. Extract symptoms
            symptom_results = await self.symptom_extractor.extract_symptoms(input_results)
            
            if symptom_results["status"] == "error":
                return {
                    "status": "error",
                    "message": "Failed to extract symptoms",
                    "details": symptom_results
                }
            
            # --- FIX: Ensure symptoms is always a list ---
            symptoms = symptom_results["symptoms"]
            if isinstance(symptoms, str):
                symptoms = [symptoms]
            elif not isinstance(symptoms, list):
                symptoms = list(symptoms)
            # -------------------------------------------

            # 3. Find similar cases in Neo4j
            similar_cases = self.neo4j_client.find_similar_cases(
                symptoms=symptoms,
                limit=3
            )
            
            # 4. Prepare final response
            return {
                "status": "success",
                "input_analysis": input_results,
                "symptoms": symptom_results,
                "similar_cases": similar_cases
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
        
    def close(self):
        """Cleanup resources."""
        self.neo4j_client.close()

async def main():
    # Example usage
    copilot = MedicalCopilot()
    
    try:
        # Example with text input
        result = await copilot.process_input(
            text="I have a severe headache and fever for the last 2 days"
        )
        print("Result:", result)
        
    finally:
        copilot.close()

if __name__ == "__main__":
    asyncio.run(main()) 