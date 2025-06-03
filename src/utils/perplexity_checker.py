from typing import Dict, Any, Optional
import requests
from src.config.settings import Settings

class PerplexityChecker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.PERPLEXITY_API_KEY
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
    def check_diagnosis(self, diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """Check a diagnosis using Perplexity Sonar"""
        if not self.api_key:
            return {
                "checked": False,
                "error": "Perplexity API key not configured"
            }
        
        # Prepare the query for Perplexity
        query = self._prepare_query(diagnosis)
        
        try:
            response = self._call_perplexity_api(query)
            confidence_score = self._extract_confidence(response)
            
            return {
                "checked": True,
                "confidence_score": confidence_score,
                "is_reliable": confidence_score >= self.confidence_threshold,
                "raw_response": response,
                "diagnosis": diagnosis
            }
        except Exception as e:
            return {
                "checked": False,
                "error": str(e),
                "diagnosis": diagnosis
            }
    
    def _prepare_query(self, diagnosis: Dict[str, Any]) -> str:
        """Prepare the query for Perplexity API"""
        return f"""
        Verify the following medical diagnosis:
        
        Symptoms: {diagnosis.get('symptoms', [])}
        Diagnosis: {diagnosis.get('potential_diagnoses', '')}
        
        Please evaluate:
        1. Is this diagnosis medically accurate?
        2. Are there any potential hallucinations or inaccuracies?
        3. What is the confidence level in this diagnosis?
        """
    
    def _call_perplexity_api(self, query: str) -> Dict[str, Any]:
        """Call the Perplexity API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json={
                "model": "sonar-medium-online",
                "messages": [{"role": "user", "content": query}]
            }
        )
        
        response.raise_for_status()
        return response.json()
    
    def _extract_confidence(self, response: Dict[str, Any]) -> float:
        """Extract confidence score from Perplexity response"""
        # This is a placeholder - actual implementation would depend on Perplexity's response format
        try:
            # Assuming the response contains a confidence score
            return float(response.get("confidence", 0.0))
        except (ValueError, TypeError):
            return 0.0 