from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

class DiagnosisGenerator:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=openai_api_key)
        
    def generate_diagnoses(self, symptoms: List[str], medical_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate potential diagnoses based on symptoms and medical context"""
        
        prompt = PromptTemplate(
            input_variables=["symptoms", "medical_context"],
            template="""
            Based on the following symptoms and medical context, generate potential diagnoses:
            
            Symptoms: {symptoms}
            
            Medical Context:
            {medical_context}
            
            For each potential diagnosis, provide:
            1. Condition name
            2. Confidence level (High/Medium/Low)
            3. Key supporting symptoms
            4. Differential diagnosis considerations
            5. Recommended diagnostic tests
            6. Treatment options
            """
        )
        
        response = self.llm.predict(
            prompt.format(
                symptoms=symptoms,
                medical_context=medical_context
            )
        )
        
        return {
            "symptoms": symptoms,
            "medical_context": medical_context,
            "potential_diagnoses": response
        }
    
    def rank_diagnoses(self, diagnoses: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank the potential diagnoses by confidence and severity"""
        prompt = PromptTemplate(
            input_variables=["diagnoses"],
            template="""
            Rank the following potential diagnoses by confidence and severity:
            
            {diagnoses}
            
            Provide a ranked list with:
            1. Primary diagnosis (highest confidence)
            2. Secondary diagnoses
            3. Differential diagnoses to consider
            """
        )
        
        ranked_diagnoses = self.llm.predict(
            prompt.format(
                diagnoses=diagnoses["potential_diagnoses"]
            )
        )
        
        return {
            "original_diagnoses": diagnoses,
            "ranked_diagnoses": ranked_diagnoses
        }
    
    def get_diagnosis_summary(self, ranked_diagnoses: Dict[str, Any]) -> str:
        """Generate a concise summary of the diagnoses"""
        prompt = PromptTemplate(
            input_variables=["ranked_diagnoses"],
            template="""
            Provide a concise summary of the following ranked diagnoses:
            
            {ranked_diagnoses}
            
            Focus on the most likely diagnosis and key considerations.
            """
        )
        
        return self.llm.predict(
            prompt.format(
                ranked_diagnoses=ranked_diagnoses["ranked_diagnoses"]
            )
        ) 