from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

class AlternativeExplanationGenerator:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=openai_api_key)
        
    def generate_alternatives(self, symptoms: List[str], primary_diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate alternative explanations for the symptoms"""
        
        prompt = PromptTemplate(
            input_variables=["symptoms", "primary_diagnosis"],
            template="""
            Based on the following symptoms and primary diagnosis, generate alternative explanations:
            
            Symptoms: {symptoms}
            
            Primary Diagnosis:
            {primary_diagnosis}
            
            For each alternative explanation, provide:
            1. Alternative condition or cause
            2. How it explains the symptoms
            3. Supporting evidence
            4. Why it might be considered
            5. How to differentiate from primary diagnosis
            """
        )
        
        response = self.llm.predict(
            prompt.format(
                symptoms=symptoms,
                primary_diagnosis=primary_diagnosis
            )
        )
        
        return {
            "symptoms": symptoms,
            "primary_diagnosis": primary_diagnosis,
            "alternative_explanations": response
        }
    
    def evaluate_alternatives(self, alternatives: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the likelihood and implications of alternative explanations"""
        prompt = PromptTemplate(
            input_variables=["alternatives"],
            template="""
            Evaluate the following alternative explanations:
            
            {alternatives}
            
            For each alternative, provide:
            1. Likelihood assessment
            2. Clinical significance
            3. Impact on treatment approach
            4. Additional considerations
            """
        )
        
        evaluation = self.llm.predict(
            prompt.format(
                alternatives=alternatives["alternative_explanations"]
            )
        )
        
        return {
            "original_alternatives": alternatives,
            "evaluation": evaluation
        }
    
    def get_alternative_summary(self, evaluated_alternatives: Dict[str, Any]) -> str:
        """Generate a concise summary of the alternative explanations"""
        prompt = PromptTemplate(
            input_variables=["evaluated_alternatives"],
            template="""
            Provide a concise summary of the following evaluated alternative explanations:
            
            {evaluated_alternatives}
            
            Focus on the most significant alternatives and their implications.
            """
        )
        
        return self.llm.predict(
            prompt.format(
                evaluated_alternatives=evaluated_alternatives["evaluation"]
            )
        ) 