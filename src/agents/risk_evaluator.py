from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

class RiskEvaluator:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=openai_api_key)
        
    def evaluate_risk(self, symptoms: List[str], medical_context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate medical risks based on symptoms and medical context"""
        
        prompt = PromptTemplate(
            input_variables=["symptoms", "medical_context"],
            template="""
            Based on the following symptoms and medical context, evaluate the potential risks:
            
            Symptoms: {symptoms}
            
            Medical Context:
            {medical_context}
            
            Provide a structured risk assessment with:
            1. Risk Level (Low/Medium/High)
            2. Immediate Concerns
            3. Potential Complications
            4. Recommended Actions
            5. Time Sensitivity
            """
        )
        
        response = self.llm.predict(
            prompt.format(
                symptoms=symptoms,
                medical_context=medical_context
            )
        )
        
        # Parse the response into a structured format
        risk_assessment = {
            "symptoms": symptoms,
            "medical_context": medical_context,
            "risk_analysis": response,
            "requires_immediate_attention": self._check_immediate_attention(response)
        }
        
        return risk_assessment
    
    def _check_immediate_attention(self, risk_analysis: str) -> bool:
        """Check if the risk analysis indicates need for immediate attention"""
        immediate_keywords = [
            "high risk", "emergency", "immediate", "urgent",
            "severe", "critical", "life-threatening"
        ]
        
        return any(keyword in risk_analysis.lower() for keyword in immediate_keywords)
    
    def get_risk_summary(self, risk_assessment: Dict[str, Any]) -> str:
        """Generate a concise summary of the risk assessment"""
        prompt = PromptTemplate(
            input_variables=["risk_assessment"],
            template="""
            Provide a concise summary of the following risk assessment:
            
            {risk_assessment}
            
            Focus on the most critical points and recommended actions.
            """
        )
        
        return self.llm.predict(
            prompt.format(
                risk_assessment=risk_assessment
            )
        ) 