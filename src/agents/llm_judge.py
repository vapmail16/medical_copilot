from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

class LLMJudge:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=openai_api_key)
        
    def evaluate_diagnosis(self, 
                          symptoms: List[str],
                          medical_context: Dict[str, Any],
                          diagnosis: Dict[str, Any],
                          alternatives: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the quality and validity of the diagnosis and alternatives"""
        
        prompt = PromptTemplate(
            input_variables=["symptoms", "medical_context", "diagnosis", "alternatives"],
            template="""
            Evaluate the following medical assessment:
            
            Symptoms: {symptoms}
            
            Medical Context:
            {medical_context}
            
            Primary Diagnosis:
            {diagnosis}
            
            Alternative Explanations:
            {alternatives}
            
            Provide a comprehensive evaluation with:
            1. Diagnosis Quality Assessment
               - Completeness
               - Evidence-based reasoning
               - Clinical relevance
            2. Alternative Explanations Review
               - Coverage of possibilities
               - Differential diagnosis quality
            3. Overall Assessment
               - Confidence in conclusions
               - Areas of uncertainty
               - Recommendations for improvement
            """
        )
        
        evaluation = self.llm.predict(
            prompt.format(
                symptoms=symptoms,
                medical_context=medical_context,
                diagnosis=diagnosis,
                alternatives=alternatives
            )
        )
        
        return {
            "symptoms": symptoms,
            "medical_context": medical_context,
            "diagnosis": diagnosis,
            "alternatives": alternatives,
            "evaluation": evaluation
        }
    
    def validate_risk_assessment(self, risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the risk assessment and its implications"""
        prompt = PromptTemplate(
            input_variables=["risk_assessment"],
            template="""
            Validate the following risk assessment:
            
            {risk_assessment}
            
            Provide a validation with:
            1. Risk Level Appropriateness
            2. Action Recommendations Review
            3. Time Sensitivity Assessment
            4. Potential Oversights
            5. Additional Considerations
            """
        )
        
        validation = self.llm.predict(
            prompt.format(
                risk_assessment=risk_assessment
            )
        )
        
        return {
            "original_assessment": risk_assessment,
            "validation": validation
        }
    
    def get_final_recommendation(self, 
                               evaluation: Dict[str, Any],
                               risk_validation: Dict[str, Any]) -> str:
        """Generate final recommendations based on all evaluations"""
        prompt = PromptTemplate(
            input_variables=["evaluation", "risk_validation"],
            template="""
            Based on the following evaluations, provide final recommendations:
            
            Diagnosis Evaluation:
            {evaluation}
            
            Risk Assessment Validation:
            {risk_validation}
            
            Provide a structured final recommendation with:
            1. Primary Action Items
            2. Follow-up Requirements
            3. Monitoring Recommendations
            4. Contingency Plans
            5. Patient Communication Points
            """
        )
        
        return self.llm.predict(
            prompt.format(
                evaluation=evaluation["evaluation"],
                risk_validation=risk_validation["validation"]
            )
        ) 