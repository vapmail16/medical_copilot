from typing import Dict, Any, List, Optional
from langgraph.graph import Graph, StateGraph
from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI

from src.agents.symptom_extractor import SymptomExtractor
from src.agents.medical_knowledge_retriever import MedicalKnowledgeRetriever
from src.agents.risk_evaluator import RiskEvaluator
from src.agents.diagnosis_generator import DiagnosisGenerator
from src.agents.alternative_explanation_generator import AlternativeExplanationGenerator
from src.agents.llm_judge import LLMJudge
from src.utils.perplexity_checker import PerplexityChecker
from src.utils.neo4j_manager import Neo4jManager
from src.utils.safety_compliance import SafetyCompliance
from src.config.settings import Settings

class MedicalWorkflow:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=settings.OPENAI_API_KEY)
        
        # Initialize all agents
        self.symptom_extractor = SymptomExtractor(settings.OPENAI_API_KEY)
        self.medical_knowledge_retriever = MedicalKnowledgeRetriever(settings.OPENAI_API_KEY)
        self.risk_evaluator = RiskEvaluator(settings.OPENAI_API_KEY)
        self.diagnosis_generator = DiagnosisGenerator(settings.OPENAI_API_KEY)
        self.alternative_generator = AlternativeExplanationGenerator(settings.OPENAI_API_KEY)
        self.llm_judge = LLMJudge(settings.OPENAI_API_KEY)
        
        # Initialize utilities
        self.perplexity_checker = PerplexityChecker(settings)
        self.neo4j_manager = Neo4jManager(settings)
        self.safety_compliance = SafetyCompliance(settings)
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> Graph:
        """Create the medical workflow graph using dict state"""
        workflow = StateGraph(dict)
        
        # Define the nodes
        def check_pii_and_sensitive_content(state: dict) -> dict:
            # Check for PII
            has_pii, pii_matches = self.safety_compliance.check_pii(state.get("patient_input", ""))
            if has_pii:
                state["patient_input"] = self.safety_compliance.redact_pii(state["patient_input"], pii_matches)
                state["pii_detected"] = True
            else:
                state["pii_detected"] = False
            # Check for sensitive content
            has_sensitive, _ = self.safety_compliance.check_sensitive_content(state.get("patient_input", ""))
            state["sensitive_content_detected"] = has_sensitive
            return state
        
        def extract_symptoms(state: dict) -> dict:
            state["symptoms"] = self.symptom_extractor.extract_symptoms(state["patient_input"])
            return state
        
        def retrieve_medical_context(state: dict) -> dict:
            state["medical_context"] = self.medical_knowledge_retriever.get_medical_context(state["symptoms"])
            # Sanitize context based on user role
            state["medical_context"] = self.safety_compliance.sanitize_medical_context(
                state["medical_context"],
                state.get("user_role", "patient")
            )
            return state
        
        def evaluate_risk(state: dict) -> dict:
            state["risk_assessment"] = self.risk_evaluator.evaluate_risk(
                state["symptoms"],
                state["medical_context"]
            )
            return state
        
        def generate_diagnosis(state: dict) -> dict:
            state["diagnosis"] = self.diagnosis_generator.generate_diagnoses(
                state["symptoms"],
                state["medical_context"]
            )
            return state
        
        def generate_alternatives(state: dict) -> dict:
            state["alternatives"] = self.alternative_generator.generate_alternatives(
                state["symptoms"],
                state["diagnosis"]
            )
            return state
        
        def evaluate_with_judge(state: dict) -> dict:
            state["evaluation"] = self.llm_judge.evaluate_diagnosis(
                state["symptoms"],
                state["medical_context"],
                state["diagnosis"],
                state["alternatives"]
            )
            return state
        
        def check_with_perplexity(state: dict) -> dict:
            state["perplexity_check"] = self.perplexity_checker.check_diagnosis(state["diagnosis"])
            return state
        
        def check_validation_required(state: dict) -> dict:
            # Doctors never require validation
            if state.get("user_role", "patient") == "doctor":
                state["requires_validation"] = False
                return state
            state["requires_validation"] = False
            if not self.settings.AUTONOMOUS_MODE:
                state["requires_validation"] = True
            elif state["perplexity_check"].get("confidence_score", 0) < self.settings.CONFIDENCE_THRESHOLD:
                state["requires_validation"] = True
            elif state.get("sensitive_content_detected", False):
                state["requires_validation"] = True
            return state
        
        def store_in_neo4j(state: dict) -> dict:
            if not state.get("requires_validation") or state.get("validation_status"):
                # Ensure sensitive data is properly handled
                case_data = {
                    "symptoms": state.get("symptoms", []),
                    "diagnosis": state.get("diagnosis", {}),
                    "confidence": state.get("perplexity_check", {}).get("confidence_score", 0),
                    "risk_level": state.get("risk_assessment", {}).get("risk_level", "unknown"),
                    "user_role": state.get("user_role", "patient"),
                    "sensitive_content": state.get("sensitive_content_detected", False)
                }
                self.neo4j_manager.store_case(case_data)
            return state
        
        def generate_final_recommendation(state: dict) -> dict:
            risk_validation = self.llm_judge.validate_risk_assessment(state["risk_assessment"])
            state["final_recommendation"] = self.llm_judge.get_final_recommendation(
                state["evaluation"],
                risk_validation
            )
            return state
        
        # Add nodes to the graph
        workflow.add_node("check_pii_and_sensitive_content", check_pii_and_sensitive_content)
        workflow.add_node("extract_symptoms", extract_symptoms)
        workflow.add_node("retrieve_medical_context", retrieve_medical_context)
        workflow.add_node("evaluate_risk", evaluate_risk)
        workflow.add_node("generate_diagnosis", generate_diagnosis)
        workflow.add_node("generate_alternatives", generate_alternatives)
        workflow.add_node("evaluate_with_judge", evaluate_with_judge)
        workflow.add_node("check_with_perplexity", check_with_perplexity)
        workflow.add_node("check_validation_required", check_validation_required)
        workflow.add_node("store_in_neo4j", store_in_neo4j)
        workflow.add_node("generate_final_recommendation", generate_final_recommendation)
        
        # Define the edges
        workflow.add_edge("check_pii_and_sensitive_content", "extract_symptoms")
        workflow.add_edge("extract_symptoms", "retrieve_medical_context")
        workflow.add_edge("retrieve_medical_context", "evaluate_risk")
        workflow.add_edge("evaluate_risk", "generate_diagnosis")
        workflow.add_edge("generate_diagnosis", "generate_alternatives")
        workflow.add_edge("generate_alternatives", "evaluate_with_judge")
        workflow.add_edge("evaluate_with_judge", "check_with_perplexity")
        workflow.add_edge("check_with_perplexity", "check_validation_required")
        workflow.add_edge("check_validation_required", "store_in_neo4j")
        workflow.add_edge("store_in_neo4j", "generate_final_recommendation")
        
        # Set the entry point
        workflow.set_entry_point("check_pii_and_sensitive_content")
        
        return workflow.compile()
    
    def process_patient_input(self, patient_input: str, user_role: str = "patient") -> Dict[str, Any]:
        """Process patient input through the complete workflow"""
        # Initialize the state as a dict
        initial_state = {
            "patient_input": patient_input,
            "user_role": user_role,
            "symptoms": [],
            "medical_context": {},
            "risk_assessment": {},
            "diagnosis": {},
            "alternatives": {},
            "evaluation": {},
            "final_recommendation": "",
            "perplexity_check": {},
            "requires_validation": False,
            "validation_status": None,
            "pii_detected": False,
            "sensitive_content_detected": False
        }
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        # Log access attempt
        self.safety_compliance.log_access_attempt(
            user_role,
            "medical_workflow",
            True
        )
        return final_state
    
    def validate_diagnosis(self, diagnosis_id: str, is_valid: bool, user_role: str) -> bool:
        """Validate a diagnosis (for controlled mode)"""
        if not self.safety_compliance.validate_user_access(user_role, {"diagnosis_id": diagnosis_id}):
            return False
        return is_valid
    
    def find_similar_cases(self, symptoms: List[str], user_role: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar cases from the database"""
        if not self.safety_compliance.validate_user_access(user_role, {"symptoms": symptoms}):
            return []
        return self.neo4j_manager.find_similar_cases(symptoms, limit)
    
    def find_comorbidities(self, diagnosis: str, user_role: str) -> List[Dict[str, Any]]:
        """Find common comorbidities for a diagnosis"""
        if not self.safety_compliance.validate_user_access(user_role, {"diagnosis": diagnosis}):
            return []
        return self.neo4j_manager.find_comorbidities(diagnosis)
    
    def get_case_statistics(self, user_role: str) -> Dict[str, Any]:
        """Get statistics about stored cases"""
        if user_role != "doctor":
            return {"error": "Unauthorized access"}
        return self.neo4j_manager.get_case_statistics()
    
    def close(self):
        """Clean up resources"""
        self.neo4j_manager.close() 