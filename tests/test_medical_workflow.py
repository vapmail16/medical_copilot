import unittest
from unittest.mock import Mock, patch, MagicMock
from src.workflow.medical_workflow import MedicalWorkflow
from src.config.settings import Settings
from src.utils.safety_compliance import SafetyCompliance
import os

class TestMedicalWorkflow(unittest.TestCase):
    def setUp(self):
        # Set environment variable for OpenAI API key
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        
        # Create mock settings with a dummy API key
        self.settings = Settings(
            OPENAI_API_KEY="dummy_key",
            PERPLEXITY_API_KEY="dummy_key",
            NEO4J_URI="bolt://localhost:7687",
            NEO4J_USER="neo4j",
            NEO4J_PASSWORD="password",
            AUTONOMOUS_MODE=False,
            CONFIDENCE_THRESHOLD=0.8
        )

        # Mock the OpenAI client
        self.openai_patcher = patch('openai.OpenAI')
        self.mock_openai = self.openai_patcher.start()
        self.mock_openai.return_value = MagicMock()

        # Mock the ChatOpenAI class
        self.chat_openai_patcher = patch('langchain_openai.ChatOpenAI')
        self.mock_chat_openai = self.chat_openai_patcher.start()
        self.mock_chat_openai.return_value = MagicMock()

        # Mock the OpenAIEmbeddings class
        self.embeddings_patcher = patch('langchain_community.embeddings.OpenAIEmbeddings')
        self.mock_embeddings = self.embeddings_patcher.start()
        self.mock_embeddings.return_value = MagicMock()

        # Patch agent methods before workflow instantiation
        self.patcher_symptom = patch('src.agents.symptom_extractor.SymptomExtractor.extract_symptoms', 
            Mock(return_value=["fever", "cough", "fatigue"]))
        self.patcher_medical = patch('src.agents.medical_knowledge_retriever.MedicalKnowledgeRetriever.get_medical_context', 
            Mock(return_value={"common_conditions": ["flu", "cold"], "risk_factors": ["age", "immunity"], "context_analysis": "Patient shows symptoms of respiratory infection"}))
        self.patcher_risk = patch('src.agents.risk_evaluator.RiskEvaluator.evaluate_risk', 
            Mock(return_value={"risk_level": "low", "factors": ["mild symptoms", "no underlying conditions"]}))
        self.patcher_diag = patch('src.agents.diagnosis_generator.DiagnosisGenerator.generate_diagnoses', 
            Mock(return_value={"primary": "common cold", "confidence": 0.85, "differential": ["flu", "allergies"]}))
        self.patcher_alt = patch('src.agents.alternative_explanation_generator.AlternativeExplanationGenerator.generate_alternatives', 
            Mock(return_value={"possible_conditions": ["flu", "allergies"], "explanations": ["viral infection", "seasonal allergies"]}))
        self.patcher_judge_eval = patch('src.agents.llm_judge.LLMJudge.evaluate_diagnosis', 
            Mock(return_value={"confidence": 0.85, "explanation": "Symptoms consistent with common cold"}))
        self.patcher_judge_val = patch('src.agents.llm_judge.LLMJudge.validate_risk_assessment', 
            Mock(return_value=True))
        self.patcher_judge_final = patch('src.agents.llm_judge.LLMJudge.get_final_recommendation', 
            Mock(return_value="Rest and stay hydrated"))
        self.patcher_perplexity = patch('src.utils.perplexity_checker.PerplexityChecker.check_diagnosis', 
            Mock(return_value={"confidence_score": 0.85, "is_reliable": True}))
        self.patcher_neo4j_store = patch('src.utils.neo4j_manager.Neo4jManager.store_case', 
            Mock(return_value=True))
        self.patcher_neo4j_similar = patch('src.utils.neo4j_manager.Neo4jManager.find_similar_cases', 
            Mock(return_value=[{"symptoms": ["fever", "cough"], "diagnosis": "cold"}]))
        self.patcher_neo4j_comorb = patch('src.utils.neo4j_manager.Neo4jManager.find_comorbidities', 
            Mock(return_value=[{"condition": "asthma", "frequency": 0.3}]))
        self.patcher_neo4j_stats = patch('src.utils.neo4j_manager.Neo4jManager.get_case_statistics', 
            Mock(return_value={"total_cases": 100, "total_symptoms": 50}))
        self.patcher_safety = patch('src.utils.safety_compliance.SafetyCompliance.check_pii', 
            Mock(return_value=(False, [])))
        self.patcher_sensitive = patch('src.utils.safety_compliance.SafetyCompliance.check_sensitive_content', 
            Mock(return_value=(False, [])))

        # Start all patchers
        self.patcher_symptom.start()
        self.patcher_medical.start()
        self.patcher_risk.start()
        self.patcher_diag.start()
        self.patcher_alt.start()
        self.patcher_judge_eval.start()
        self.patcher_judge_val.start()
        self.patcher_judge_final.start()
        self.patcher_perplexity.start()
        self.patcher_neo4j_store.start()
        self.patcher_neo4j_similar.start()
        self.patcher_neo4j_comorb.start()
        self.patcher_neo4j_stats.start()
        self.patcher_safety.start()
        self.patcher_sensitive.start()

        # Create workflow instance
        self.workflow = MedicalWorkflow(self.settings)
    
    def tearDown(self):
        # Stop all patchers
        self.patcher_symptom.stop()
        self.patcher_medical.stop()
        self.patcher_risk.stop()
        self.patcher_diag.stop()
        self.patcher_alt.stop()
        self.patcher_judge_eval.stop()
        self.patcher_judge_val.stop()
        self.patcher_judge_final.stop()
        self.patcher_perplexity.stop()
        self.patcher_neo4j_store.stop()
        self.patcher_neo4j_similar.stop()
        self.patcher_neo4j_comorb.stop()
        self.patcher_neo4j_stats.stop()
        self.patcher_safety.stop()
        self.patcher_sensitive.stop()
        self.chat_openai_patcher.stop()
        self.openai_patcher.stop()
        self.embeddings_patcher.stop()
        self.workflow.close()
        
        # Clean up environment variable
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
    
    def test_basic_workflow_patient(self):
        """Test basic workflow with patient role"""
        result = self.workflow.process_patient_input(
            "I have a fever and cough",
            user_role="patient"
        )
        
        # Verify basic workflow steps
        self.assertIn("symptoms", result)
        self.assertIn("medical_context", result)
        self.assertIn("diagnosis", result)
        self.assertIn("final_recommendation", result)
        
        # Verify patient-specific restrictions
        self.assertFalse(result["sensitive_content_detected"])
        self.assertTrue(result["requires_validation"])
    
    def test_workflow_with_pii(self):
        """Test workflow with PII in input"""
        pii_matches = [
            {"start": 24, "end": 39, "text": "test@example.com"},
            {"start": 52, "end": 64, "text": "123-456-7890"}
        ]
        with patch('src.utils.safety_compliance.SafetyCompliance.check_pii', return_value=(True, pii_matches)):
            with patch('src.utils.safety_compliance.SafetyCompliance.redact_pii', return_value="I have a fever. My email is [REDACTED] and phone is [REDACTED]"):
                result = self.workflow.process_patient_input(
                    "I have a fever. My email is test@example.com and phone is 123-456-7890",
                    user_role="patient"
                )
                self.assertTrue(result["pii_detected"])
                self.assertNotIn("test@example.com", result["patient_input"])
                self.assertNotIn("123-456-7890", result["patient_input"])
    
    def test_workflow_with_sensitive_content(self):
        """Test workflow with sensitive medical content"""
        with patch('src.utils.safety_compliance.SafetyCompliance.check_sensitive_content', return_value=(True, [
            {"start": 28, "end": 31, "text": "HIV"},
            {"start": 52, "end": 61, "text": "depression"}
        ])):
            with patch('src.utils.safety_compliance.SafetyCompliance.sanitize_medical_context', return_value="Sanitized context"):
                result = self.workflow.process_patient_input(
                    "I have been diagnosed with HIV and experiencing depression",
                    user_role="patient"
                )
                self.assertTrue(result["sensitive_content_detected"])
                self.assertTrue(result["requires_validation"])
    
    def test_doctor_access(self):
        """Test workflow with doctor role"""
        with patch('src.utils.safety_compliance.SafetyCompliance.check_sensitive_content', return_value=(True, [
            {"start": 28, "end": 31, "text": "HIV"},
            {"start": 52, "end": 61, "text": "depression"}
        ])):
            with patch('src.utils.safety_compliance.SafetyCompliance.sanitize_medical_context', return_value="Sanitized context"):
                result = self.workflow.process_patient_input(
                    "Patient reports HIV positive status and depression",
                    user_role="doctor"
                )
                self.assertTrue(result["sensitive_content_detected"])
                # Doctor should not require validation for sensitive content
                self.assertFalse(result["requires_validation"])
                stats = self.workflow.get_case_statistics(user_role="doctor")
                self.assertIn("total_cases", stats)
    
    def test_validation_flow(self):
        """Test diagnosis validation flow"""
        # Process input requiring validation
        result = self.workflow.process_patient_input(
            "I have been diagnosed with cancer",
            user_role="patient"
        )
        
        self.assertTrue(result["requires_validation"])
        
        # Validate diagnosis
        is_valid = self.workflow.validate_diagnosis(
            diagnosis_id="test_id",
            is_valid=True,
            user_role="doctor"
        )
        self.assertTrue(is_valid)
    
    def test_similar_cases_retrieval(self):
        """Test similar cases retrieval"""
        cases = self.workflow.find_similar_cases(
            symptoms=["fever", "cough"],
            user_role="doctor"
        )
        
        self.assertIsInstance(cases, list)
        self.assertGreater(len(cases), 0)
    
    def test_comorbidity_detection(self):
        """Test comorbidity detection"""
        comorbidities = self.workflow.find_comorbidities(
            diagnosis="asthma",
            user_role="doctor"
        )
        
        self.assertIsInstance(comorbidities, list)
        self.assertGreater(len(comorbidities), 0)
    
    def test_unauthorized_access(self):
        """Test unauthorized access attempts"""
        # Try to access statistics as patient
        stats = self.workflow.get_case_statistics(user_role="patient")
        self.assertIn("error", stats)
        
        # Try to access sensitive cases as patient
        with patch('src.utils.neo4j_manager.Neo4jManager.find_similar_cases', return_value=[]):
            cases = self.workflow.find_similar_cases(
                symptoms=["HIV", "depression"],
                user_role="patient"
            )
            self.assertEqual(len(cases), 0)
    
    def test_autonomous_mode(self):
        """Test workflow in autonomous mode"""
        self.settings.AUTONOMOUS_MODE = True
        workflow = MedicalWorkflow(self.settings)
        result = workflow.process_patient_input(
            "I have a fever and cough",
            user_role="patient"
        )
        self.assertFalse(result["requires_validation"])

    def test_empty_input(self):
        """Test workflow with empty input"""
        with patch('src.agents.symptom_extractor.SymptomExtractor.extract_symptoms', return_value=[]):
            result = self.workflow.process_patient_input("", user_role="patient")
            self.assertIn("symptoms", result)
            self.assertEqual(len(result["symptoms"]), 0)
            self.assertIn("diagnosis", result)
            self.assertIn("final_recommendation", result)

    def test_invalid_user_role(self):
        """Test workflow with invalid user role"""
        result = self.workflow.process_patient_input(
            "I have a fever",
            user_role="invalid_role"
        )
        self.assertIn("symptoms", result)
        self.assertTrue(result["requires_validation"])

    def test_high_risk_scenario(self):
        """Test workflow with high-risk symptoms"""
        # Mock risk evaluator to return high risk
        self.patcher_risk.stop()
        self.patcher_risk = patch('src.agents.risk_evaluator.RiskEvaluator.evaluate_risk',
            Mock(return_value={"risk_level": "high", "factors": ["severe symptoms", "underlying conditions"]}))
        self.patcher_risk.start()

        result = self.workflow.process_patient_input(
            "I have severe chest pain and difficulty breathing",
            user_role="patient"
        )
        
        self.assertEqual(result["risk_assessment"]["risk_level"], "high")
        self.assertTrue(result["requires_validation"])

    def test_low_confidence_diagnosis(self):
        """Test workflow with low confidence diagnosis"""
        # Mock perplexity checker to return low confidence
        self.patcher_perplexity.stop()
        self.patcher_perplexity = patch('src.utils.perplexity_checker.PerplexityChecker.check_diagnosis',
            Mock(return_value={"confidence_score": 0.4, "is_reliable": False}))
        self.patcher_perplexity.start()

        result = self.workflow.process_patient_input(
            "I have some unusual symptoms that are hard to describe",
            user_role="patient"
        )
        
        self.assertLess(result["perplexity_check"]["confidence_score"], self.settings.CONFIDENCE_THRESHOLD)
        self.assertTrue(result["requires_validation"])

    def test_workflow_with_multiple_conditions(self):
        """Test workflow with multiple medical conditions"""
        result = self.workflow.process_patient_input(
            "I have diabetes and high blood pressure, and recently developed a persistent cough",
            user_role="patient"
        )
        
        self.assertIn("symptoms", result)
        self.assertIn("medical_context", result)
        self.assertIn("risk_assessment", result)
        self.assertTrue(result["requires_validation"])

    def test_workflow_with_medication_mention(self):
        """Test workflow with medication mentions"""
        result = self.workflow.process_patient_input(
            "I'm taking metformin for diabetes and have been experiencing side effects",
            user_role="patient"
        )
        
        self.assertIn("symptoms", result)
        self.assertIn("medical_context", result)
        self.assertTrue(result["requires_validation"])

    def test_workflow_with_family_history(self):
        """Test workflow with family medical history"""
        result = self.workflow.process_patient_input(
            "My father had heart disease and I'm experiencing similar symptoms",
            user_role="patient"
        )
        
        self.assertIn("symptoms", result)
        self.assertIn("medical_context", result)
        self.assertIn("risk_assessment", result)
        self.assertTrue(result["requires_validation"])

if __name__ == '__main__':
    unittest.main() 