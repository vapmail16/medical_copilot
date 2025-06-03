from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase
from src.config.settings import Settings

class Neo4jManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        
    def store_case(self, case_data: Dict[str, Any]) -> bool:
        """Store a medical case in Neo4j"""
        with self.driver.session(database=self.settings.NEO4J_DATABASE) as session:
            try:
                # Create case node
                result = session.write_transaction(
                    self._create_case,
                    case_data
                )
                return True
            except Exception as e:
                print(f"Error storing case: {str(e)}")
                return False
    
    def _create_case(self, tx, case_data: Dict[str, Any]):
        """Create a case node and its relationships"""
        query = """
        CREATE (c:Case {
            id: apoc.create.uuid(),
            timestamp: datetime(),
            symptoms: $symptoms,
            diagnosis: $diagnosis,
            confidence: $confidence,
            risk_level: $risk_level
        })
        WITH c
        UNWIND $symptoms as symptom
        MERGE (s:Symptom {name: symptom})
        CREATE (c)-[:HAS_SYMPTOM]->(s)
        WITH c
        UNWIND $diagnosis as diag
        MERGE (d:Diagnosis {name: diag.name})
        CREATE (c)-[:HAS_DIAGNOSIS {confidence: diag.confidence}]->(d)
        RETURN c
        """
        
        return tx.run(query, {
            "symptoms": case_data["symptoms"],
            "diagnosis": case_data["diagnosis"],
            "confidence": case_data.get("confidence", 0.0),
            "risk_level": case_data.get("risk_level", "unknown")
        })
    
    def find_similar_cases(self, symptoms: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar cases based on symptoms"""
        with self.driver.session(database=self.settings.NEO4J_DATABASE) as session:
            return session.read_transaction(
                self._find_similar_cases,
                symptoms,
                limit
            )
    
    def _find_similar_cases(self, tx, symptoms: List[str], limit: int):
        """Find similar cases using symptom overlap"""
        query = """
        MATCH (c:Case)-[:HAS_SYMPTOM]->(s:Symptom)
        WHERE s.name IN $symptoms
        WITH c, count(s) as matching_symptoms
        MATCH (c)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
        RETURN c, collect(d) as diagnoses, matching_symptoms
        ORDER BY matching_symptoms DESC
        LIMIT $limit
        """
        
        result = tx.run(query, {"symptoms": symptoms, "limit": limit})
        return [dict(record) for record in result]
    
    def find_comorbidities(self, diagnosis: str) -> List[Dict[str, Any]]:
        """Find common comorbidities for a diagnosis"""
        with self.driver.session(database=self.settings.NEO4J_DATABASE) as session:
            return session.read_transaction(
                self._find_comorbidities,
                diagnosis
            )
    
    def _find_comorbidities(self, tx, diagnosis: str):
        """Find comorbidities using case co-occurrence"""
        query = """
        MATCH (c:Case)-[:HAS_DIAGNOSIS]->(d1:Diagnosis {name: $diagnosis})
        MATCH (c)-[:HAS_DIAGNOSIS]->(d2:Diagnosis)
        WHERE d1 <> d2
        WITH d2, count(*) as co_occurrence
        RETURN d2.name as diagnosis, co_occurrence
        ORDER BY co_occurrence DESC
        """
        
        result = tx.run(query, {"diagnosis": diagnosis})
        return [dict(record) for record in result]
    
    def get_case_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored cases"""
        with self.driver.session(database=self.settings.NEO4J_DATABASE) as session:
            return session.read_transaction(self._get_statistics)
    
    def _get_statistics(self, tx):
        """Get various statistics about the case database"""
        query = """
        MATCH (c:Case)
        WITH count(c) as total_cases
        MATCH (s:Symptom)
        WITH total_cases, count(s) as total_symptoms
        MATCH (d:Diagnosis)
        WITH total_cases, total_symptoms, count(d) as total_diagnoses
        MATCH (c:Case)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
        WITH total_cases, total_symptoms, total_diagnoses,
             count(DISTINCT c) as cases_with_diagnosis
        RETURN {
            total_cases: total_cases,
            total_symptoms: total_symptoms,
            total_diagnoses: total_diagnoses,
            cases_with_diagnosis: cases_with_diagnosis
        } as stats
        """
        
        result = tx.run(query)
        return dict(result.single()["stats"])
    
    def close(self):
        """Close the Neo4j driver connection"""
        self.driver.close() 