from typing import Dict, Any, List
from neo4j import GraphDatabase
import os

class Neo4jClient:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.driver = None

    def connect(self):
        """Connect to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            return True
        except Exception as e:
            print(f"Failed to connect to Neo4j: {str(e)}")
            return False

    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()

    def find_similar_cases(self, symptoms: List[str], limit: int = 3) -> List[Dict[str, Any]]:
        """Find similar cases based on symptoms."""
        query = """
        MATCH (s:Symptom)-[:PART_OF]->(d:Diagnosis)
        WHERE s.name IN $symptoms
        WITH d, count(s) as symptom_count
        ORDER BY symptom_count DESC
        LIMIT $limit
        RETURN d.name as diagnosis, d.confidence as confidence, symptom_count
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, symptoms=symptoms, limit=limit)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Error finding similar cases: {str(e)}")
            return []

    def store_case(self, symptoms: List[str], diagnosis: str, confidence: float):
        """Store a new case in the graph."""
        query = """
        MERGE (d:Diagnosis {name: $diagnosis, confidence: $confidence})
        WITH d
        UNWIND $symptoms as symptom
        MERGE (s:Symptom {name: symptom})
        MERGE (s)-[:PART_OF]->(d)
        """
        
        try:
            with self.driver.session() as session:
                session.run(query, symptoms=symptoms, diagnosis=diagnosis, confidence=confidence)
                return True
        except Exception as e:
            print(f"Error storing case: {str(e)}")
            return False

    def get_symptom_relationships(self, symptom: str) -> List[Dict[str, Any]]:
        """Get relationships between a symptom and other symptoms/diagnoses."""
        query = """
        MATCH (s:Symptom {name: $symptom})-[r]-(related)
        RETURN type(r) as relationship_type, 
               labels(related)[0] as related_type,
               related.name as related_name
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, symptom=symptom)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Error getting symptom relationships: {str(e)}")
            return [] 