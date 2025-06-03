from typing import List, Dict, Any
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import json

class MedicalKnowledgeRetriever:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=openai_api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Initialize vector store with medical knowledge
        # This is a placeholder - you would need to load your medical knowledge base
        self.vector_store = None
        
    def setup_vector_store(self, medical_knowledge: List[Dict[str, Any]]):
        """Setup the vector store with medical knowledge"""
        texts = [json.dumps(knowledge) for knowledge in medical_knowledge]
        self.vector_store = FAISS.from_texts(texts, self.embeddings)
    
    def retrieve_relevant_knowledge(self, symptoms: List[str]) -> List[Dict[str, Any]]:
        """Retrieve relevant medical knowledge based on symptoms"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call setup_vector_store first.")
        
        # Combine symptoms into a single query
        query = " ".join(symptoms)
        
        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(query, k=3)
        
        # Parse the documents back into dictionaries
        knowledge = [json.loads(doc.page_content) for doc in docs]
        
        return knowledge
    
    def get_medical_context(self, symptoms: List[str]) -> Dict[str, Any]:
        """Get comprehensive medical context for the given symptoms"""
        knowledge = self.retrieve_relevant_knowledge(symptoms)
        
        prompt = PromptTemplate(
            input_variables=["symptoms", "knowledge"],
            template="""
            Based on the following symptoms and medical knowledge, provide a comprehensive medical context:
            
            Symptoms: {symptoms}
            
            Medical Knowledge:
            {knowledge}
            
            Provide a structured response with:
            1. Relevant medical conditions
            2. Key diagnostic criteria
            3. Important medical considerations
            4. Potential risk factors
            """
        )
        
        response = self.llm.predict(
            prompt.format(
                symptoms=symptoms,
                knowledge=json.dumps(knowledge, indent=2)
            )
        )
        
        return {
            "symptoms": symptoms,
            "medical_knowledge": knowledge,
            "context_analysis": response
        } 