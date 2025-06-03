from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    OPENAI_API_KEY: str
    NEO4J_URI: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str
    
    # Mode Configuration
    AUTONOMOUS_MODE: bool = False  # Default to controlled mode
    
    # Perplexity Configuration
    PERPLEXITY_API_KEY: Optional[str] = None
    CONFIDENCE_THRESHOLD: float = 0.8  # Minimum confidence threshold
    
    # Neo4j Configuration
    NEO4J_DATABASE: str = "medical_copilot"
    
    # Deepgram Configuration
    DEEPGRAM_API_KEY: Optional[str] = None  # Allow extra env var
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow" 