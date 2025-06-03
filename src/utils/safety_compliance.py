from typing import Dict, Any, List, Optional, Tuple
import re
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from src.config.settings import Settings

class SafetyCompliance:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=settings.OPENAI_API_KEY)
        
        # Initialize PII patterns
        self.pii_patterns = {
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            'date_of_birth': r'\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-]\d{4}\b'
        }
        
        # Initialize sensitive conditions list
        self.sensitive_conditions = [
            "HIV", "AIDS", "mental health", "suicide", "abuse",
            "substance abuse", "STD", "STI", "pregnancy", "abortion",
            "cancer", "terminal", "palliative", "hospice"
        ]
    
    def check_pii(self, text: str) -> Tuple[bool, List[str]]:
        """Check for PII in the text"""
        found_pii = []
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                found_pii.append({
                    "type": pii_type,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        return len(found_pii) > 0, found_pii
    
    def redact_pii(self, text: str, pii_matches: List[Dict[str, Any]]) -> str:
        """Redact PII from text"""
        redacted_text = text
        for match in sorted(pii_matches, key=lambda x: x["start"], reverse=True):
            redacted_text = (
                redacted_text[:match["start"]] +
                f"[REDACTED {match['type'].upper()}]" +
                redacted_text[match["end"]:]
            )
        return redacted_text
    
    def check_sensitive_content(self, text: str) -> Tuple[bool, List[str]]:
        """Check for sensitive medical conditions"""
        found_conditions = []
        
        for condition in self.sensitive_conditions:
            if re.search(rf'\b{condition}\b', text, re.IGNORECASE):
                found_conditions.append(condition)
        
        return len(found_conditions) > 0, found_conditions
    
    def validate_user_access(self, user_role: str, content: Dict[str, Any]) -> bool:
        """Validate if user has access to the content based on their role"""
        if user_role == "doctor":
            return True
        
        # For non-doctors, check if content contains sensitive information
        has_sensitive, _ = self.check_sensitive_content(str(content))
        return not has_sensitive
    
    def sanitize_medical_context(self, context: Dict[str, Any], user_role: str) -> Dict[str, Any]:
        """Sanitize medical context based on user role"""
        if user_role == "doctor":
            return context
        
        # For non-doctors, filter out sensitive information
        sanitized_context = context.copy()
        
        # Convert context to string for checking
        context_str = str(context)
        has_sensitive, sensitive_conditions = self.check_sensitive_content(context_str)
        
        if has_sensitive:
            # Use LLM to generate a sanitized version
            prompt = PromptTemplate(
                input_variables=["context", "sensitive_conditions"],
                template="""
                Sanitize the following medical context by removing or generalizing sensitive information.
                Sensitive conditions to handle: {sensitive_conditions}
                
                Original context:
                {context}
                
                Provide a sanitized version that:
                1. Maintains medical accuracy
                2. Removes specific sensitive conditions
                3. Uses appropriate generalizations
                4. Preserves non-sensitive information
                """
            )
            
            sanitized_text = self.llm.predict(
                prompt.format(
                    context=context_str,
                    sensitive_conditions=sensitive_conditions
                )
            )
            
            # Update the context with sanitized version
            sanitized_context["context_analysis"] = sanitized_text
        
        return sanitized_context
    
    def validate_diagnosis_access(self, diagnosis: Dict[str, Any], user_role: str) -> bool:
        """Validate if user has access to the diagnosis"""
        if user_role == "doctor":
            return True
        
        # For non-doctors, check if diagnosis contains sensitive information
        diagnosis_str = str(diagnosis)
        has_sensitive, _ = self.check_sensitive_content(diagnosis_str)
        return not has_sensitive
    
    def get_access_level(self, user_role: str) -> str:
        """Get the access level for a user role"""
        access_levels = {
            "doctor": "full",
            "nurse": "partial",
            "patient": "limited",
            "researcher": "anonymized"
        }
        return access_levels.get(user_role.lower(), "limited")
    
    def log_access_attempt(self, user_role: str, content_type: str, success: bool) -> None:
        """Log access attempts for audit purposes"""
        # This would typically write to a secure log
        print(f"Access attempt - Role: {user_role}, Content: {content_type}, Success: {success}") 