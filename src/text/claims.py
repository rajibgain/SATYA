"""Claim extraction for the SATYA text forensics pipeline."""

from typing import List

def extract_claims(text: str) -> List[str]:
    """Extract falsifiable claims from the input text.
    
    In a full implementation, this would use a specialized NLP model or LLM 
    to isolate atomic claims from prose. For the skeleton, we treat the entire 
    input as a single claim, or split by simple sentences if needed.
    """
    text = text.strip()
    if not text:
        return []
    
    # Placeholder: Treat the entire text as a single claim for now.
    return [text]
