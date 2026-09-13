"""Inference entry point for the SATYA text forensics pipeline."""

from typing import Dict, Any

from .preprocess import TextPreprocessor
from .claims import extract_claims
from .evidence import check_evidence

def infer_text(text_path: str) -> Dict[str, Any]:
    """Run text analysis by extracting claims and checking evidence."""
    preprocessor = TextPreprocessor()
    
    # Load and validate
    raw_text = preprocessor.validate_and_load(text_path)
    clean_text = preprocessor.preprocess(raw_text)
    
    # Extract claims
    claims = extract_claims(clean_text)
    
    # Evidence check
    evidence_results = []
    for claim in claims:
        evidence_results.append(check_evidence(claim))
        
    # Aggregate verdict
    # Default to INSUFFICIENT EVIDENCE since we have no real provider
    verdict = "INSUFFICIENT EVIDENCE"
    
    result = {
        "verdict": verdict,
        "claims_extracted": len(claims),
        "evidence_results": evidence_results,
        "model_name": "Claim Verification Pipeline",
        "warning": "This module is a skeleton for claim verification. It does not contain an active search provider."
    }
    
    return result
