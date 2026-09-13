"""Evidence checking for the SATYA text forensics pipeline."""

from typing import Dict, Any

def check_evidence(claim: str) -> Dict[str, Any]:
    """Check a claim against an evidence provider.
    
    Currently returns INSUFFICIENT EVIDENCE as no live search provider is implemented.
    """
    return {
        "claim": claim,
        "evidence_found": False,
        "verdict": "INSUFFICIENT EVIDENCE",
        "confidence": 0.0,
        "sources": []
    }
