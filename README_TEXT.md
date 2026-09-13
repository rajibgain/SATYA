# SATYA Text Forensics Module

This module analyzes text to verify falsifiable claims. It is designed to evaluate raw text for forensic authenticity rather than simple classification.

## Architecture

1. **Preprocessing (`preprocess.py`)**: Loads the file, validates UTF-8 encoding, enforces length constraints, and normalizes unicode and whitespace.
2. **Claim Extraction (`claims.py`)**: Parses the normalized text to extract distinct falsifiable claims.
3. **Evidence Checking (`evidence.py`)**: Cross-references claims against a search provider or knowledge base.
4. **Inference (`inference.py`)**: Orchestrates the pipeline and aggregates evidence to produce a final verdict.

## Notes

- Currently, this module acts as a structural skeleton. It intentionally defaults to `INSUFFICIENT EVIDENCE` because an active search provider is not yet integrated.
- It will NOT return `TRUE` or `FALSE` unless the evidence layer explicitly supports such a conclusion.
