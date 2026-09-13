import unicodedata
import re

class TextPreprocessor:
    def __init__(self, max_length=10000, min_length=10):
        self.max_length = max_length
        self.min_length = min_length

    def validate_and_load(self, filepath):
        """Loads and validates text from a file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        except UnicodeDecodeError:
            raise ValueError(f"File {filepath} is not valid UTF-8.")
        except Exception as e:
            raise ValueError(f"Error reading file {filepath}: {e}")

        # Basic validation
        if not raw_text:
            raise ValueError(f"File {filepath} is empty.")
        
        if not raw_text.strip():
            raise ValueError(f"File {filepath} contains only whitespace.")
            
        if len(raw_text.strip()) < self.min_length:
            raise ValueError(f"File {filepath} is too short (min {self.min_length} chars).")
            
        if len(raw_text) > self.max_length:
            raise ValueError(f"File {filepath} exceeds max length ({self.max_length} chars). Truncation is disabled for safety.")
            
        return raw_text

    def preprocess(self, text):
        """
        Deterministic preprocessing:
        - Unicode normalization (NFC)
        - Whitespace normalization (convert tabs/newlines to single space conditionally, 
          but for forensic analysis we might want to keep paragraphs. We will standardize 
          excessive whitespace while keeping paragraph breaks).
        """
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Normalize whitespace but preserve paragraphs
        # Replace 3+ newlines with 2 newlines (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Replace multiple spaces/tabs with single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        text = text.strip()
        return text

    def tokenize(self, text):
        """
        Basic deterministic word tokenization preserving punctuation.
        This is a rudimentary tokenizer for feature extraction and basic embedding.
        """
        # Split on whitespace and punctuation boundaries
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text, re.UNICODE)
        return tokens

    def segment_sentences(self, text):
        """
        Basic sentence segmentation using regex.
        """
        # Split on . ! ? followed by space and uppercase letter, or end of string
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
