pip install spacy tqdm --quiet
python -m spacy download en_core_web_trf --quiet

import spacy
import pandas as pd
import re
import time as time
from typing import Dict, Optional
from tqdm import tqdm  # for progress tracking

# define anonymization class and function

class PIIMasker:
    def __init__(self, model_name: str = 'en_core_web_trf'):

        """Initialize the PII masker with specified spaCy model."""

        self.nlp = spacy.load(model_name)
        self.nlp.max_length = 2000000  # Increase max text length
        self.max_cell_length = 32000

        # Compile regex patterns once
        self.hardcoded_replacements: Dict[str, str] = {
            r"\b(Chief\s+Executive\s+Officer|CEO)\b": "LEADERSHIP_ROLE",
            r"\b(Chief\s+Financial\s+Officer|CFO)\b": "LEADERSHIP_ROLE",
            r"\b(Chief\s+Marketing\s+Officer|CMO)\b": "LEADERSHIP_ROLE",
            r"\b(Chief\s+Technology\s+Officer|CTO)\b": "LEADERSHIP_ROLE",
            r"\b(Vice\s+President|VP)(?:\s+of\s+[A-Za-z\s]+)?\b": "LEADERSHIP_ROLE",  # flexible VP of X
            r"\b(He|She|Him|Her)\b": "THEY",
            r"\b(TCCC)\b": "COMPANY" # hardcode acronym in
        }
        self.compiled_patterns = {
            re.compile(pattern, re.IGNORECASE): replacement
            for pattern, replacement in self.hardcoded_replacements.items()
        }

    def normalize_text(self, text: Optional[str]) -> str:

        """Normalize text by replacing special characters."""

        if not isinstance(text, str):
            return ""

        try:
            replacements = [
                ("\\t\\r\\n\\r\\n\\r\\n", ". "),
                ("\\t\\r\\n", ". "),
                ("\\r\\n", " "),
                ("\\r", " "),
                ("\\t", " "),
                ("\\n", ". "),
                ("â€™", "'")
            ]

            normalized = text
            for old, new in replacements:
                normalized = normalized.replace(old, new)

            normalized = normalized.strip()

            # Excel length safety
            if len(normalized) > self.max_cell_length:
                normalized = normalized[:self.max_cell_length] + " [TRUNCATED]"

            return normalized

        except Exception as e:
            print(f"Error normalizing text: {str(e)}")
            return text  # Return original if something goes wrong

    def anonymize_text(self, text: Optional[str]) -> str:

        """Anonymize text by masking PII."""

        if not isinstance(text, str) or not text.strip():
            return ""

        try:
            # Apply hardcoded replacements
            anonymized = text
            for pattern, replacement in self.compiled_patterns.items():
                anonymized = pattern.sub(replacement, anonymized)

            # Process with spaCy for NER
            doc = self.nlp(anonymized)

            # Create entity mapping
            entity_map = {}
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
                    entity_map[ent.text] = {
                        "PERSON": "NAME",
                        "ORG": "COMPANY",
                        "GPE": "PLACE",
                        "LOC": "LOCATION"
                    }[ent.label_]

            # Apply entity replacements
            if entity_map:
                pattern = r'\b(' + '|'.join(re.escape(k) for k in entity_map.keys()) + r')\b'
                anonymized = re.sub(pattern, lambda m: entity_map[m.group()], anonymized)

            return anonymized

        except Exception as e:
            print(f"Error processing text: {str(e)[:200]}")
            return f"ERROR_PROCESSING_TEXT: {str(e)[:50]}"  # Return error message instead of original text

    def process_dataframe(self, df: pd.DataFrame,
                          text_columns: Optional[list] = None) -> pd.DataFrame:

        """
        Process entire DataFrame.

           Args:
               df: Input DataFrame
               text_columns: Optional list of column names to process
        """

        df_copy = df.copy()

        # If no columns specified, process all object columns
        if text_columns is None:
            text_columns = df_copy.select_dtypes(include=['object']).columns

        # Process each column
        for col in text_columns:
            if col not in df_copy.columns:
                print(f"Warning: Column '{col}' not found in DataFrame")
                continue

            print(f"Processing column: {col}")
            tqdm.pandas(desc=f"Anonymizing {col}")

            # First normalize
            df_copy[col] = df_copy[col].progress_apply(self.normalize_text)

            # Then anonymize
            df_copy[col] = df_copy[col].progress_apply(self.anonymize_text)

        return df_copy
