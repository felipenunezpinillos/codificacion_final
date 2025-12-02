# main.py

import os
import json
import logging
import time
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np
from typing import Dict, List, Optional, Tuple

from Scripts.classification import build_labeled_examples_from_codebook, classify_fragment_cosine, normalize_text
from Scripts.vectorize import load_cache, get_embedding, save_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path("C:/Users/Felipe Nunez/Documents/Machine Learning Work/JER/codificacion_final")
CLEANED_DIR = BASE_DIR / "assets/output/interviews/coding/cleaned"
SEGMENTED_DIR = BASE_DIR / "assets/output/interviews/coding/segmented"
OUTPUT_DIR = BASE_DIR / "assets/output/interviews/coding/classified"
ANALYSIS_DIR = BASE_DIR / "assets/output/interviews/coding/analysis"
ALL_INTERVIEWS_PATH = BASE_DIR / "assets/output/interviews/coding/all_interviews.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Critical parameters for perfect classification
MAX_RETRIES = 5  # Increased for robustness
RETRY_DELAY = 10  # Increased delay for better API stability
API_TIMEOUT = 60  # Explicit timeout
MIN_CONFIDENCE_FOR_CLASSIFICATION = 0.80  # Strict confidence threshold
MAX_API_ATTEMPTS_PER_BATCH = 3  # Batch processing for API efficiency

# Quality control thresholds
MIN_FRAGMENT_QUALITY_SCORE = 0.7
MAX_CATEGORIES_PER_FRAGMENT = 3
SEMANTIC_COHERENCE_THRESHOLD = 0.65

def load_fragment_questions_mapping() -> Dict[str, str]:
    """
    Load the mapping of fragments to questions from all_interviews.json.
    Returns a dictionary with fragment text as key and question as value.
    """
    logger.info("Loading fragment-to-question mapping...")
    fragment_to_question = {}
    
    try:
        with open(ALL_INTERVIEWS_PATH, 'r', encoding='utf-8') as f:
            all_interviews = json.load(f)
        
        for entry in all_interviews:
            fragment = entry.get("fragment", "").strip()
            question = entry.get("question", "").strip()
            
            if fragment and question:
                # Normalize fragment text for matching
                normalized_fragment = normalize_text(fragment)
                fragment_to_question[normalized_fragment] = question
        
        logger.info(f"Loaded {len(fragment_to_question)} fragment-question mappings")
        return fragment_to_question
        
    except FileNotFoundError:
        logger.warning(f"Could not find {ALL_INTERVIEWS_PATH}, continuing without question mapping")
        return {}
    except Exception as e:
        logger.error(f"Error loading fragment-question mapping: {e}")
        return {}

def find_question_for_fragment(fragment: str, fragment_to_question: Dict[str, str]) -> str:
    """
    Find the corresponding question for a given fragment.
    Uses exact matching first, then fuzzy matching if needed.
    """
    if not fragment_to_question:
        return "Question not available"
    
    # Normalize the fragment for matching
    normalized_fragment = normalize_text(fragment)
    
    # Try exact match first
    if normalized_fragment in fragment_to_question:
        return fragment_to_question[normalized_fragment]
    
    # Try fuzzy matching for slight variations
    from difflib import get_close_matches
    close_matches = get_close_matches(normalized_fragment, fragment_to_question.keys(), n=1, cutoff=0.95)
    
    if close_matches:
        return fragment_to_question[close_matches[0]]
    
    # Try partial matching for truncated fragments
    for stored_fragment, question in fragment_to_question.items():
        if len(normalized_fragment) > 50 and normalized_fragment in stored_fragment:
            return question
        elif len(stored_fragment) > 50 and stored_fragment in normalized_fragment:
            return question
    
    return "Question not found"

def validate_fragment_quality(fragment: str) -> float:
    """
    Advanced fragment quality validation to prevent noise classification.
    Returns quality score between 0.0 and 1.0.
    """
    if not fragment or len(fragment.strip()) < 50:
        return 0.0
    
    # Check for meaningful content
    words = fragment.split()
    if len(words) < 10:
        return 0.0
    
    # Check for educational/institutional context keywords
    contextual_keywords = [
        "estudiante", "profesor", "colegio", "escuela", "institución", "educativo",
        "convivencia", "conflicto", "paz", "restaurativo", "diálogo", "comunidad",
        "aprendizaje", "formación", "pedagógico", "académico"
    ]
    
    context_score = sum(1 for word in words if any(kw in word.lower() for kw in contextual_keywords))
    context_ratio = context_score / len(words)
    
    # Penalize fragments that are mostly names, dates, or meaningless content
    meaningless_patterns = ["nombre persona", "fecha", "hora", "día", "sí", "no", "eh", "um"]
    meaningless_score = sum(1 for pattern in meaningless_patterns if pattern in fragment.lower())
    meaningless_ratio = meaningless_score / len(words)
    
    # Calculate quality score
    base_score = min(len(fragment) / 200, 1.0)  # Length component
    quality_score = base_score * (1 + context_ratio) * (1 - meaningless_ratio)
    
    return max(0.0, min(1.0, quality_score))

def validate_classification_coherence(fragment: str, categories: List[str], 
                                    confidence_scores: List[float]) -> bool:
    """
    Validates semantic coherence between fragment and assigned categories.
    """
    if not categories or not confidence_scores:
        return False
    
    # Check confidence distribution
    if max(confidence_scores) < MIN_CONFIDENCE_FOR_CLASSIFICATION:
        return False
    
    # Check for category overlap/redundancy
    if len(set(categories)) != len(categories):
        return False
    
    # Additional semantic validation could be added here
    return True

def classify_files():
    """
    Enhanced classification with interpretation focus, comprehensive logging, and question mapping.
    """
    logger.info("Starting enhanced fragment classification with interpretation focus and question mapping...")
    
    # Load fragment-to-question mapping
    fragment_to_question = load_fragment_questions_mapping()
    
    # Load cache and build labeled examples
    cache = load_cache()
    labeled_examples = build_labeled_examples_from_codebook()

    if not labeled_examples:
        logger.error("No labeled examples could be built from codebook.")
        return
    
    logger.info(f"Built {len(labeled_examples)} labeled examples from codebook.")
    
    # Process all segmented files
    segmented_files = list(SEGMENTED_DIR.glob("*.json"))
    logger.info(f"Found {len(segmented_files)} segmented files to process.")
    
    for file_path in segmented_files:
        logger.info(f"Processing file: {file_path.name}")
        
        try:
            # Load fragments
            with open(file_path, 'r', encoding='utf-8') as f:
                fragments = json.load(f)
            
            if not fragments:
                logger.warning(f"No fragments found in {file_path.name}")
                continue

            logger.info(f"Processing {len(fragments)} fragments from {file_path.name}")
            
            classified_fragments = []
            document_name = file_path.stem  # Extract document name from filename
            
            for i, fragment_text in enumerate(fragments, 1):
                try:
                    # Get embedding
                    fragment_embedding = cache.get(fragment_text)
                    if fragment_embedding is None:
                        fragment_embedding = get_embedding(fragment_text)
                        if fragment_embedding is not None:
                            cache[fragment_text] = fragment_embedding
                    
                    if fragment_embedding is None:
                        logger.warning(f"Could not get embedding for fragment {i} in {file_path.name}")
                        continue
                    
                    # Find corresponding question
                    question = find_question_for_fragment(fragment_text, fragment_to_question)
                    
                    # Classify with document tracking and interpretation focus
                    fragment_id = f"F{i:03d}"
                    result = classify_fragment_cosine(
                        fragment_text, 
                        fragment_embedding, 
                        labeled_examples,
                        document_name=document_name,
                        fragment_id=fragment_id
                    )
                    
                    if result["category"]:  # Only keep classified fragments
                        # Enhanced result with question included
                        enhanced_result = {
                            "fragment": result["fragment"],
                            "question": question,
                            "category": result["category"], 
                            "confidence": result["confidence"]
                        }
                        classified_fragments.append(enhanced_result)
                        
                        logger.info(f"Fragment {fragment_id} classified with {len(result['category'])} categories, confidence: {result['confidence']:.3f}")
                    else:
                        logger.debug(f"Fragment {fragment_id} rejected (no classification)")
                
                except Exception as e:
                    logger.error(f"Error processing fragment {i} in {file_path.name}: {e}")
                    continue

            # Save results
            output_file = OUTPUT_DIR / f"{file_path.stem}.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(classified_fragments, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(classified_fragments)} classified fragments to {output_file.name}")
            
            # Log summary for this document
            if classified_fragments:
                categories_count = {}
                total_confidence = 0
                total_categories = 0
                question_mapped_count = sum(1 for frag in classified_fragments if frag["question"] != "Question not found")
                
                for frag in classified_fragments:
                    for cat in frag["category"]:
                        categories_count[cat] = categories_count.get(cat, 0) + 1
                    total_confidence += frag["confidence"]
                    total_categories += len(frag["category"])
                
                avg_confidence = total_confidence / len(classified_fragments)
                avg_categories_per_fragment = total_categories / len(classified_fragments)
                question_mapping_rate = (question_mapped_count / len(classified_fragments)) * 100
                
                logger.info(f"""
                === DOCUMENT SUMMARY: {document_name} ===
                Total fragments processed: {len(fragments)}
                Total fragments classified: {len(classified_fragments)} ({len(classified_fragments)/len(fragments)*100:.1f}%)
                Questions successfully mapped: {question_mapped_count}/{len(classified_fragments)} ({question_mapping_rate:.1f}%)
                Average confidence: {avg_confidence:.3f}
                Average categories per fragment: {avg_categories_per_fragment:.1f}
                Most frequent categories:
                {chr(10).join([f"  - {cat}: {count} fragments" for cat, count in sorted(categories_count.items(), key=lambda x: x[1], reverse=True)[:5]])}
                =======================================
                """)
            else:
                logger.warning(f"No fragments were classified in {document_name}")
        
        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}")
            continue
    
    # Save updated cache
    save_cache(cache)
    logger.info("Enhanced classification with interpretation focus and question mapping completed!")
    logger.info("Check 'classification_results.log' for detailed fragment-by-fragment logging.")
    logger.info("Final results now include: fragment, question, category, and confidence.")

if __name__ == "__main__":
    classify_files()
