# Scripts/classification.py

import numpy as np
import logging
import openai
import json
import re
from difflib import get_close_matches, SequenceMatcher
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from config.codebook_def_def import FINAL_CODEBOOK_JER
from config.config import GPT_MODEL
from Scripts.vectorize import get_embedding

logger = logging.getLogger(__name__)

# Set up classification logging
classification_logger = logging.getLogger('classification_results')
classification_handler = logging.FileHandler('classification_results.log', encoding='utf-8')
classification_formatter = logging.Formatter('%(asctime)s - %(message)s')
classification_handler.setFormatter(classification_formatter)
classification_logger.addHandler(classification_handler)
classification_logger.setLevel(logging.INFO)

# RELAXED QUALITY THRESHOLDS - Allow shorter but meaningful content
SIMILARITY_THRESHOLD = 0.65  # More lenient than 0.70
API_CONFIDENCE_THRESHOLD = 0.80  # More lenient than 0.85  
QUALITY_SCORE_THRESHOLD = 0.65   # More lenient than 0.75
MAX_CATEGORIES = 3
TOP_N_CANDIDATES = 40
MIN_TEXT_LENGTH = 50  # Much more reasonable than 120
MIN_WORDS = 8  # Much more reasonable than 12

# Additional precision parameters
SEMANTIC_CONSISTENCY_THRESHOLD = 0.75
CATEGORY_OVERLAP_PENALTY = 0.15
CONFIDENCE_CALIBRATION_FACTOR = 0.95

def normalize_category_name(category_name: str) -> str:
    """
    Enhanced category name normalization for robust matching.
    """
    if not category_name:
        return ""
    
    # Convert to lowercase and strip
    normalized = category_name.lower().strip()
    
    # Normalize spaces (multiple spaces to single space)
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Remove extra punctuation but keep important dots and periods
    normalized = re.sub(r'[^\w\s\.\-]', '', normalized)
    
    # Normalize period patterns
    normalized = re.sub(r'\.{2,}', '.', normalized)  # Multiple dots to single
    normalized = re.sub(r'\s*\.\s*', '. ', normalized)  # Standardize period spacing
    
    # Remove trailing period if present
    if normalized.endswith('.'):
        normalized = normalized[:-1]
    
    return normalized.strip()

def find_best_category_match(api_category: str, available_categories: List[str]) -> Optional[str]:
    """
    Enhanced category matching with multiple fallback strategies.
    """
    if not api_category or not available_categories:
        return None
    
    # Normalize the API category
    normalized_api = normalize_category_name(api_category)
    
    # Create normalized lookup
    normalized_lookup = {normalize_category_name(cat): cat for cat in available_categories}
    
    # Strategy 1: Exact match after normalization
    if normalized_api in normalized_lookup:
        logger.debug(f"Exact match found: {api_category} -> {normalized_lookup[normalized_api]}")
        return normalized_lookup[normalized_api]
    
    # Strategy 2: Check for prefix matches (common for truncated categories)
    for norm_cat, orig_cat in normalized_lookup.items():
        if normalized_api in norm_cat or norm_cat.startswith(normalized_api):
            logger.debug(f"Prefix match found: {api_category} -> {orig_cat}")
            return orig_cat
    
    # Strategy 3: Check for partial matches with high similarity
    best_match = None
    best_score = 0.0
    
    for norm_cat, orig_cat in normalized_lookup.items():
        # Use sequence matcher for fuzzy matching
        similarity = SequenceMatcher(None, normalized_api, norm_cat).ratio()
        
        if similarity > best_score and similarity >= 0.85:  # High threshold for fuzzy matching
            best_match = orig_cat
            best_score = similarity
    
    if best_match:
        logger.debug(f"Fuzzy match found: {api_category} -> {best_match} (score: {best_score:.3f})")
        return best_match
    
    # Strategy 4: Check for key phrase matches
    api_words = set(normalized_api.split())
    for norm_cat, orig_cat in normalized_lookup.items():
        cat_words = set(norm_cat.split())
        
        # Calculate word overlap ratio
        if len(api_words) > 3:  # Only for reasonably long category names
            overlap = len(api_words & cat_words)
            overlap_ratio = overlap / min(len(api_words), len(cat_words))
            
            if overlap_ratio >= 0.8:  # High word overlap
                logger.debug(f"Word overlap match found: {api_category} -> {orig_cat} (overlap: {overlap_ratio:.3f})")
                return orig_cat
    
    # Strategy 5: Traditional fuzzy matching with lower cutoff as final fallback
    fuzzy_matches = get_close_matches(normalized_api, normalized_lookup.keys(), n=1, cutoff=0.75)
    if fuzzy_matches:
        matched_cat = normalized_lookup[fuzzy_matches[0]]
        logger.debug(f"Traditional fuzzy match found: {api_category} -> {matched_cat}")
        return matched_cat
    
    # Log the failure for analysis
    logger.warning(f"No match found for category: '{api_category}' (normalized: '{normalized_api}')")
    logger.debug(f"Available categories: {list(normalized_lookup.keys())[:5]}...")  # Show first 5 for debugging
    
    return None

def normalize_text(s: str) -> str:
    """
    Enhanced normalization with better handling of special characters and spaces.
    """
    # Remove extra whitespace and normalize
    normalized = " ".join(s.strip().lower().split())
    # Remove special characters but keep meaningful punctuation
    normalized = re.sub(r'[^\w\s\.\,\;\:\!\?]', ' ', normalized)
    return " ".join(normalized.split())

def is_meaningful_content(text: str) -> bool:
    """
    ULTRA-RELAXED qualitative content validation allowing most educational fragments.
    Only reject pure greetings/administrative content or very short fragments.
    """
    text = normalize_text(text)
    
    # Basic length check
    if len(text) < MIN_TEXT_LENGTH:
        logger.debug(f"REJECTED for length: {len(text)} < {MIN_TEXT_LENGTH}")
        return False
    
    words = text.split()
    if len(words) < MIN_WORDS:
        logger.debug(f"REJECTED for word count: {len(words)} < {MIN_WORDS}")
        return False
    
    # VERY MINIMAL REJECTION - Only pure greetings/admin
    pure_admin_patterns = [
        "mi nombre es", "me llamo", "soy el", "soy la",
        "buenos días", "buenas tardes", "mucho gusto"
    ]
    
    admin_count = sum(1 for pattern in pure_admin_patterns if pattern in text.lower())
    
    # Only reject if it's clearly ONLY administrative and very short
    if admin_count >= 2 and len(words) < 15:
        logger.debug(f"REJECTED as pure admin: admin_count={admin_count}, words={len(words)}")
        return False
        
    # ULTRA-PERMISSIVE: Accept almost everything else
    logger.debug(f"ACCEPTED: length={len(text)}, words={len(words)}, admin={admin_count}")
    return True

def cosine_similarity(vec1, vec2):
    """Enhanced cosine similarity with numerical stability."""
    v1 = np.array(vec1, dtype=np.float64)
    v2 = np.array(vec2, dtype=np.float64)
    
    # Check for zero vectors
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    v1_normalized = v1 / norm1
    v2_normalized = v2 / norm2
    
    similarity = np.dot(v1_normalized, v2_normalized)
    return float(np.clip(similarity, -1.0, 1.0))

def calculate_semantic_consistency(categories: List[str]) -> float:
    """
    Calculate semantic consistency between assigned categories to prevent conflicting assignments.
    """
    if len(categories) <= 1:
        return 1.0
    
    # Define category groups that should not appear together
    conflicting_groups = [
        # Process vs outcome categories
        ["0.1. Conocimiento de la JER", "9.3 Cambios significativos en la convivencia escolar a partir del enfoque restaurativo"],
        # Individual vs institutional focus
        ["3.1. Cambios en capacidades socioemocionales", "11.1 Transformaciones al PEI"],
        # Problem vs solution focus
        ["13.3 Retos y dificultades", "9.1 Percepción de la convivencia escolar tras implementar  las experiencias pedagógicas con enfoque restaurativo."]
    ]
    
    consistency_score = 1.0
    
    # Check for conflicting categories
    for group in conflicting_groups:
        overlap = len(set(categories) & set(group))
        if overlap > 1:
            consistency_score -= CATEGORY_OVERLAP_PENALTY * overlap
    
    # Check for hierarchical consistency (e.g., specific implementation should align with general knowledge)
    hierarchy_checks = [
        ("0.1. Conocimiento de la JER", ["0.2. Procesos de la JER en la IED", "2.1 Reconocimiento del enfoque restaurativo para la reconciliación"]),
        ("1.1. Paz como derecho", ["1.2. Defensa de la paz", "1.4. Paz y convivencia"])
    ]
    
    for parent, children in hierarchy_checks:
        if parent in categories:
            child_count = sum(1 for child in children if child in categories)
            if child_count > 1:  # Too many specific categories with general one
                consistency_score -= 0.1
    
    return max(0.0, consistency_score)

def build_labeled_examples_from_codebook():
    """
    Enhanced codebook example building with better representation.
    """
    labeled = []
    for category, details in FINAL_CODEBOOK_JER.items():
        # Build comprehensive representation
        definition = details.get("definition", "").strip()
        keywords = " ".join(details.get("keywords", []))
        synonyms = " ".join(details.get("synonyms", []))
        phrases = " ".join(details.get("phrases", []))
        
        # Weight definition more heavily
        rep = f"{definition} {definition} {keywords} {synonyms} {phrases}".strip()
        
        emb = get_embedding(rep)
        if emb is not None:
            labeled.append((rep, category, emb, True))
            
        # Add negative examples with proper weighting
        for neg in details.get("negative_examples", []):
            neg_emb = get_embedding(neg)
            if neg_emb is not None:
                labeled.append((neg, category, neg_emb, False))
    
    logger.info(f"Built {len(labeled)} labeled examples from codebook.")
    return labeled

def classify_by_similarity(fragment_embedding, labeled_examples):
    """
    Enhanced similarity classification with better scoring.
    """
    matches = []
    fragment_emb = np.array(fragment_embedding, dtype=np.float64)
    
    for text_rep, category, code_emb, is_positive in labeled_examples:
        code_emb = np.array(code_emb, dtype=np.float64)
        
        # Calculate multiple similarity metrics
        cos_score = cosine_similarity(fragment_emb, code_emb)
        
        # Enhanced Euclidean score with better normalization
        euclidean_dist = np.linalg.norm(fragment_emb - code_emb)
        euclidean_score = 1 / (1 + euclidean_dist / np.sqrt(len(fragment_emb)))
        
        # Weighted combination favoring cosine similarity
        combined = 0.85 * cos_score + 0.15 * euclidean_score
        
        # Handle negative examples
        if not is_positive:
            combined = max(0.0, 1 - combined)
        
        # Apply stricter threshold
        if combined >= SIMILARITY_THRESHOLD:
            matches.append((category, combined, text_rep))
    
    # Sort and filter top matches
    matches.sort(key=lambda x: x[1], reverse=True)
    
    # Remove very low scores but with slightly more lenient threshold
    matches = [(cat, score, text) for cat, score, text in matches if score >= 0.70]  # Lowered from 0.75 to 0.70
    
    return matches

def extract_complete_json(text: str) -> str:
    """
    Extract complete JSON from text, stopping exactly when the JSON structure ends.
    This handles cases where explanations follow the JSON.
    """
    # Find the starting position of JSON
    start_pos = -1
    for i, char in enumerate(text):
        if char in '[{':
            start_pos = i
            break
    
    if start_pos == -1:
        return ""
    
    # Track bracket/brace nesting to find the end of JSON
    bracket_stack = []
    in_string = False
    escape_next = False
    i = start_pos
    
    while i < len(text):
        char = text[i]
        
        if escape_next:
            escape_next = False
            i += 1
            continue
            
        if char == '\\' and in_string:
            escape_next = True
            i += 1
            continue
            
        if char == '"':
            in_string = not in_string
            i += 1
            continue
        
        if not in_string:
            if char in '[{':
                bracket_stack.append(char)
            elif char in ']}':
                if not bracket_stack:
                    # Mismatched bracket - invalid JSON
                    return ""
                
                opening = bracket_stack.pop()
                # Check for matching brackets
                if (char == ']' and opening != '[') or (char == '}' and opening != '{'):
                    # Mismatched bracket type - invalid JSON
                    return ""
                
                # If stack is empty, we've found the end of the JSON
                if not bracket_stack:
                    return text[start_pos:i+1]
        
        i += 1
    
    # If we reach here, JSON was not properly closed
    return ""

def extract_codes_fallback(text: str) -> List[Dict[str, any]]:
    """
    Fallback method to extract codes from malformed API responses using regex patterns.
    """
    results = []
    available_categories = list(FINAL_CODEBOOK_JER.keys())
    
    # Pattern to match code structures in various formats
    patterns = [
        r'"código"\s*:\s*"([^"]+)".*?"confianza"\s*:\s*([0-9.]+)',
        r'"código"\s*:\s*"([^"]+)".*?"confianza"\s*:\s*([0-9.]+)',
        r'código.*?([0-9]+\..*?[^\s,}]+).*?confianza.*?([0-9.]+)',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            try:
                code = match.group(1).strip()
                confidence = float(match.group(2))
                
                # Find best category match
                matched_category = find_best_category_match(code, available_categories)
                if matched_category and confidence >= 0.75:
                    results.append({"code": matched_category, "confidence": confidence})
                    
            except (ValueError, IndexError):
                continue
    
    # Remove duplicates and sort by confidence
    seen_codes = set()
    unique_results = []
    for result in sorted(results, key=lambda x: x["confidence"], reverse=True):
        if result["code"] not in seen_codes:
            seen_codes.add(result["code"])
            unique_results.append(result)
    
    return unique_results[:MAX_CATEGORIES]

def analyze_candidates_with_api(fragment, candidate_categories):
    """
    STAGE 2: ENHANCED EXPERT analysis with detailed explanations for each candidate.
    Special attention to code 2.4 without ban. Emphasizes balanced 0-3 range.
    """
    if not candidate_categories:
        return None
        
    # BUILD CANDIDATES BLOCK - ONLY filtered categories
    candidates_block = ""
    for category_code in candidate_categories:
        if category_code in FINAL_CODEBOOK_JER:
            category_details = FINAL_CODEBOOK_JER.get(category_code, {})
            definition = category_details.get("definition", "")
            candidates_block += f"• {category_code}\n  {definition}\n\n"

    # STAGE 2: ENHANCED EXPERT ANALYSIS WITH EXPLANATIONS
    system_msg = (
        "Eres un EXPERTO ACADÉMICO en JER con criterio equilibrado y metodológico.\n\n"
        
        "🎯 METODOLOGÍA DE ANÁLISIS:\n"
        "1. EVALÚA CADA CANDIDATO individualmente con justificación específica\n"
        "2. CORRESPONDENCIA CONCEPTUAL: ¿Desarrolla el fragmento aspectos del concepto?\n"
        "3. SUSTANCIA EDUCATIVA: ¿Aporta contenido específico del tema?\n"
        "4. ANÁLISIS EQUILIBRADO: 0, 1, 2 o 3 categorías son IGUALMENTE VÁLIDAS\n\n"
        
        "⚠️ ATENCIÓN ESPECIAL CÓDIGO 2.4:\n"
        "- Código: '2.4 Prácticas de restauración y resolución de conflictos'\n"
        "- ALTA FRECUENCIA de relación aparente pero no siempre genuina\n"
        "- EXIGE EVIDENCIA CLARA de prácticas restaurativas específicas\n"
        "- NO es suficiente mencionar conflictos o problemas generales\n"
        "- REQUIERE: procesos restaurativos, círculos, mediación, reparación del daño\n\n"
        
        "❌ RECHAZAR SI:\n"
        "- Solo información administrativa/biográfica\n"
        "- Menciones superficiales sin desarrollo conceptual\n"
        "- Para 2.4: conflictos generales sin prácticas restaurativas\n"
        "- Relaciones muy débiles o forzadas\n\n"
        
        "✅ ASIGNAR SI:\n"
        "- Desarrollo conceptual específico del tema\n"
        "- Correspondencia semántica clara con la definición\n"
        "- Para 2.4: evidencia de prácticas restaurativas concretas\n"
        "- Contenido educativo que elabora el tema apropiadamente\n\n"
        
        "🎯 RANGO VÁLIDO 0-3 CATEGORÍAS:\n"
        "- 0 categorías: Perfectamente válido si no hay correspondencia clara\n"
        "- 1-2 categorías: Lo más común para desarrollo conceptual específico\n"
        "- 3 categorías: Válido para fragmentos excepcionalmente ricos y multitemáticos\n"
        "- CALIDAD sobre cantidad - mejor pocas pero correctas\n\n"
        
        "📊 FORMATO RESPUESTA - OBLIGATORIO PARA CADA CANDIDATO:\n"
        "[\n"
        "  {\n"
        "    \"código\": \"CÓDIGO_EXACTO\",\n"
        "    \"confianza\": 0.XX,\n"
        "    \"justificación\": \"Razón específica por la cual SÍ corresponde conceptualmente\"\n"
        "  },\n"
        "  // Solo incluir códigos que SÍ corresponden\n"
        "  // Para código 2.4: exigir evidencia de prácticas restaurativas concretas\n"
        "]\n\n"
        "IMPORTANTE: Evalúa TODOS los candidatos pero incluye SOLO los que realmente corresponden."
    )

    user_msg = (
        f"🔍 FRAGMENTO A ANALIZAR:\n\"{fragment}\"\n\n"
        f"🎯 CANDIDATOS PRE-FILTRADOS ({len(candidate_categories)} códigos):\n{candidates_block}"
        
        "⚡ ANÁLISIS DETALLADO REQUERIDO:\n"
        f"Evalúa cada uno de los {len(candidate_categories)} candidatos individualmente. "
        "Para cada código, determina si el fragmento desarrolla genuinamente aspectos del concepto definido. "
        "Da ATENCIÓN ESPECIAL al código 2.4 si está presente - requiere evidencia de prácticas restaurativas concretas. "
        "Incluye SOLO los códigos que realmente corresponden (puede ser 0, 1, 2 o 3):"
    )

    try:
        resp = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.07,  # Slightly higher for nuanced analysis
            max_tokens=800,    # More space for detailed explanations of each candidate
            top_p=0.08,        # Focused but allowing some creativity in explanations
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Stage 2 enhanced analysis error: {e}")
        return None

def enhanced_parse_refined_categories(raw_json: str) -> List[Dict[str, any]]:
    """
    Enhanced JSON parsing for BALANCED EXPERT two-stage analysis results.
    10% less strict than before, with special attention to code 2.4.
    """
    if not raw_json or not raw_json.strip():
        return []
    
    try:
        # Enhanced cleaning with better JSON extraction
        s = raw_json.strip()
        
        # Extract JSON from code blocks if present
        if "```" in s:
            json_parts = s.split("```")
            for part in json_parts:
                cleaned_part = part.strip()
                if cleaned_part.startswith('[') or cleaned_part.startswith('{'):
                    s = cleaned_part
                    break
        
        # EXTRACT JSON for balanced expert analysis
        json_text = extract_complete_json(s)
        
        if not json_text:
            logger.warning(f"No valid JSON structure found in expert analysis response: {s[:100]}...")
            return []
        
        # Parse the extracted JSON
        parsed = json.loads(json_text)
        
        # Handle balanced expert format - direct array expected
        if isinstance(parsed, list):
            # Direct array format: [{"código": "...", "confianza": 0.XX, "justificación": "..."}]
            final_classifications = parsed
        elif isinstance(parsed, dict) and 'clasificacion_final' in parsed:
            # Fallback to old format if API still returns it
            final_classifications = parsed.get('clasificacion_final', [])
        else:
            logger.warning(f"Unexpected expert analysis format: {type(parsed)}")
            return []

        # Process final classifications - 10% LESS STRICT VALIDATION
        available_categories = list(FINAL_CODEBOOK_JER.keys())
        results = []
        
        for item in final_classifications:
            if isinstance(item, dict):
                code = item.get("código", "").strip()
                conf = float(item.get("confianza", 0.0))
                # Handle justification from expert analysis
                justification = item.get("justificación", item.get("análisis", item.get("evidencia", "")))
                
            elif isinstance(item, str):
                code = item.strip()
                conf = 0.8
                justification = ""
            else:
                continue
            
            # Ensure complete code preservation
            matched_category = find_best_category_match(code, available_categories)
            
            if not matched_category:
                partial_matches = [cat for cat in available_categories if code.lower() in cat.lower()]
                if len(partial_matches) == 1:
                    matched_category = partial_matches[0]
                    logger.debug(f"Using partial match: '{code}' -> '{matched_category}'")
                else:
                    logger.warning(f"Category not found after enhanced matching: '{code}'")
                    continue
            
            # 10% LESS STRICT QUALITY ASSESSMENT
            quality_score = conf * 0.40  # Increased from 0.35 to 0.40 (more lenient)
            
            # SPECIAL ATTENTION TO CODE 2.4 - Extra scrutiny without ban
            is_code_24 = "2.4" in matched_category and "restauración" in matched_category.lower()
            
            # ENHANCED JUSTIFICATION ASSESSMENT (10% more lenient)
            if justification and len(justification.strip()) > 8:  # Lowered from 10
                justification_lower = justification.lower()
                
                # Strong semantic indicators (highest quality)
                strong_semantic = [
                    "desarrolla específicamente", "elabora el concepto", "corresponde semánticamente",
                    "significado central", "desarrollo conceptual", "sustancia específica",
                    "concepto se manifiesta", "contenido conceptual específico"
                ]
                
                # Good semantic indicators (good quality)
                good_semantic = [
                    "desarrolla", "elabora", "corresponde", "significado", "reflexiona sobre",
                    "enfoque específico", "contenido educativo", "propósito claro"
                ]
                
                # Weak indicators - Penalized but not elimination
                weak_indicators = [
                    "menciona", "aparece", "contiene", "incluye", "palabra", "término", 
                    "similar", "relacionado", "hace referencia", "se refiere"
                ]
                
                strong_count = sum(1 for indicator in strong_semantic if indicator in justification_lower)
                good_count = sum(1 for indicator in good_semantic if indicator in justification_lower)
                weak_count = sum(1 for indicator in weak_indicators if indicator in justification_lower)
                
                if strong_count > 0:
                    # Strong semantic analysis - high bonus (10% more generous)
                    bonus = min(strong_count * 0.40, 0.55)  # Increased from 0.35->0.50
                    quality_score += bonus
                    logger.debug(f"Strong semantic analysis for '{matched_category}': +{bonus:.2f}")
                elif good_count > 0 and weak_count <= 1:
                    # Good semantic analysis with minimal weak indicators (10% more generous)
                    bonus = min(good_count * 0.30, 0.40)  # Increased from 0.25->0.35
                    quality_score += bonus
                    logger.debug(f"Good semantic analysis for '{matched_category}': +{bonus:.2f}")
                elif weak_count > good_count + strong_count:
                    # More weak than strong/good - penalty but less severe (10% more lenient)
                    quality_score *= 0.6  # Increased from 0.5 to 0.6
                    logger.debug(f"Too many weak indicators for '{matched_category}' - penalized")
                else:
                    # Neutral analysis - bigger bonus (10% more generous)
                    bonus = min(len(justification.strip()) / 150, 0.20)  # Increased from 200->150, 0.15->0.20
                    quality_score += bonus
            else:
                # NO JUSTIFICATION - Less severe penalty (10% more lenient)
                quality_score *= 0.7  # Increased from 0.6 to 0.7
                logger.debug(f"No justification for '{matched_category}' - penalized")
            
            # SPECIAL SCRUTINY FOR CODE 2.4 - Higher standards without ban
            if is_code_24:
                logger.debug(f"Special scrutiny for 2.4 code: '{matched_category}'")
                if strong_count == 0 and good_count == 0:
                    # For 2.4, require at least some semantic indicators
                    quality_score *= 0.7
                    logger.debug("2.4 code: No strong/good semantic indicators - additional penalty")
                if weak_count > 1:
                    # 2.4 with multiple weak indicators gets extra penalty
                    quality_score *= 0.8
                    logger.debug("2.4 code: Multiple weak indicators - extra penalty")
            
            # 10% LESS STRICT THRESHOLDS - More accessible
            if quality_score >= 0.63 and conf >= 0.76:  # Lowered from 0.70 & 0.85 (about 10% reduction)
                # STORE ESSENTIAL INFORMATION ONLY
                results.append({
                    "code": matched_category, 
                    "confidence": conf,
                    "quality_score": quality_score
                })
                logger.debug(f"ACCEPTED '{matched_category}' - quality_score: {quality_score:.3f}, confidence: {conf:.3f}")
            else:
                logger.debug(f"REJECTED '{matched_category}' - quality_score: {quality_score:.3f}, confidence: {conf:.3f} (thresholds: quality=0.63, conf=0.76)")

        # Sort by quality score
        results.sort(key=lambda x: x["quality_score"], reverse=True)
        
        # BALANCED CATEGORY LIMITS (10% more generous)
        if len(results) > 3:
            # Keep top 3 if many high-quality results
            results = results[:3]
            logger.info(f"Reduced to top 3 categories due to excess")
        
        if len(results) > 2:
            # For 3 categories, slightly lower quality requirement (10% more lenient)
            if results[2]["quality_score"] < 0.68:  # Lowered from 0.75
                results = results[:2]
                logger.info(f"Reduced to 2 categories - third category quality insufficient")
        
        if len(results) > 1:
            # For 2 categories, slightly lower quality requirement (10% more lenient)
            if results[1]["quality_score"] < 0.65:  # Lowered from 0.72
                results = results[:1]
                logger.info(f"Reduced to 1 category - second category quality insufficient")
        
        # Slightly more lenient consistency check (10% more lenient)
        if len(results) > 1:
            categories = [r["code"] for r in results]
            consistency = calculate_semantic_consistency(categories)
            if consistency < 0.72:  # Lowered from 0.80
                logger.info(f"Low consistency ({consistency:.3f}), keeping only top result")
                results = results[:1]
        
        # Return MINIMAL essential fields
        return [{"code": r["code"], "confidence": r["confidence"]} for r in results]
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in expert analysis: {e}, attempting fallback from: {raw_json[:200]}...")
        
        # Fallback: try to extract codes using regex patterns
        try:
            return extract_codes_fallback(raw_json)
        except Exception as fallback_error:
            logger.error(f"Fallback extraction also failed: {fallback_error}")
            return []
            
    except Exception as e:
        logger.error(f"Unexpected error in expert analysis parsing: {e}")
        return []

def classify_fragment_cosine(fragment, fragment_embedding, labeled_examples, 
                           document_name: str = "unknown", fragment_id: str = "unknown"):
    """
    Enhanced fragment classification with BALANCED expert two-stage analysis.
    10% less strict - finds the right balance between rigor and accessibility.
    """
    if fragment_embedding is None:
        log_classification_result(fragment, [], 0.0, document_name, fragment_id)
        return {"fragment": fragment, "category": [], "confidence": 0.0}
        
    # Enhanced content validation
    if not is_meaningful_content(fragment):
        logger.debug(f"Fragment rejected as non-meaningful: {fragment[:50]}...")
        log_classification_result(fragment, [], 0.0, document_name, fragment_id)
        return {"fragment": fragment, "category": [], "confidence": 0.0}

    # Two-stage balanced expert analysis
    logger.debug(f"Starting balanced expert two-stage analysis: {fragment[:50]}...")
    
    # Direct API analysis with all 55 categories
    all_categories_list = list(FINAL_CODEBOOK_JER.keys())
    raw = refine_candidates_with_api(fragment, all_categories_list)
    
    if raw:
        refined = enhanced_parse_refined_categories(raw)
        if refined:
            codes = [r["code"] for r in refined]
            confidences = [r["confidence"] for r in refined]
            
            # Apply slightly more lenient confidence calibration (10% more generous)
            calibrated_confidences = [c * 0.97 for c in confidences]  # Increased from 0.95
            max_confidence = max(calibrated_confidences) if calibrated_confidences else 0.0
            
            # 10% more lenient final validation
            if max_confidence >= 0.72:  # Lowered from 0.80
                log_classification_result(fragment, codes, max_confidence, document_name, fragment_id)
                return {
                    "fragment": fragment, 
                    "category": codes, 
                    "confidence": max_confidence
                }

    # If expert API analysis didn't work, use REASONABLE similarity-based fallback
    logger.debug("Expert API analysis found no valid assignments, using reasonable similarity fallback...")
    
    # Get similarity candidates with reasonable standards
    candidates = classify_by_similarity(fragment_embedding, labeled_examples)
    if not candidates:
        log_classification_result(fragment, [], 0.0, document_name, fragment_id)
        return {"fragment": fragment, "category": [], "confidence": 0.0}

    # 10% LESS STRICT FALLBACK
    reasonable_threshold = 0.84  # Lowered from 0.88 (about 10% reduction)
    expert_fallback = []
    
    for cat, score, _ in candidates[:5]:  # Top 5 candidates
        if score >= reasonable_threshold:
            expert_fallback.append({"code": cat, "confidence": score * 0.92})  # Increased from 0.90
        if len(expert_fallback) >= 2:  # Allow up to 2 categories from similarity fallback
            break
    
    if expert_fallback:
        codes = [r["code"] for r in expert_fallback]
        confidence = expert_fallback[0]["confidence"]
        log_classification_result(fragment, codes, confidence, document_name, fragment_id)
        return {"fragment": fragment, "category": codes, "confidence": confidence}

    log_classification_result(fragment, [], 0.0, document_name, fragment_id)
    return {"fragment": fragment, "category": [], "confidence": 0.0}

# Update the parse function reference
parse_refined_categories = enhanced_parse_refined_categories

def log_classification_result(fragment: str, categories: List[str], confidence: float, 
                            document_name: str = "unknown", fragment_id: str = "unknown"):
    """
    Log classification results - STREAMLINED for essential info only.
    """
    if categories:
        categories_str = ", ".join(categories)
        classification_logger.info(
            f"✅ {document_name}[{fragment_id}] → [{categories_str}] (conf: {confidence:.3f}) | {fragment[:60]}{'...' if len(fragment) > 60 else ''}"
        )
    else:
        classification_logger.info(
            f"❌ {document_name}[{fragment_id}] → [NO_CLASSIFICATION] | {fragment[:60]}{'...' if len(fragment) > 60 else ''}"
        )

def filter_candidates_with_api(fragment, all_categories_list):
    """
    STAGE 1: Rapid filtering to identify real candidate categories.
    Discards obviously irrelevant categories based on semantic quick-scan.
    """
    # BUILD DEFINITIONS BLOCK
    all_categories_block = ""
    for category_code in FINAL_CODEBOOK_JER.keys():
        category_details = FINAL_CODEBOOK_JER.get(category_code, {})
        definition = category_details.get("definition", "")
        all_categories_block += f"• {category_code}\n  {definition}\n\n"

    # STAGE 1: RAPID FILTERING SYSTEM
    system_msg = (
        "Eres un clasificador EXPERTO en JER. Tu tarea: FILTRAR RÁPIDAMENTE categorías "
        "que podrían tener correspondencia semántica con el fragmento.\n\n"
        
        "🎯 OBJETIVO: Descartar categorías obviamente irrelevantes.\n"
        "- NO hagas análisis profundo, solo identifica posibles correspondencias\n"
        "- Enfócate en el SIGNIFICADO CENTRAL del fragmento vs DEFINICIONES\n"
        "- Descarta presentaciones personales, datos administrativos, menciones superficiales\n\n"
        
        "✅ INCLUIR si hay POSIBLE correspondencia semántica\n"
        "❌ DESCARTAR si es obviamente irrelevante\n\n"
        
        "📊 RESPUESTA: Lista de códigos candidatos (5-8 máximo):\n"
        "[\"código1\", \"código2\", \"código3\"]\n\n"
        "Si NO hay candidatos viables: []"
    )

    user_msg = (
        f"🔍 FRAGMENTO: \"{fragment}\"\n\n"
        f"📋 TODAS LAS CATEGORÍAS:\n{all_categories_block}"
        
        "⚡ FILTRADO RÁPIDO: Identifica 5-8 categorías que PODRÍAN tener "
        "correspondencia semántica. Descarta obviamente irrelevantes:"
    )

    try:
        resp = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1,
            max_tokens=200,  # Very short response needed
            top_p=0.1,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Stage 1 filtering error: {e}")
        return None

def refine_candidates_with_api(fragment, all_categories_list):
    """
    TWO-STAGE EXPERT API CLASSIFICATION:
    Stage 1: Rapid filtering to identify real candidates
    Stage 2: Expert-level brutal analysis with PhD-level standards
    """
    logger.debug(f"Starting expert two-stage analysis for fragment: {fragment[:50]}...")
    
    # STAGE 1: RAPID FILTERING
    logger.debug("STAGE 1: Rapid filtering of all categories...")
    filtering_result = filter_candidates_with_api(fragment, all_categories_list)
    
    if not filtering_result:
        logger.debug("Stage 1 filtering failed")
        return None
    
    # Parse filtered candidates
    try:
        # Extract candidate list from response
        import json
        import re
        
        # Clean response and extract JSON array
        cleaned = filtering_result.strip()
        
        # Find JSON array pattern
        json_match = re.search(r'\[([^\]]*)\]', cleaned)
        if json_match:
            json_text = json_match.group(0)
            candidates = json.loads(json_text)
            logger.debug(f"Stage 1 identified {len(candidates)} candidates: {candidates}")
        else:
            logger.debug("No JSON array found in filtering response")
            return None
            
        if not candidates:
            logger.debug("Stage 1 found no viable candidates")
            return None
            
    except Exception as e:
        logger.error(f"Error parsing Stage 1 candidates: {e}")
        return None
    
    # STAGE 2: EXPERT-LEVEL BRUTAL ANALYSIS of candidates only
    logger.debug(f"STAGE 2: Expert-level brutal analysis of {len(candidates)} candidates...")
    analysis_result = analyze_candidates_with_api(fragment, candidates)
    
    if not analysis_result:
        logger.debug("Stage 2 expert analysis failed")
        return None
    
    logger.debug(f"Stage 2 expert analysis completed. Result: {analysis_result[:100]}...")
    return analysis_result

if __name__ == "__main__":
    labeled = build_labeled_examples_from_codebook()
    dummy = "Los estudiantes han mejorado significativamente sus habilidades de resolución de conflictos a través del programa de justicia restaurativa implementado en el colegio."
    dummy_emb = get_embedding(dummy)
    if dummy_emb:
        result = classify_fragment_cosine(dummy, dummy_emb, labeled)
        print(f"Classification result: {result}")
        log_classification_result(dummy, result["category"], result["confidence"])
    else:
        print("Could not get embedding for test fragment")
