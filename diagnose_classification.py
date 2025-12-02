#!/usr/bin/env python3
"""
Diagnostic script to understand why everything is being classified as 2.4
"""

import logging
from config.codebook_def_def import FINAL_CODEBOOK_JER
from Scripts.classification import refine_candidates_with_api, classify_fragment_cosine, build_labeled_examples_from_codebook
from Scripts.vectorize import get_embedding

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_api_classification():
    """Test if API classification is working properly"""
    
    print("=== CLASSIFICATION DIAGNOSTIC ===")
    
    # Test different types of fragments
    test_fragments = [
        "Los estudiantes desarrollaron estrategias de reconciliación mediante círculos de diálogo",
        "El rector implementó cambios en el PEI para incluir el enfoque restaurativo",
        "Los profesores recibieron formación específica en justicia restaurativa",
        "Hay indicadores que muestran el impacto positivo del programa",
        "Se establecieron mecanismos de participación estudiantil en el gobierno escolar"
    ]
    
    labeled_examples = build_labeled_examples_from_codebook()
    print(f"Built {len(labeled_examples)} labeled examples")
    
    for i, fragment in enumerate(test_fragments, 1):
        print(f"\n--- TEST {i} ---")
        print(f"Fragment: {fragment}")
        
        # Get embedding
        embedding = get_embedding(fragment)
        if not embedding:
            print("❌ Failed to get embedding")
            continue
            
        # Test API call directly
        print("\n🔍 Testing API call directly:")
        try:
            all_categories_list = list(FINAL_CODEBOOK_JER.keys())
            api_result = refine_candidates_with_api(fragment, all_categories_list)
            print(f"API Response: {api_result}")
            
            if not api_result:
                print("⚠️ API call returned None/empty")
            elif "2.4" in str(api_result):
                print("🚨 API defaulting to 2.4!")
            else:
                print("✅ API seems to be working")
                
        except Exception as e:
            print(f"❌ API call failed: {e}")
            
        # Test full classification
        print("\n🔍 Testing full classification:")
        result = classify_fragment_cosine(fragment, embedding, labeled_examples, "test_doc", f"F{i:03d}")
        print(f"Full result: {result}")
        
        if result['category'] and len(result['category']) > 0:
            if "2.4" in result['category'][0]:
                print("🚨 Full classification also defaulting to 2.4!")
            else:
                print("✅ Full classification working properly")
        else:
            print("⚠️ No classification returned")

def analyze_similarity_bias():
    """Check if similarity vectors are biased toward 2.4"""
    
    print("\n=== SIMILARITY BIAS ANALYSIS ===")
    
    # Check what 2.4 looks like
    category_2_4 = "2.4 Prácticas de restauración y resolución de conflictos"
    if category_2_4 in FINAL_CODEBOOK_JER:
        details = FINAL_CODEBOOK_JER[category_2_4]
        print(f"2.4 Definition: {details.get('definition', '')[:200]}...")
        print(f"2.4 Keywords: {details.get('keywords', [])[:5]}")
        
        # Get its embedding
        definition = details.get("definition", "").strip()
        keywords = " ".join(details.get("keywords", []))
        rep = f"{definition} {definition} {keywords}".strip()
        embedding_2_4 = get_embedding(rep)
        
        if embedding_2_4:
            print("✅ 2.4 has valid embedding")
            
            # Test against a few other categories
            other_categories = ["0.1. Conocimiento de la JER", "11.1 Transformaciones al PEI", "1.1. Paz como derecho"]
            
            test_fragment = "Los estudiantes participan en círculos de diálogo"
            test_embedding = get_embedding(test_fragment)
            
            if test_embedding:
                from Scripts.classification import cosine_similarity
                
                print(f"\nTest fragment: {test_fragment}")
                print("Similarity scores:")
                
                for cat in other_categories:
                    if cat in FINAL_CODEBOOK_JER:
                        cat_details = FINAL_CODEBOOK_JER[cat]
                        cat_def = cat_details.get("definition", "").strip()
                        cat_keywords = " ".join(cat_details.get("keywords", []))
                        cat_rep = f"{cat_def} {cat_def} {cat_keywords}".strip()
                        cat_embedding = get_embedding(cat_rep)
                        
                        if cat_embedding:
                            similarity = cosine_similarity(test_embedding, cat_embedding)
                            print(f"  {cat[:30]}: {similarity:.3f}")
                
                # Compare with 2.4
                similarity_2_4 = cosine_similarity(test_embedding, embedding_2_4)
                print(f"  2.4 (current): {similarity_2_4:.3f}")
                
                if similarity_2_4 > 0.8:
                    print("🚨 2.4 has unusually high similarity!")

if __name__ == "__main__":
    test_api_classification()
    analyze_similarity_bias() 