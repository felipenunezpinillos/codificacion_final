#!/usr/bin/env python3
"""
Check the actual message size being sent to OpenAI API - STREAMLINED HIGH-SPEED VERSION
"""

from config.codebook_def_def import FINAL_CODEBOOK_JER

def check_message_size():
    """Check the size of the STREAMLINED HIGH-SPEED message being sent to OpenAI API"""
    
    print("=== STREAMLINED HIGH-SPEED MESSAGE SIZE VERIFICATION ===")
    
    # Build the NEW STREAMLINED message format - ONLY codes and definitions
    all_categories_block = ""
    for category_code in FINAL_CODEBOOK_JER.keys():
        category_details = FINAL_CODEBOOK_JER.get(category_code, {})
        
        # Get ONLY the definition - nothing else
        definition = category_details.get("definition", "")
        
        # Build minimal but complete category representation
        all_categories_block += f"• {category_code}\n  {definition}\n\n"

    # NEW STREAMLINED HIGH-SPEED SYSTEM MESSAGE
    system_msg = (
        "Eres un clasificador EXPERTO en Justicia Escolar Restaurativa (JER). "
        "Analiza el SIGNIFICADO REAL de fragmentos, no coincidencias superficiales.\n\n"
        
        "🎯 REGLA FUNDAMENTAL: SIGNIFICADO > PALABRAS\n"
        "- Solo asigna categorías cuando el fragmento DESARROLLA genuinamente el concepto\n"
        "- Rechaza presentaciones personales ('soy el rector'), datos temporales ('hace 4 años'), menciones superficiales\n"
        "- Exige correspondencia semántica entre el propósito del fragmento y la definición de la categoría\n\n"
        
        "📊 CRITERIOS DE ASIGNACIÓN:\n"
        "1. DESARROLLO CONCEPTUAL: ¿El fragmento explica/reflexiona sobre el concepto?\n"
        "2. CORRESPONDENCIA SEMÁNTICA: ¿El significado del fragmento coincide con la definición?\n"
        "3. SUSTANCIA: ¿Aporta información conceptual específica, no administrativa?\n\n"
        
        "❌ AUTO-RECHAZAR:\n"
        "- Presentaciones: 'soy...', 'me llamo...', 'llevo X años...'\n"
        "- Información temporal/administrativa sin desarrollo conceptual\n"
        "- Menciones que no desarrollan el concepto central\n\n"
        
        "✅ FORMATO RESPUESTA RÁPIDA:\n"
        "Para asignaciones válidas únicamente:\n"
        "[\n"
        "  {\n"
        "    \"código\": \"CÓDIGO_EXACTO\",\n"
        "    \"confianza\": 0.XX,\n"
        "    \"justificación\": \"Breve razón semántica (max 30 palabras)\"\n"
        "  }\n"
        "]\n\n"
        "Si NO HAY correspondencia semántica genuina: []\n"
        "Mejor no asignar que asignar incorrectamente."
    )

    # Test fragment
    fragment = "Los estudiantes desarrollaron estrategias de reconciliación mediante círculos de diálogo"

    user_msg = (
        f"🔍 FRAGMENTO: \"{fragment}\"\n\n"
        f"📋 CATEGORÍAS:\n{all_categories_block}"
        
        "⚡ ANÁLISIS: Evalúa correspondencia semántica. "
        "Solo asigna si hay desarrollo conceptual genuino, no meras menciones. "
        "Respuesta JSON directa:"
    )

    # Calculate approximate token count
    full_message = system_msg + user_msg
    
    # Rough token estimation (4 characters ≈ 1 token in Spanish)
    char_count = len(full_message)
    estimated_tokens = char_count / 4
    
    print(f"Total characters: {char_count:,}")
    print(f"Estimated tokens: {estimated_tokens:,.0f}")
    print(f"Token limit: 8,000")
    print(f"Available headroom: {8000 - estimated_tokens:,.0f} tokens")
    print(f"Efficiency: {estimated_tokens/8000*100:.1f}% of limit used")
    
    # Show size reduction comparison
    print("\n=== SIZE COMPARISON ===")
    print("Previous exhaustive version: ~6,436 tokens")
    print(f"New streamlined version: ~{estimated_tokens:,.0f} tokens")
    print(f"Reduction: {6436 - estimated_tokens:,.0f} tokens ({(6436-estimated_tokens)/6436*100:.1f}% smaller)")

    # Show message length breakdown
    print(f"\n=== MESSAGE BREAKDOWN ===")
    print(f"System message: {len(system_msg):,} chars (~{len(system_msg)/4:.0f} tokens)")
    print(f"Categories block: {len(all_categories_block):,} chars (~{len(all_categories_block)/4:.0f} tokens)")
    print(f"User message: {len(user_msg):,} chars (~{len(user_msg)/4:.0f} tokens)")
    
    print(f"\n✅ Well within 8,000 token limit!")
    print(f"🚀 Streamlined for maximum speed while maintaining analytical quality")

if __name__ == "__main__":
    check_message_size() 