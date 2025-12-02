# main_interview.py

import os
import json
import logging

from config.config                   import INPUT_DIR, OUTPUT_DIR
from Scripts.loader                  import load_fragments_with_question
from Scripts.cleaning                import clean_text
from Scripts.segmentation            import segment_text
from Scripts.vectorize               import load_cache, get_embedding, save_cache
from Scripts.classification          import classify_fragment_cosine, build_labeled_examples_from_codebook

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_input_directories():
    """Get all input directories to process."""
    base_dir = INPUT_DIR
    subdirs = ['directivos', 'docentes', 'estudiantes', 'familia', 'sed']
    return [os.path.join(base_dir, subdir) for subdir in subdirs]

# Cargar cache de embeddings y ejemplos etiquetados
cache  = load_cache()
codes  = build_labeled_examples_from_codebook()
all_results = []

# Preparar carpetas de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)
CLEANED_DIR   = os.path.join(OUTPUT_DIR, 'cleaned')
SEGMENTED_DIR = os.path.join(OUTPUT_DIR, 'segmented')
PARTIAL_DIR   = os.path.join(OUTPUT_DIR, 'partial')

os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(SEGMENTED_DIR, exist_ok=True)
os.makedirs(PARTIAL_DIR, exist_ok=True)

# Process each input directory
for input_dir in get_input_directories():
    logger.info("Processing directory: %s", input_dir)
    
    for fn in os.listdir(input_dir):
        if not fn.lower().endswith('.txt'):
            continue
        path = os.path.join(input_dir, fn)
        basename = os.path.splitext(fn)[0]
        logger.info("Processing %s", fn)

        try:
            # 1) Cargar pares pregunta–respuesta
            qa_pairs = load_fragments_with_question(path)
            file_results = []
            
            # Store all cleaned responses and segmented fragments for this file
            all_cleaned_responses = []
            all_segmented_fragments = []

            for idx, qa in enumerate(qa_pairs):
                q    = qa['question']
                resp = qa['response']

                try:
                    # ---- 2a) CLEANING ----
                    cleaned = clean_text(resp)
                    all_cleaned_responses.append(cleaned)

                    # ---- 2b) SEGMENTATION ----
                    fragments = segment_text(cleaned)
                    all_segmented_fragments.extend(fragments)

                    # ---- 3) Para cada fragmento, obtener embedding y clasificar ----
                    for frag in fragments:
                        frag = frag.strip()
                        if not frag:
                            continue

                        emb = cache.get(frag) or get_embedding(frag)
                        if emb is None:
                            logger.error("No se pudo obtener embedding para el fragmento. Saltando.")
                            continue
                            
                        cache[frag] = emb

                        cls = classify_fragment_cosine(frag, emb, codes)
                        subcats = cls.get('category', [])

                        entry = {
                            'question':   q,
                            'fragment':   frag,
                            'codigos':    subcats
                        }
                        file_results.append(entry)
                        all_results.append(entry)

                except Exception as e:
                    logger.error("Error procesando QA pair %d: %s", idx, e)
                    continue

            # Save complete cleaned file
            cleaned_filename = os.path.join(CLEANED_DIR, f"{basename}_cleaned.txt")
            with open(cleaned_filename, 'w', encoding='utf-8') as cf:
                cf.write('\n\n'.join(all_cleaned_responses))
            logger.info("Saved complete cleaned file → %s", cleaned_filename)

            # Save complete segmented file
            segmented_filename = os.path.join(SEGMENTED_DIR, f"{basename}_segmented.json")
            with open(segmented_filename, 'w', encoding='utf-8') as sf:
                json.dump(all_segmented_fragments, sf, indent=2, ensure_ascii=False)
            logger.info("Saved complete segmented file → %s", segmented_filename)

            # 4) Escribir JSON parcial para este archivo de entrada
            partial_path = os.path.join(PARTIAL_DIR, f"{basename}.json")
            with open(partial_path, 'w', encoding='utf-8') as pf:
                json.dump(file_results, pf, indent=2, ensure_ascii=False)
            logger.info("Saved partial results for %s → %s", fn, partial_path)

        except Exception as e:
            logger.error("Error procesando archivo %s: %s", fn, e)
            continue

# 5) Escribir JSON completo de todos los archivos procesados
full_path = os.path.join(OUTPUT_DIR, 'all_interviews.json')
with open(full_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
logger.info("Full JSON saved → %s", full_path)

# 6) Guardar cache de embeddings actualizada
save_cache(cache)
logger.info("Embeddings cache saved → %s", os.path.join(OUTPUT_DIR, "embeddings_cache.pkl"))
