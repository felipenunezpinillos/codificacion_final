# Scripts/vectorize.py

import openai
import logging
import os
import pickle
import time
from config.config import OPENAI_API_KEY, EMBEDDING_MODEL, OUTPUT_DIR

logger = logging.getLogger(__name__)
openai.api_key = OPENAI_API_KEY

CACHE_PATH = os.path.join(OUTPUT_DIR, "embeddings_cache.pkl")

def load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
                logger.debug("Persistent cache loaded with %d items", len(cache))
                return cache
        except Exception as e:
            logger.error("Error loading persistent cache: %s", e)
    return {}

def save_cache(cache):
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
            logger.debug("Persistent cache saved with %d items", len(cache))
    except Exception as e:
        logger.error("Error saving persistent cache: %s", e)

embedding_cache = load_cache()

def get_embedding(text, max_retries=3):
    if text in embedding_cache:
        logger.debug("Using cached embedding for text (length %d)", len(text))
        return embedding_cache[text]
    logger.debug("Generating embedding for text (length %d)", len(text))
    
    for attempt in range(max_retries):
        try:
            response = openai.Embedding.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            embedding = response['data'][0]['embedding']
            embedding_cache[text] = embedding
            save_cache(embedding_cache)
            logger.debug("Embedding generated (length %d)", len(embedding))
            return embedding
        except Exception as e:
            wait_time = 2 ** attempt  # Exponential backoff
            logger.error("Error generating embedding (attempt %d/%d): %s. Retrying in %ds", 
                        attempt + 1, max_retries, e, wait_time)
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                logger.error("Failed to generate embedding after %d attempts", max_retries)
                return None

def vectorize_fragments(fragments):
    logger.info("Vectorizing %d fragmentos", len(fragments))
    results = []
    for fragment in fragments:
        emb = get_embedding(fragment)
        if emb is not None:
            results.append((fragment, emb))
        else:
            logger.warning("No se generó embedding para fragmento (primeros 30 caracteres): %s", fragment[:30])
    logger.info("Vectorización completada para %d fragmentos", len(results))
    return results

if __name__ == "__main__":
    sample_fragments = [
        "This is a sample fragment about logistics.",
        "Otro fragmento que habla de experiencia de usuario."
    ]
    vectorized = vectorize_fragments(sample_fragments)
    for frag, vec in vectorized:
        print("Fragment:", frag)
        print("Embedding length:", len(vec))
