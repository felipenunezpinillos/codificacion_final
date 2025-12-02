# Scripts/segmentation.py

import time
import openai
import logging
from config.config import OPENAI_API_KEY, GPT_MODEL, API_DELAY, INITIAL_DELAY, MIN_FRAGMENT_LENGTH

logger = logging.getLogger(__name__)
openai.api_key = OPENAI_API_KEY

def segment_text(text, retries=3):
    """
    Segmenta el texto completo en fragmentos lógicamente coherentes, donde cada fragmento
    representa una idea continua. No se divide simplemente por signos de puntuación; en su lugar,
    se agrupan las oraciones que pertenecen juntas. El objetivo de esta segmentación es facilitar
    una codificación cualitativa tipo Atlas.ti, por lo que se realizan las particiones pensando 
    en que cada fragmento podría corresponder a un código específico, mientras que el siguiente fragmento
    podría corresponder a otro distinto según su significado. Devuelve cada fragmento en una línea nueva.
    
    Reglas de segmentación:
    1. No se fragmentan textos menores a 500 caracteres
    2. Se fragmenta en máximo 3 partes solo cuando es necesario (cuando múltiples códigos podrían aplicarse)
    3. Se descartan fragmentos menores a MIN_FRAGMENT_LENGTH caracteres
    
    Args:
        text (str): Texto completo a segmentar.
        retries (int): Número de reintentos en caso de error de la API.
    
    Returns:
        List[str]: Cada línea de la respuesta de OpenAI se considera un fragmento separado.
        Máximo 3 fragmentos.
    """
    # Si el texto es muy corto, devolverlo como un solo fragmento
    if len(text) < 500:
        logger.debug("Text shorter than 500 characters, returning as single fragment")
        return [text]
        
    prompt = (
        "Segmenta el siguiente texto en fragmentos lógicamente coherentes, donde cada fragmento "
        "represente una idea continua. No te limites a dividir por signos de puntuación; en su lugar, "
        "agrupa las oraciones que pertenezcan juntas. El objetivo de esta segmentación es facilitar "
        "una codificación cualitativa tipo Atlas.ti, por lo que debes realizar las particiones pensando "
        "en que cada fragmento podría corresponder a un código específico, mientras que el fragmento "
        "siguiente podría corresponder a otro distinto según su significado.\n\n"
        "IMPORTANTE:\n"
        "1. Solo fragmenta el texto si es necesario (cuando diferentes partes podrían recibir códigos distintos)\n"
        "2. Genera como máximo 3 fragmentos, agrupando las ideas más relacionadas\n"
        "3. Cada fragmento debe tener al menos 150 caracteres\n"
        "4. Si el texto es coherente y podría recibir un solo código, devuélvelo como un solo fragmento\n\n"
        f"Texto:\n'''{text}'''"
    )
    logger.debug("Segmentation request for text of length %d", len(text))
    
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "Eres un asistente experto en segmentación de texto para codificación cualitativa."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            raw_output = response.choices[0].message.content.strip()
            logger.debug("Successful segmentation on attempt %d", attempt + 1)
            # Cada línea de la respuesta se toma como un fragmento
            fragments = [line.strip() for line in raw_output.split("\n") if line.strip()]
            # Limitar a máximo 3 fragmentos
            fragments = fragments[:3]
            # Filtrar fragmentos muy cortos
            fragments = [f for f in fragments if len(f) >= MIN_FRAGMENT_LENGTH]
            time.sleep(API_DELAY)
            return fragments
        except Exception as e:
            wait_time = INITIAL_DELAY * (2 ** attempt)
            logger.error("Error en segmentación (intento %d): %s. Reintentando en %.2fs", attempt + 1, e, wait_time)
            time.sleep(wait_time)
    logger.warning("Devolviendo todo el texto como un solo fragmento tras %d intentos fallidos", retries)
    return [text]
