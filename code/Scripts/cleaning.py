# Scripts/cleaning.py

"""
Module for cleaning text using the OpenAI API.
Removes unwanted content (headers, footers, image captions, etc.) from text.
Processes large texts in chunks with debugging, retries, and exponential back-off.
"""

import time
import openai
import logging
from config.config import OPENAI_API_KEY, GPT_MODEL, MAX_CHUNK_LENGTH, API_DELAY, INITIAL_DELAY
from utils.utils import split_text_into_chunks

logger = logging.getLogger(__name__)
openai.api_key = OPENAI_API_KEY

def clean_text_chunk(chunk, retries=3):
    """
    Cleans a single text chunk using the OpenAI API.
    Only removes specific unwanted elements while preserving all actual content.
    Uses exponential back-off for retries and adds a fixed delay after a successful call.
    """
    prompt = (
        "You are a text cleaning assistant. Your task is to clean the following text by ONLY removing:\n"
        "1. Page numbers (e.g., 'Page 1 of 10')\n"
        "2. Document headers/footers that repeat on every page\n"
        "3. Image captions that are clearly not part of the main text\n"
        "4. Technical metadata or formatting markers\n\n"
        "IMPORTANT: DO NOT remove or modify:\n"
        "- Any actual content from the speaker/participant\n"
        "- Any part of the actual responses or statements\n"
        "- Any meaningful text, even if it seems repetitive\n"
        "- Any context or background information\n\n"
        "Preserve the complete content exactly as spoken/written, only removing the technical elements listed above.\n\n"
        f"Text to clean: '''{chunk}'''"
    )
    logger.debug("Cleaning chunk with length %d", len(chunk))
    
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a conservative text cleaning assistant that preserves all meaningful content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            cleaned_chunk = response.choices[0].message.content.strip()
            logger.debug("Successfully cleaned chunk on attempt %d", attempt + 1)
            time.sleep(API_DELAY)
            return cleaned_chunk
        except Exception as e:
            wait_time = INITIAL_DELAY * (2 ** attempt)
            logger.error("Error cleaning text chunk (attempt %d): %s. Retrying in %.2fs", attempt + 1, e, wait_time)
            time.sleep(wait_time)
    logger.warning("Returning original chunk after %d failed attempts", retries)
    return chunk

def clean_text(text):
    """
    Cleans a large text by splitting it into chunks, cleaning each, and reassembling.
    
    Args:
        text (str): Raw input text.
        
    Returns:
        str: Fully cleaned text.
    """
    logger.info("Starting cleaning process for text with length %d", len(text))
    chunks = split_text_into_chunks(text, MAX_CHUNK_LENGTH)
    logger.info("Text split into %d chunks", len(chunks))
    cleaned_chunks = []
    for i, chunk in enumerate(chunks):
        logger.info("Cleaning chunk %d/%d", i + 1, len(chunks))
        cleaned_chunks.append(clean_text_chunk(chunk))
    full_cleaned_text = "\n".join(cleaned_chunks)
    logger.info("Completed cleaning process; final text length: %d", len(full_cleaned_text))
    return full_cleaned_text

if __name__ == "__main__":
    sample_text = "Header info\nThis is the main text. Footer info\n" * 10
    print("Cleaned Text:", clean_text(sample_text))
