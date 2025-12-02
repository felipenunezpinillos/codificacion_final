# Scripts/loader.py

import re
import logging
from config.config import MIN_FRAGMENT_LENGTH
from utils.utils import read_text_file

logger = logging.getLogger(__name__)

def load_fragments_with_question(filepath: str) -> list:
    """
    Reads a transcript and groups all consecutive participant lines under each question into one response.
    Only returns pairs where the combined response length >= MIN_FRAGMENT_LENGTH.
    """
    text = read_text_file(filepath)
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    q_pat = re.compile(r'^E[0-9A-Za-z]*:\s*(.+)')  # question identifier
    p_pat = re.compile(r'^P[0-9A-Za-z]*:\s*(.+)')  # participant identifier

    result = []
    current_q = None
    current_resp = []

    def flush():
        if current_q is not None and current_resp:
            resp = " ".join(current_resp).strip()
            if len(resp) >= MIN_FRAGMENT_LENGTH:
                result.append({"question": current_q, "response": resp})

    for line in lines:
        qm = q_pat.match(line)
        pm = p_pat.match(line)
        if qm:
            # new question: flush previous
            flush()
            current_q = qm.group(1).strip()
            current_resp = []
        elif pm and current_q:
            current_resp.append(pm.group(1).strip())
    # flush last
    flush()
    logger.info("Found %d Q–A pairs", len(result))
    return result
