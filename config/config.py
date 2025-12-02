# config/config.py

import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
env = os.getenv

# OpenAI settings
OPENAI_API_KEY   = env("OPENAI_API_KEY", "")
GPT_MODEL        = env("GPT_MODEL", "gpt-4.1-nano")   # chat model por defecto
EMBEDDING_MODEL  = env("EMBEDDING_MODEL", "text-embedding-ada-002")
MAX_CHUNK_LENGTH = int(env("MAX_CHUNK_LENGTH", "3000"))
API_DELAY        = float(env("API_DELAY", "1.8"))
INITIAL_DELAY    = float(env("INITIAL_DELAY", "2"))

# Folders
INPUT_DIR           = env(
    "INPUT_DIR",
    r"C:\Users\Felipe Nunez\Documents\Machine Learning Work\JER\codificacion_final\assets\input\interviews\txt"
)
OUTPUT_DIR          = env(
    "OUTPUT_DIR",
    r"C:\Users\Felipe Nunez\Documents\Machine Learning Work\JER\codificacion_final\assets\output\interviews\coding"
)
MIN_FRAGMENT_LENGTH = int(env("MIN_FRAGMENT_LENGTH", "150"))
