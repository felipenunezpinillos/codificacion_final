import json
import time
import importlib.util
import os
from config.config import OPENAI_API_KEY, GPT_MODEL, API_DELAY

# Ruta absoluta al codebook original
CODEBOOK_PATH = r"C:\Users\Felipe Nunez\Documents\Machine Learning Work\JER\codificacion_final\config\codebook_def.py"
OUTPUT_PATH = r"C:\Users\Felipe Nunez\Documents\Machine Learning Work\JER\codificacion_final\config\codebook_def_def.py"
CODEBOOK_VAR_NAME = "FINAL_CODEBOOK_JER"

def import_codebook_from_py(py_path, var_name):
    """Importa el codebook como módulo desde un .py con una variable global."""
    spec = importlib.util.spec_from_file_location("codebook", py_path)
    codebook_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codebook_module)
    return getattr(codebook_module, var_name)

def get_shortest_examples(examples, n=15):
    return sorted(examples, key=len)[:n]

def build_prompt(code_name, definition, keywords, examples, old_phrases):
    prompt = f"""
    Eres un experto en análisis cualitativo y construcción de libros de códigos.

    Te enviaré los detalles de un código en formato JSON: nombre, definición, keywords actuales, phrases actuales y hasta 15 ejemplos reales (ordenados de más largo a más corto). NO modifiques la definición ni los ejemplos.

    Tu tarea es:

    1. Añadir a 'keywords' 10 palabras clave nuevas, relacionadas DIRECTAMENTE con la definición y los ejemplos (no repitas las existentes).
    2. Borrar TODOS los synonyms actuales y crear 10 palabras importantes pero NO tan directas como los keywords (igual muy importantes para capturar relaciones indirectas), usando los ejemplos y definición, ignorando las palabras que ya están en y sin repetir los keywords.
    3. Añadir a 'phrases' 3 frases nuevas (máx 2-5 palabras, extraídas literalmente de los ejemplos, muy representativas del código) y dejar las 3 primeras que ya estaban (si había).
    4. Retorna SOLO un snippet JSON, con los campos: keywords (10), synonyms (10), phrases (6).
    5. NO uses conectores ni palabras vacías. NO modifiques la definición ni los ejemplos.

    Ejemplo de formato de salida:
    {{
    "keywords": [/*10 palabras*/],
    "synonyms": [/*10 palabras*/],
    "phrases": [/*6 frases*/]
    }}

    Aquí está el código a procesar:
    nombre: {code_name}
    definition: {definition}
    keywords: {json.dumps(keywords)}
    phrases: {json.dumps(old_phrases)}
    examples: {json.dumps(examples)}

    Retorna solo el snippet JSON.
    """
    return prompt

def refine_codebook(codebook, model=GPT_MODEL):
    import openai
    openai.api_key = OPENAI_API_KEY

    refined_codebook = {}

    for code_name, entry in codebook.items():
        definition = entry.get("definition", "")
        keywords = entry.get("keywords", [])
        synonyms = entry.get("synonyms", [])
        phrases = entry.get("phrases", [])
        examples = entry.get("examples", [])

        selected_examples = get_shortest_examples(examples, n=15)
        prompt = build_prompt(
            code_name=code_name,
            definition=definition,
            keywords=keywords,
            examples=selected_examples,
            old_phrases=phrases
        )
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512
            )
            response_json = json.loads(response["choices"][0]["message"]["content"])
            entry["keywords"] = response_json["keywords"]
            entry["synonyms"] = response_json["synonyms"]
            entry["phrases"] = response_json["phrases"]
            refined_codebook[code_name] = entry
            print(f"Procesado: {code_name}")
            time.sleep(API_DELAY)
        except Exception as e:
            print(f"Error procesando {code_name}: {e}")
            print("Respuesta bruta:", response.choices[0].message.content if 'response' in locals() else "")

    return refined_codebook

def save_codebook_py(codebook, py_path, var_name):
    with open(py_path, "w", encoding="utf-8") as f:
        f.write(f"{var_name} = ")
        json.dump(codebook, f, ensure_ascii=False, indent=2)
        f.write("\n")

if __name__ == "__main__":
    codebook = import_codebook_from_py(CODEBOOK_PATH, CODEBOOK_VAR_NAME)
    refined = refine_codebook(codebook)
    save_codebook_py(refined, OUTPUT_PATH, CODEBOOK_VAR_NAME)
    print(f"\nNuevo codebook guardado como: {OUTPUT_PATH}")
