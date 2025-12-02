import pandas as pd
import json
import re
from pathlib import Path

# Ajusta estas rutas a tu entorno
EXCEL_PATH = Path(r"C:\Users\Felipe Nunez\Documents\Machine Learning Work\coder JER\assets\Reporte de codificación JER.xlsx")
CODEBOOK_INPUT_PATH = Path(r"C:\Users\Felipe Nunez\Documents\Machine Learning Work\JER\codificacion_final\config\codebook.py")
CODEBOOK_OUTPUT_PATH = Path(r"C:\Users\Felipe Nunez\Documents\Machine Learning Work\JER\codificacion_final\config\codebook_final.py")
MISSING_EXAMPLES_OUTPUT_PATH = Path(r"C:\Users\Felipe Nunez\Documents\Machine Learning Work\JER\codificacion_final\config\missing_examples.json")

# -------------------------------------------------------------------
# 1. Cargar el codebook original (supone que define FINAL_CODEBOOK_JER como dict)
# -------------------------------------------------------------------
import importlib.util

spec = importlib.util.spec_from_file_location("codebook", CODEBOOK_INPUT_PATH)
codebook_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codebook_mod)

# Asumimos que en codebook.py existe una variable global llamada FINAL_CODEBOOK_JER
codebook = codebook_mod.FINAL_CODEBOOK_JER

# -------------------------------------------------------------------
# 2. Leer el Excel y agrupar ejemplos por código
# -------------------------------------------------------------------
df = pd.read_excel(EXCEL_PATH)

# Nos aseguramos de solo tomar filas con Texto y Códigos no nulos
df = df[["Contenido de texto", "Códigos"]].dropna(subset=["Contenido de texto", "Códigos"])

# Función para limpiar fragmentos: eliminar speaker‐labels al inicio de cada línea
def clean_fragment(text: str) -> str:
    """
    Elimina prefijos de línea como 'P:', 'E:', 'P2-', 'P6-', 'E(number)-', etc.,
    cuando aparecen al comienzo de cualquier línea. Preserva el resto del texto y sus saltos de línea.
    """
    # (?m) activa modo multilinea, ^ representa el inicio de cada línea
    # Patrón: cualquier cantidad de espacios + 'P' o 'E' + opcionalmente dígitos + ':' o '-'
    return re.sub(r'(?m)^\s*[PE]\d*[\-:]\s*', '', text).strip()

# Función para extraer cada código basándose en su patrón numérico (e.g., "10.3")
def split_codes_by_number(codes_cell: str) -> list:
    """
    Dada una cadena con uno o varios códigos concatenados (p.ej. "10.3 Construcción, desarrollo y seguimiento del acuerdo restaurativo. 11.4 Ajuste organizacional IED"),
    devuelve una lista de cadenas, cada una comenzando con el número de código hasta justo antes del siguiente número de código.
    """
    text = str(codes_cell).strip()
    # Buscar todos los bloques que inician con dígitos+punto+dígitos hasta el próximo dígitos+punto+dígitos o fin de cadena.
    pattern = re.compile(r'(\d+\.\d+[^0-9]*?)(?=\s*\d+\.\d+|$)')
    matches = pattern.findall(text)
    # Limpiar cada match: eliminar espacios en extremos y comas/puntos finales sobrantes
    cleaned = [m.strip().rstrip(',.') for m in matches]
    return cleaned

# Aplicar la extracción por número en lugar de split por comas
df["Códigos"] = df["Códigos"].apply(split_codes_by_number)

# “Explotar” la lista de códigos para tener una fila por cada par (fragmento, código)
df_exploded = df.explode("Códigos").rename(columns={"Contenido de texto": "example", "Códigos": "code"})

# Antes de agrupar, limpiamos cada fragmento
df_exploded["example"] = df_exploded["example"].apply(clean_fragment)

# Agrupar ejemplos en listas por código
examples_by_code = df_exploded.groupby("code")["example"].apply(list).to_dict()

# -------------------------------------------------------------------
# 3. Insertar los ejemplos agrupados en el codebook y recolectar faltantes
# -------------------------------------------------------------------
missing_examples = {}

for code_key, example_list in examples_by_code.items():
    if code_key in codebook:
        codebook[code_key]["examples"] = example_list
    else:
        # Guardar ejemplos de códigos que no están en el codebook
        missing_examples[code_key] = example_list

# -------------------------------------------------------------------
# 4. Guardar el codebook actualizado en un nuevo archivo .py
# -------------------------------------------------------------------
with open(CODEBOOK_OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("FINAL_CODEBOOK_JER = ")
    json.dump(codebook, f, ensure_ascii=False, indent=2)

# -------------------------------------------------------------------
# 5. Guardar los ejemplos de códigos faltantes en un JSON separado
# -------------------------------------------------------------------
with open(MISSING_EXAMPLES_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(missing_examples, f, ensure_ascii=False, indent=2)

print(f"Codebook actualizado guardado en: {CODEBOOK_OUTPUT_PATH}")
print(f"Ejemplos de códigos faltantes guardados en: {MISSING_EXAMPLES_OUTPUT_PATH}")
