#!/usr/bin/env python3
"""
scripts/clean_codebook.py

Este script carga el diccionario FINAL_CODEBOOK_JER desde config/codebook.py,
elimina etiquetas de hablante ("P:", "E:", "P<number>:", etc.) en los ejemplos,
los ordena de más largo a más corto, y genera una copia limpia en
config/codebook_cleaned.py con la misma estructura pero con los ejemplos
limpiados y reordenados.
"""

import re
import os
import importlib.util
import textwrap

# Ajusta estas rutas según la estructura de tu proyecto:
CODEBOOK_PATH = os.path.join(os.path.dirname(__file__), "config", "codebook.py")
CLEANED_CODEBOOK_PATH = os.path.join(os.path.dirname(__file__), "config", "codebook_cleaned.py")


def load_original_codebook(path):
    """
    Carga dinámicamente el módulo config.codebook desde la ruta dada
    y retorna el diccionario FINAL_CODEBOOK_JER definido en él.
    """
    spec = importlib.util.spec_from_file_location("codebook", path)
    codebook_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codebook_module)
    try:
        return codebook_module.FINAL_CODEBOOK_JER
    except AttributeError:
        raise RuntimeError("No se encontró FINAL_CODEBOOK_JER en el módulo codebook.")


def clean_example(text):
    """
    Limpia el texto de ejemplo:
    1. Elimina etiquetas de hablante al inicio de líneas (p. ej. "P:", "E:", "P1:", "E2:", etc.)
    2. Elimina saltos de línea múltiples
    3. Normaliza los espacios
    4. Elimina espacios al inicio y final de cada línea
    5. Une todo en una sola línea
    """
    # Primero eliminamos las etiquetas de hablante
    text = re.sub(r'(?m)^\s*[PE]\d*[:-]?\s*', '', text)
    
    # Dividimos en líneas y limpiamos cada línea
    lines = [line.strip() for line in text.splitlines()]
    
    # Filtramos líneas vacías y unimos con un espacio
    return ' '.join(line for line in lines if line)


def sort_examples_descending(examples_list):
    """
    Dada una lista de ejemplos (strings), devuelve todos los ejemplos
    tras limpiarlos (sin etiquetas) y ordenados de mayor a menor longitud.
    """
    cleaned = [clean_example(e) for e in examples_list]
    # Filtrar posibles cadenas vacías
    cleaned = [c for c in cleaned if c]
    # Ordenar de mayor a menor longitud
    cleaned_sorted = sorted(cleaned, key=len, reverse=True)
    return cleaned_sorted


def build_cleaned_codebook(original_codebook):
    """
    Recorre cada categoría en original_codebook y construye uno nuevo con:
      - "definition" idéntico al original.
      - "keywords", "synonyms" y "phrases" idénticos al original.
      - "examples": todos los ejemplos originales, pero limpios y ordenados
         de mayor a menor longitud.
    Retorna el nuevo diccionario.
    """
    cleaned = {}
    for code_key, details in original_codebook.items():
        new_entry = {
            "definition": details.get("definition", ""),
            "keywords": details.get("keywords", []),
            "synonyms": details.get("synonyms", []),
            "phrases": details.get("phrases", []),
            "examples": sort_examples_descending(details.get("examples", []))
        }
        cleaned[code_key] = new_entry
    return cleaned


def dict_to_python_literal(d, indent=0):
    """
    Convierte un diccionario de Python (que contenga solo dicts, listas y strings)
    en una cadena con literal Python válido, respetando indentación.
    """
    spacing = " " * indent
    if isinstance(d, dict):
        items = []
        for k, v in d.items():
            key_repr = repr(k)
            val_repr = dict_to_python_literal(v, indent + 4)
            items.append(f"{' ' * (indent + 4)}{key_repr}: {val_repr}")
        inner = ",\n".join(items)
        return "{\n" + inner + f"\n{spacing}}}"
    elif isinstance(d, list):
        if not d:
            return "[]"
        items = []
        for item in d:
            item_repr = dict_to_python_literal(item, indent + 4)
            items.append(f"{' ' * (indent + 4)}{item_repr}")
        inner = ",\n".join(items)
        return "[\n" + inner + f"\n{spacing}]"
    else:
        # Asumimos que es str, int, etc.
        return repr(d)


def write_cleaned_codebook_file(cleaned_codebook, dest_path):
    """
    Escribe un archivo Python en dest_path que define FINAL_CODEBOOK_JER_CLEANED
    como el diccionario cleaned_codebook.
    """
    header = textwrap.dedent("""\
        # -*- coding: utf-8 -*-
        \"\"\"
        config/codebook_cleaned.py

        Este archivo ha sido generado automáticamente por scripts/clean_codebook.py.
        Contiene la versión 'limpia' de FINAL_CODEBOOK_JER, en la que cada categoría:
          - Conserva 'definition', 'keywords', 'synonyms' y 'phrases' tal cual.
          - En 'examples', todos los ejemplos se limpian (sin etiquetas P:/E:) y
            se ordenan de mayor a menor longitud para facilitar seleccionar los
            tres más cortos posteriormente.
        \"\"\"

        FINAL_CODEBOOK_JER_CLEANED = 
        """)

    dict_literal = dict_to_python_literal(cleaned_codebook, indent=0)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(dict_literal)
        f.write("\n")


if __name__ == "__main__":
    try:
        original = load_original_codebook(CODEBOOK_PATH)
    except Exception as e:
        print(f"[Error] No se pudo cargar config/codebook.py: {e}")
        exit(1)

    cleaned_dict = build_cleaned_codebook(original)
    write_cleaned_codebook_file(cleaned_dict, CLEANED_CODEBOOK_PATH)
    print(f"Se ha creado la copia limpia: {CLEANED_CODEBOOK_PATH}")
