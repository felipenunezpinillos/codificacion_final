import json
from collections import Counter
from pathlib import Path

def analyze_code_frequencies(json_path):
    """
    Analizes the frequency of codes in the interviews JSON file.

    Args:
        json_path (str or Path): Path to the JSON file containing interview data

    Returns:
        Counter: Counter object with code frequencies
    """
    # Leer el archivo JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Inicializar contador para los códigos
    code_counter = Counter()

    # Contar códigos en todos los fragmentos
    for entry in data:
        # Ahora la clave es "category" en lugar de "codigos"
        codes = entry.get('category', [])
        code_counter.update(codes)

    return code_counter

def print_code_frequencies(counter):
    """
    Imprime las frecuencias de códigos de forma formateada.

    Args:
        counter (Counter): Counter object con las frecuencias de códigos
    """
    print("\nCode Frequencies:")
    print("-" * 50)

    if not counter:
        print("No codes found in the data.")
        return

    # Ordenar por frecuencia (más frecuente primero)
    for code, count in counter.most_common():
        print(f"{code}: {count} occurrences")

    print("\nSummary:")
    print(f"Total unique codes: {len(counter)}")
    print(f"Total coded fragments: {sum(counter.values())}")

def main():
    # Ruta al archivo JSON generado tras clasificación
    json_path = Path("assets/output/interviews/coding/classified/all_classified.json")

    if not json_path.exists():
        print(f"Error: File not found at {json_path}")
        return

    # Analizar frecuencias
    code_counter = analyze_code_frequencies(json_path)

    # Imprimir resultados
    print_code_frequencies(code_counter)

if __name__ == "__main__":
    main()
