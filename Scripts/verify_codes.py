import json
from collections import Counter
from pathlib import Path

def analyze_code_frequencies(json_path):
    """
    Analiza la frecuencia de códigos en el archivo JSON de entrevistas.
    
    Args:
        json_path (str o Path): Ruta al archivo JSON con los datos de las entrevistas.
        
    Returns:
        Counter: Objeto Counter con la frecuencia de cada código encontrado.
    """
    # Leer el JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Contador para los códigos
    code_counter = Counter()
    
    # Recorre todas las entradas y cuenta los códigos en el campo "category"
    for entry in data:
        codes = entry.get("category", [])
        # Asegurarse de que sea una lista antes de actualizar
        if isinstance(codes, list):
            code_counter.update(codes)
        else:
            # Si "category" no es lista, saltar o convertir a lista
            # Por ejemplo: code_counter.update([codes])
            continue
    
    return code_counter

def print_code_frequencies(counter):
    """
    Imprime las frecuencias de códigos de forma formateada.
    
    Args:
        counter (Counter): Objeto Counter con las frecuencias de códigos.
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
    # Ajusta esta ruta al lugar donde tengas tu JSON
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
