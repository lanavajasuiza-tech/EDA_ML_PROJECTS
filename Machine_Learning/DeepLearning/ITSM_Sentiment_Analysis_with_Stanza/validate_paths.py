import os
import nltk

# Definir las rutas principales que deben existir
REQUIRED_PATHS = [
    "data"
    "data/raw",
    "data/processed",
    "data/embeddings",
    "processing",
    "decorators",
    "utils",
    "tests"
]

# Definir los recursos de NLTK necesarios
REQUIRED_NLTK_RESOURCES = ['punkt', 'stopwords', 'wordnet']

def validate_paths(base_path=None):
    """
    Valida que las rutas requeridas existen dentro de la estructura del proyecto.
    Args:
        base_path (str): Ruta base del proyecto. Si no se proporciona, toma la ruta actual.
    """
    if base_path is None:
        base_path = os.getcwd()

    print(f"Validando rutas desde: {os.path.abspath(base_path)}")

    for path in REQUIRED_PATHS:
        full_path = os.path.join(base_path, path)
        if os.path.exists(full_path):
            print(f"✔ Ruta encontrada: {full_path}")
        else:
            print(f"✘ Ruta faltante: {full_path}")
            create = input(f"¿Deseas crearla? (s/n): ").strip().lower()
            if create == "s":
                os.makedirs(full_path, exist_ok=True)
                print(f"✔ Ruta creada: {full_path}")
            else:
                print(f"✘ Ruta no creada: {full_path}")

def validate_nltk_resources():
    """
    Verifica y descarga los recursos necesarios de NLTK.
    """
    print("\nValidando recursos de NLTK...")
    for resource in REQUIRED_NLTK_RESOURCES:
        try:
            nltk.data.find(resource)
            print(f"✔ Recurso de NLTK '{resource}' ya está disponible.")
        except LookupError:
            print(f"✘ Recurso de NLTK '{resource}' no encontrado. Descargando...")
            try:
                nltk.download(resource)
                print(f"✔ Recurso de NLTK '{resource}' descargado correctamente.")
            except Exception as e:
                print(f"✘ Error al descargar '{resource}': {e}")
