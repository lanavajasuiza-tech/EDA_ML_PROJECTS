'''
import os
import sys

################### VERIFICACIÓN DE RUTAS EN  SYS ##########################3

import sys

# Ruta del proyecto
project_root = r"C:/Users/rportatil112/Documents/EDA_Analysis/WF_Plantilla_PreAnálisis"

# Añadir solo la ruta si no está presente
if project_root not in sys.path:
    sys.path.append(project_root)

print("Rutas de búsqueda de módulos (limpiadas y actualizadas):")
for path in sys.path:
    print(path)
    '''

import os
import sys

# ------------------- CONFIGURACIÓN DE LA RUTA RAÍZ ------------------- #

import sys
project_root = r"C:/Users/rportatil112/Documents/EDA_ML_PROJECTS/WF_Plantilla_PreAnálisis"

if project_root not in sys.path:
    sys.path.append(project_root)

print("Rutas en sys.path:")
for path in sys.path:
    print(path)


# Añade la ruta raíz al sys.path si no está presente
if project_root not in sys.path:
    sys.path.append(project_root)

# ------------------- VERIFICACIÓN DE SUBMÓDULOS ------------------- #

# Define las subcarpetas principales que contienen los módulos
submodules = ["utils", "dataSet"]

# Añade las subcarpetas al sys.path
for submodule in submodules:
    submodule_path = os.path.join(project_root, submodule)
    if submodule_path not in sys.path:
        sys.path.append(submodule_path)

# ------------------- COMPROBACIÓN DE LA ESTRUCTURA ------------------- #

# Verifica que cada subcarpeta contiene un archivo __init__.py
for submodule in submodules:
    submodule_path = os.path.join(project_root, submodule)
    # Asegúrate de que la carpeta exista
    if not os.path.exists(submodule_path):
        print(f"⚠️ Falta el subdirectorio {submodule}. Creándolo automáticamente...")
        os.makedirs(submodule_path)
    # Verifica que el archivo __init__.py exista
    init_file = os.path.join(submodule_path, "__init__.py")
    if not os.path.exists(init_file):
        print(f"⚠️ Falta el archivo __init__.py en {submodule}. Creándolo automáticamente...")
        with open(init_file, "w") as f:
            pass  # Crear un archivo vacío


# ------------------- VALIDACIÓN FINAL ------------------- #

# Comprobar que los módulos clave están accesibles
try:
    from utils.processing import DataLoader
    from utils.analyzer import DataAnalyzer
    from utils.optimizing import DataCleaner
    from utils.visualization import (
        HeatmapVisualizer,
        ScatterplotVisualizer,
        BarplotVisualizer,
        BoxplotVisualizer,
        DataVisualizationCoordinator,
    )
    print("✅ Todos los módulos importados correctamente.")
except ModuleNotFoundError as e:
    print(f"❌ Error al importar módulos: {e}")

# ------------------- INFORMACIÓN DE DEPURACIÓN ------------------- #

# Imprime las rutas actuales de búsqueda en sys.path
print("\nRutas de búsqueda de módulos (actualizadas):")
for path in sys.path:
    print(path)
