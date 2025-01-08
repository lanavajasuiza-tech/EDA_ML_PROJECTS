
# Visualizar las rutas raíz
import sys
print("Rutas en sys.path:")
for path in sys.path:
    print(path)
    
# Ruta al directorio raíz del proyecto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


import os

# Ruta del script actual
current_path = os.path.abspath(__file__)
print("Ruta del script actual:", current_path)

# Directorio del script actual
current_dir = os.path.dirname(current_path)
print("Directorio del script:", current_dir)
