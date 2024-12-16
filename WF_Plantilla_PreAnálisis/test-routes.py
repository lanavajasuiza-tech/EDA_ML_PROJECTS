import os
import sys

################### VERIFICACIÓN DE RUTAS EN  SYS ##########################3

import sys

# Ruta del proyecto
project_root = r"C:\ruta_de_tu_proyecto"

# Añadir solo la ruta si no está presente
if project_root not in sys.path:
    sys.path.append(project_root)

print("Rutas de búsqueda de módulos (limpiadas y actualizadas):")
for path in sys.path:
    print(path)