import os
import sys

################### VERIFICACIÓN DE RUTAS EN  SYS ##########################3

import sys

# Ruta del proyecto
project_root = r"C:\Users\rportatil112\Documents\CURSO-DATA-SCIENCE\MACHINE_LEARNING_E_INTELIGENCIA_ARTIFICIAL\Curso_ML_Laner\17-Modelos\NoSupervisados\WF_ML_kmeans"

# Añadir solo la ruta si no está presente
if project_root not in sys.path:
    sys.path.append(project_root)

print("Rutas de búsqueda de módulos (limpiadas y actualizadas):")
for path in sys.path:
    print(path)