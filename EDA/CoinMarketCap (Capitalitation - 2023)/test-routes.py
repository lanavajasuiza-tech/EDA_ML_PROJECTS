import os
import sys

################### VERIFICACIÓN DE RUTAS EN  SYS ##########################3

current_directory = os.getcwd()
print(f"Directorio actual: {current_directory}")

# Ruta del proyecto
project_root = os.path.join(os.getcwd(), "/home/ana/Documentos/EDA_ML_PROJECTS/EDA/CoinMarketCap")
export_path = os.path.join(project_root, "dataSet")
print(f"Ruta absoluta del proyecto: {project_root}. {export_path}")


# Añadir solo la ruta si no está presente
if project_root not in sys.path:
    sys.path.append(project_root)

print("Rutas de búsqueda de módulos (limpiadas y actualizadas):")
for path in sys.path:
    print(path)


    
    
# Eliminar duplicados en sys.path
sys.path = list(set(sys.path))
print("\n".join(sys.path))

import sys

# Eliminar ruta que no quieres
ruta_no_deseada = "EDA/CoinMarketCap/dataSet"
sys.path = [ruta for ruta in sys.path if ruta != ruta_no_deseada]
print("\n".join(sys.path))


