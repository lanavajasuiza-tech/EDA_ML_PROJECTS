"""
Archivo principal del proyecto de clustering y análisis de datos.

Este archivo sirve como punto de entrada del programa, donde se importan
y ejecutan los modelos de clustering y las funciones de preprocesamiento
y visualización. Aquí se cargan los datos, se normalizan y se ejecutan
los modelos en función de los requisitos.

Responsabilidades:
- Coordinar la ejecución de diferentes modelos de clustering.
- Integrar preprocesamiento y visualización.
- Servir como interfaz principal del proyecto.

Estructura:
1. Cargar datos.
2. Preprocesamiento.
3. Aplicación de modelos de clustering.
4. Visualización de resultados.
"""
import os
import sys
from utils.processing import DataLoader
from utils.analyzer import DataAnalyzer



# Ajustar el directorio raíz para garantizar que los módulos sean accesibles
sys.path.append(r"C:\Users\rportatil112\Documents\CURSO-DATA-SCIENCE\MACHINE_LEARNING_E_INTELIGENCIA_ARTIFICIAL\Curso_ML_Laner\17-Modelos\NoSupervisados\WF_ML_kmeans")
print("Directorio actual:", os.getcwd())

# Configuración: Definimos la ruta y nombre del dataSet
df_path = "dataSet"
df = "wholesale.csv"
new_df = "incluir_el_nombre" # En caso de jugar con otro función change_file en processing.py

#---------------- CARGAR Y PROCESAR LOS DATOS -------------------#
try:
    loader = DataLoader(df_path=df_path, df=df)
    df = loader.load_data()
    print("\n--- Dataset cargado correctamente ---")
except FileNotFoundError as e:
    print(f"Error al cargar el dataset: {e}")
    df = None
except ValueError as e:
    print(f"Error de valor en el dataset: {e}")
    df = None

# Si los datos se cargaron, proceder al análisis
if df is not None:
    # Instanciar el analizador
    analyzer = DataAnalyzer(df)

    # Llamadas a los métodos del analizador para verificar si se están cargando o no
    #print(dir(analyzer))  # Verifica que 'nan_summary' esté listado

    analyzer.overview()
    analyzer.duplicates_analysis()
    analyzer.missing_values_analysis()
    analyzer.data_types_analysis()
    print("Columnas categóricas detectadas:", analyzer.columns_cat)
    print("Columnas numéricas detectadas:", analyzer.columns_num)
    analyzer.combined_statistics()
    analyzer.nan_summary()
else:
    print("\n--- No se pudo cargar el dataset. Análisis abortado ---")


