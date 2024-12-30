"""
Author: Ana Ndongo
Date: December 2024
Description:

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
from utils.optimizing import DataCleaner
from utils.visualization import HeatmapVisualizer, ScatterplotVisualizer, BarplotVisualizer, BoxplotVisualizer, DataVisualizationCoordinator


#---------------- CARGAR DATASET -------------------#

# Detectar dinámicamente el directorio raíz del proyecto
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
print("Directorio raíz detectado dinámicamente:", project_root)


# Ruta simplificada al dataset
df_path = os.path.join(project_root, "dataSet")
print(f"Ruta del dataset: {df_path}")
df = "wholesale.csv"

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
    analyzer.nan_summary()
else:
    print("\n--- No se pudo cargar el dataset. Análisis abortado ---")

    #---------------- OPTIMIZAR LOS DATOS -------------------#

from utils.optimizing import DataCleaner


if df is not None:
    print("\n--- Optimizando los datos ---")

    # Instanciar DataCleaner
    cleaner = DataCleaner(df)

    # ----------------- Validaciones y Ejecuciones ---------------- #

#-----> 1. Eliminar columnas con muchos valores faltantes

cleaner = DataCleaner(df)
df_cleaned = cleaner.handle_missing_values_interactively()


    # 3. Eliminar duplicados (Elimina solo filas y columnas exactas)

cleaner.drop_duplicates_rows_and_columns()
df_cleaned = cleaner.get_cleaned_data()
print("\n--- DataFrame limpio ---")
print(df_cleaned.head())



    # 4. Binarizar columnas categóricas

cleaner = DataCleaner(df)

cleaner.preprocess_for_binarization()

df_cleaned = cleaner.get_cleaned_data()
print("\n--- DataFrame después de binarizar ---")
print(df_cleaned.head())

'''Estructura del Flujo de Binarización
Verificación del Tipo de Dato:

Identifica columnas con dtype == 'object' o category.
Excluye columnas numéricas y booleanas.
Identificación de Columnas Relevantes:

Ignora columnas con solo dos categorías si ya están representadas como binarias (e.g., Sí/No).
Genera una lista con columnas categóricas candidatas, indicando el número de categorías únicas.
Evitar Problemas de Dimensionalidad:

Si una columna tiene demasiadas categorías (por ejemplo, >10), sugiere agrupar las menos frecuentes en una categoría "Raro".
Manejo de Valores Faltantes:

Completa valores NaN antes de binarizar (por ejemplo, con la moda).
Propuesta de Binarización Personalizada:

Muestra las columnas categóricas candidatas y permite al usuario elegir:
Binarizar con todas las categorías.
Binarizar eliminando una categoría (drop_first=True).
Agrupar categorías raras antes de binarizar.
No realizar ninguna acción.'''



    # 5. Normalizar columnas numéricas

cleaner = DataCleaner(df)
cleaner.interactive_normalization()
df_cleaned = cleaner.get_cleaned_data()
print("\n--- DataFrame después de la normalización ---")
print(df_cleaned.head())


'''Puntos Clave al Normalizar
Verificación del Tipo de Dato:

Solo tiene sentido normalizar columnas numéricas (dtype == 'int' o float).
Evitar aplicar normalización a columnas categóricas o booleanas.
Identificación de Columnas Relevantes:

Algunas columnas numéricas podrían no requerir normalización, como aquellas que ya están en un rango uniforme (e.g., entre 0 y 1).
Columnas con escalas muy grandes o variables importantes para análisis deben ser priorizadas.
Selección de Estrategia de Normalización:

Min-Max Scaling: Escala los valores a un rango entre 0 y 1.
Z-Score (Estandarización): Escala los valores a una distribución con media 0 y desviación estándar 1.
Logaritmo o Escalado Robust: Opciones para manejar valores atípicos.
Manejo de Valores Faltantes:

Completar valores faltantes antes de normalizar, ya que métodos como Min-Max o Z-Score no los manejan directamente.
Propuesta Personalizada:

Proponer columnas a normalizar basándose en el tipo de dato y la distribución de los valores.
Elegir la estrategia de normalización para cada columna interactivamente.'''

    # 6. Detectar y manejar outliers

cleaner = DataCleaner(df)

cleaner.detect_and_handle_outliers()

df_cleaned = cleaner.get_cleaned_data()
print("\n--- DataFrame después de manejar outliers ---")
print(df_cleaned.head())


'''Puntos Clave al Detectar y Manejar Outliers
Verificación del Tipo de Dato:

Los outliers solo se detectan en columnas numéricas (dtype == 'int' o float).
Selección del Método de Detección:

IQR (Rango Intercuartílico):
Detecta valores fuera del rango [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
Z-Score:
Detecta valores con una desviación estándar significativa (por ejemplo, >3 desviaciones estándar).
Propuesta de Columnas Relevantes:

Identificar columnas numéricas y su distribución para decidir si necesitan detección de outliers.
Opciones de Manejo:

Eliminar las filas con outliers.
Reemplazar los outliers por un valor (mediana, media, etc.).
Dejar los outliers sin cambios.'''

    # 7. Reducir dimensiones
cleaner = DataCleaner(df)
cleaner.reduce_dimensions_interactively()
df_cleaned = cleaner.get_cleaned_data()
print("\n--- DataFrame después de la Reducción de Dimensiones ---")
print(df_cleaned.head())

'''Puntos Clave al Aplicar PCA
Verificación del Tipo de Dato:

PCA solo funciona con columnas numéricas (dtype == 'float' o int).
Identificación de Columnas Relevantes:

Necesitas al menos 2 columnas numéricas para aplicar PCA.
Si hay valores faltantes, deben completarse antes de aplicar PCA.
Selección de Número de Componentes:

Permitir al usuario elegir el número de componentes principales (n_components).
Mostrar la proporción de varianza explicada para ayudar en la decisión.
Escalado de Datos:

Es fundamental escalar las columnas antes de aplicar PCA (por ejemplo, con Z-Score), ya que PCA es sensible a las escalas de los datos.
Interactividad:

Proponer columnas candidatas y confirmar con el usuario si desea incluirlas en el PCA.
'''

    # Obtener el DataFrame limpio
df_cleaned = cleaner.get_cleaned_data()


    #---------------- VISUALIZAR LOS DATOS -------------------#

from utils.visualization import DataVisualizationCoordinator

print("\n--- Visualización Combinada ---")

# Columnas seleccionadas para las visualizaciones
x_col = "Grocery"      # Columna para el eje X del scatterplot
y_col = "Milk"         # Columna para el eje Y del scatterplot
bar_box_col = "Grocery"  # Columna para el barplot y boxplot
cluster_col = "Channel"  # Columna de clusters

# Instanciar el coordinador de visualizaciones
viz_coordinator = DataVisualizationCoordinator(df)

# Generar todas las visualizaciones en una sola ventana
try:
    viz_coordinator.plot_all(x=x_col, y=y_col, column=bar_box_col, clusters=cluster_col)
except KeyError as e:
    print(f"Error en las visualizaciones: {e}")
    print("Columnas disponibles en el DataFrame:", df.columns.tolist())
    