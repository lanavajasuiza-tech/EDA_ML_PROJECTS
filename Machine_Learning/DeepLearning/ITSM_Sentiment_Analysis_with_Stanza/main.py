# Análisis de Sentimientos en ITSM usando Stanza
#
# Fuente: https://www.kaggle.com/datasets/tobiasbueck/capterra-reviews
# Autoría: Ana Ndongo (24/01/2024)
#
# Objetivo:
# Evaluar diferencias entre dos datasets relacionados para decidir cuál utilizar como base para un análisis de sentimientos.
# Comparar y analizar comentarios del proyecto ZooKeeper (Jira).
#
# Nota sobre el dataset:
# Este dataset incluye columnas derivadas como Benchmark_results_of_politeness y Benchmark_results_of_sentiment, 
# generadas previamente mediante herramientas de análisis de sentimientos y cortesía.
#
# Dado que el objetivo del ejercicio es analizar y visualizar sentimientos, se parte de estas métricas ya calculadas 
# para centrarse en los resultados y su interpretación, en lugar de en la generación de las mismas.


import pandas as pd
import chardet
from difflib import unified_diff

# Cargar archivos
path_excel = 'data/ZooKeeper_Project_Dataset.xlsx'
path_csv = 'data/data_results.csv'

df1 = pd.read_excel(path_excel)  # Dataset RAW
df2 = pd.read_csv(path_csv, encoding='MacRoman')  # Dataset procesado

# Resumen inicial
print(f"Columnas comunes: {set(df1.columns).intersection(df2.columns)}")
print(f"Tamaño del Excel: {df1.shape}, CSV: {df2.shape}")

# Verificar igualdad en columnas clave
for col in ['Comment', 'Benchmark_results_of_politeness', 'Benchmark_results_of_sentiment']:
    iguales = df1[col].equals(df2[col])
    print(f"¿'{col}' es igual en ambos datasets?: {iguales}")
'''
¿'Comment' es igual en ambos datasets?: False
¿'Benchmark_results_of_politeness' es igual en ambos datasets?: True
¿'Benchmark_results_of_sentiment' es igual en ambos datasets?: True

Se profundiza un poco más en Comment
'''

# Limpieza y normalización para la columna Comment
df1['Comment'] = df1['Comment'].str.replace(r'(_x000D_|\n)', '', regex=True).str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)
df2['Comment'] = df2['Comment'].str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)

# Comparación después de limpiar
print(f"¿Comentarios iguales tras normalización?: {df1['Comment'].equals(df2['Comment'])}")
'''
Se siguen detectando diferencias
'''

# Mostrar primera diferencia significativa
for i, (c1, c2) in enumerate(zip(df1['Comment'], df2['Comment'])):
    if c1 != c2:
        print(f"Diferencia en fila {i}:\n" + '\n'.join(unified_diff(c1.splitlines(), c2.splitlines())))
        break

'''
Conclusión: Se usará el Excel como base (RAW) por ser el dataset más completo y adecuado para este análisis.
Se hace este ejercicio para ratificar la información con la que vamos a trabajar
'''


#--- ANALISIS DE CONTENIDO DEL DATASET ---#
# --- Análisis inicial del dataset ---
print(f"Número de filas: {df1.shape[0]}")
print(f"Número de columnas: {df1.shape[1]}")
print("Nombres de las columnas:", df1.columns.tolist())
print("\nInformación del dataset:")
print(df1.info())
print("\nValores nulos por columna:")
print(df1.isnull().sum())
print("\n¿Hay columnas duplicadas?", df1.columns.duplicated().any())
print("\nNúmero de filas duplicadas:", df1.duplicated().sum())

# Identificación de tipos de columnas
print("\nColumnas categóricas:", df1.select_dtypes(include=['object']).columns.tolist())
print("\nColumnas numéricas:", df1.select_dtypes(include=['number']).columns.tolist())

# Análisis de valores únicos y distribución
print("\nValores únicos por columna:")
for col in df1.columns:
    print(f"{col}: {df1[col].nunique()} únicos")
print("\nValores únicos en 'Benchmark_results_of_politeness':")
print(df1['Benchmark_results_of_politeness'].value_counts())
print("\nEstadísticas descriptivas de 'Benchmark_results_of_sentiment':")
print(df1['Benchmark_results_of_sentiment'].describe())
print("\nDistribución de valores en 'Benchmark_results_of_sentiment':")
print(df1['Benchmark_results_of_sentiment'].value_counts(bins=10))

# Análisis de NaN
print("\nFilas con valores nulos en 'Comment':")
print(df1[df1['Comment'].isnull()])

# --- Análisis de duplicados ---
duplicados = df1[df1.duplicated(subset=['Comment'], keep=False)]
print(f"\nDuplicados encontrados: {len(duplicados)}")
print("\nEjemplo de filas duplicadas:")
print(duplicados.head())

# Identificar diferencias significativas entre duplicados
diferencias_duplicados = duplicados.groupby('Comment').agg({
    'Benchmark_results_of_politeness': pd.Series.nunique,
    'Benchmark_results_of_sentiment': pd.Series.nunique
}).reset_index()

diferencias_significativas = diferencias_duplicados[
    (diferencias_duplicados['Benchmark_results_of_politeness'] > 1) |
    (diferencias_duplicados['Benchmark_results_of_sentiment'] > 1)
]

print(f"\nNúmero de comentarios duplicados con diferencias significativas: {len(diferencias_significativas)}")
print(diferencias_significativas)

# Ejemplo de diferencias significativas
if not diferencias_significativas.empty:
    ejemplo = duplicados[duplicados['Comment'] == diferencias_significativas.iloc[0]['Comment']]
    print("\nEjemplo de diferencias significativas:")
    print(ejemplo)

# Filtrar duplicados con inconsistencias o NaN
duplicados_con_inconsistencias_y_nan = duplicados.groupby('Comment').filter(
    lambda x: (
        len(x['Benchmark_results_of_politeness'].unique()) > 1 or
        len(x['Benchmark_results_of_sentiment'].unique()) > 1 or
        x['Comment'].isnull().any()
    )
)

print("\nDuplicados con inconsistencias o NaN:")
print(duplicados_con_inconsistencias_y_nan)

# Número de comentarios afectados
num_comentarios_afectados = duplicados_con_inconsistencias_y_nan['Comment'].nunique()
print(f"\nNúmero de comentarios afectados por ambigüedad o NaN: {num_comentarios_afectados}")

#--- Procedemos a eliminar las columnas que no aportan valor

# Eliminar filas con valores NaN en la columna 'Comment'
df1 = df1.dropna(subset=['Comment'])

# Eliminar filas duplicadas (incluyendo aquellas con inconsistencias detectadas)
df = df1.drop_duplicates()

# Mostrar el nuevo tamaño del dataset
print(f"Número de filas después de limpiar: {df.shape[0]}")
print(f"Número esperado de filas eliminadas: {13644 - df.shape[0]}")
'''
Tras analisis se procede al borrado (decisión comentada en el documento Análisis.md)
de aquellas filas que no nos aportan, un total del 2% del total del dataSet
'''

# Mostrar el tamaño del dataset limpio
print(f"Número de filas finales en el dataset limpio: {df.shape[0]}")


#---TRATAMIENTO DE LA COLUMNA CATEGÓRICA

# Crear una nueva columna con categorías basadas en los rangos de 'Benchmark_results_of_sentiment'
def categorizar_sentimiento(valor):
    if -1 <= valor <= -0.6:
        return 'Muy negativo'
    elif -0.6 < valor <= -0.2:
        return 'Negativo'
    elif -0.2 < valor <= 0.2:
        return 'Neutral'
    elif 0.2 < valor <= 0.6:
        return 'Positivo'
    elif 0.6 < valor <= 1:
        return 'Muy positivo'

# Aplicar la función para crear la nueva columna
df['Sentiment_category'] = df['Benchmark_results_of_sentiment'].apply(categorizar_sentimiento)

# Mostrar un resumen rápido
print(df['Sentiment_category'].value_counts())

df.describe
'''Información correcta y esperada, existe una nueva columna basada en la clasificacion del preprocesamiento'''

#--- REVIsAMOS LA CALIDAD DE LOS COMENTARIOS (evitaremos un preprocesaminto si procede)
import pandas as pd
import re

# Ver un fragmento antes de extraer fechas y usuarios
print("Antes de extraer fechas y usuarios:")
print(df['Comment'].head())

# Extraer las fechas y crear una nueva columna 'Date'
df['Date'] = df['Comment'].apply(lambda x: re.findall(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', x))

# Convertir la nueva columna 'Date' a datetime (se asume que las fechas están en formato correcto)
df['Date'] = pd.to_datetime(df['Date'].str[0], errors='coerce')

# Extraer los nombres de los usuarios que están antes de "wrote:"
df['User'] = df['Comment'].apply(lambda x: re.findall(r'([a-zA-Z]+(?: [a-zA-Z]+)*)(?= wrote:)', x))

# Tomamos solo el primer nombre encontrado (en caso de que haya más de un nombre)
df['User'] = df['User'].apply(lambda x: x[0] if len(x) > 0 else None)

# Ver un fragmento después de extraer fechas, usuarios y limpiar el texto
print("\nDespués de extraer y limpiar las fechas y usuarios:")
print(df[['Comment', 'Date', 'User']].head())

# Eliminamos de la columna Comments cositas que no aportan valor en dicha columna

# Ajustar el máximo de caracteres a mostrar por columna y ver el total del contenido para identificar patrones y eliminar ocntenido inecesario
pd.set_option('display.max_colwidth', None)
print(df[['Comment']].head())

import re

# Patrones mejorados
patterns = [
        r'bq\.',  # "bq."
        r'on \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}, [a-zA-Z\s]+ wrote:', # Elimina el patron "on fecha, usuario wrote:""
        r'<https://reviews.apache.org/[^\s]+>', # Elimina cualquier patrón tras "<https://reviews.apache.org/"
        r'src/[\w/.-]+',  # Esto captura la ruta del archivo
        r',?\s*line\s*\d+\s*>',  # Esto captura "line" con espacios alrededor y el número de línea
        r'\s*>\s*', # Nuevo patrón para eliminar los ">" respetando los espacios entre palabras

]

# Función para limpiar el texto
def clean_text(text):
    for pattern in patterns:
        text = re.sub(pattern, ' ', text)
    return text.strip()

# Aplicamos la limpieza a la columna de comentarios
df['Cleaned_Comment'] = df['Comment'].apply(clean_text)

# Verificamos el resultado después de aplicar la limpieza
print(df[['Comment','Cleaned_Comment']])

# Asumiendo que df es tu DataFrame con las columnas 'Comment' y 'Cleaned_Comment'
# Exportar a Excel
df[['Comment', 'Cleaned_Comment']].to_excel('data/cleaned_comments_output.xlsx', index=False)

print("El archivo Excel se ha guardado correctamente.")

print("Nombres de las columnas:", df.columns.tolist())

