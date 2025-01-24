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

# Se vuelve a verificar la consistencia de las columnas existentes
print("Nombres de las columnas:", df.columns.tolist())

# --- VISUALIZAMOS LA INFORMACION HASTA AHORA ---

import matplotlib.pyplot as plt
import seaborn as sns

# Crear la figura y los subgráficos
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# --- 1. DISTRIBUCIÓN DE LOS SENTIMIENTOS ---
sentiment_counts = df['Sentiment_category'].value_counts()
sentiment_counts.plot(kind='bar', color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'], ax=axes[0, 0])
axes[0, 0].set_title('Distribución de los Sentimientos', fontsize=14)
axes[0, 0].set_xlabel('Categoría de Sentimiento', fontsize=12)
axes[0, 0].set_ylabel('Número de Comentarios', fontsize=12)
axes[0, 0].tick_params(axis='x', rotation=45)

# --- 2. FRECUENCIA DE COMENTARIOS POR FECHA ---
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
comments_by_date = df.groupby(df['Date'].dt.date).size()
comments_by_date.plot(kind='line', color='teal', marker='o', ax=axes[0, 1])
axes[0, 1].set_title('Frecuencia de Comentarios por Fecha', fontsize=14)
axes[0, 1].set_xlabel('Fecha', fontsize=12)
axes[0, 1].set_ylabel('Número de Comentarios', fontsize=12)
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True)

# --- 3. COMENTARIOS POR USUARIO Y SENTIMIENTO (BARRAS APILADAS) ---
user_comments = df.groupby('User')['Sentiment_category'].value_counts().unstack().fillna(0)
user_comments.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='tab20c', ax=axes[1, 0])
axes[1, 0].set_title('Comentarios por Usuario y Sentimiento', fontsize=14)
axes[1, 0].set_xlabel('Usuario', fontsize=12)
axes[1, 0].set_ylabel('Número de Comentarios', fontsize=12)
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].legend(title='Sentimiento', bbox_to_anchor=(1.05, 1), loc='upper left')

# --- 4. MAPA DE CALOR DE COMENTARIOS POR USUARIO Y SENTIMIENTO ---
user_sentiment_matrix = pd.crosstab(df['User'], df['Sentiment_category'])
sns.heatmap(user_sentiment_matrix, annot=True, fmt='d', cmap='coolwarm', cbar=False, ax=axes[1, 1])
axes[1, 1].set_title('Mapa de Calor de Comentarios por Usuario y Sentimiento', fontsize=14)
axes[1, 1].set_xlabel('Sentimiento', fontsize=12)
axes[1, 1].set_ylabel('Usuario', fontsize=12)

# Ajustar el layout para que no se sobrepongan los elementos
plt.tight_layout()

# Mostrar todos los gráficos
plt.show()


#### APLICAMOS STANZA y LDA ###
import stanza
import pandas as pd
from decorators.timing import timeit  # Decorador importado correctamente

# Inicializar Stanza
stanza.download('en')  # Solo la primera vez
nlp = stanza.Pipeline('en', processors='tokenize,lemma')  # Reducir procesadores si es necesario

# Función para procesar con Stanza
@timeit
def process_with_stanza(text):
    if not text or pd.isna(text):  # Manejar valores vacíos o nulos
        return ""
    doc = nlp(text)
    lemmatized_text = " ".join([word.lemma for sent in doc.sentences for word in sent.words])
    return lemmatized_text
'''
# Crear DataFrame de ejemplo (para verificar que estanza funciona)
df = pd.DataFrame({'Cleaned_Comment': ["This is a test.", "Stanza works well!", "Lemmatization is useful.", None]})

# Aplicar la función con Stanza
df['Lemmatized_Comment'] = df['Cleaned_Comment'].apply(process_with_stanza)

# Verificar los resultados
print(df)
'''
# Cargar el dataset desde el archivo Excel
file_path = 'data/cleaned_comments_output.xlsx'  # Ruta del archivo
df = pd.read_excel(file_path)

# Verificar que las columnas existen
print("Columnas disponibles:", df.columns.tolist())

# Aplicar la lematización a la columna 'Cleaned_Comment'
df['Lemmatized_Comment'] = df['Cleaned_Comment'].apply(process_with_stanza)

# Verificar los resultados (opcional)
print(df[['Cleaned_Comment', 'Lemmatized_Comment']].head())

# Exportar el DataFrame actualizado al mismo archivo
output_path = 'data/cleaned_comments_output.xlsx'  # Archivo original
df.to_excel(output_path, index=False)
print(f"Archivo actualizado con éxito: {output_path}")


# Se vuelve a verificar la consistencia de las columnas existentes
print("Nombres de las columnas:", df.columns.tolist())

#### SE PROCEDE A HACER EL ANÁLISIS DE SENTIMIENTO ####

# Se incluye en el pipeline de Stanza incluye el procesador sentiment
nlp = stanza.Pipeline('en', processors='tokenize,lemma,sentiment')

@timeit
def analyze_sentiment_with_stanza(text):
    """
    Analiza el sentimiento del texto usando Stanza.
    Devuelve un puntaje promedio de sentimiento (0: Negativo, 1: Neutral, 2: Positivo).
    """
    if not text or pd.isna(text):  # Manejar valores vacíos
        return 0  # Neutral
    doc = nlp(text)
    sentiment_scores = [sentence.sentiment for sentence in doc.sentences]
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    return avg_sentiment

# Aplicar el análisis de sentimientos a la columna 'Lemmatized_Comment'
df['Sentiment_Score'] = df['Lemmatized_Comment'].apply(analyze_sentiment_with_stanza)

# Categorizar sentimientos
def categorize_sentiment(score):
    if score >= 1.5:
        return "Positive"
    elif score >= 0.5:
        return "Neutral"
    else:
        return "Negative"

df['Sentiment_Category'] = df['Sentiment_Score'].apply(categorize_sentiment)

# Verificar los resultados
print(df[['Lemmatized_Comment', 'Sentiment_Score', 'Sentiment_Category']].head())

df.to_excel('data/sentiment_analysis_output.xlsx', index=False)
print("Archivo con análisis de sentimientos exportado con éxito.")

# Combinamos el resultado de sentiment_analysys_output.xlsx con el original de Kaggle para comparar resultados

import pandas as pd

# Leer los archivos
df1 = pd.read_excel('data/ZooKeeper_Project_Dataset.xlsx')
df2 = pd.read_excel('data/sentiment_analysis_output.xlsx')

# Ver los nombres de las columnas
print(df1.columns)
print(df2.columns)

# Concatenar los DataFrames
df_concatenado = pd.concat([df1, df2], axis=1)
print(df_concatenado.columns) 

# Guardar el resultado
df_concatenado.to_excel('data/Análisis_final.xlsx', index=False)

#### VISUALIZAMOS RESULTADOS STANZA ####

import matplotlib.pyplot as plt
import seaborn as sns

# ... (tu código para cargar los datos y crear los DataFrames) ...

# Crear una figura con 1 fila y 2 columnas
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Gráfico 1: Distribución de Sentimientos
sentiment_counts = df_concatenado['Sentiment_Category'].value_counts()
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'], ax=axes[0])
axes[0].set_title('Distribución de Sentimientos')

# Gráfico 2: Histograma de Puntajes de Sentimiento
sns.histplot(df_concatenado['Sentiment_Score'], bins=20, kde=True, ax=axes[1])
axes[1].set_title('Distribución de Puntajes de Sentimiento')

# Ajustar el espaciado entre subplots
plt.tight_layout()
plt.show()