import os
import pandas as pd

#### CARGAMOS EL DATASET ######

def load_csv(file_path):
    """
    Verifica si el archivo existe y lo carga como un DataFrame.

    Args:
        file_path (str): Ruta completa del archivo CSV.

    Returns:
        DataFrame: El DataFrame cargado.
        None: Si el archivo no existe.
    """
    # Verificar si el archivo existe
    if not os.path.exists(file_path):
        print("El archivo no se encontró. Verifica el path.")
        return None
    
    print("El archivo existe y está listo para ser cargado.")
    
    # Cargar el archivo
    df = pd.read_csv(file_path)
    print("Archivo cargado con éxito. Mostrando las primeras filas:")
    print(df.head())

    
    # Mostrar total de filas y columnas
    print(f"El DataFrame tiene {df.shape[0]} filas y {df.shape[1]} columnas.")
    
    return df

# Define el path del archivo
file_path = r"C:/Users/rportatil112/Documents/CURSO-DATA-SCIENCE/REDES_NEURONALES/PNL/6.Ejercicio_Reseñas_Amazon/1429_1.csv"

# Llama a la función
df = load_csv(file_path)

# Verificar si se cargó correctamente
if df is not None:
    print("El DataFrame está listo para usarse.")
else:
    print("Hubo un problema al cargar el archivo.")


#### ANALIZAMOS LOS DATOS ######

import pandas as pd

def analyze_dataframe(df):
    
    # Tipos de datos
    print("\nTipos de datos:")
    print(df.dtypes)
    
    # Valores NaN
    print("\nCantidad de valores nulos por columna:")
    print(df.isnull().sum())
    
    # Filas duplicadas
    print(f"\nNúmero de filas duplicadas: {df.duplicated().sum()}")
    
    # Dividir en categóricos y numéricos
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print("\nColumnas categóricas:")
    print(cat_columns)
    
    print("\nColumnas numéricas:")
    print(num_columns)

    # Mostrar total de filas y columnas
    print(f"El DataFrame tiene {df.shape[0]} filas y {df.shape[1]} columnas.")

  
    return cat_columns, num_columns


# Llama a la función para analizar tu DataFrame
cat_columns, num_columns = analyze_dataframe(df)

#-------- Exámino la información

# Contar valores únicos por cada columna
unique_values = df.nunique()

# Mostrar los resultados como una tabla ordenada
print("Valores únicos por columna:")
print(unique_values.sort_values(ascending=False))
'''Agregamos esta información a la tabla de Análsis.md y comentamos resultados'''

# Se procede a borrar las columnas irrelevantes tras análisis anterior documentado el motivo en Análsis.md
# Columnas a eliminar
columns_to_drop = [
    "id", "asins", "manufacturer", "reviews.id", 
    "reviews.sourceURLs", "reviews.userProvince", 
    "reviews.didPurchase", "reviews.userCity", 
    "reviews.userProvince", "reviews.id", 
    "reviews.dateAdded", "reviews.dateSeen"
]

# Eliminar columnas
df = df.drop(columns=columns_to_drop)

# Confirmar las nuevas dimensiones del DataFrame
print(f"Columnas eliminadas. Nuevas dimensiones del DataFrame: {df.shape}")
print(f"Columnas restantes: {df.columns.tolist()}")

# Rellenar nulos en la columna reviews.text, solo tiene un valor NaN
df['reviews.text'] = df['reviews.text'].fillna("Sin información")

#<----- Tratamos los NaN

# Hay solo 39 NaN en reviews.date, por lo que procedemos a aplicar una media y convertir las columnas de fecha en datetime

import pandas as pd

# Convertir reviews.date a datetime
df['reviews.date'] = pd.to_datetime(df['reviews.date'], errors='coerce')

# Calcular la media de las fechas (ignorando NaN)
mean_date = df['reviews.date'].dropna().mean()

# Rellenar valores nulos con la media calculada
df['reviews.date'] = df['reviews.date'].fillna(mean_date)

# Validar el resultado
print(f"Rango de fechas: {df['reviews.date'].min()} a {df['reviews.date'].max()}")
print(f"Media de fechas utilizada para rellenar: {mean_date}")
print(f"Valores nulos restantes en 'reviews.date': {df['reviews.date'].isnull().sum()}")


# <---- Hago un punto de situación

# Resumen de NaN, tipos de datos y valores únicos
import pandas as pd

def dataframe_summary(df):
    """
    Genera un resumen del estado actual de un DataFrame, incluyendo:
    - Nombre de las columnas
    - Tipos de datos
    - Cantidad de valores nulos
    - Porcentaje de valores nulos
    - Cantidad de valores únicos

    Args:
        df (pd.DataFrame): El DataFrame a analizar.

    Returns:
        pd.DataFrame: Resumen del DataFrame.
    """
    summary = pd.DataFrame({
        'Columna': df.columns,
        'Tipo de Dato': df.dtypes,
        'Valores NaN': df.isnull().sum(),
        'Porcentaje NaN': (df.isnull().sum() / len(df)) * 100,
        'Valores Únicos': df.nunique()
    }).reset_index(drop=True)
    
    # Ordenar por porcentaje de valores NaN
    summary = summary.sort_values(by='Porcentaje NaN', ascending=False)

    return summary

# Generar el resumen
summary = dataframe_summary(df)
print(summary)

# <---- Voy a tratar los NaN en la columna name, deducir el producto a partir de categories/brand y si no se puede
# rellenar con sin información

# Completar valores nulos de 'name' usando 'categories' o 'brand'
df['name'] = df['name'].fillna(
    df['categories'].str.split(">").str[-1].str.strip()  # Deducir nombre a partir de categorías
)

# Si aún quedan valores nulos, rellenarlos con un marcador genérico
df['name'] = df['name'].fillna("Producto desconocido")

# Validar resultados
print(f"Valores nulos restantes en 'name': {df['name'].isnull().sum()}")
print(df['name'].value_counts().head(10))  # Ver los nombres más frecuentes
'''
¡Perfecto! Vamos a implementar ambas estrategias para tratar los valores nulos en la columna name. Si no es posible deducir el nombre del producto usando otras columnas como categories o brand, utilizaremos un marcador genérico ("Producto desconocido").

Código para Tratar name
python
Copiar
Editar
# Completar valores nulos de 'name' usando 'categories' o 'brand'
df['name'] = df['name'].fillna(
    df['categories'].str.split(">").str[-1].str.strip()  # Deducir nombre a partir de categorías
)

# Si aún quedan valores nulos, rellenarlos con un marcador genérico
df['name'] = df['name'].fillna("Producto desconocido")

# Validar resultados
print(f"Valores nulos restantes en 'name': {df['name'].isnull().sum()}")
print(df['name'].value_counts().head(10))  # Ver los nombres más frecuentes
Explicación del Código
Intentar deducir el nombre del producto:

Usa categories para extraer la última categoría como posible nombre del producto.
Por ejemplo, si categories es "Electronics > Tablets", se usará "Tablets".
Rellenar con un marcador genérico:

Para las filas donde no sea posible deducir el nombre, se asigna "Producto desconocido".
Validación:

Se verifica que no queden valores nulos en name.
Se imprimen los nombres más frecuentes para validar los resultados.'''

# VOY A VERIFICAR QUE SE HA HECHO BIEN

# Paso 1: Crear una columna auxiliar para registrar el origen de los datos
df['name_origen'] = 'original'  # Por defecto, marcar como original

# Paso 2: Marcar las filas donde el nombre se deduce de categorías
df.loc[df['name'].isnull() & df['categories'].notnull(), 'name_origen'] = 'deducido de categories'
df.loc[df['name_origen'] == 'deducido de categories', 'name'] = df['categories'].str.split(">").str[-1].str.strip()

# Paso 3: Marcar las filas donde el nombre se rellena con el marcador genérico
df.loc[df['name'].isnull(), 'name_origen'] = 'relleno genérico'
df['name'] = df['name'].fillna("Producto desconocido")

# Paso 4: Verificar la distribución del origen de los datos
print("Distribución del origen de 'name':")
print(df['name_origen'].value_counts())

# Paso 5: Ver ejemplos de nombres deducidos de categorías
print("\nEjemplos de nombres deducidos de 'categories':")
deducidos = df[df['name_origen'] == 'deducido de categories']
print(deducidos[['name', 'categories']].head(10))

# Paso 6: Ver ejemplos de nombres rellenados con el marcador genérico
print("\nEjemplos de nombres rellenados con el marcador genérico:")
rellenos_genericos = df[df['name_origen'] == 'relleno genérico']
print(rellenos_genericos[['name', 'categories']].head(10))

# Paso 7: Mostrar una muestra aleatoria de los cambios realizados
print("\nEjemplo de cambios realizados (muestra aleatoria):")
df_sample = df.sample(10)
print(df_sample[['name', 'categories', 'name_origen']])

# Verificamos que ya no tenemos NaN en name
# Generar el resumen
summary = dataframe_summary(df)
print(summary)

# Hago el tratamiento del resto de columnas con NaN

df['reviews.doRecommend'] = df['reviews.doRecommend'].fillna("No especificado") # Rellenar 'reviews.doRecommend' con "No especificado" ya que es una decisión del usuario y faltan pocos valores.
df['reviews.numHelpful'] = df['reviews.numHelpful'].fillna(0) # Rellenar 'reviews.numHelpful' con 0, asumiendo que NaN significa que no hubo votos útiles.
df['reviews.rating'] = df['reviews.rating'].fillna(df['reviews.rating'].median()) # Rellenar 'reviews.rating' con la mediana para mantener la tendencia general de las calificaciones.
df['reviews.username'] = df['reviews.username'].fillna("Usuario desconocido") # Rellenar 'reviews.username' con "Usuario desconocido" para preservar las filas sin un nombre específico.
df['reviews.title'] = df['reviews.title'].fillna("Sin título") # Rellenar 'reviews.title' con "Sin título" para evitar filas sin un encabezado en la reseña.
print("Valores nulos restantes:") # Verificar que no queden valores nulos
print(df.isnull().sum())


# <---- Reviso caracteres extraños y unifico los textos (mayus/minus)

import re
# Identificar columnas de texto
text_columns = df.select_dtypes(include=["object"]).columns

# Revisar caracteres únicos en cada columna de texto
for col in text_columns:
    text_data = " ".join(df[col].dropna().astype(str))  # Combina todos los textos en un único string
    unique_chars = sorted(set(text_data))  # Extrae caracteres únicos
    print(f"Caracteres únicos en la columna '{col}': {unique_chars[:100]}")  # Muestra los primeros 100 caracteres
    print(f"Total de caracteres únicos: {len(unique_chars)}\n")

# Veo que hay emojis, pruebo la librería emoji para analizar el "sentimiento"
import emoji
import re

# Función para extraer emojis de un texto
def extract_emojis(text):
    emoji_pattern = re.compile("[" 
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"  # Símbolos y pictogramas
        u"\U0001F680-\U0001F6FF"  # Transporte y símbolos
        u"\U0001F1E0-\U0001F1FF"  # Banderas (iOS)
        u"\U00002700-\U000027BF"  # Otros símbolos
        u"\U000024C2-\U0001F251"  # Misceláneos
        "]+", flags=re.UNICODE)
    return emoji_pattern.findall(text) if isinstance(text, str) else []

# Extraer emojis de todas las columnas de texto
def extract_all_emojis_with_count(df):
    text_columns = df.select_dtypes(include=["object"]).columns  # Identificar columnas de texto
    emoji_counts = {}

    for col in text_columns:
        print(f"Procesando columna '{col}'...")
        emojis_in_col = df[col].dropna().apply(extract_emojis).explode().dropna()
        for emoji_char in emojis_in_col:
            emoji_counts[emoji_char] = emoji_counts.get(emoji_char, 0) + 1
    
    return emoji_counts

# Obtener emojis únicos y sus conteos
emoji_counts = extract_all_emojis_with_count(df)

# Clasificar emojis automáticamente
def classify_emoji(emoji_char):
    description = emoji.demojize(emoji_char)
    if "heart" in description or "smile" in description:
        return "positivo"
    elif "angry" in description or "cry" in description:
        return "negativo"
    else:
        return "neutral"

# Clasificación de emojis únicos
emoji_sentiment = {e: classify_emoji(e) for e in emoji_counts.keys()}

# Mostrar clasificación con conteo
print("\nClasificación y conteo de emojis en todo el dataset:")
for e, sentiment in emoji_sentiment.items():
    print(f"{e}: {sentiment}, aparece {emoji_counts[e]} veces")


# Para practicar con esta librería me decido a limpiar lo que no nos ineresa y mantener el resto de emojis,
# habrá que tener en cuenta esto en la tokenización posterior, nos dará una salida con el conteo de lo realizado

import re
from collections import Counter

# Función para limpiar texto, contar caracteres eliminados y unificar a minúsculas
def clean_text_with_stats(text):
    # Expresión regular para extraer emojis
    emoji_pattern = re.compile("[" 
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"  # Símbolos y pictogramas
        u"\U0001F680-\U0001F6FF"  # Transporte y símbolos
        u"\U0001F1E0-\U0001F1FF"  # Banderas (iOS)
        u"\U00002700-\U000027BF"  # Otros símbolos
        u"\U000024C2-\U0001F251"  # Misceláneos
        "]+", flags=re.UNICODE)

    if not isinstance(text, str):
        return "", 0, 0  # Manejo de valores NaN

    # Extraer emojis
    emojis = emoji_pattern.findall(text)

    # Contar caracteres eliminados
    original_length = len(text)
    clean_text = re.sub(r"[^\w\s]", "", text)  # Elimina caracteres especiales
    cleaned_length = len(clean_text)

    # Convertir a minúsculas y añadir emojis preservados
    clean_text = clean_text.strip().lower() + " " + " ".join(emojis)

    # Calcular estadísticos
    removed_count = original_length - cleaned_length - len(emojis)  # Caracteres eliminados
    preserved_count = len(emojis)  # Caracteres preservados

    return clean_text, removed_count, preserved_count


# LLAMAMOS A LA FUNCIÓN ANTERIOR

# Crear columnas limpias y estadísticas
text_columns = df.select_dtypes(include=["object"]).columns  # Identificar columnas de texto

# Diccionario para almacenar estadísticas
cleaning_stats = Counter()

for col in text_columns:
    df[f'{col}_cleaned'], removed, preserved = zip(*df[col].dropna().apply(clean_text_with_stats))
    cleaning_stats[col] = {
        "total_removed": sum(removed),
        "total_preserved": sum(preserved)
    }

# Mostrar resultados de limpieza
print("Estadísticas de limpieza:")
for col, stats in cleaning_stats.items():
    print(f"Columna '{col}': {stats['total_removed']} caracteres eliminados, {stats['total_preserved']} emojis preservados.")

### TOKENIZAMOS ####
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import emoji

# Descargar recursos necesarios
nltk.download('punkt')
nltk.download('stopwords')

# Tokenizar el texto y preservar emojis
def tokenize_with_emojis(text):
    if not isinstance(text, str):
        return []
    # Extraer tokens
    tokens = word_tokenize(text.lower())
    # Filtrar stopwords
    stop_words = set(stopwords.words('english'))  # Cambiar a 'spanish' si es en español
    tokens = [t for t in tokens if t not in stop_words]
    # Añadir emojis si los hay
    emojis = emoji.emoji_list(text)  # Devuelve emojis presentes en el texto
    tokens.extend([e['emoji'] for e in emojis])
    return tokens

# Aplicar tokenización al texto
df['reviews.text_tokens'] = df['reviews.text'].apply(tokenize_with_emojis)

# Verificar los resultados
print(df[['reviews.text', 'reviews.text_tokens']].head())

#### APLICAMOS EL ANÁLISIS DE SENTIMIENTO ####

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Inicializar el analizador de VADER
vader_analyzer = SentimentIntensityAnalyzer()

# Función para calcular el puntaje de sentimiento
def sentiment(document):
    score = vader_analyzer.polarity_scores(document)
    return score  # Retorna el diccionario completo con 'neg', 'neu', 'pos', 'compound'

# Aplicar la función a la columna 'reviews.text'
df['sentiment_scores'] = df['reviews.text'].apply(sentiment)

# Crear una nueva columna con el puntaje 'compound'
df['compound_score'] = df['sentiment_scores'].apply(lambda x: x['compound'])

# Clasificar en positivo, negativo o neutral
def classify_sentiment(compound):
    if compound >= 0.05:
        return 'positivo'
    elif compound <= -0.05:
        return 'negativo'
    else:
        return 'neutral'

# Crear la columna 'sentiment_class'
df['sentiment_class'] = df['compound_score'].apply(classify_sentiment)

# Verificar los resultados
print(df[['reviews.text', 'compound_score', 'sentiment_class']].head())

### CONTESTAMOS LAS PREGUNTAS Y VISUALIZAMOS

import matplotlib.pyplot as plt

# 1.-¿Qué productos deben ser mantenidos?

# Preparar los datos truncando nombres largos
positive_products = df[df['sentiment_class'] == 'positivo'].groupby('name').size().sort_values(ascending=False).head(10)
positive_products.index = positive_products.index.map(lambda x: x[:30] + '...' if len(x) > 30 else x)  # Truncar nombres largos

# Visualización ajustada
plt.figure(figsize=(6, 8))  # Tamaño más pequeño
positive_products.plot(kind='barh', color='green', alpha=0.7)
plt.title('Top 10 Productos con Más Reseñas Positivas', fontsize=12)
plt.xlabel('Número de Reseñas Positivas', fontsize=10)
plt.ylabel('Producto', fontsize=10)
plt.xticks(fontsize=9)
plt.yticks(fontsize=8)
plt.gca().invert_yaxis()  # Invertir el eje y para que el mejor producto esté arriba
plt.tight_layout()  # Ajustar los márgenes para evitar superposición
plt.show()
'''
Enfoque: Productos con alto número de reseñas positivas.
Gráfico: Barras con los productos más populares (reseñas positivas).
'''



# 2.-¿Qué productos deben ser descartados?

# Calcular reseñas negativas por producto
negative_products = df[df['sentiment_class'] == 'negativo'].groupby('name').size().sort_values(ascending=False).head(10)

# Truncar nombres largos para mejorar la legibilidad
negative_products.index = negative_products.index.map(lambda x: x[:30] + '...' if len(x) > 30 else x)

# Visualización ajustada
plt.figure(figsize=(6, 8))  # Tamaño más pequeño para una visualización compacta
negative_products.plot(kind='barh', color='red', alpha=0.7)
plt.title('Top 10 Productos con Más Reseñas Negativas', fontsize=12)
plt.xlabel('Número de Reseñas Negativas', fontsize=10)
plt.ylabel('Producto', fontsize=10)
plt.xticks(fontsize=9)  # Ajustar tamaño de las etiquetas del eje x
plt.yticks(fontsize=8)  # Ajustar tamaño de las etiquetas del eje y
plt.gca().invert_yaxis()  # Invertir el eje y para que el peor producto esté arriba
plt.tight_layout()  # Ajustar márgenes para evitar superposiciones
plt.show()
'''
Enfoque: Productos con alto número de reseñas negativas.
Gráfico: Barras con los productos más criticados.
'''

# 3.-¿Qué productos son basura?

# Calcular proporción de reseñas negativas
product_summary = df.groupby('name').agg(
    total_reviews=('sentiment_class', 'count'),
    negative_reviews=('sentiment_class', lambda x: (x == 'negativo').sum())
).reset_index()
product_summary['negative_percentage'] = (product_summary['negative_reviews'] / product_summary['total_reviews']) * 100

# Seleccionar los productos con mayor proporción de reseñas negativas
top_negative_percentage = product_summary.sort_values(by='negative_percentage', ascending=False).head(10)

# Truncar nombres largos de productos para la visualización
top_negative_percentage['name'] = top_negative_percentage['name'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)

# Visualización ajustada
plt.figure(figsize=(8, 5))  # Tamaño ligeramente más grande para acomodar nombres largos
plt.barh(top_negative_percentage['name'], top_negative_percentage['negative_percentage'], color='darkred', alpha=0.7)
plt.title('Top 10 Productos con Mayor Proporción de Reseñas Negativas', fontsize=12)
plt.xlabel('Porcentaje de Reseñas Negativas (%)', fontsize=10)
plt.ylabel('Producto', fontsize=10)
plt.xticks(fontsize=9)  # Ajustar tamaño de las etiquetas del eje x
plt.yticks(fontsize=8)  # Ajustar tamaño de las etiquetas del eje y
plt.gca().invert_yaxis()  # Invertir el eje y para que el producto con mayor porcentaje esté arriba
plt.tight_layout()  # Ajustar márgenes para evitar superposiciones
plt.show()
'''
Enfoque: Productos con alta proporción de reseñas negativas respecto al total.
Gráfico: Barras mostrando el porcentaje de reseñas negativas por producto.
'''

# 4.-¿Qué producto debería ser recomendado al cliente?

# Calcular productos recomendados
recommended_products = df[df['sentiment_class'] == 'positivo'].groupby('name').agg(
    avg_rating=('reviews.rating', 'mean'),
    positive_reviews=('sentiment_class', 'count')
).reset_index()

# Truncar nombres largos de productos para mejorar la visualización en el gráfico
recommended_products['name'] = recommended_products['name'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)

# Ordenar por número de reseñas positivas
recommended_products = recommended_products.sort_values(by='positive_reviews', ascending=False).head(10)

# Visualización con barras
plt.figure(figsize=(10, 6))
plt.barh(recommended_products['name'], recommended_products['positive_reviews'], color='green', alpha=0.8)

# Títulos y etiquetas
plt.title('Top 10 Productos con Más Reseñas Positivas', fontsize=12)
plt.xlabel('Número de Reseñas Positivas', fontsize=10)
plt.ylabel('Productos', fontsize=10)
plt.gca().invert_yaxis()  # Invertir el eje Y para que el producto con más reseñas esté arriba

# Añadir etiquetas a las barras
for index, value in enumerate(recommended_products['positive_reviews']):
    plt.text(value + 1, index, str(value), fontsize=10, va='center')

# Ajustar diseño
plt.tight_layout()
plt.show()
'''
Enfoque: Producto con la mejor calificación promedio y mayor número de reseñas positivas.
Gráfico: Dispersión entre calificación promedio y reseñas positivas.
'''

# 5.-¿Cuáles son los mejores productos para los consumidores?

# Agrupación y cálculo de sentimiento promedio y calificación promedio
best_products = df.groupby('name').agg(
    avg_sentiment=('compound_score', 'mean'),
    avg_rating=('reviews.rating', 'mean')
).reset_index()

# Seleccionar los mejores productos por sentimiento promedio
top_best_products = best_products.sort_values(by='avg_sentiment', ascending=False).head(10)

# Truncar nombres largos de productos para mejorar la visualización en el gráfico
top_best_products['name'] = top_best_products['name'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)

# Visualización con barras horizontales
plt.figure(figsize=(10, 6))
plt.barh(top_best_products['name'], top_best_products['avg_sentiment'], color='green', alpha=0.7)

# Añadir etiquetas de los valores al final de cada barra
for index, value in enumerate(top_best_products['avg_sentiment']):
    plt.text(value + 0.01, index, f"{value:.2f}", fontsize=10, va='center')

# Títulos y etiquetas
plt.title('Top 10 Mejores Productos por Sentimiento Promedio', fontsize=14)
plt.xlabel('Sentimiento Promedio', fontsize=12)
plt.ylabel('Producto', fontsize=12)
plt.gca().invert_yaxis()  # Invertir el eje Y para mostrar los mejores productos arriba

# Ajustar diseño
plt.tight_layout()
plt.show()
'''
Enfoque: Productos con las calificaciones y sentimientos promedio más altos.
Gráfico: Barras mostrando calificación promedio.
'''

# 6.-¿Qué productos deberían ser planeados para el inventario del próximo invierno?

# Convertir la columna de fecha a datetime y filtrar meses de invierno
df['reviews.date'] = pd.to_datetime(df['reviews.date'], errors='coerce')
winter_reviews = df[df['reviews.date'].dt.month.isin([12, 1, 2])]

# Calcular número de reseñas por mes y producto
winter_trends_by_product = winter_reviews.groupby([winter_reviews['reviews.date'].dt.to_period('M'), 'name']).size().reset_index()
winter_trends_by_product.columns = ['Month', 'Product', 'Reviews']

# Seleccionar los 5 productos con más reseñas para hacer la visualización más clara
top_products = winter_trends_by_product.groupby('Product')['Reviews'].sum().nlargest(5).index
filtered_trends = winter_trends_by_product[winter_trends_by_product['Product'].isin(top_products)]

# Visualización con líneas para cada producto
plt.figure(figsize=(12, 7))
for product in top_products:
    product_data = filtered_trends[filtered_trends['Product'] == product]
    plt.plot(
        product_data['Month'].astype(str), 
        product_data['Reviews'], 
        marker='o', linestyle='-', label=product[:30] + '...' if len(product) > 30 else product, alpha=0.7
    )

# Títulos y etiquetas
plt.title('Tendencia de Reseñas en Invierno por Producto', fontsize=14)
plt.xlabel('Mes', fontsize=12)
plt.ylabel('Número de Reseñas', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.legend(title="Productos", fontsize=9, loc='upper left')
plt.grid(alpha=0.5)

# Ajustar diseño
plt.tight_layout()
plt.show()
'''
Enfoque: Productos con alta popularidad en reseñas durante meses de invierno.
Gráfico: Línea temporal mostrando tendencias de reseñas por producto.
'''

# 7.-¿Qué productos requieren publicidad?


# Ajustar el criterio de "pocas reseñas" a menos de 1000
promising_products = recommended_products[(recommended_products['positive_reviews'] < 1000) & 
                                          (recommended_products['avg_rating'] >= 4.5)]

# Verificar si ahora hay datos
print(promising_products.shape)

# Truncar nombres largos para una mejor visualización
promising_products['name'] = promising_products['name'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))
plt.barh(promising_products['name'], promising_products['avg_rating'], color='orange', alpha=0.7)

# Añadir etiquetas directamente sobre las barras
for index, value in enumerate(promising_products['avg_rating']):
    plt.text(value + 0.05, index, f"{value:.2f}", fontsize=10, va='center')

# Títulos y etiquetas
plt.title('Productos que Requieren Publicidad (Alta Calificación, Menos de 1000 Reseñas)', fontsize=14)
plt.xlabel('Calificación Promedio', fontsize=12)
plt.ylabel('Producto', fontsize=12)
plt.gca().invert_yaxis()  # Mostrar el producto con mejor calificación en la parte superior

# Mostrar el gráfico
plt.tight_layout()
plt.show()
'''
Enfoque: Productos con alta calificación promedio pero pocas reseñas.
Gráfico: Barras mostrando productos prometedores.
'''