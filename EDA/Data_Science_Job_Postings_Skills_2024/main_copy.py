"""
Author: Ana Ndongo
Date: 27 December 2024
Kaggle dataSet: https://www.kaggle.com/datasets/asaniczka/data-science-job-postings-and-skills
Description:

LinkedIn is a popular professional networking platform with millions of job postings across various industries.

This dataset provides a raw dump of data science-related job postings collected from LinkedIn. It includes information about job titles, companies, locations, search parameters, and other relevant details.

The main objective of this dataset is not only to provide insights into the data science job market and the skills required by professionals in this field but also to offer users an opportunity to practice their data cleaning skills.

By working with this dataset, users can gain hands-on experience in cleaning and preprocessing raw data, a critical skill for aspiring data scientists.
"""
import os
import sys
from utils.processing import DataLoader
from utils.analyzer import DataAnalyzer


#---------------- CARGAR DATASET -------------------#

# Detectar dinámicamente el directorio raíz del proyecto
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
print("Directorio raíz detectado dinámicamente:", project_root)


# Ruta simplificada al dataset
df_path = os.path.join(project_root, "dataSet")
print(f"Ruta del dataset: {df_path}")
df = "Linkedin_Data_Jobs_2024.csv"

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
    #analyzer.nan_summary()
else:
    print("\n--- No se pudo cargar el dataset. Análisis abortado ---")

    #---------------- PROCESAR LOS DATOS -------------------#

# Revisión rápida de los valores NaN
print(df.isna().sum())
print(df[df.isna().any(axis=1)])  # Filas que contienen valores NaN

df.dropna(subset=['job_skills', 'job_location'], inplace=True)

analyzer.nan_summary()

# Calcular la frecuencia de cada habilidad en la columna 'job_skills'
skills_count = df['job_skills'].value_counts()
print(skills_count)

# Separar las habilidades individuales
df['job_skills'] = df['job_skills'].str.split(', ')

# Aplanar la lista y contar las habilidades únicas
from itertools import chain
import pandas as pd

all_skills = list(chain.from_iterable(df['job_skills'].dropna()))
skills_count = pd.Series(all_skills).value_counts()
print(skills_count)

# Separar habilidades únicas (aparecen solo una vez) y habilidades comunes
unique_skills = skills_count[skills_count == 1]
common_skills = skills_count[skills_count > 1]
print(f"Número de habilidades únicas: {len(unique_skills)}")
print(f"Número de habilidades comunes: {len(common_skills)}")
'''La salida apunta a excesivas habilidades, vamos a intentar reducir'''


all_skills_cleaned = [skill.strip().lower() for skill in all_skills] # Limpiar y normalizar las habilidades
skills_count_cleaned = pd.Series(all_skills_cleaned).value_counts() # Contar las habilidades después de la limpieza

# Separar habilidades únicas y comunes nuevamente
unique_skills_cleaned = skills_count_cleaned[skills_count_cleaned == 1]
common_skills_cleaned = skills_count_cleaned[skills_count_cleaned > 1]
print(f"Número de habilidades únicas (limpias): {len(unique_skills_cleaned)}")
print(f"Número de habilidades comunes (limpias): {len(common_skills_cleaned)}")
'''Tras normalizar en minusculas y elminar espacios, se reduce la cantidad pero sigue siendo excesivo'''

import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Cargar el modelo preentrenado de spaCy
nlp = spacy.load('en_core_web_md')

# Limitar a las primeras 5000 habilidades
limited_skills = skills_count_cleaned.index[:5000]

# Convertir habilidades a vectores
skills_vectors = np.array([nlp(skill).vector for skill in limited_skills])

# Calcular similitudes entre todos los vectores
similarity_matrix = cosine_similarity(skills_vectors)

# Identificar pares con similitud alta
similarities = {}
for i in range(len(limited_skills)):
    for j in range(i + 1, len(limited_skills)):
        if similarity_matrix[i, j] > 0.8:
            similarities[(limited_skills[i], limited_skills[j])] = similarity_matrix[i, j]

# Mostrar resultados
print(f"Pares similares encontrados: {len(similarities)}")
for pair, sim in list(similarities.items())[:10]:
    print(f"{pair}: {sim:.2f}")
'''Ha encontrado 107687 pares similares, python + skill, con una similitud perfecta de 1.00,
es decir, que python es esencial acompañado de otra tecnología haremos un dicionari manual para 
clasificar  en tres categorías: "Lenguajes de programación":, "Infraestructura y sistemas": y Análisis de datos y machine learning":'''

# Diccionario actualizado de categorías
categories = {
    "Gestión y proyectos": ["teamwork", "teamwork", "problem solving", "project management", "scrum", "agile", "kanban", "product owner", "team lead"],
    "Infraestructura y sistemas": ["data engineering", "data warehousing", "spark", "hadoop", "aws", "azure", "docker", "linux", "git"],
    "Análisis de datos y machine learning": ["tableau", "data management", "data modeling", "data science", "data visualization", "machine learning", "ai", "data analysis", "statistics", "deep learning", "time series"],
    "Lenguajes de programación": ["python", "java", "sql", "r", "c++", "javascript", "scala", "ruby"]
}

# Crear un diccionario para almacenar las habilidades clasificadas
classified_skills = {cat: [] for cat in categories}
unclassified_skills = []

# Clasificar las habilidades con coincidencias exactas
for skill in skills_count_cleaned.index:
    found = False
    for category, keywords in categories.items():
        if any(skill.lower() == keyword for keyword in keywords):
            classified_skills[category].append(skill)
            found = True
    if not found:
        unclassified_skills.append(skill)

# Mostrar las habilidades clasificadas
for category, skills in classified_skills.items():
    print(f"\n{category} ({len(skills)} habilidades):")
    print(skills[:10])

# Mostrar las habilidades no clasificadas
print(f"\nHabilidades no clasificadas ({len(unclassified_skills)}):")
print(unclassified_skills[:100])

# vamos a importar estas 64925 habilidades para tratarlas en un scritp aparte y ampliar las categorias en este script
import pandas as pd

# Exportar las habilidades no clasificadas a un archivo CSV
unclassified_df = pd.DataFrame(unclassified_skills, columns=["Habilidades no clasificadas"])
unclassified_df.to_csv("dataSet/unclassified_skills.csv", index=False)

print("Archivo 'unclassified_skills.csv' exportado correctamente.")
'''Este archivo se merece otro análisis exhaustivo de como se busca / solicitan habilidades técnicas,
para este análisis de ejercicio vamos a seguir con lo que tenemos hasta ahora '''

    #---------------- VISUALIZAR LOS DATOS -------------------#

#graficamos para buscar un patrón a partir de un heatmap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter

# Cargar el archivo CSV
skills_df = pd.read_csv("dataSet/unclassified_skills.csv")

# Crear combinaciones de habilidades
combinations_list = []
for skills in skills_df["Habilidades no clasificadas"].dropna():
    skills_split = skills.split(", ")
    combinations_list.extend(combinations(skills_split, 2))

# Contar las combinaciones más frecuentes
combinations_counter = Counter(combinations_list)

# Convertir el contador a un DataFrame
combinations_df = pd.DataFrame(combinations_counter.items(), columns=["Combination", "Frequency"])
combinations_df[["Skill1", "Skill2"]] = pd.DataFrame(combinations_df["Combination"].tolist(), index=combinations_df.index)
combinations_df = combinations_df.drop(columns=["Combination"])

# Crear una matriz para el heatmap
pivot_table = combinations_df.pivot("Skill1", "Skill2", "Frequency").fillna(0)

# Dibujar el heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(pivot_table, cmap="Blues", linewidths=0.5)
plt.title("Mapa de Calor: Combinaciones de Habilidades Más Frecuentes")
plt.show()
