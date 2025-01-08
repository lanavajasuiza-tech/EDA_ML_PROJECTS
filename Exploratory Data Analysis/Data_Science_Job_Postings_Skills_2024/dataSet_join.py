# Juntamos los tres dataSet en uno en formato horizontal

import pandas as pd
import os

base_path = r'C:/Users/rportatil112/Documents/EDA_ML_PROJECTS/EDA/Data_Science_Job_Postings_Skills_2024/dataSet'

# Listar archivos en el directorio
print("Archivos disponibles en el directorio:")
print(os.listdir(base_path))

# Cargar los datasets desde el subdirectorio
df1 = pd.read_csv(os.path.join(base_path, 'job_postings.csv'))
df2 = pd.read_csv(os.path.join(base_path, 'job_skills.csv'))
df3 = pd.read_csv(os.path.join(base_path, 'job_summary.csv'))

# Mostrar las columnas de cada dataset
print("Columnas en df1:", df1.columns)
print("Columnas en df2:", df2.columns)
print("Columnas en df3:", df3.columns)

print("Ejemplo de valores en df1['job_link']:", df1['job_link'].head())
print("Ejemplo de valores en df2['job_link']:", df2['job_link'].head())
print("Ejemplo de valores en df3['job_link']:", df3['job_link'].head())

# Combinar los datasets en función de la columna 'job_link'
df_merged = df1.merge(df2, on='job_link').merge(df3, on='job_link')

# Guardar el dataset combinado
output_path = r'C:/Users/rportatil112/Documents/EDA_ML_PROJECTS/EDA/Data_Science_Job_Postings_Skills_2024/dataSet/Linkedin_Data_Jobs_2024.csv'
df_merged.to_csv(output_path, index=False)

print(f"Archivo combinado guardado en: {output_path}")

# Mostrar columnas de cada dataset
print("Columnas en df_merged:", df_merged.columns.tolist())

