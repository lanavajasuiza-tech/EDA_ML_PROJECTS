# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 09:29:29 2024

@author: rportatil112
"""

#######################################
# Analisis de Componentes Principales #
#######################################

import os
import pandas as pd
import numpy as np

os.chdir(r"C:\Users\rportatil112\Documents\CURSO-DATA-SCIENCE\MACHINE_LEARNING_&_INTELIGENCIA_ARTIFICIAL\Curso_ML_Laner\17-Modelos\Modelos_Regresion")
df = pd.read_csv("hormigon.csv")

# Aquí vemos que tenemos 9 columnas por lo tanto serían 9 categorías?
df.describe()
print(df.dtypes) # Revisar tipos de datos para descartar númericos de categóricos

# Muestro a que se refieren los valores de cada columna
print("Títulos de las columnas:", df.columns.tolist())


df.isnull().sum() # Comprobamos que no hay valores perdidos.
df.isnull().any().any() # Con esta funcion nos vemos si existen (TRUE) o no (FALSE) datos perdidos
duplicados = df.duplicated().sum() # Verificar duplicados
print(f"Número de filas duplicadas: {duplicados}")
#--> hay 25 duplicados

# Analizo el porqué de los duplicados verifcando los duplicados exactos
duplicados_exactos = df[df.duplicated()]
print(duplicados_exactos)

# Verificar duplicados basados en algunas columnas
duplicados_parciales = df[df.duplicated(subset=['cement', 'slag', 'ash'])]
print(duplicados_parciales)
# son duplicados númeriocos no tiene sentido elmnarlos

# Identificar posibles valores anómalos
for columna in df.select_dtypes(include=['float64', 'int64']).columns:
    print(f"{columna}: Mín={df[columna].min()}, Máx={df[columna].max()}")
'''
Aquí estamos observando los tipos de valores y cuales son los valores míni,os y máximos,
para ver si hay alguna anomalía o valor extraño por ejemplo un datos disparatado en números
'''

correlacion = df.corr() # Calcular la matriz de correlación, un previo de los datos para luego comparar
print(correlacion['water'])


# Separamos la variable dependiente ("y") de las explicativas ("X").

from sklearn.decomposition import PCA

y=df['water'] #defino en una nueva variable (y)la variable objetivo es un standar
X = df.drop(['water'], axis = 1) # devino las variables 'predictoras' y elmino la objetivo

pca = PCA(n_components=5, svd_solver='full') # elijo 5 variables para el PCA , el otro parámetro es un standard de optimización
pca.fit(X) #no es necesario incluir la variable y ya que es la objetivo

df_reducido = pca.transform(X) # genera un dataSet de tipo Array NumPy (no vamos a poder visualizar nada)

# Convertir el array NumPy a un DataFrame
df_reducido = pd.DataFrame(df_reducido, columns=[f'PC{i+1}' for i in range(df_reducido.shape[1])])
print(df_reducido.head())
'''aquí podemos observar que efectivamente 
ahora renemso 5 categorías en lugar de 8
'''

# Muestro a que se refieren los valores de cada columna
print("Títulos de las columnas:", df_reducido.columns.tolist())
'''
"Parece" que hemos perdido los títulos de las columnas pero en realidad cada componente principal 
es una combinación lineal (ponderada) de las columnas originales y están diseñadas para capturar 
la mayor cantidad de variabilidad en los datos.
'''

# Imprimir los pesos de cada variable original en los componentes principales
print("Pesos de las variables originales en cada componente principal:")
pd.DataFrame(
    pca.components_,   # La matriz de componentes principales (pesos de las variables originales)
    columns=X.columns, # Los nombres de las columnas originales (cement, slag, etc.)
    index=[f'PC{i+1}' for i in range(pca.n_components_)]  # Nombres de los componentes principales (PC1, PC2, ...)
)

# Antes
print("Varianza explicada por cada componente:", pca.explained_variance_ratio_)
# Después
print("Varianza acumulada:", pca.explained_variance_ratio_.cumsum())


#A partir de aquí tenemos datos optimizados y podemos utilizar un modelo que nos interese

