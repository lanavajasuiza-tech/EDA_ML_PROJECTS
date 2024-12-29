"""
This dataset contains information about cryptocurrency prices, market capitalization, and other metrics. The data is collected from CoinMarketCap (
https://coinmarketcap.com/), a popular website that tracks cryptocurrency prices.

This dataset can be used to:

Analyze the price trends of different cryptocurrencies.
Compare the market capitalization of different cryptocurrencies. -
Look at the circulating supply of different cryptocurrencies.
Analyze the trading volume of different cryptocurrencies.
Look at the volatility of different cryptocurrencies.
Compare the performance of different cryptocurrencies against each other or against a benchmark index.
Identify correlations between different cryptocurrency prices.
Use the data to build models to predict future prices or other trends.

+Info: https://www.kaggle.com/datasets/harshalhonde/coinmarketcap-cryptocurrency-dataset-2023
"""


import sys
import os
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
df = "currencies_data_Kaggle_2023_unique.csv"

#---------------- CARGAR Y ANALIZAR LOS DATOS -------------------#
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
    analyzer.missing_values_analysis() # Tarda entre 7 y 10 minutos, paciencia ...
    analyzer.data_types_analysis()
else:
    print("\n--- No se pudo cargar el dataset. Análisis abortado ---")

#---------------- PROCESAR LOS DATOS -------------------#

'''
Vamos a tratar las fechas, los NaN: y los valores categóricos
'''

# Por lo pronto, nos cargamos name.1 que está duplicada

if 'name.1' in df.columns:
    df.drop(columns=['name.1'], inplace=True)
    print("Columna 'name.1' eliminada.")
    analyzer.data_types_analysis()


# Veamos que columnas tienen Nan
nan_por_columna = df.isnull().sum()
print(nan_por_columna[nan_por_columna > 0])
'''La columna maxSupply contiene todos los NaN
y se debe a que no hay datos así que lo vamos a rellenar con 0'''

df.fillna(0, inplace=True)
print(f"Valores NaN restantes: {df.isnull().sum().sum()}") # confirmamos que ya no hay NaN
analyzer.missing_values_analysis()


# Convetiremos las fechas a formato datetime y preparamos para trabajar como serie temporal

import pandas as pd

# Se convierten las columnas de fechas a formato datetime
df['lastUpdated'] = pd.to_datetime(df['lastUpdated'], errors='coerce')
df['dateAdded'] = pd.to_datetime(df['dateAdded'], errors='coerce')
analyzer.overview()

# Se crea un índice temporal sin eliminar la columna dateAdded (por si queremos trabajar una serie temporal en algún momento)
df.set_index('dateAdded', inplace=True , drop=False)
print(df.index)


# Ordenar el DataFrame por el índice (dateAdded)
df.sort_index(inplace=True)
print(df.index.is_monotonic_increasing)  # Debe devolver True si está ordenado


# Crear columnas derivadas de 'dateAdded', para estudiar el momento en que se agregaron estas
df['year_added'] = df.index.year
df['month_added'] = df.index.month
df['day_added'] = df.index.day
df['weekday_added'] = df.index.weekday  # 0 = Lunes, 6 = Domingo
print(df[['year_added', 'month_added', 'day_added', 'weekday_added']].head())
analyzer.overview()


# Normalizamos la info para ver si hay más duplicados
df['name'] = df['name'].str.strip().str.title()  # Títulos con mayúscula inicial
df['symbol'] = df['symbol'].str.strip().str.upper()  # Símbolos en mayúsculas
print(df[['name', 'symbol']].head())

# Verificar duplicados entre 'name' y 'symbol'
duplicados = df[df.duplicated(subset=['name', 'symbol'], keep=False)]
print(duplicados)
print(f"Duplicados encontrados: {duplicados.shape[0]}")
'''Y vemos que efectivamente tras poner la primera en mayúscula y todos los symbol en mayúscula también
aparecen 62 duplicados para Symbol, que es el USD, que indica el valor en esta moneda para esta cripto como
valor par, esto no nos interesa por lo que las eliminamos, nos interesa el valor en su symbolo'''


# Filtrar filas donde el symbol no sea 'USD'
pares_no_deseados = ['USD', 'EUR', 'GBP']
df = df[~df['symbol'].isin(pares_no_deseados)]
print(f"Filas restantes después de eliminar pares no deseados: {df.shape[0]}")


# Tratamos las dos categóricas que nos faltan name y symbol
'''La estrategia es la siguente:
Crearmos un diccionario que mapee el nombre y simbolo con el label_enconder
de esta manera podemos referenciar este archvio a futuras visualizacionciones o mapeos'''

from sklearn.preprocessing import LabelEncoder

# Crear el codificador para 'symbol'
le = LabelEncoder()
df['symbol_encoded'] = le.fit_transform(df['symbol'])

# Crear el diccionario
mapping_df = df[['name', 'symbol', 'symbol_encoded']].drop_duplicates()

# Guardar el diccionario en un archivo CSV
mapping_df.to_csv('dataSet/symbol_name_encoded_mapping.csv', index=False)
print("Diccionario de mapeo creado y guardado como 'symbol_name_encoded_mapping.csv'")

# Eliminar las columnas 'name' y 'symbol' del dataset principal
df = df.drop(columns=['name', 'symbol'])
analyzer.data_types_analysis()
analyzer.update_data(df)
