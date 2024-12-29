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
df = "currencies_data_DIC_2024.csv"

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
    analyzer.missing_values_analysis() # Tarda entre 7 y 10 minutos, paciencia ...
    analyzer.data_types_analysis()
else:
    print("\n--- No se pudo cargar el dataset. Análisis abortado ---")


'''
El DataSet es muy grande  vamos a reducirlo siguiendo estos criterios:
0. Convertir los dato tipo fecha para manejar la información y prepararlo para predicciones temporales.
1. Volumen de mercado (volume_24h): Las criptomonedas con mayor actividad.
2. Capitalización de mercado (marketCap): Las criptomonedas más valiosas.
3. Antigüedad (date_added): Las criptomonedas con más historia.
4. Ranking (cmc_rank): Seleccionar las criptomonedas mejor posicionadas.'''


# Filtrar criptomonedas según múltiples criterios

from datetime import datetime
import pandas as pd

# Convertir dataset_end_date a UTC, de cara a manejar la fecha
dataset_end_date = pd.Timestamp(datetime(2023, 12, 31), tz="UTC")  # Asegurarse de que sea UTC
one_year_ago = dataset_end_date - pd.Timedelta(days=365)

min_volume = 1e4  # 1e4 = 10,000. Solo incluir criptomonedas con al menos 10,000 unidades transaccionadas en 24h
min_market_cap = 1e6  # 1e6 = 1,000,000. Solo incluir criptomonedas con capitalización >= 1,000,000
top_n_rank = 50  # Considerar las 50 mejores posicionadas según cmc_rank

# Asegurarse de que las fechas en la columna 'date_added' estén en UTC
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')  # Convertir a datetime
df['date_added'] = df['date_added'].dt.tz_convert("UTC")  # Convertir a UTC si no lo está

# Filtrar el dataset según los criterios definidos
df_combined_filtered = df[
    (df['volume_24h'] >= min_volume) &  # Criptomonedas con volumen >= 10,000
    (df['market_cap'] >= min_market_cap) &  # Criptomonedas con capitalización >= 1,000,000
    (df['date_added'] <= one_year_ago) &  # Criptomonedas añadidas hace más de un año
    (df['cmc_rank'] <= top_n_rank)  # Criptomonedas en el Top 50
]

# Eliminar duplicados
df_combined_filtered = df_combined_filtered.drop_duplicates()

# Mostrar las dimensiones del dataset reducido
print("\nDataset reducido basado en múltiples criterios:")
print(df_combined_filtered.shape)
print("\nPrimeras filas del dataset reducido:")
print(df_combined_filtered.head())

# Exportar el dataset reducido a la ruta indicada
export_path = "EDA/CoinMarketCap/dataSet/currencies_data_DIC_2024_reduced.csv"
df_combined_filtered.to_csv(export_path, index=False)
print(f"\nDataset reducido exportado correctamente a: {export_path}")

# Importamos el nuevo dataSet para trabajar con él
df_path = os.path.join(project_root, "dataSet")
print(f"Ruta del dataset: {df_path}")
df = "currencies_data_DIC_2024_reduced.csv"

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
    #analyzer.nan_summary()
else:
    print("\n--- No se pudo cargar el dataset. Análisis abortado ---")

# Rellenamos los NaN en Maxsuply

# Rellenar NaN con 0 (mantiene el conteo de NaN pero es un 0)
df_combined_filtered['maxSupply'] = df_combined_filtered['maxSupply'].fillna(0)
print("\nPrimeras filas después de rellenar NaN en maxSupply:")
print(df_combined_filtered['maxSupply'].head())


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
    