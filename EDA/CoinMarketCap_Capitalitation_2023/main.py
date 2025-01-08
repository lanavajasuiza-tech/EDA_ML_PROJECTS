"""
Author: Ana Ndongo
Date:  15th December 2024
Description:

This dataset contains information about cryptocurrency prices, market capitalization, and other metrics. 
The data is collected from CoinMarketCap (https://coinmarketcap.com/), a popular website that tracks cryptocurrency prices.

This dataset can be used to:
- Analyze the price trends of different cryptocurrencies.
- Compare the market capitalization of different cryptocurrencies.
- Examine the circulating supply of different cryptocurrencies.
- Analyze the trading volume of different cryptocurrencies.
- Study the volatility of different cryptocurrencies.
- Compare the performance of different cryptocurrencies against each other or against a benchmark index.
- Identify correlations between different cryptocurrency prices.
- Use the data to build models to predict future prices or other trends.

+Info: https://www.kaggle.com/datasets/harshalhonde/coinmarketcap-cryptocurrency-dataset-2023
"""

import sys
import os
from utils.processing import DataLoader
from utils.analyzer import DataAnalyzer

# ---------------- LOAD DATASET -------------------#

# Dynamically detect the project's root directory
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
print("Dynamically detected root directory:", project_root)

# Simplified dataset path
df_path = os.path.join(project_root, "dataSet")
print(f"Dataset path: {df_path}")
df = "currencies_data_Kaggle_2023_unique.csv"

# ---------------- LOAD AND ANALYZE DATA -------------------#
try:
    loader = DataLoader(df_path=df_path, df=df)
    df = loader.load_data()
    print("\n--- Dataset successfully loaded ---")
except FileNotFoundError as e:
    print(f"Error loading dataset: {e}")
    df = None
except ValueError as e:
    print(f"Dataset value error: {e}")
    df = None

# Proceed with analysis if data is loaded
if df is not None:
    # Instantiate the analyzer
    analyzer = DataAnalyzer(df)

    # Call analyzer methods to verify functionality
    analyzer.overview()
    analyzer.duplicates_analysis()
    analyzer.missing_values_analysis()  # Takes 7-10 minutes; please be patient...
    analyzer.data_types_analysis()
else:
    print("\n--- Could not load the dataset. Analysis aborted ---")

# ---------------- PROCESS DATA -------------------#

'''
We will handle dates, NaN values, and categorical variables
'''

# For now, drop 'name.1', which is duplicated

if 'name.1' in df.columns:
    df.drop(columns=['name.1'], inplace=True)
    print("Column 'name.1' removed.")
    analyzer.data_types_analysis()

# Check columns with NaN values
nan_by_column = df.isnull().sum()
print(nan_by_column[nan_by_column > 0])
'''The column maxSupply contains all NaN values
and this is because the data is unavailable, so we will fill it with 0.'''

df.fillna(0, inplace=True)
print(f"Remaining NaN values: {df.isnull().sum().sum()}")  # Confirm no NaN values remain
analyzer.missing_values_analysis()

# Convert dates to datetime format and prepare for time series analysis

import pandas as pd

# Convert date columns to datetime format
df['lastUpdated'] = pd.to_datetime(df['lastUpdated'], errors='coerce')
df['dateAdded'] = pd.to_datetime(df['dateAdded'], errors='coerce')
analyzer.overview()

# Create a temporal index without dropping the column dateAdded (in case we want to work with time series later)
df.set_index('dateAdded', inplace=True, drop=False)
print(df.index)

# Sort the DataFrame by the index (dateAdded)
df.sort_index(inplace=True)
print(df.index.is_monotonic_increasing)  # Should return True if sorted

# Create derived columns from 'dateAdded' to study when cryptocurrencies were added
df['year_added'] = df.index.year
df['month_added'] = df.index.month
df['day_added'] = df.index.day
df['weekday_added'] = df.index.weekday  # 0 = Monday, 6 = Sunday
print(df[['year_added', 'month_added', 'day_added', 'weekday_added']].head())
analyzer.overview()
df.head()

# Normalize the data to check for more duplicates
df['name'] = df['name'].str.strip().str.title()  # Title case for names
df['symbol'] = df['symbol'].str.strip().str.upper()  # Uppercase for symbols
print(df[['name', 'symbol']].head())

# Check for duplicates between 'name' and 'symbol'
duplicates = df[df.duplicated(subset=['name', 'symbol'], keep=False)]
print(duplicates)
print(f"Found duplicates: {duplicates.shape[0]}")
'''After normalizing to title case for names and uppercase for symbols,
we found 62 duplicates for Symbol, which corresponds to USD, indicating the value in dollars
for these cryptocurrencies as a pair value. This is not relevant, so we remove them, focusing on their symbol value.'''

# Handle the two remaining categorical variables: name and symbol
'''The strategy is as follows:
Create a dictionary mapping names to their LabelEncoder values.
This allows us to reference this file for future visualizations or mappings.'''

from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder for 'name'
le = LabelEncoder()
df['name_encoded'] = le.fit_transform(df['name'])

# Create a dictionary mapping 'name' -> 'name_encoded'
name_to_encoded = dict(zip(df['name'], df['name_encoded']))

# Verify the result
print("First encoded values:")
print(df[['name', 'name_encoded']].head())

# Save the dictionary to a CSV file
mapping_df = pd.DataFrame(list(name_to_encoded.items()), columns=['name', 'name_encoded'])
mapping_df.to_csv('dataSet/name_encoded_mapping.csv', index=False)
print("Mapping dictionary created and saved as 'name_encoded_mapping.csv'")

# Remove columns 'name' and 'symbol'
df = df.drop(columns=['name', 'symbol'])
print("Remaining columns after removing 'name' and 'symbol':")
print(df.columns.tolist())

# Save the cleaned dataset ready for further analysis and/or training
df.to_csv('dataSet/currencies_data_ready.csv', index=False)
print("Dataset ready and saved as 'currencies_data_ready.csv'")


# -------> VISUALIZAMOS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Simulación de un DataFrame similar al tuyo (puedes reemplazarlo por tu dataset real)
np.random.seed(42)  # Establecer una semilla para reproducibilidad
date_range = pd.date_range(start='2009-01-01', end='2023-12-31', freq='M')  # Crear un rango de fechas mensual desde enero de 2009 hasta diciembre de 2023

# Los valores de maxSupply, circulatingSupply y price fueron elegidos para reflejar un rango realista basado en valores históricos de criptomonedas. Se seleccionaron estos valores para ilustrar tendencias típicas de suministro y precio en el mercado de criptomonedas, que pueden variar ampliamente. Las monedas seleccionadas ('BTC', 'ETH', 'XRP', 'LTC', 'ADA', 'DOGE', 'SOL', 'DOT') representan algunas de las criptomonedas más conocidas y con diferentes características de mercado.
sample_data = {
    'dateAdded': date_range,
    'maxSupply': np.random.uniform(1e7, 1e9, len(date_range)),  # Entre 10 millones y 1 billón
    'circulatingSupply': np.random.uniform(1e6, 1e8, len(date_range)),  # Entre 1 millón y 100 millones
    'price': np.random.uniform(1, 50000, len(date_range)),  # Entre $1 y $50,000
    'symbol': np.random.choice(['BTC', 'ETH'], len(date_range))  # Monedas simuladas
}

# Crear datos aleatorios para cada mes dentro de ese rango
filtered_df = pd.DataFrame(sample_data)  # Convertir los datos en un DataFrame
print(filtered_df.head())  # Mostrar un vistazo al DataFrame

# Convertimos 'dateAdded' a datetime (si no está ya en ese formato)
filtered_df['dateAdded'] = pd.to_datetime(filtered_df['dateAdded'], errors='coerce')

# Agrupamos por símbolo y calculamos la mediana de maxSupply y circulatingSupply por año
# Usamos as_index=False para evitar conflictos con índices duplicados
grouped_df = filtered_df.groupby([filtered_df['dateAdded'].dt.year, 'symbol'], as_index=False).median()

# Renombramos la columna para evitar conflictos
grouped_df.rename(columns={'dateAdded': 'year'}, inplace=True)

# Inicializamos el gráfico
plt.figure(figsize=(16, 10))

# Scatter plot con líneas de tendencia para cada símbolo
sns.scatterplot(
    data=grouped_df,
    x='year',
    y='circulatingSupply',
    hue='symbol',
    style='symbol',
    palette='tab10',
    s=100,
    alpha=0.8
)

# Añadimos líneas de tendencia usando lineplot
sns.lineplot(
    data=grouped_df,
    x='year',
    y='circulatingSupply',
    hue='symbol',
    palette='tab10',
    legend=False
)

# Mejoramos los títulos y etiquetas
plt.title('Median Circulating Supply by Year and Cryptocurrency Symbol', fontsize=18, pad=20)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Circulating Supply (Median)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Cryptocurrency Symbol', fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 0.5))  # Ajustamos la leyenda a formato vertical para evitar desorden
plt.tight_layout()

# Mostramos el gráfico
plt.show()
