import pandas as pd

# Cargar el dataset
df = pd.read_csv('EDA/CoinMarketCap/dataSet/currencies_data_Kaggle_2023.csv')

# Eliminar duplicados basados en la columna 'name'
df_unicos = df.drop_duplicates(subset='name', keep='first')  # Mantiene la primera aparición de cada criptomoneda

# Guardar el dataset limpio
df_unicos.to_csv('EDA/CoinMarketCap/dataSet/currencies_data_unique.csv', index=False)

# Verificar el resultado
print(f"Filas originales: {df.shape[0]}")
print(f"Filas después de eliminar duplicados: {df_unicos.shape[0]}")
