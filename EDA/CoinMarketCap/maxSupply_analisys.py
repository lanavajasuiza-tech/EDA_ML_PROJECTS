"""
Análisis DataSet: currencies_data_2023_reduced

El DataSet es muy grande, vamos a reducirlo siguiendo estos criterios:

0. Convertir los dato tipo fecha para manerjar la información y prepararlo para predicciones temporales.
1. Volumen de mercado (volume24h): Las criptomonedas con mayor actividad.
2. Capitalización de mercado (marketCap): Las criptomonedas más valiosas.
3. Antigüedad (dateAdded): Las criptomonedas con más historia.
4. Ranking (cmcRank): Seleccionar las criptomonedas mejor posicionadas.


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
df = "currencies_data_2023_reduced.csv"

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

print(df.head())

# Filtrar criptomonedas según múltiples criterios

from datetime import datetime
import pandas as pd

# Convertir dataset_end_date a UTC, de cara a manejar la fecha
dataset_end_date = pd.Timestamp(datetime(2023, 12, 31), tz="UTC")  # Asegurarse de que sea UTC
one_year_ago = dataset_end_date - pd.Timedelta(days=365)

min_volume = 1e4  # 1e4 = 10,000. Solo incluir criptomonedas con al menos 10,000 unidades transaccionadas en 24h
min_market_cap = 1e6  # 1e6 = 1,000,000. Solo incluir criptomonedas con capitalización >= 1,000,000
top_n_rank = 50  # Considerar las 50 mejores posicionadas según cmcRank

# Asegurarse de que las fechas en la columna 'dateAdded' estén en UTC
df['dateAdded'] = pd.to_datetime(df['dateAdded'], errors='coerce')  # Convertir a datetime
df['dateAdded'] = df['dateAdded'].dt.tz_convert("UTC")  # Convertir a UTC si no lo está

# Filtrar el dataset según los criterios definidos
df_combined_filtered = df[
    (df['volume24h'] >= min_volume) &  # Criptomonedas con volumen >= 10,000
    (df['marketCap'] >= min_market_cap) &  # Criptomonedas con capitalización >= 1,000,000
    (df['dateAdded'] <= one_year_ago) &  # Criptomonedas añadidas hace más de un año
    (df['cmcRank'] <= top_n_rank)  # Criptomonedas en el Top 50
]

# Eliminar duplicados
df_combined_filtered = df_combined_filtered.drop_duplicates()
print("Dimensiones del dataset filtrado:", df_combined_filtered.shape)

      #------- ANALISIS DE LA RELACIÓN MAXSUPPLY VS CIRCULATINGSUP
# Vamos a intentar detallar que nos puede aclarar la columna maxSupply en relación a circulatingSupply ya 
# qué ambos conceptos están muy relacionados de esta manera podremos detectar alguna información que nos pueda
# servir para manejar los datos 

'''
1. ¿Qué porcentaje del suministro máximo está en circulación?
Esto indica cuánto del suministro total planificado ya está disponible.
Un alto porcentaje puede significar madurez del proyecto, mientras que un bajo porcentaje puede indicar emisión futura pendiente.
'''

# Verificar contenido inicial del dataset
print("\nDimensiones de df_combined_filtered:", df_combined_filtered.shape)
print("\nPrimeras filas de df_combined_filtered:")
print(df_combined_filtered.head())

# Excluir criptomonedas con maxSupply igual a 0 o NaN
df_valid_supply = df_combined_filtered[df_combined_filtered['maxSupply'] > 0]
print("\nDimensiones de df_valid_supply después del filtro maxSupply > 0:", df_valid_supply.shape)

# Calcular el porcentaje emitido solo para criptomonedas con maxSupply válido
df_valid_supply['percent_emitted'] = (
    df_valid_supply['circulatingSupply'] / df_valid_supply['maxSupply']
) * 100

# Manejar valores NaN e infinitos
df_valid_supply['percent_emitted'] = df_valid_supply['percent_emitted'].fillna(0).replace([float('inf')], 0)

# Ordenar por 'percent_emitted' en orden descendente
df_valid_supply_sorted = df_valid_supply.sort_values(by='percent_emitted', ascending=False)

# Mostrar las primeras filas del resultado ordenado
print("\nPorcentaje del suministro máximo en circulación calculado correctamente (ordenado):")
print(df_valid_supply_sorted[['name', 'symbol', 'circulatingSupply', 'maxSupply', 'percent_emitted']].head(46))

   #---> Visualizamos

import matplotlib.pyplot as plt

# Seleccionar las 10 principales criptomonedas
top_10 = df_valid_supply_sorted[['name', 'percent_emitted']].head(10)

# Crear el gráfico de tarta
plt.figure(figsize=(8, 8))
plt.pie(
    top_10['percent_emitted'],
    labels=top_10['name'],
    autopct='%1.1f%%',  # Mostrar porcentajes
    startangle=90,      # Girar el inicio del gráfico
    colors=plt.cm.tab10.colors  # Paleta de colores
)
plt.title('Porcentaje Emitido - Top 10 Criptomonedas')
plt.show()


'''
2. ¿La criptomoneda está cerca de alcanzar su límite de emisión?
Si circulatingSupply ≈ maxSupply, es probable que la emisión esté casi completa, lo que podría influir en la escasez y el precio.
'''

# Calcular el porcentaje emitido solo para valores válidos de maxSupply (> 0)
df_combined_filtered = df_combined_filtered[df_combined_filtered['maxSupply'] > 0]

df_combined_filtered['percent_emitted'] = (
    df_combined_filtered['circulatingSupply'] / df_combined_filtered['maxSupply']
) * 100

# Manejar valores problemáticos: NaN e infinitos (aunque ya no deberían existir)
df_combined_filtered['percent_emitted'] = df_combined_filtered['percent_emitted'].fillna(0).replace([float('inf')], 0)

# Verificar las primeras filas después del filtrado
print("\nPrimeras filas del DataFrame filtrado:")
print(df_combined_filtered[['name', 'symbol', 'circulatingSupply', 'maxSupply', 'percent_emitted']].head())

import matplotlib.pyplot as plt

# Ordenar por 'percent_emitted' y seleccionar el top 10
top_10_emitted = df_combined_filtered.sort_values(by='percent_emitted', ascending=False).head(10)

# Crear el gráfico de barras
plt.figure(figsize=(12, 7))
plt.barh(top_10_emitted['name'], top_10_emitted['percent_emitted'], color='skyblue')
plt.title('Top 10 Criptomonedas por Porcentaje Emitido')
plt.xlabel('Porcentaje Emitido (%)')
plt.ylabel('Criptomonedas')
plt.gca().invert_yaxis()  # Invertir el eje Y para ordenar de mayor a menor
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()



'''
3. ¿Qué criptomonedas no tienen un suministro máximo definido?
Si maxSupply es NaN o 0, la criptomoneda puede no tener un límite fijo, lo que podría diluir su valor con el tiempo.
'''
# Filtrar criptomonedas con maxSupply NaN o igual a 0
no_max_supply = df_combined_filtered[df_combined_filtered['maxSupply'].isnull() | (df_combined_filtered['maxSupply'] == 0)]

print("\nCriptomonedas sin suministro máximo definido:")
print(no_max_supply[['name', 'symbol', 'circulatingSupply', 'maxSupply']])



'''
4. ¿Qué criptomonedas tienen un alto suministro máximo pero bajo suministro circulante?
Esto podría sugerir proyectos en una etapa temprana o con un modelo inflacionario que emite nuevas monedas gradualmente.
'''
# Filtrar criptomonedas con bajo porcentaje emitido (< 10%)
low_emission = df_combined_filtered[df_combined_filtered['percent_emitted'] < 10]

print("\nCriptomonedas con alto suministro máximo pero bajo suministro circulante:")
print(low_emission[['name', 'symbol', 'circulatingSupply', 'maxSupply', 'percent_emitted']])

'''
5. ¿Cuál es la relación entre el porcentaje emitido y el mercado?
¿Las monedas con alta emisión tienen mayores capitalizaciones o precios más estables?
¿Las monedas con baja emisión están sobrevaloradas?
'''
import matplotlib.pyplot as plt

# Gráfico de dispersión: Porcentaje emitido vs Capitalización de mercado
plt.figure(figsize=(10, 6))
plt.scatter(df_combined_filtered['percent_emitted'], df_combined_filtered['marketCap'], alpha=0.5)
plt.title("Relación entre Porcentaje Emitido y Capitalización de Mercado")
plt.xlabel("Porcentaje Emitido (%)")
plt.ylabel("Capitalización de Mercado (MarketCap)")
plt.xscale("log")  # Escala logarítmica para visualizar mejor
plt.yscale("log")  # Escala logarítmica para visualizar mejor
plt.show()

'''
6. ¿Hay discrepancias significativas entre circulatingSupply y maxSupply?
Esto podría indicar problemas de datos o características únicas del proyecto.
'''
# Crear una columna de discrepancia
top_discrepancies = df_combined_filtered['supply_discrepancy'] = df_combined_filtered['maxSupply'] - df_combined_filtered['circulatingSupply']

# Filtrar discrepancias mayores a un umbral
large_discrepancies = df_combined_filtered[df_combined_filtered['supply_discrepancy'] > 1e8]  # Ajustar el umbral según necesidad

print("\nCriptomonedas con discrepancias significativas entre circulatingSupply y maxSupply:")
print(large_discrepancies[['name', 'symbol', 'circulatingSupply', 'maxSupply', 'supply_discrepancy']])

# Crear el gráfico de barras
plt.figure(figsize=(12, 7))
plt.barh(top_discrepancies['name'], top_discrepancies['supply_discrepancy'], color='gold')
plt.title('Top 10 Criptomonedas con Discrepancias Significativas')
plt.xlabel('Diferencia (maxSupply - circulatingSupply)')
plt.ylabel('Criptomonedas')
plt.gca().invert_yaxis()  # Invertir el eje Y para ordenar correctamente
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

'''
7. ¿Qué criptomonedas tienen proporciones extremas (muy altas o bajas)?
Criptomonedas con proporciones cercanas a 0% o 100% pueden ser puntos de interés para análisis adicionales.
'''
# Filtrar criptomonedas con porcentaje emitido muy alto (> 99%)
extremely_high_emission = df_combined_filtered[df_combined_filtered['percent_emitted'] > 99]

# Filtrar criptomonedas con porcentaje emitido muy bajo (< 1%)
extremely_low_emission = df_combined_filtered[df_combined_filtered['percent_emitted'] < 1]

print("\nCriptomonedas con proporciones extremadamente altas (> 99%):")
print(extremely_high_emission[['name', 'symbol', 'circulatingSupply', 'maxSupply', 'percent_emitted']])

print("\nCriptomonedas con proporciones extremadamente bajas (< 1%):")
print(extremely_low_emission[['name', 'symbol', 'circulatingSupply', 'maxSupply', 'percent_emitted']])

#-----------------  CONCLUSINES DE ESTE ANÁLISIS maxSupply vs circulatingSupply -------------------#

"""
Exactamente, tu conclusión es muy acertada y resume perfectamente las observaciones clave:

Relación entre maxSupply y circulatingSupply como indicador de madurez:

- Cuando el porcentaje emitido es alto, muestra que el proyecto ya tiene una base estable y posiblemente ha ganado confianza en el mercado.
Estos proyectos suelen ser percibidos como más sólidos y menos propensos a fluctuaciones bruscas debido a una oferta controlada.
Inflación controlada como estrategia clave:

- Los proyectos con un alto porcentaje emitido suelen tener una política de emisión clara y controlada, lo que les da una ventaja a 
largo plazo al minimizar el riesgo de inflación excesiva.
Riesgos asociados con un bajo porcentaje emitido:

- Si el porcentaje emitido es bajo, puede deberse a estrategias específicas (como incentivos futuros o reservas para desarrollo), 
pero hay un riesgo latente si el suministro retenido se libera abruptamente, lo que podría saturar el mercado y generar inflación.
Impacto en la revalorización a largo plazo:

- Un proyecto con un porcentaje emitido alto, combinando madurez y control en la emisión, tiene más posibilidades de revalorización a 
medida que aumenta su adopción y utilidad en el mercado.
Tu análisis combina los aspectos técnicos del comportamiento de estas variables con un enfoque estratégico, lo que es clave para 
entender no solo el estado actual de las criptomonedas, sino también su potencial a largo plazo. 
"""