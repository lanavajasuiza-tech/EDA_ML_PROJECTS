# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:29:29 2024

@author: Ana Ndongo
"""

###################################################
#        Análisis HousePrices (PriceSale)         #
###################################################

# IMPORTAMOS LIBRERÍAS BÁSICAS Y EL DATASET

import os
import pandas as pd
import numpy as np

os.chdir(r"MACHINE_LEARNING_&_INTELIGENCIA_ARTIFICIAL\Curso_ML_Laner\17-Modelos\HousePrices")
df = pd.read_csv("train.csv")



# PRIMERA TOMA DE CONTACTO CON EL DATA SET

df.info()
df.isnull().sum()
df.describe()

# UN MAPA DE CALOR PARA DECIRME POR LAS VARIABLES QUE MAS SE RELACIONAN CON EL PRICESALES

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Crear un subconjunto de las columnas numéricas
numericas = df.select_dtypes(include=['float64', 'int64'])

# Calcular la matriz de correlación
correlacion = numericas.corr()
# Generar el mapa de calor
plt.figure(figsize=(12, 10))
sns.heatmap(correlacion, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)

# Mostrar el gráfico
plt.title("Mapa de calor de correlaciones", fontsize=16)
plt.show()

#  UN GRÁFICO DE DISPERSIÓN (scatterplots - NUMÉRICOS) PARA DECIRME POR LAS VARIABLES QUE MAS SE RELACIONAN CON EL PRICESALES

import matplotlib.pyplot as plt
import seaborn as sns

# Selección de las principales variables numéricas correlacionadas
variables_relevantes = ['GrLivArea', 'OverallQual', 'GarageArea', 'TotalBsmtSF']

# Crear un gráfico de dispersión para cada variable
plt.figure(figsize=(12, 8))
for i, var in enumerate(variables_relevantes):
    plt.subplot(2, 2, i + 1)
    sns.scatterplot(x=df[var], y=df['SalePrice'])
    plt.title(f'Relación entre {var} y SalePrice')
    plt.xlabel(var)
    plt.ylabel('SalePrice')

plt.tight_layout()
plt.show()

# AQUÍ EL "RELLENO DE NONE" LO HACEMOS DE MANERA MANUAL

# reemplazar los valores nAn de la columna "NOMBRE_COLUMNA" por la palabra "No NOMBRE_COLUMNA" access
df['Alley'] = df['Alley'].fillna('No alley access')
df['BsmtQual'] = df['BsmtQual'].fillna('No Basement')
df['BsmtCond'] = df['BsmtCond'].fillna('No Basement')
df['BsmtExposure'] = df['BsmtExposure'].fillna('No Basement')
df['BsmtFinType1'] = df['BsmtFinType1'].fillna('No Basement')
df['BsmtFinType2'] = df['BsmtFinType2'].fillna('No Basement')
df['FireplaceQu'] = df['FireplaceQu'].fillna('No Fireplace')
df['MasVnrType']=df['MasVnrType'].fillna('none')
df['GarageType'] = df['GarageType'].fillna('No Garage')
df['GarageFinish'] = df['GarageFinish'].fillna('No Garage')
df['GarageQual'] = df['GarageQual'].fillna('No Garage')
df['GarageCond'] = df['GarageCond'].fillna('No Garage')
df['PoolQC'] = df['PoolQC'].fillna('No Pool')
df['Fence'] = df['Fence'].fillna('No Fence')
df['MiscFeature'] = df['MiscFeature'].fillna('None')
df.info()

# Verificar el total de valores NaN en el DataFrame a nivel general, no por categorías
total_nan = df.isnull().sum().sum()
print(f"Total de valores NaN: {total_nan}")

# AQUÍ UTILIZAMOS FANCYIMPUTE PARA AUTOMATIZAR EL "RELLENO DE NONES" PARA TRES COLUMNAS (ADEMÁS DE BORRAR DOS DE ELLAS)

from fancyimpute  import IterativeImputer

# Eliminamos aquellas columnas que no son representativas
df = df.drop(columns=['Electrical'])
df= df.drop(columns=['Id'])

# iterar los valores LotFrontage, MasVnrType, GarageYrBlt
from fancyimpute  import IterativeImputer
imputer=IterativeImputer(random_state=42)
df[['LotFrontage']]=imputer.fit_transform(df[['LotFrontage']])
df[['MasVnrArea']]=imputer.fit_transform(df[['MasVnrArea']])
df[['GarageYrBlt']]=imputer.fit_transform(df[['GarageYrBlt']])

#verificar que todos los datos perdidos se han completado
df.info()
df.isnull().any().any()

#BINOMIALIZAR LAS VARIABLES
df = pd.get_dummies(df, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)
df

# CARGAMOS LAS LIBRERÍAS ASOCIADAS A LOS ENTRENAMIENTOS


from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# AQUÍ VAMOS A DIVIDIR EL DATA SET PARA YA TENERLO AJUSTADO Y EMPEZAR A ENTRENAR DISTINTOS MODELOS 

# Realizamos la particion dejando un 80% para entrenar y un 20% de test.
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.20)

train_data_X = X_train.drop(['SalePrice'], axis = 1)
train_data_y = X_train['SalePrice']

test_data_X = X_test.drop(['SalePrice'], axis = 1)
test_data_y = X_test['SalePrice']

# AQUÍ VAMOS A EMPEZAR A ENTRENAR LOS MODELOS

## Modelo regresión lineal
from sklearn import linear_model
regr = linear_model.LinearRegression()

# Dividir dataset en entrenamiento y prueba
regr.fit(train_data_X, train_data_y)

# Evaluar desempeño en entrenamiento y prueba
train_score = regr.score(train_data_X, train_data_y)  # R² en entrenamiento
test_score = regr.score(test_data_X, test_data_y)  # R² en conjunto de prueba

# Predicciones
predicciones = regr.predict(test_data_X)


# Calcular métricas adicionales
mae = mean_absolute_error(test_data_y, predicciones)
mse = mean_squared_error(test_data_y, predicciones)
rmse = mse ** 0.5

# Evaluar R² con texto descriptivo
def evaluar_r2(valor):
    return "Correcto ✅" if valor > 0.8 else "Mejorable ⚠️" if valor > 0.5 else "Revisión urgente ❌"

# Resultados en tabla extendida
resultados = pd.DataFrame({
    "Métrica": ["R² (Entrenamiento)", "R² (Prueba)", "MAE", "MSE", "RMSE"],
    "Valor": [train_score, test_score, mae, mse, rmse],
    "Nota": [
        f"El R² mide qué tan bien el modelo explica los datos. {evaluar_r2(test_score)}",
        f"El R² mide qué tan bien el modelo explica los datos. {evaluar_r2(test_score)}",
        "El promedio de errores absolutos entre las predicciones y los valores reales.",
        "El promedio de errores al cuadrado, penalizando más los errores grandes.",
        "La raíz del MSE para interpretar los errores en la misma escala que la variable objetivo."
    ],
    "Evaluación": [
        evaluar_r2(train_score),
        evaluar_r2(test_score),
        "Debe ser lo más bajo posible.",
        "Debe ser lo más bajo posible.",
        "Debe ser lo más bajo posible."
    ]
})

# Mostrar tabla de resultados
print(resultados)

# Visualización de predicciones
plt.figure(figsize=(8, 6))
plt.scatter(test_data_y, predicciones, alpha=0.5, label="Predicciones")
plt.plot([test_data_y.min(), test_data_y.max()], [test_data_y.min(), test_data_y.max()], 'r--', label="Ideal")
plt.title("Valores reales vs Predicciones")
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.legend()
plt.grid(True)
plt.show()

## OLS (Ordinary Least Squares)



## Modelo árbol de decisión



## Modelo RANDOM Forest



## Modelo XGBoost



# AQUÍ VAMOS A LANZAR UN LAZY* PARA COMPARAR CON LOS RESULTADOS ANTERIORES

from pyforest import * 
from lazypredict.Supervised import LazyRegressor

reg = LazyRegressor(verbose=0,ignore_warnings=True)
models, predictions = reg.fit(train_data_X, test_data_X, train_data_y, test_data_y)
models
pd.set_option('display.max_columns', None)  # Mostrar todas las columnas
pd.set_option('display.width', 1000)       # Ajustar el ancho de la salida en consola
print(models.columns)





