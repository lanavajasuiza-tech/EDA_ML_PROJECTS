# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:29:29 2024

@author: Ana Ndongo
"""

############################################################
#        IMPORTAMOS LIBRERÍAS BÁSICAS Y EL DATASET         #
############################################################


import os
import pandas as pd
import numpy as np


os.chdir(r"/home/ana/Documentos/CURSO_DATA_SCIENCE/MACHINE_LEARNING_E_INTELIGENCIA_ARTIFICIAL/Curso_ML_Laner/17-Modelos/HousePrices/")
df = pd.read_csv("train.csv")

############################################################
#         PRIMERA TOMA DE CONTACTO CON EL DATA SET         #
############################################################



df.describe().T
print(df.dtypes) 
print("Títulos de las columnas:", df.columns.tolist())
print("Número de columnas:", len(df.columns))
print("Número de filas:", len(df))

# Si quisieramos analizar una variable en concreto
df[['MSZoning', 'GarageArea']].describe().T


##########################################################################
#         VAMOS A ANALIZAR SI HAY DUPLICADOS DE FILAS Y COLUMNAS         #
##########################################################################

## Para filas
duplicados_filas = df[df.index.duplicated(keep=False) & df.duplicated(keep=False)]

print(f"Número de filas duplicadas (incluyendo títulos): {duplicados_filas.shape[0]}")
if not duplicados_filas.empty:
    print("Filas duplicadas exactas:")
    print(duplicados_filas)
else:
    print("No se encontraron filas duplicadas exactas.")
temp_df = df.copy()
temp_df.loc["Columna_Titulo"] = temp_df.columns  # Añadir títulos como valores

## Para columnas
duplicados_columnas = temp_df.T[temp_df.T.duplicated(keep=False)].T

print(f"Número de columnas duplicadas (incluyendo títulos): {duplicados_columnas.shape[1]}")
if not duplicados_columnas.empty:
    print("Columnas duplicadas exactas:")
    print(duplicados_columnas.columns.tolist())
else:
    print("No se encontraron columnas duplicadas exactas.")


##########################################################################
#     VISUALIZACION DE DATOS NUMÉRICOS Y CATEGORICS EN UNA TABLA         #
##########################################################################

columnas_numericas = df.select_dtypes(include=['number']).columns.tolist()
columnas_categoricas = df.select_dtypes(exclude=['number']).columns.tolist()
print(columnas_categoricas)

## Creamos la tabla y visualizamos

tabla_tipos = pd.DataFrame({
    "Tipo de columna": ["Numéricas", "Categóricas"],
    "Cantidad": [len(columnas_numericas), len(columnas_categoricas)],
    "Columnas": [columnas_numericas, columnas_categoricas],
    "Tipo": [
        [df[col].dtype for col in columnas_numericas],  # Tipos para columnas numéricas
        [df[col].dtype for col in columnas_categoricas] # Tipos para columnas categóricas
    ]
})
tabla_exploded = tabla_tipos.explode(['Columnas', 'Tipo']).reset_index(drop=True)
print(tabla_exploded)


###################################################################
#      VAMOS A VER VALORES PERDIDOS NaN Y SUS PORCENTAJES         #
###################################################################

total_nan = df.isnull().sum().sum()
print(f"La tabla contiene {total_nan} valores NaN.")
if df.isnull().any().any():
    print("Existen valores NaN en la tabla. Es necesario analizarlos y manejarlos.")
else:
    print("La tabla no contiene valores NaN. Todo está limpio.")


## Crear un DataFrame con los valores nulos (NaN)

valores_nan = df.isnull()
total_filas = len(df)
tabla_nan = pd.DataFrame({
    "Columna": valores_nan.columns,
    "Total NaN": valores_nan.sum(),
    "Porcentaje NaN": (valores_nan.sum() / total_filas) * 100  # Calcular porcentaje
})
tabla_nan = tabla_nan[tabla_nan["Total NaN"] > 0] # Filtrar para mostrar solo las columnas con valores NaN
tabla_nan = tabla_nan.sort_values(by="Porcentaje NaN", ascending=False)
print(tabla_nan)

########################################################################################
#      AQUÍ COMPARAMOS VALORES CONJUNTAMENTE PARA CATEGÓRICAS Y PARA NUMÉRICAS         #
########################################################################################


categoricas = ['MasVnrType', 'FireplaceQu', 'LotFrontage']  # Añade aquí más columnas categóricas
numericas = ['MasVnrType', 'FireplaceQu', 'LotFrontage']  # Añade aquí más columnas numéricas

categ_stats = df[categoricas].describe().T # Estadísticas para columnas categóricas
print("Estadísticas para columnas categóricas:")
print(categ_stats)
num_stats = df[numericas].describe().T # Estadísticas para columnas numéricas
print("\nEstadísticas para columnas numéricas:")
print(num_stats)

# Combinar ambas tablas
combined_stats = pd.concat([categ_stats, num_stats], axis=0)
print("\nEstadísticas combinadas:")
print(combined_stats)


########################################################################################################
#      UN MAPA DE CALOR PARA DECIRME POR LAS VARIABLES QUE MAS SE RELACIONAN CON EL PRICESALES         #
########################################################################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Crear un subconjunto de las columnas numéricas
numericas = df.select_dtypes(include=['float64', 'int64'])

# Calcular la matriz de correlación
correlacion = numericas.corr()
# Generar el mapa de calor
plt.figure(figsize=(36, 34))
sns.heatmap(correlacion, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)

# Mostrar el gráfico
plt.title("Mapa de calor de correlaciones", fontsize=16)
plt.show()


##########################################################################################################################################
#     UN GRÁFICO DE DISPERSIÓN (scatterplots - NUMÉRICOS) PARA DECIRME POR LAS VARIABLES QUE MAS SE RELACIONAN CON EL PRICESALES         #
##########################################################################################################################################
import matplotlib.pyplot as plt
import seaborn as sns

# Selección de las principales variables numéricas correlacionadas (me baso en las relaciones del mapa de calor)
variables_relevantes = ['GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']

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


###############################################################################################
#      AQUÍ VAMOS A OPTIMIZAR LOS DATOS MANUALMENTE IMPUTANDO NaN, Mediana, Modas, ...        #
###############################################################################################

# Lista de columnas a procesar de NaN a NONE según revisión anterior (osn de tipo object)
columns_to_fill_with_NONE= [
    "Alley", "MiscFeature", "GarageFinish", "GarageQual", 
    "Fence", "GarageType", "GarageCond", "PoolQC", 
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType2", "BsmtFinType1"
]
df[columns_to_fill_with_NONE] = df[columns_to_fill_with_NONE].fillna("NONE")
print(df[columns_to_fill_with_NONE].head())

# Lista de columnas a procesar de NaN a MODA según revisión anterior
columns_to_fill_with_mode = ["FireplaceQu", "MasVnrType", "Electrical"]
for column in columns_to_fill_with_mode:
    mode_value = df[column].mode()[0]  # Calcular la moda (valor más frecuente)
    df[column].fillna(mode_value, inplace=True)
print(df[columns_to_fill_with_mode].head())

# Lista de columnas a procesar de NaN a MEDIANA según revisión anterior
median_value = df["LotFrontage"].median()
df["LotFrontage"].fillna(median_value, inplace=True)
print(df["LotFrontage"].head())

# Lista de columnas a procesar de NaN a 0 según revisión anterior (son de tipo int/float)
columns_to_fill_with_zero = ["GarageYrBlt", "MasVnrArea"]
df[columns_to_fill_with_zero] = df[columns_to_fill_with_zero].fillna(0)
print(df[columns_to_fill_with_zero].head())

# Volvemos a verificar el total de NaN, es decir, que no haya

valores_nan = df.isnull()
total_filas = len(df)

tabla_nan = pd.DataFrame({
    "Columna": valores_nan.columns,
    "Total NaN": valores_nan.sum(),
    "Porcentaje NaN": (valores_nan.sum() / total_filas) * 100  # Calcular porcentaje
})

tabla_nan = tabla_nan[tabla_nan["Total NaN"] > 0]
if not tabla_nan.empty:
    tabla_nan = tabla_nan.sort_values(by="Porcentaje NaN", ascending=False)
    print("Columnas con valores NaN en el DataFrame:")
    print(tabla_nan)
else:
    print("No hay NaN en este Data Set.")


#########################################
#      BINOMIALIZAR LAS VARIABLES       #
#########################################

df = pd.get_dummies(df, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)
df


##############################################################################################################
#     AQUÍ VAMOS A DIVIDIR EL DATA SET PARA YA TENERLO AJUSTADO Y EMPEZAR A ENTRENAR DISTINTOS MODELOS       #
##############################################################################################################

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Separar la variable objetivo y las características
y = df['SalePrice']  # Variable objetivo
X = df.drop(['SalePrice'], axis=1)  # Conjunto de características

# Realizamos la particion dejando un 80% para entrenar y un 20% de test.
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.20)

train_data_X = X_train.drop(['SalePrice'], axis = 1)
train_data_y = X_train['SalePrice']

test_data_X = X_test.drop(['SalePrice'], axis = 1)
test_data_y = X_test['SalePrice']



#########################################################
#     AQUÍ VAMOS A EMPEZAR A ENTRENAR LOS MODELOS       #
#########################################################

## Modelo regresión lineal
from sklearn import linear_model
regr = linear_model.LinearRegression()

# Entrenamos y evaluamos con los datos de train
regr.fit(train_data_X, train_data_y)
regr.score(train_data_X,train_data_y)

# Realizamos la prediccion Y Evaluamos en los datos de test.
prediccion = regr.predict(test_data_X)
regr.score(test_data_X,test_data_y)

import matplotlib.pyplot as plt

# Gráfico de predicciones vs valores reales
plt.figure(figsize=(8, 6))
plt.scatter(test_data_y, prediccion, alpha=0.6)
plt.plot([test_data_y.min(), test_data_y.max()], [test_data_y.min(), test_data_y.max()], color='red', linestyle='--')
plt.title('Predicciones vs Valores Reales')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.grid(True)
plt.show()

#####################################
#   OLS (Ordinary Least Squares)    #
#####################################



#####################################
#     Modelo Árbol de Decisión      #
#####################################

# Creamos el modelo.# Importamos la libreria necesaria.
from sklearn.tree import DecisionTreeRegressor

# Definimos el modelo y el numero maximo de ramas del arbol.
arbol = DecisionTreeRegressor(criterion='squared_error', max_depth=8, max_features=None, max_leaf_nodes=None,min_impurity_decrease=0.0,min_samples_leaf=10, min_samples_split=2, min_weight_fraction_leaf=0.0, random_state=None, splitter='best')

# Entrenamos y evaluamos con los datos de train
arbol=arbol.fit(train_data_X, train_data_y)
arbol.score(train_data_X, train_data_y)

# Realizamos la prediccion y evaluamos en los datos de test.
prediccionarbol = arbol.predict(test_data_X)
arbol.score(test_data_X, test_data_y)


###################################
#     Modelo RANDOM Forest        #
###################################

# Definimos el modelo con todos sus parametros
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

RF= RandomForestRegressor(n_estimators=500, criterion='squared_error' ,max_features='sqrt' ,max_depth=500, min_samples_split=5, min_samples_leaf=3, max_leaf_nodes=None,min_impurity_decrease=0, bootstrap=True, oob_score=True, n_jobs=1, random_state=None, verbose=0, warm_start=False)

RF.fit(train_data_X, train_data_y)

# Analizamos la fiabilidad sobre los datos utilizados para crear el modelo.
RF.score(test_data_X, test_data_y)
df

from sklearn.model_selection import cross_val_score

# Separamos la variable dependiente ("y") de las explicativas ("X").

y=df['SalePrice']
X = df.drop(['SalePrice'], axis = 1)

# Procedemos a realizar validacion cruzada
from sklearn.model_selection import cross_val_score
scores = cross_val_score(RF, X, y, cv=5)

print(scores.mean())

# El ultimo paso es determinar la importancia de cada una de las variables
importancias=pd.DataFrame(RF.feature_importances_)
importancias.index=(X.columns)
importancias


#########################
#    Modelo XGBoost     #
#########################


import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo XGBoost
xgb_model = xgb.XGBRegressor(
    base_score=0.5,                 # Valor inicial para las predicciones
    colsample_bylevel=0.8,          # Submuestreo por nivel
    colsample_bytree=0.8,           # Submuestreo por árbol
    gamma=0,                        # Tolerancia mínima de pérdida para dividir un nodo
    learning_rate=0.1,              # Tasa de aprendizaje
    max_delta_step=0,               # Paso máximo para ajustar el peso
    max_depth=3,                    # Profundidad máxima del árbol
    min_child_weight=1,             # Peso mínimo de los nodos hijos
    n_estimators=200,               # Número de árboles
    objective='reg:squarederror',   # Objetivo para regresión
    reg_alpha=0,                    # Regularización L1
    reg_lambda=1,                   # Regularización L2
    scale_pos_weight=1,             # Escalado para datos desbalanceados
    seed=42,                        # Semilla para reproducibilidad
    verbosity=1,                    # Nivel de información
    subsample=1                     # Submuestreo de filas
)

# Entrenar el modelo
modeloxgb = xgb_model.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de prueba
y_pred = modeloxgb.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Validación cruzada
scores = cross_val_score(xgb_model, X, y, cv=5, scoring='r2')  # Cambiar métrica si es necesario
print(f"Cross-validated R²: {scores.mean():.2f}")

# Importancia de las características
from xgboost import plot_importance
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
ax = plt.gca()  # Obtener el eje actual
plot_importance(modeloxgb, max_num_features=10, importance_type="weight", ax=ax)
plt.title("Importancia de las Características")
plt.show()


#importancias=pd.DataFrame(modeloxgb.feature_importances_)
#importancias.index=(X.columns)

#import pandas as pd
#importacia=pd.concat(X.columns,importancias)

plot_importance(modeloxgb)


######################################################################################
#     AQUÍ VAMOS A LANZAR UN LAZY* PARA COMPARAR CON LOS RESULTADOS ANTERIORES       #
######################################################################################

from pyforest import * 
from lazypredict.Supervised import LazyRegressor

reg = LazyRegressor(verbose=0,ignore_warnings=True)
models, predictions = reg.fit(train_data_X, test_data_X, train_data_y, test_data_y)
models
pd.set_option('display.max_columns', None)  # Mostrar todas las columnas
pd.set_option('display.width', 1000)       # Ajustar el ancho de la salida en consola
print(models.columns)





