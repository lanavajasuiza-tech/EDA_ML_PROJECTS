# -*- coding: utf-8 -*-
"""
End-to-End Workflow para Machine Learning
Adaptable a diferentes datasets
"""

# --- 1. Importar Librerías ---
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- 2. Configuración Inicial ---
# Ruta al dataset
os.chdir(r"/ruta/del/dataset/")  # Cambia esto según la ubicación de tu archivo
file_name = "train.csv"  # Nombre del archivo CSV

# Leer dataset
df = pd.read_csv(file_name)
print("Dataset cargado con éxito.")

# --- 3. Análisis Exploratorio ---
print("\n--- Análisis Exploratorio ---")
print("Primeras filas del dataset:")
print(df.head())
print("\nDescripción estadística:")
print(df.describe().T)
print("\nInformación general:")
print(df.info())

# Visualizar correlaciones entre variables numéricas
plt.figure(figsize=(20, 16))
sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap="coolwarm")
plt.title("Mapa de calor de correlaciones")
plt.show()

# Visualizar relaciones específicas (modificar según el dataset)
variables_relevantes = ['GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']  # Ajustar según dataset
plt.figure(figsize=(16, 12))
for i, var in enumerate(variables_relevantes):
    plt.subplot(2, 2, i + 1)
    sns.scatterplot(x=df[var], y=df['SalePrice'])  # Ajustar variable objetivo
    plt.title(f'Relación entre {var} y SalePrice')
plt.tight_layout()
plt.show()

# --- 4. Limpieza de Datos ---
print("\n--- Limpieza de Datos ---")

# Manejo de valores faltantes categóricos (ajustar columnas según dataset)
columns_to_fill_with_NONE = [
    "Alley", "MiscFeature", "GarageFinish", "GarageQual", 
    "Fence", "GarageType", "GarageCond", "PoolQC", 
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType2", "BsmtFinType1"
]
if set(columns_to_fill_with_NONE).issubset(df.columns):
    df[columns_to_fill_with_NONE] = df[columns_to_fill_with_NONE].fillna("NONE")

# Imputación por moda
columns_to_fill_with_mode = ["FireplaceQu", "MasVnrType", "Electrical"]
for column in columns_to_fill_with_mode:
    if column in df.columns:
        mode_value = df[column].mode()[0]
        df[column].fillna(mode_value, inplace=True)

# Imputación por mediana
if "LotFrontage" in df.columns:
    median_value = df["LotFrontage"].median()
    df["LotFrontage"].fillna(median_value, inplace=True)

# Imputación por cero
columns_to_fill_with_zero = ["GarageYrBlt", "MasVnrArea"]
if set(columns_to_fill_with_zero).issubset(df.columns):
    df[columns_to_fill_with_zero] = df[columns_to_fill_with_zero].fillna(0)

# Verificar valores faltantes
if df.isnull().sum().sum() == 0:
    print("No hay valores faltantes tras la limpieza.")

# --- 5. Enriquecimiento de Datos --- (Este paso dependerá de cada Data Set)
print("\n--- Enriquecimiento de Datos ---")
if "YearBuilt" in df.columns and "YrSold" in df.columns:
    df["Crisis"] = df["YrSold"] - df["YearBuilt"]
    df["TiempoPasado"] = 2024 - df["YearBuilt"]

# Binarización de variables categóricas
df = pd.get_dummies(df, drop_first=True)

# --- 6. División en Entrenamiento y Prueba ---
print("\n--- División de Datos ---")
y = df['SalePrice']  # Variable objetivo
X = df.drop(['SalePrice', 'Id'], axis=1, errors='ignore')  # Ajustar según el dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 7. Escalado y PCA ---
print("\n--- Escalado y PCA ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=10)  # Ajustar componentes según necesidades
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print("Varianza explicada por componentes principales:", pca.explained_variance_ratio_)

# --- 8. Modelos y Evaluación ---
print("\n--- Modelos y Evaluación ---")

# 1. Regresión Lineal
lr = LinearRegression()
lr.fit(X_train_pca, y_train)
y_pred_lr = lr.predict(X_test_pca)
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f"MSE - Regresión Lineal: {mse_lr}")

# 2. Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"MSE - Random Forest: {mse_rf}")

# 3. XGBoost
xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f"MSE - XGBoost: {mse_xgb}")

# --- 9. Visualización de Resultados ---
print("\n--- Visualización de Resultados ---")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.6, label="Random Forest")
plt.scatter(y_test, y_pred_xgb, alpha=0.6, label="XGBoost")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label="Perfect Fit")
plt.title("Comparación de Predicciones")
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.legend()
plt.grid(True)
plt.show()

# --- 10. Comparación de Modelos con LazyPredict ---
from lazypredict.Supervised import LazyRegressor
import pandas as pd
import matplotlib.pyplot as plt

# Asegúrate de que los datos estén divididos correctamente
X_train_lazy = X_train
X_test_lazy = X_test
y_train_lazy = y_train
y_test_lazy = y_test

# Inicializar LazyRegressor
reg = LazyRegressor(verbose=0, ignore_warnings=True)

# Ajustar modelos y obtener predicciones
models, predictions = reg.fit(X_train_lazy, X_test_lazy, y_train_lazy, y_test_lazy)

# Mostrar tabla completa en consola
pd.set_option('display.max_columns', None)  # Mostrar todas las columnas
pd.set_option('display.width', 1000)       # Ajustar ancho de salida
print("\n--- Resultados de LazyPredict ---")
print(models)

# --- Visualización de Resultados ---
# Ordenar modelos por R²
models_sorted = models.sort_values(by="R-Squared", ascending=False)

# Visualización en un gráfico de barras
plt.figure(figsize=(12, 8))
sns.barplot(x=models_sorted.index, y=models_sorted["R-Squared"])
plt.title("Comparación de Modelos (LazyPredict)")
plt.xlabel("Modelos")
plt.ylabel("R² Score")
plt.xticks(rotation=45, ha="right")
plt.grid(True)
plt.tight_layout()
plt.show()

# Si necesitas guardar los resultados en un archivo CSV
models.to_csv("resultados_lazypredict.csv", index=True)
