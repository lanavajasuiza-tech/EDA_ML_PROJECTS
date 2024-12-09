# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 10:25:44 2023

@author: borja
"""

# --- Librerías necesarias ---
import pandas as pd
import os
import matplotlib.pyplot as plt
import pmdarima
from sklearn.impute import SimpleImputer
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron as PP, DFGLS
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

# --- Configuración inicial ---
# Directorio y carga de datos
os.chdir(r"..\CURSO-DATA-SCIENCE\MACHINE_LEARNING_E_INTELIGENCIA_ARTIFICIAL\Curso_ML_Laner\17-Modelos\SeriesTemporales\ARIMAX")
datos = pd.read_excel("Inflation.xlsx")

'''
Descripción de las variables:
- Monthly_CPI: Índice de Precios al Consumidor.
- Monthly_Inflation: Tasa de inflación/deflación mensual.
- Monthly_Rate: Indicador económico (eliminado porque no lo usamos).
- Bank Rate: Tasa de interés oficial.
'''

# Eliminamos la variable que no nos interesa
del datos["Monthly_Rate"]

# Convertimos la fecha al formato adecuado y la usamos como índice
datos["observation_date"] = pd.to_datetime(datos["observation_date"], format="%Y-%m-%d %H:%M:%S")
datos.index = datos["observation_date"]
del datos["observation_date"]

# --- Análisis exploratorio ---
plt.figure(figsize=(12, 6))
plt.plot(datos["Monthly_Inflation"], label="Inflación Mensual", color="blue")
plt.axhline(0, color="red", linestyle="--", label="Línea de referencia (0%)")
plt.title("Evolución Mensual de la Inflación", fontsize=14)
plt.xlabel("Fecha", fontsize=12)
plt.ylabel("Inflación (%)", fontsize=12)
plt.legend()
plt.grid()
plt.show()

'''
Observaciones:
- La serie parece estacionaria, con oscilaciones alrededor de 1.0%.
- Hay valores perdidos (NaN) al inicio y final.
'''

# --- Tratamiento de valores perdidos ---
datosNoComp = datos[datos.isna().any(axis=1)]  # Localizamos los valores NaN

# Imputación con la mediana
imputer = SimpleImputer(strategy='median')
datos.iloc[:, :] = imputer.fit_transform(datos)

# --- Comprobación de estacionariedad ---
# Test de diferenciación
print("Diferenciación mínima necesaria (ndiffs):", pmdarima.arima.ndiffs(datos["Monthly_Inflation"]))
print("Diferenciación estacional mínima necesaria (nsdiffs):", pmdarima.arima.nsdiffs(datos["Monthly_Inflation"], m=12))

# Test de Dickey-Fuller (ADF)
ADF = adfuller(datos["Monthly_Inflation"])
print(f'ADF Statistic: {ADF[0]} | p-value: {ADF[1]}')

# Test KPSS
KPSS = kpss(datos["Monthly_Inflation"], regression='c')
print(f'KPSS Statistic: {KPSS[0]} | p-value: {KPSS[1]}')

# Test de Phillips-Perron (PP)
pp_test = PP(datos["Monthly_Inflation"])
print(f'PP Statistic: {pp_test.stat:.6f} | p-value: {pp_test.pvalue:.6f}')

# Test de DFGLS (ERS)
ers_test = DFGLS(datos["Monthly_Inflation"])
print(f'ERS Statistic: {ers_test.stat:.6f} | p-value: {ers_test.pvalue:.6f}')

# --- Descomposición de la serie ---
SerieDescompuesta = seasonal_decompose(datos["Monthly_Inflation"], model='additive')

# Graficamos los componentes
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
components = ["Observada", "Tendencia", "Estacionalidad", "Residuos"]
colors = ["blue", "orange", "green", "red"]
for i, component in enumerate([SerieDescompuesta.observed, SerieDescompuesta.trend,
                                SerieDescompuesta.seasonal, SerieDescompuesta.resid]):
    axes[i].plot(component, color=colors[i])
    axes[i].set_title(components[i])
plt.tight_layout()
plt.show()

# --- Correlogramas (ACF y PACF) ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
plot_acf(datos["Monthly_Inflation"], lags=40, ax=axes[0])
axes[0].set_title("ACF - Autocorrelación")
plot_pacf(datos["Monthly_Inflation"], lags=40, ax=axes[1])
axes[1].set_title("PACF - Autocorrelación Parcial")
plt.tight_layout()
plt.show()

'''
Observación:
- Se aprecia periodicidad estacional (12 meses).
'''

# --- División del dataset ---
train = datos.loc[:"2014-12-01"]
test = datos["2015-01-01":]

# --- Modelo SARIMAX ---
model = sm.tsa.statespace.SARIMAX(
    train["Monthly_Inflation"],
    order=(1, 0, 2),
    seasonal_order=(1, 0, 2, 12)
)

# Entrenamiento del modelo
ArimaModel = model.fit()

# --- Predicciones ---
test["Predicciones"] = ArimaModel.forecast(len(test))

# --- Visualización de predicciones ---
plt.figure(figsize=(12, 6))
plt.plot(train.index, train["Monthly_Inflation"], label="Entrenamiento", color="blue")
plt.plot(test.index, test["Monthly_Inflation"], label="Test", color="green")
plt.plot(test.index, test["Predicciones"], label="Predicciones", color="orange", linestyle="--")
plt.title("Inflación Real vs Predicciones", fontsize=16)
plt.xlabel("Fecha", fontsize=14)
plt.ylabel("Inflación (%)", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --- Diagnóstico de residuos ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
diagnostics = ["Residuos", "Histograma", "QQ-Plot", "ACF de Residuos"]
functions = [
    lambda ax: ax.plot(ArimaModel.resid, color="blue"),
    lambda ax: ax.hist(ArimaModel.resid, bins=30, density=True, color="skyblue", edgecolor="black"),
    lambda ax: sm.qqplot(ArimaModel.resid, line="s", ax=ax),
    lambda ax: plot_acf(ArimaModel.resid, lags=40, ax=ax)
]
for i, func in enumerate(functions):
    func(axes.flatten()[i])
    axes.flatten()[i].set_title(diagnostics[i])
plt.tight_layout()
plt.show()

# --- ECM (Error Cuadrático Medio) ---
ECM = sum((test["Monthly_Inflation"] - test["Predicciones"])**2) / len(test)
print(f"Error Cuadrático Medio (ECM): {ECM:.6f}")

# --- Comparación de métricas ---
print(f"AIC: {ArimaModel.aic} | BIC: {ArimaModel.bic}")
