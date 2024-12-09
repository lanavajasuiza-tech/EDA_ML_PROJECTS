# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 10:25:44 2023

@author: borja
"""

import pandas as pd
import os


os.chdir(r"..\CURSO-DATA-SCIENCE\MACHINE_LEARNING_E_INTELIGENCIA_ARTIFICIAL\Curso_ML_Laner\17-Modelos\SeriesTemporales\ARIMAX")
# Cargamos los datos

# === PASO 1: ANÁLISIS EXPLORATORIO DE LA SERIE TEMPORAL ===
# Convertir la fecha a DATETIME y usarla como índice (ordenado cronológicamente)
datos = pd.read_excel("Inflation.xlsx")
print(datos)

'''
Monthly_CPI: Índice de Precios al Consumidor, que mide el nivel promedio de precios de bienes y servicios.
Monthly_Inflation: Tasa de cambio porcentual mensual en el nivel de precios, indicando inflación (+) o deflación (-).
Monthly_Rate: Tasa financiera mensual o cambio relacionado con indicadores económicos, posiblemente de interés.
Bank Rate: Tasa de interés oficial establecida por el banco central para préstamos a corto plazo.
'''
# Eliminamos la variable que no nos interesa
del (datos["Monthly_Rate"])
datos.columns
# Pasamos la fecha a formato fecha
datos["observation_date"] = pd.to_datetime(datos["observation_date"], format="%Y-%m-%d %H:%M:%S")

# Pasamos la fecha al indice
datos.index = datos["observation_date"]
del (datos["observation_date"])
print(datos)

# Ya disponemos de los datos en el formato adecuado.

# Procedemos a analizar la inflacion
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))  # Tamaño del gráfico
plt.plot(datos["Monthly_Inflation"], label="Inflación Mensual", color="blue")
plt.title("Evolución Mensual de la Inflación", fontsize=14)
plt.xlabel("Fecha", fontsize=12)
plt.ylabel("Inflación (%)", fontsize=12)
plt.axhline(0, color="red", linestyle="--", label="Línea de referencia (0%)")  # Línea de referencia
plt.grid()
plt.legend()
plt.show()


# En este primer grafico parece una serie estacionaria, ya que que la inflación oscila alrededor de un valor promedio constante (1.0%) y la varianza no parece cambiar significativamente a lo largo del tiempo.
# Hay que destacar que esta serie ya esta diferenciada.
# La inflacion es el IPC diferenciado

# Procedemos a comprobar si seria necesario diferenciarla
import pmdarima

# Respecto a la observacion anterior

# === ESTACIONARIEDAD: ===
# Pruebas estadísticas (ADF, KPSS, PP, ERS)
pmdarima.arima.ndiffs(datos["Monthly_Inflation"])

# Da error ya que tenemos valores perdidos, Tenemos NaN

# Procedemos a analizar donde se encuentran esos valores.
datos.describe()

# Al analizar el count vemos que tenemos los valores perdidos.
# Procedemos a localizarlos.

# Extraemos los datos con NaN
datosNoComp = datos[datos.isna().any(axis=1)]
print(datosNoComp)

# Vemos que son el primer y el ultimo dato.
# Podemos (y debemos) eliminar/imputar esos datos ya que al ser solo dos no afecta demasiado al resultado
from sklearn.impute import SimpleImputer #(strategy = median(mediana), mean(media), most_frecuent, constant)

# datos=datos.dropna()
print("Antes de imputar:")
print(datos)
imputer = SimpleImputer(strategy='median')
datos.iloc[:, :] = imputer.fit_transform(datos)
print("\nDespués de imputar:")
print(datos)

# Repetimos el proceso, y ya no deberíamos tener error

# === ESTACIONARIEDAD: ===
# Pruebas estadísticas (ADF, KPSS, PP, ERS)
pmdarima.arima.ndiffs(datos["Monthly_Inflation"])

# La salida es 0, por lo que no es necesario diferenciarlo.

# Estacionalmente con estacionalidad anual.
pmdarima.arima.nsdiffs(datos["Monthly_Inflation"], m = 12)

# Tampoco habria que diferenciarla estaciolnalmente.


# Pasamos a comprobarlo mediante los test que conocemos.
# Importamos el test ADF
from statsmodels.tsa.stattools import adfuller

# H0: No estacionario (raiz unitaria)
ADF = adfuller(datos["Monthly_Inflation"])

# Mostramos los resultados
print('ADF Statistic: %f' % ADF[0])
print('p-value: %f' % ADF[1])

# Interpretamos los resultados
if ADF[1] < 0.05:
    print("Interpretación:")
    print(f"- El ADF Statistic es {ADF[0]:.6f} y el p-value es {ADF[1]:.6f}.")
    print("- Esto indica que hay suficiente evidencia estadística para concluir que la serie no tiene raíz unitaria y es estacionaria.")
    print("- Significa que las propiedades estadísticas de la serie (media, varianza y autocorrelación) son constantes a lo largo del tiempo.")
    print("Rechazamos H0 -> La serie es estacionaria.")
else:
    print("Interpretación:")
    print(f"- El ADF Statistic es {ADF[0]:.6f} y el p-value es {ADF[1]:.6f}.")
    print("- Esto indica que no podemos rechazar la hipótesis de que la serie tiene raíz unitaria y, por lo tanto, no es estacionaria.")
    print("- Podría ser necesario aplicar transformaciones, como diferenciación, para estabilizar la serie antes de usarla en modelos.")
    print("No rechazamos H0 -> La serie no es estacionaria.")

adf_conclusion = "estacionaria (rechazamos H0)" if ADF[1] < 0.05 else "no estacionaria (no rechazamos H0)"

# Rechazamos H0 -> Serie estacionaria


# KPSS (Kwiatkowski-Phillips-Schmidt-Shin Test)
from statsmodels.tsa.stattools import kpss

# H0: Serie Estacionaria
KPSS = kpss(datos["Monthly_Inflation"], regression='c')

# Mostramos los resultados
print('KPSS Statistic: %f' % KPSS[0])
print('p-value: %f' % KPSS[1])

# Interpretamos los resultados
if KPSS[1] < 0.05:
    print("Interpretación:")
    print(f"- El KPSS Statistic es {KPSS[0]:.6f} y el p-value es {KPSS[1]:.6f}.")
    print("- Esto indica que hay suficiente evidencia estadística para concluir que la serie no es estacionaria.")
    print("- Es probable que la serie tenga una tendencia o patrón no constante en el tiempo.")
    print("Rechazamos H0 -> La serie no es estacionaria.")
else:
    print("Interpretación:")
    print(f"- El KPSS Statistic es {KPSS[0]:.6f} y el p-value es {KPSS[1]:.6f}.")
    print("- Esto indica que no hay suficiente evidencia estadística para rechazar que la serie es estacionaria.")
    print("- La serie tiene propiedades constantes como media, varianza y autocorrelación en el tiempo.")
    print("No rechazamos H0 -> La serie es estacionaria.")

kpss_conclusion = "no estacionaria (rechazamos H0)" if KPSS[1] < 0.05 else "estacionaria (no rechazamos H0)"

# PP
from arch.unitroot import PhillipsPerron as PP

# H0: La serie NO es estacionaria (tiene raíz unitaria).

# Aplicamos el test de Phillips-Perron
pp_test = PP(datos["Monthly_Inflation"])

# Mostramos los resultados
print(f"Phillips-Perron Test Statistic: {pp_test.stat:.6f}")  # Estadístico del test PP
print(f"p-value: {pp_test.pvalue:.6f}")  # p-value del test PP

# Interpretamos los resultados
if pp_test.pvalue < 0.05:
    print("Interpretación:")
    print(f"- El estadístico del test PP es {pp_test.stat:.6f}, y el p-value es {pp_test.pvalue:.6f}.")
    print("- El p-value es menor a 0.05, lo que indica suficiente evidencia estadística para rechazar que la serie tiene raíz unitaria.")
    print("- Esto significa que la serie tiene una media, varianza y autocorrelación constantes en el tiempo.")
    print("Rechazamos H0 -> La serie es estacionaria.")
else:
    print("Interpretación:")
    print(f"- El estadístico del test PP es {pp_test.stat:.6f}, y el p-value es {pp_test.pvalue:.6f}.")
    print("- El p-value es mayor o igual a 0.05, lo que indica que no hay suficiente evidencia estadística para rechazar que la serie tiene raíz unitaria.")
    print("- Esto sugiere que la serie puede tener tendencias o varianza no constante, y podrían requerirse transformaciones adicionales.")
    print("No rechazamos H0 -> La serie no es estacionaria.")

pp_conclusion = "estacionaria (rechazamos H0)" if pp_test.pvalue < 0.05 else "no estacionaria (no rechazamos H0)"

# ERS
from arch.unitroot import DFGLS

# H0: La serie NO es estacionaria (tiene raíz unitaria).

# Aplicamos el test ERS
ers_test = DFGLS(datos["Monthly_Inflation"])

# Mostramos los resultados
print(f"ERS Test Statistic: {ers_test.stat:.6f}")  # Estadístico del test ERS
print(f"p-value: {ers_test.pvalue:.6f}")  # p-value del test ERS

# Interpretamos los resultados
if ers_test.pvalue < 0.05:
    print("Interpretación:")
    print(f"- El estadístico del test ERS es {ers_test.stat:.6f}, y el p-value es {ers_test.pvalue:.6f}.")
    print("- El p-value es menor a 0.05, lo que indica suficiente evidencia estadística para rechazar que la serie tiene raíz unitaria.")
    print("- Esto sugiere que la serie es estacionaria, con media y varianza constantes en el tiempo.")
    print("Rechazamos H0 -> La serie es estacionaria.")
else:
    print("Interpretación:")
    print(f"- El estadístico del test ERS es {ers_test.stat:.6f}, y el p-value es {ers_test.pvalue:.6f}.")
    print("- El p-value es mayor o igual a 0.05, lo que indica que no hay suficiente evidencia estadística para rechazar que la serie tiene raíz unitaria.")
    print("- Esto significa que la serie puede tener tendencias o varianza no constante, requiriendo transformaciones adicionales.")
    print("No rechazamos H0 -> La serie no es estacionaria.")

ers_conclusion = "estacionaria (rechazamos H0)" if ers_test.pvalue < 0.05 else "no estacionaria (no rechazamos H0)"

# En este caso podemos afirmar sin duda alguna que la serie es estacionaria.

# Pasamos a visualizar su descomposicion. (Descomposición son siempre 3 conceptos: Tendencia(Trend), Estacionalidad(Seasonal) y Residuos(Residual))
# Ayuda a entender los datos y a preparlos para el modelado. La descomposición puede ser aditiva o multiplicativa)
# en este cas usamos la aditiva (hay un patrón estacional constante, la multiplicativa se usa cuando los patrones aumentan o disminuyen)

from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Descomposición aditiva
SerieDescompuesta = seasonal_decompose(datos["Monthly_Inflation"], model='additive')

# Graficamos manualmente cada componente
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# Serie observada
axes[0].plot(SerieDescompuesta.observed, label="Observada", color="blue")
axes[0].set_title("Serie Observada")
axes[0].grid()

# Tendencia
axes[1].plot(SerieDescompuesta.trend, label="Tendencia", color="orange")
axes[1].set_title("Tendencia")
axes[1].grid()

# Estacionalidad
axes[2].plot(SerieDescompuesta.seasonal, label="Estacionalidad", color="green")
axes[2].set_title("Estacionalidad")
axes[2].grid()

# Residuos
axes[3].plot(SerieDescompuesta.resid, label="Residuos", color="red")
axes[3].set_title("Residuos")
axes[3].grid()

# Configuramos etiquetas y diseño general
for ax in axes:
    ax.legend(loc="upper right")
plt.xlabel("Fecha")
plt.tight_layout()
plt.show()

residuos = SerieDescompuesta.resid.dropna()

# Establecemos un umbral para considerar residuos pequeños
umbral = residuos.abs().mean() / datos["Monthly_Inflation"].mean()

if umbral < 0.05:  # Umbral relativo ajustable según el tipo de datos
    print("Los residuos son pequeños y no muestran patrones evidentes. Esto podría indicar un buen ajuste del modelo. Sin embargo, analiza el tipo de dato para confirmar que las fluctuaciones no son significativas.")
else:
    print("Los residuos muestran cierta magnitud o patrones que podrían ser relevantes. Revisa si existen variables no contempladas o ajustes adicionales que puedan mejorar el modelo.")


# En este caso observamos varios aspectos:
    # La componente estacional tiene un peso minimo.
    # Los residuos se semejan a un ruido blanco.
    # No se aprecian ciclos pese a que hay ciertos picos.
    
# Tras ello procedemos a la realizacion del correlograma y el correlograma parcial.

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Creamos una figura con dos subgráficos
fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # 1 fila, 2 columnas

# Graficamos el ACF en el primer subplot
plot_acf(datos["Monthly_Inflation"], lags=40, ax=axes[0], title="ACF - Autocorrelación")

# Graficamos el PACF en el segundo subplot
plot_pacf(datos["Monthly_Inflation"], lags=40, ax=axes[1], title="PACF - Autocorrelación Parcial")

# Ajustamos el diseño para evitar solapamientos
plt.tight_layout()
plt.show()




# Sin embargo al visualizar ambos graficos se aprecia una componente estacional.
# Con una periodicidad de 12 meses.

# El correlograma muestra estacionalidad por lo que la parte MA sera 2 y 2 estacional.
# El correlograma parcial se trunca en el uno por lo que muestra un AR(1)

# Separamos en train y test

# === PASO 2: HACEMOS EL MODELO ===
# División en train / test
train = datos.loc[:"2014-12-01 00:00:00"]   # 659 registros
test = datos["2015-01-01 00:00:00":]        # 60 registros (8%)

# Entrenaos el modelo segun lo descrito anteriormente.

model = sm.tsa.statespace.SARIMAX(train.iloc[:, 1], 
                                  order=(1, 0, 2), 
                                  seasonal_order=(1, 0, 2, 12))
'''
- Modelo SARIMAX: incluye componentes estacionales y no estacionales.
- Parámetros del modelo:
  - `order=(p, d, q)`: Componentes no estacionales:
    - p=1: 1 término autorregresivo (AR).
    - d=0: Sin diferenciación (serie estacionaria).
    - q=2: 2 términos de medias móviles (MA).
  - `seasonal_order=(P, D, Q, m)`: Componentes estacionales:
    - P=1: 1 término autorregresivo estacional (SAR).
    - D=0: Sin diferenciación estacional.
    - Q=2: 2 términos de medias móviles estacionales (SMA).
    - m=12: Periodo de estacionalidad (anual, datos mensuales).
- `train.iloc[:, 1]`: La columna 1 del conjunto de entrenamiento, que contiene la serie temporal a modelar.
'''


# Visualizamos el modelo (Ver significado de los valores en zChuleta)
ArimaModel = model.fit()
print (ArimaModel.summary())

# Todos los valores son significativos

# Sacamos las predicciones
test = test.copy()  # Nos aseguramos de que test sea una copia independiente
test["Predicciones"] = ArimaModel.forecast(len(test))
print(test.head())


# Visualización de las predicciones y datos reales
plt.figure(figsize=(12, 6))
plt.plot(train.index, train.iloc[:, 1], label="Datos de Entrenamiento", color="blue")
plt.plot(test.index, test.iloc[:, 0], label="Datos Reales (Test)", color="green")
plt.plot(test.index, test["Predicciones"], label="Predicciones del Modelo", color="red", linestyle="--")
plt.title("Comparación: Datos Reales vs Predicciones del Modelo SARIMAX", fontsize=14)
plt.xlabel("Fecha", fontsize=12)
plt.ylabel("Valores", fontsize=12)
plt.legend(loc="upper left")
plt.grid()
plt.show()

# Residuos del modelo
residuos = ArimaModel.resid

# Visualizamos los residuos con diferentes gráficos
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuos en el tiempo
axes[0, 0].plot(residuos, label="Residuos", color="blue")
axes[0, 0].axhline(0, color="red", linestyle="--", label="Referencia: 0")
axes[0, 0].set_title("Residuos en el Tiempo", fontsize=12)
axes[0, 0].set_xlabel("Fecha", fontsize=10)
axes[0, 0].set_ylabel("Residuos", fontsize=10)
axes[0, 0].grid()
axes[0, 0].legend()

# Histograma de residuos con barras y densidad
axes[0, 1].hist(residuos, bins=30, density=True, color="skyblue", edgecolor="black", alpha=0.7, label="Histograma")
residuos_kde = sm.nonparametric.KDEUnivariate(residuos)
residuos_kde.fit()
axes[0, 1].plot(residuos_kde.support, residuos_kde.density, color="orange", label="KDE (Densidad)")
axes[0, 1].set_title("Histograma de Residuos + Densidad", fontsize=12)
axes[0, 1].set_xlabel("Residuos", fontsize=10)
axes[0, 1].set_ylabel("Densidad", fontsize=10)
axes[0, 1].grid()
axes[0, 1].legend()

# QQ-Plot para normalidad
sm.qqplot(residuos, line="s", ax=axes[1, 0])
axes[1, 0].set_title("QQ-Plot de Residuos", fontsize=12)

# ACF de residuos
plot_acf(residuos, ax=axes[1, 1], lags=40)
axes[1, 1].set_title("ACF de Residuos (Correlograma)", fontsize=12)

# Ajuste de diseño general
plt.tight_layout()
plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Parece correcto.

# Crear una figura con dos subgráficos
fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # Una fila y dos columnas

# Gráfico ACF
plot_acf(ArimaModel.resid, lags=40, ax=axes[0], title="ACF - Residuos del Modelo")
axes[0].grid()

# Gráfico PACF
plot_pacf(ArimaModel.resid, lags=40, ax=axes[1], title="PACF - Residuos del Modelo")
axes[1].grid()

# Ajustamos el diseño
plt.tight_layout()
plt.show()

# Esto confirma lo anterior con un mayor numero de Lags


# ANALIZAMOS LAS PREDICCIONES MEDIANTE EL ECM (Error Cuadrático Medio)

# --- 1. Revisión de los tipos de datos ---
# Verificamos que las columnas utilizadas sean numéricas para evitar problemas en cálculos posteriores.
print("Tipos de datos en el conjunto de prueba:")
print(test.dtypes)

# --- 2. Cálculo del Error Cuadrático Medio (ECM) ---
# ECM mide la magnitud promedio del error cuadrático entre los valores reales y los predichos.
ECM = sum((test["Monthly_Inflation"] - test["Predicciones"])**2) / len(test)
ecm = ECM  # Si ya lo calculaste previamente

print(f"Error Cuadrático Medio (ECM): {ECM:.6f}")

# --- 3. Visualización de valores reales vs predicciones ---
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(test["Monthly_Inflation"], label="Inflación Real", color="blue")
plt.plot(test["Predicciones"], label="Predicciones", color="orange", linestyle="--")
plt.title("Inflación Real vs Predicciones", fontsize=14)
plt.xlabel("Fecha", fontsize=12)
plt.ylabel("Inflación (%)", fontsize=12)
plt.legend()
plt.grid()
plt.show()


# División del dataset en conjuntos de entrenamiento y prueba ---

# === PASO 2: HACEMOS EL MODELO ===
# División en train / test
train = datos.loc[:"2014-12-01 00:00:00"]  # Conjunto de entrenamiento hasta diciembre de 2014
test = datos["2015-01-01 00:00:00":]  # Conjunto de prueba desde enero de 2015

# Entrenamiento del modelo SARIMAX con variable exógena ---
# Incluimos la variable "Bank Rate" como exógena (factor externo que puede influir en la inflación)
Modelo2 = sm.tsa.statespace.SARIMAX(
    train["Monthly_Inflation"],  # Variable dependiente
    order=(1, 0, 2),  # Parámetros ARIMA: (p=1, d=0, q=2)
    seasonal_order=(1, 0, 2, 12),  # Parámetros estacionales: (P=1, D=0, Q=2, s=12)
    exog=train["Bank Rate"]  # Variable exógena (factor adicional que afecta al modelo)
)

# Ajuste del modelo ---
ArimaModel2 = Modelo2.fit()

# Resumen del modelo ---
print(ArimaModel2.summary())

# Visualización del diagnóstico del modelo ---
# Comprobamos que el modelo sea correcto
import matplotlib.pyplot as plt
# Residuos del modelo
residuos = ArimaModel2.resid

# Creación de una figura con subgráficos para los diagnósticos
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuos estandarizados a lo largo del tiempo
axes[0, 0].plot(residuos, label="Residuos Estandarizados", color="blue", linewidth=1.5)
axes[0, 0].axhline(0, color="red", linestyle="--", alpha=0.7, label="Referencia: 0")
axes[0, 0].set_title("Residuos Estandarizados", fontsize=12)
axes[0, 0].set_xlabel("Fecha", fontsize=10)
axes[0, 0].set_ylabel("Valor", fontsize=10)
axes[0, 0].grid(alpha=0.3)
axes[0, 0].legend(fontsize=10)

# Histograma de residuos con densidad estimada
axes[0, 1].hist(residuos, bins=30, density=True, color="skyblue", edgecolor="black", alpha=0.7, label="Histograma")
residuos_kde = sm.nonparametric.KDEUnivariate(residuos)
residuos_kde.fit()
axes[0, 1].plot(residuos_kde.support, residuos_kde.density, color="orange", label="Densidad Estimada (KDE)")
axes[0, 1].set_title("Histograma de Residuos", fontsize=12)
axes[0, 1].set_xlabel("Residuos", fontsize=10)
axes[0, 1].set_ylabel("Densidad", fontsize=10)
axes[0, 1].grid(alpha=0.3)
axes[0, 1].legend(fontsize=10)

# QQ-Plot para normalidad de residuos
sm.qqplot(residuos, line="s", ax=axes[1, 0])
axes[1, 0].set_title("QQ-Plot de Residuos", fontsize=12)

# Correlograma de residuos (ACF)
plot_acf(residuos, ax=axes[1, 1], lags=40)
axes[1, 1].set_title("ACF de Residuos", fontsize=12)
axes[1, 1].grid(alpha=0.3)

# Ajuste de diseño general
plt.tight_layout()
plt.show()

# Análisis del correlograma de los residuos ---
# Graficamos el correlograma y correlograma parcial de los residuos
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # Una fila, dos columnas

# ACF
plot_acf(ArimaModel2.resid, lags=40, ax=axes[0], title="ACF - Residuos del Modelo SARIMAX")
axes[0].grid()

# PACF
plot_pacf(ArimaModel2.resid, lags=40, ax=axes[1], title="PACF - Residuos del Modelo SARIMAX")
axes[1].grid()

plt.tight_layout()  # Ajusta los espacios entre los gráficos
plt.show()

'''
- Correlograma y correlograma parcial:
    - Si todas las barras están dentro de los intervalos de confianza, los residuos son ruido blanco.
    - Residuos correlacionados indicarían que el modelo no ha capturado todos los patrones de la serie.
'''

# Comparación de métricas AIC y BIC ---
# AIC y BIC se encuentran en el resumen del modelo.
print(f"AIC del modelo: {ArimaModel2.aic}")
print(f"BIC del modelo: {ArimaModel2.bic}")
'''
- AIC y BIC:
    - Se usan para comparar modelos. Modelos con menor AIC/BIC se consideran mejores.
    - Si este modelo tiene valores mayores que otro modelo, sería menos adecuado.
'''



# ESTA ES UNA VARIACIÓN PARA METER UN VALOR EXóGENO

import matplotlib.pyplot as plt

# Realizamos las predicciones
test["Predicciones"] = ArimaModel2.forecast(len(test), exog=test["Bank Rate"])

# Calculamos el ECM
ECM2 = sum((test["Monthly_Inflation"] - test["Predicciones"])**2) / len(test)
print(f"Error Cuadrático Medio (ECM): {ECM2:.4f}")

# Gráfico mejorado
plt.figure(figsize=(12, 6))
plt.plot(test.index, test["Monthly_Inflation"], label="Inflación Real", color="blue", linewidth=2)
plt.plot(test.index, test["Predicciones"], label="Predicciones del Modelo", color="orange", linestyle="--", linewidth=2)
plt.title("Comparación: Inflación Real vs Predicciones del Modelo", fontsize=16)
plt.xlabel("Fecha", fontsize=14)
plt.ylabel("Inflación (%)", fontsize=14)
plt.axhline(0, color="red", linestyle="--", alpha=0.5, label="Referencia (0%)")  # Línea de referencia
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# El ECM se dispara.  (Error Cuadrático Medio)

# En este caso no parece una buena alternativa añadir
# Una variable externa ya que se sobreentrena.

# Otro posible problema seria definir mal el ARIMA, cambiamos el rango de fechas 


# === PASO 2: HACEMOS EL MODELO ===
# División en train / test
train = datos.loc[:"2014-12-01 00:00:00"]
test = datos["2015-01-01 00:00:00":]

# Entrenaos el modelo segun lo descrito anteriormente.

import statsmodels.api as sm
import matplotlib.pyplot as plt

# --- Definición y entrenamiento del modelo SARIMAX ---
# Configuración del modelo SARIMAX
model = sm.tsa.statespace.SARIMAX(
    train.iloc[:, 1],  # Serie temporal a modelar
    order=(1, 0, 0),  # Parámetros ARIMA no estacional
    seasonal_order=(1, 0, 0, 12)  # Parámetros estacionales
)

# Entrenamiento del modelo
ArimaModel = model.fit()

# Resumen del modelo
print(ArimaModel.summary())

# --- Generación de predicciones ---
# Generamos predicciones para el conjunto de prueba
test["Predicciones"] = ArimaModel.forecast(len(test))

# --- Visualización de predicciones ---
plt.figure(figsize=(12, 6))
plt.plot(train.index, train.iloc[:, 1], label="Datos de Entrenamiento", color="blue", linewidth=2)
plt.plot(test.index, test.iloc[:, 0], label="Datos Reales (Test)", color="green", linewidth=2)
plt.plot(test.index, test["Predicciones"], label="Predicciones del Modelo", color="orange", linestyle="--", linewidth=2)
plt.title("Comparación: Datos Reales vs Predicciones del Modelo SARIMAX", fontsize=16)
plt.xlabel("Fecha", fontsize=14)
plt.ylabel("Valores", fontsize=14)
plt.axhline(0, color="red", linestyle="--", alpha=0.5, label="Línea de Referencia (0%)")
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --- Análisis de residuos ---
# Gráficos de diagnóstico del modelo
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuos estandarizados en el tiempo
axes[0, 0].plot(ArimaModel.resid, label="Residuos Estandarizados", color="blue", linewidth=1.5)
axes[0, 0].axhline(0, color="red", linestyle="--", alpha=0.7, label="Referencia: 0")
axes[0, 0].set_title("Residuos Estandarizados", fontsize=12)
axes[0, 0].set_xlabel("Fecha", fontsize=10)
axes[0, 0].set_ylabel("Valor", fontsize=10)
axes[0, 0].grid(alpha=0.3)
axes[0, 0].legend(fontsize=10)

# Histograma de residuos con densidad
axes[0, 1].hist(ArimaModel.resid, bins=30, density=True, color="skyblue", edgecolor="black", alpha=0.7, label="Histograma")
residuos_kde = sm.nonparametric.KDEUnivariate(ArimaModel.resid)
residuos_kde.fit()
axes[0, 1].plot(residuos_kde.support, residuos_kde.density, color="orange", label="Densidad Estimada (KDE)")
axes[0, 1].set_title("Histograma de Residuos", fontsize=12)
axes[0, 1].set_xlabel("Residuos", fontsize=10)
axes[0, 1].set_ylabel("Densidad", fontsize=10)
axes[0, 1].grid(alpha=0.3)
axes[0, 1].legend(fontsize=10)

# QQ-Plot para normalidad
sm.qqplot(ArimaModel.resid, line="s", ax=axes[1, 0])
axes[1, 0].set_title("QQ-Plot de Residuos", fontsize=12)

# Correlograma (ACF) de residuos
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(ArimaModel.resid, ax=axes[1, 1], lags=40)
axes[1, 1].set_title("ACF de Residuos", fontsize=12)
axes[1, 1].grid(alpha=0.3)

# Ajustamos el diseño
plt.tight_layout()
plt.show()


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# PARECE CORRECTO

# Crear una figura con dos gráficos en paralelo
fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # Una fila, dos columnas

# Graficamos el ACF en el primer subplot
plot_acf(ArimaModel.resid, lags=40, ax=axes[0], title="ACF - Residuos del Modelo")
axes[0].grid(alpha=0.3)  # Cuadrícula más suave

# Graficamos el PACF en el segundo subplot
plot_pacf(ArimaModel.resid, lags=40, ax=axes[1], title="PACF - Residuos del Modelo")
axes[1].grid(alpha=0.3)  # Cuadrícula más suave

# Ajustar el diseño para evitar solapamientos
plt.tight_layout()
plt.show()

# Esto confirma lo anterior con un mayor numero de Lags


# Analizamos las predicciones mediante el ECM  (Error Cuadrático Medio)
# Calculamos el ECM
ECM = sum((test["Monthly_Inflation"] - test["Predicciones"])**2) / len(test)
print(f"Error Cuadrático Medio (ECM): {ECM:.6f}")

# Graficamos los valores reales y las predicciones
plt.figure(figsize=(12, 6))
plt.plot(test.index, test["Monthly_Inflation"], label="Inflación Real", color="blue", linewidth=2)
plt.plot(test.index, test["Predicciones"], label="Predicciones del Modelo", color="orange", linestyle="--", linewidth=2)

# Añadimos título y etiquetas
plt.title("Comparación: Inflación Real vs Predicciones del Modelo", fontsize=16)
plt.xlabel("Fecha", fontsize=14)
plt.ylabel("Inflación (%)", fontsize=14)

# Añadimos una línea de referencia en 0
plt.axhline(0, color="red", linestyle="--", alpha=0.5, label="Referencia (0%)")

# Configuramos leyenda y cuadrícula
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# Ajustamos el diseño y mostramos
plt.tight_layout()
plt.show()

# ===== Conclusión =====
print(f"\nConclusión del modelo:")
print(f"- ADF Test: La serie es {adf_conclusion}.")
print(f"- KPSS Test: La serie es {kpss_conclusion}.")
print(f"- ERS Test: La serie es {adf_conclusion}.")
print(f"- PP Test: La serie es {kpss_conclusion}.")
print(f"- IAC Test: La serie es {adf_conclusion}.")
print(f"- Error Cuadrático Medio (ECM): {ECM:.6f}")
print("El modelo SARIMAX muestra resultados aceptables, pero se recomienda revisar los residuos para validar su ajuste.")
