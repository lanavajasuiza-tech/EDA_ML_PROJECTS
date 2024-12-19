# Los modelos de clasificacion son modelos que buscan predecir si una observacion
# pertenc¡ece a un determinado grupo o a otro en funcion de sus caracteristicas


# Importamos los datos

#%reset

# EL primer modelo de calsificacion es la regresion logistica.
# Este modelo es una regresion que mide la probabilidad de que una observacion
# tome el valor 1 o 0.

# Para aplicar este modelo es necesario que todos los valores tengan un formato numerico
# Para ello hacemos las siguientes transformaciones.

import os
import pandas as pd
import numpy as np

os.chdir(r"/home/ana/Documentos/CURSO_DATA_SCIENCE/MACHINE_LEARNING_E_INTELIGENCIA_ARTIFICIAL/Curso_ML_Laner/17-Modelos/Modelos_Clasificacion/titanic/")
data_train = pd.read_csv("train.csv")

# Obtener el número de filas
numero_filas = data_train.shape[0]
print(f"El dataset tiene {numero_filas} filas.")


# Información básica
print(data_train.info())
print(data_train.describe())
print(data_train.head())
print(data_train.tail())


# Comprobamos que no hay valores perdidos.
data_train.isnull().sum()
'''
En este caso vemos que en las campos columna Age y Cabin hay valores NaN de 177 y 687 respectivamente
'''
# Decidimos que valores tienen importancia para medir la supervivencia: Passenger_Id, Cabin, Ticket, ...

data_train.drop("PassengerId", axis=1, inplace=True)
data_train.drop("Cabin", axis=1, inplace=True)
data_train.drop("Ticket", axis=1, inplace=True)
data_train.drop("Name", axis=1, inplace=True)
'''
El id del pasajero no es relevante para la superviviencia
Demasiados valores NaN en este campo
A priori, el ticket tampoco es relevante, el nivel adquisitov está asociado y tiene que ver pero tenemos el campo Pclass...
'''
data_train.isnull().sum()
# Con esta funcion nos vemos si existen (TRUE) o no (FALSE) datos perdidos
data_train.isnull().any().any()
'''
Volvemos a verificar
'''

# Calcular y rellenar con la mediana
data_train["Age"].fillna(data_train["Age"].median(), inplace=True)
print("Valores de 'Age' rellenados con la mediana.")

#Verificamos los tipos que tenemos
print(data_train.info())

#Decidimos qué valores necesitamos convertir. En este caso: Sex y Embarked:
data_train["Sex"] = data_train["Sex"].map({"male": 0, "female": 1})
# Rellenar valores faltantes con la moda
data_train["Embarked"].fillna(data_train["Embarked"].mode()[0], inplace=True)

# Convertir a variables dummies
data_train = pd.get_dummies(data_train, columns=["Embarked"], drop_first=True)

# Volvemos a verificar
print(data_train.info())
print(data_train[["Embarked_Q", "Embarked_S"]].head()) # Verificamos que son valores consistentes
print(data_train.describe())

# Ahora dividimos el data set en un 80% para Train y un 20% para test

from sklearn.model_selection import train_test_split

# Separar características (X) y variable objetivo (y)
X = data_train.drop("Survived", axis=1)  # Todas las columnas excepto "Survived"
y = data_train["Survived"]  # Variable objetivo

# Dividir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificar tamaños
print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} filas")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} filas")

# Aplicamos el modelo de clasificación de Regresión Logística y creamos un RegLog simplificado

from sklearn.linear_model import LogisticRegression

RegLog = LogisticRegression(
    C=1.0,
    class_weight="balanced",
    solver="liblinear",
    max_iter=100,
    random_state=42
)

# Entrenar el modelo
RegLog.fit(X_train, y_train)

# Generar predicciones sobre el conjunto de prueba
prediccion = RegLog.predict(X_test)
print("Predicciones del modelo:")
print(prediccion)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, prediccion)

# Mostrar la matriz en texto
print("Matriz de confusión:")
print(conf_matrix)



# Crear un DataFrame para la matriz de confusión
conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=["Real: No sobreviven (VN)", "Real: Sobreviven (FP)"],
    columns=["Predicción: No sobreviven (FN)", "Predicción: Sobreviven (VP)"]
)

# Mostrar la matriz de confusión como tabla
print("Matriz de Confusión (Formato Tabular):\n")
print(conf_matrix_df.to_string())


# Analizamos con KNN
# pendiente de hacer
# Cogemos la columna sobre la que vamos a predecir



# MODELO NAIVE BAYES

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# Cargar el dataset
os.chdir(r"C:\Users\rportatil112\Documents\CURSO-DATA-SCIENCE\MACHINE_LEARNING_&_INTELIGENCIA_ARTIFICIAL\Curso_ML_Laner\17-Modelos\Modelos_Clasificacion\titanic")
data_train = pd.read_csv("train.csv")

# Preprocesar los datos, normalizar y hacer el balance de los datos
data_train["Sex"] = data_train["Sex"].map({"male": 0, "female": 1})
data_train = pd.get_dummies(data_train, columns=["Embarked"], drop_first=True)
data_train.drop(["Name", "PassengerId", "Ticket", "Cabin"], axis=1, inplace=True)
data_train["Age"].fillna(data_train["Age"].median(), inplace=True)

# Separar características (X) y variable objetivo (y)
X = data_train.drop("Survived", axis=1)
y = data_train["Survived"]

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
gnb = GaussianNB()
Bayes = gnb.fit(X_train, y_train)

# Hacer predicciones y evaluar en el conjunto de prueba
predicciones = gnb.predict(X_test)
print("Precisión en el conjunto de prueba:", accuracy_score(y_test, predicciones))

# Validación cruzada (el train test lo hace intermanente)
print("Precisión promedio (10-fold CV):", cross_val_score(gnb, X, y, cv=10).mean())
print("Precisión (precision):", cross_val_score(gnb, X, y, cv=10, scoring='precision').mean())
print("Recall (recall):", cross_val_score(gnb, X, y, cv=10, scoring='recall').mean())
'''
Interpretación de las Métricas
Precisión en el Conjunto de Prueba (Accuracy):

Precisión en el conjunto de prueba: 0.770949720670391
Significado: El modelo clasifica correctamente el 77.1% de los casos en el conjunto de prueba.
Es una métrica general que muestra el porcentaje de aciertos totales.
Precisión Promedio en Validación Cruzada (10-fold CV):

Precisión promedio (10-fold CV): 0.7822846441947565
Significado: El modelo tiene una precisión promedio de 78.2% a través de 10 particiones del dataset.
Valida la estabilidad del modelo en diferentes divisiones de los datos.
Precisión (Precision):

Precisión (precision): 0.7203479247079275
Significado: De todas las instancias que el modelo predijo como positivas, el 72.0% fueron correctas.
Es importante cuando los falsos positivos tienen un costo alto.
Recall:

Recall (recall): 0.7136134453781514
Significado: De todas las instancias positivas reales, el modelo identificó correctamente el 71.4%.
Es crucial cuando los falsos negativos tienen un costo alto.
'''

# Crear un DataFrame con las métricas
metrics_data = {
    "Métrica": ["Precisión (Prueba)", "Precisión Promedio (10-fold CV)", "Precisión", "Recall"],
    "Valor": [0.7709, 0.7823, 0.7203, 0.7136]
}

metrics_df = pd.DataFrame(metrics_data)

# Mostrar la tabla
print(metrics_df)

#  MODELO DE DECISIONES ÁRBOL DE DECISIONES

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd

# Creación del modelo
modelo = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=8, random_state=42, splitter='best')

# Entrenamiento del modelo
arbol = modelo.fit(X_train, y_train)

# Predicción sobre el conjunto de prueba
predicciones = arbol.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predicciones))

# Evaluación usando validación cruzada
scores = cross_val_score(arbol, X, y, cv=10, scoring=None)  # None para usar la métrica por defecto de la estimación
print("CV Accuracy Score: Mean =", scores.mean())
print("CV Precision:", cross_val_score(arbol, X, y, cv=10, scoring='precision').mean())
print("CV Recall:", cross_val_score(arbol, X, y, cv=10, scoring='recall').mean())

# Importancia de las características
importanciasarbol = pd.DataFrame(arbol.feature_importances_, index=X.columns, columns=['Importancia'])
print(importanciasarbol)
