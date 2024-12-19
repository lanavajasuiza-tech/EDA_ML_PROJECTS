#############################################################
# Este método de flujo de trabajo para revisar cual el
# el modelo de ML que mejor puede adaptarse a nuestro estudio
#############################################################

import os
import pandas as pd
from pyforest import *   
'''
Acumula las librerías básicas EDA (pandas, os, numpy, mathlib, etc) solo funciona cuando se llama a alguna de estas
'''
print(active_imports()) 
'''
Para ver que librerías está usando, en este caso saldrá vacía porque no esta llamando a nada
'''

os.chdir(r"/home/ana/Documentos/CURSO DATA SCIENCE/MACHINE_LEARNING_&_INTELIGENCIA_ARTIFICIAL/Curso_ML_Laner/17-Modelos/Modelos_Clasificacion/")
data = pd.read_csv("churn.csv")
print(active_imports()) 

# SUPONEMOS QUE TENEMOS UN DATA SET PERFECTO Y LIMPIO
data.head(n=10)

# Para esta primera modelizacion hacemos la siguiente particion. 

train_data = data[:3000] # Hasta que fila será train
test_data = data[3000:]  # Desde que fila será test

# Separamos las variables explicativas y la variable a predecir en train

train_data_X = train_data.drop(['Variable_objetivo'], axis = 1) # todos los valors excepto los de la columna churn que es la predicción usuarios susceptibles de irse, del total 3000
train_data_y = train_data['Variable_objetivo'] # valor predictivo, sobre la columna churn es la que va a predecir cuantos usuarios son susceptibles de irse, del total 3000

# Separamos las vaariables explicativas y la variable a predecir en test

test_data_X = test_data.drop(['Variable_objetivo'], axis = 1) # todos los valors excepto los de la columna churn que es la predicción usuarios susceptibles de irse, del test 033300
test_data_y = test_data['Variable_objetivo'] # valor predictivo, sobre la columna churn es la que va a predecir cuantos usuarios son susceptibles de irse, del test 333

# Creamos el modelo

#from sklearn.linear_model import LogisticRegression

import lazypredict
from lazypredict.Supervised import LazyClassifier
print("LazyPredict instalado correctamente")


clf = LazyClassifier(verbose=0,ignore_warnings=True)
models, predictions = clf.fit(train_data_X, test_data_X, train_data_y, test_data_y)
models
'''
La tabla contiene las siguientes columnas: 
Accuracy (Precisión): Proporción de predicciones correctas en relación con el total de datos, ideal para conjuntos de datos balanceados. 
Balanced Accuracy: Media de las sensibilidades (recall) de cada clase, útil para conjuntos de datos desbalanceados. 
ROC AUC: Evalúa qué tan bien separa el modelo las clases; un valor cercano a 1.0 indica buena separación, ideal para problemas binarios. 
F1 Score: Promedio armónico entre precisión (precision) y sensibilidad (recall), útil para evitar tanto falsos positivos como falsos negativos. 
Time Taken: Tiempo que tardó cada modelo en entrenarse y evaluarse, útil para medir la eficiencia computacional.
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Entrenar el modelo
model = RandomForestClassifier(random_state=42)
model.fit(train_data_X, train_data_y)

# Predecir en el conjunto de prueba
y_pred = model.predict(test_data_X)

# Generar y mostrar la matriz de confusión
print("Matriz de confusión:")
print(confusion_matrix(test_data_y, y_pred))


