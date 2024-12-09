# Fuente: https://planetachatbot.com/ejecutar-30-modelos-aprendizaje-automatico-lineas-codigo/
## #
# Pyforest es una biblioteca de Python diseñada para facilitar el trabajo con datos al reducir el tiempo dedicado a 
# importar las librerías más comunes de análisis y visualización de datos. Es especialmente útil para quienes trabajan 
# en proyectos rápidos o exploratorios, ya que permite comenzar a escribir código sin necesidad de declarar 
# explícitamente las importaciones.
#######################################################################################################################



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
'''
En este caso imprime las librerías usadas: ['import os', 'import pandas as pd']
'''

data["State"].value_counts()
data["Churn"].value_counts()/3333

# El primer paso es conocer nuestros datos.
# Realizamos una primera visualizacion.
# EL objetivo es precedir que clientes van a marchar de la compaía telefónica.

print(data)
print(active_imports()) 

# Visualizamos los 10 primero datos, de una manera mas comoda.

data.head(n=10)

# Realizamos un resumen estadistico de las variables.

data.describe()


# Comprobamos que no hay valores perdidos.

data.isnull().sum()


# Con esta funcion nos vemos si existen (TRUE) o no (FALSE) datos perdidos

data.isnull().any().any()

# Tras comporbar que no existen valores perdidos analizamos las variables.
# El objetico es determinar si todas las variables son utiles para la modelizacion.
# Hay una serie de variables que no nos proporcionan informacion.
# El estado en el que vive un cliente no aporta informacion sobre su comportamiento.
# Lo mismo ocurre con los codigos identificativos y con el numero de telefono.
# Eliminamos esas variables

len(data['State'].value_counts())
del data['State']
del data['Account Length']
del data['Area Code']
del data['Phone']

# El reto de las variables pueden ser relevantes para el analisis.
# Tenemos variables numericas y variables categoricas.
# Las variables categoricas deben ser binominalizadas o dumified o binarizada, es decir que tome dos valores, 0 y 1 normalmente.
# En este caso dado que las variables solo toman dos opciones se puede hacer "a mano".

data.loc[:,"IntPlan"]=0
data.loc[data["Int'l Plan"]=="yes","IntPlan"]=1

# Si tuviesemos mas categorias esto seria un gran trabajo.
# Con la funcion get dummies creamos un nuevo conjunto de datos en el que cada 
# posible opcion de una variable pasa a ser una variable dummy.

vmailplan2 = pd.get_dummies(data['VMail Plan'], dtype="uint8")
churn2 = pd.get_dummies(data['Churn'], dtype="uint8")

# Al crear los nombres de las variables con un "." al final tenemos problemas.
# Por ello cambiamos el nombre de estas variables

churn2.columns=("Falsee","Truee")

# Una vez que ya hemos hecho la transformacion juntamos todas las variables en un data frame.

datosfinal=pd.concat([data, vmailplan2.yes,churn2.Truee], axis=1)

#Eliminamos las variables originales que hemos transformado

del datosfinal['Churn']
del datosfinal['VMail Plan']
del datosfinal["Int'l Plan"]

# Cambiamos el nombre a aquellas variables que nos interesa.

datosfinal.columns.values[15] = "VmailPlan"
datosfinal.columns.values[16] = "Churn"

# Volvemos a visualizar los datos. Que ya están listo para modelaizar ya que tienen todos datos númericos y los títulos más aclarados.

datosfinal.head(n=10)

# Para esta primera modelizacion hacemos la siguiente particion. Del total de 3333 filas tenemos que:
# Para entrenar: 3000 observaciones
# PAra test 333 observaciones

train_data = datosfinal[:3000]
test_data = datosfinal[3000:]

# Separamos las variables explicativas y la variable a predecir en train

train_data_X = train_data.drop(['Churn'], axis = 1) # todos los valors excepto los de la columna churn que es la predicción usuarios susceptibles de irse, del total 3000
train_data_y = train_data['Churn'] # valor predictivo, sobre la columna churn es la que va a predecir cuantos usuarios son susceptibles de irse, del total 3000

# Separamos las vaariables explicativas y la variable a predecir en test

test_data_X = test_data.drop(['Churn'], axis = 1) # todos los valors excepto los de la columna churn que es la predicción usuarios susceptibles de irse, del test 033300
test_data_y = test_data['Churn'] # valor predictivo, sobre la columna churn es la que va a predecir cuantos usuarios son susceptibles de irse, del test 333

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

# Ordenar por la columna 'Accuracy' en orden descendente

models_sorted = models.sort_values(by='Accuracy', ascending=False) # orden por una columna
models_sorted = models.sort_values(by=['Accuracy', 'Time Taken'], ascending=[False, True]) # orden por varias columnas

'''
Con los datos de la tabla podemos ordenar según preferencia de columna para elgier el modelo que mejor se ajuste al estudio
'''

# Mostrar la tabla ordenada

print(models_sorted)



