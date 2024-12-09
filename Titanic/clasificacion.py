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

os.chdir(r"C:\Users\rportatil112\Documents\CURSO-DATA-SCIENCE\MACHINE_LEARNING_&_INTELIGENCIA_ARTIFICIAL\Curso_ML_Laner\17-Modelos\Modelos_Clasificacion")
data = pd.read_csv("churn.csv")
data.describe()

data["State"].value_counts()
data["Churn"].value_counts()/3333 #False se queda y True se va, en esta línea saca el porcentajo a partir de la columna churn
# El primer paso es conocer nuestros datos.
# Realizamos una primera visualizacion.
# EL objetivo es precedir que clientes van a marchar de la compaía telefónica.
print(data)

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

import pandas as pd

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

from sklearn.linear_model import LogisticRegression

RegLog = LogisticRegression(C=1.0, class_weight="balanced", dual=False, fit_intercept=True,intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,penalty='l2', random_state=None, solver='liblinear', tol=0.0001,verbose=0, warm_start=False)
'''
Explicación de cada elemento, RegLog es una sentencia bastante estandar, se podrá ajustar valores depende del escenaro pero básicamente esa es la línea para este modelo.
RegLog = LogisticRegression(
    C=1.0,                  # Control de regularización (1.0 es el valor predeterminado; menor valor = mayor regularización)
    class_weight="balanced",# Ajusta el peso automáticamente según la frecuencia de las clases (útil para datos desbalanceados)
    dual=False,             # Solo aplica para problemas muy grandes; desactivado aquí porque no es necesario
    fit_intercept=True,     # Calcula el término independiente (intercepto) del modelo
    intercept_scaling=1,    # Escala del intercepto (solo afecta si se usa 'liblinear' como solver)
    max_iter=100,           # Máximo número de iteraciones para que el algoritmo de optimización converja
    multi_class='ovr',      # Clasificación multiclase: "One-vs-Rest" (una clase contra todas las demás)
    n_jobs=1,               # Número de procesadores a usar; 1 significa usar solo un núcleo
    penalty='l2',           # Tipo de regularización: 'l2' (Ridge), útil para prevenir sobreajuste
    random_state=None,      # Semilla para la generación de números aleatorios; None significa que es aleatorio
    solver='liblinear',     # Algoritmo de optimización (liblinear es eficiente para problemas pequeños y binarios)
    tol=0.0001,             # Tolerancia para la convergencia del algoritmo
    verbose=0,              # Nivel de mensajes de salida; 0 significa que no se muestran mensajes
    warm_start=False        # Si es True, reutiliza soluciones previas como punto de partida; False reinicia el modelo
)

'''
# Aplicamos el modelo a los datos de train

RegLog.fit(train_data_X, train_data_y) 
'''
fit: Es el método universal en Scikit-learn para entrenar modelos. 
Ajusta el modelo a los datos de entrenamiento.
'''
RegLog.coef_
'''
Devuelve los coeficientes que representan el impacto de las variables predictoras en el modelo.
'''

# Realizamos la prediccion sobre los datos de test

prediccion = RegLog.predict(test_data_X)
print(prediccion) 
'''la predicción esperada en este punto son un motón de 0 y 1'''

#score(test_data_y, prediccion)
# Calculamos l¡el nivel de acierto en train y en test

from sklearn.metrics import confusion_matrix

confusion_matrix(test_data_y, prediccion)
''' Y aquí tenemos la predicción!!! y es lo que tenemos que interpretar a partir de la matriz de confusión binaria
la salida nos indica que según la matriz ha 64 falsos positivos y 7 falsos negativos, los falsos positivos son los
usuarios que sé que se van a ir por lo que puedo hacer campaña de márketin para que se queden pero los ocultos son los 
falsos negativos, que son aquello que no sé porqué se van a ir (ver gráfica explicación en zDoc.md)
'''


# class_weight = None
(278+6)/333 # Acierto
(278 + 8)/333 # Prop No Fugados.
278/(278 + 8) # Specificity
6/(41 + 6) # Sensitivity
6/(6 + 8) # Precision

# class_weight = Balanced
(221+40)/333 # Acierto
(221 + 65)/333 # Prop No Fugados.
221/(221 + 65) # Specificity
40/(7 + 40) # Sensitivity
40/(65 + 40) # Precision



# 278 predice que no se fugan y no lo hacen
# 41 precide que no se fugan y si lo hacen
# 8 poredice que se fugan y no lo hacen
# 6 Predice que se fugan y lo hacen.

# �Que os parecen estos resultados?

# Repetimos el proceso normalizando los datos

# El primer punto es ver el formato que tienen los datos

datosfinal.dtypes


# Vemos que Churn, IntPlan y VMail Plan son variables categoriacas (si o no).
# Por ello no deben de ser normalizadas.
# Sin embargo IntPlan no lo reconoce como integer por lo que hay que cambiarlo.

datosfinal["IntPlan"]=datosfinal["IntPlan"].astype(np.uint8)


# El primer paso es separar los numericos de los categoricos
# Hay otras variables que podemos cambiarles el formato.

datosfinal.dtypes

import pandas as pd
import numpy as np

a=pd.DataFrame(datosfinal.dtypes)
NumNames=a[(a[0]=="float")]
NumNamesInt=a[(a[0]=="int64")]
NumNamesObj=a[(a[0]=="uint8")]

datosNum=pd.DataFrame(datosfinal.select_dtypes(include=[float]).to_numpy(),columns=NumNames.index)
datosNumInt=pd.DataFrame(datosfinal.select_dtypes(include=[np.int64]).to_numpy(),columns=NumNamesInt.index)
datosNumObj=pd.DataFrame(datosfinal.select_dtypes(include=[np.uint8]).to_numpy(),columns=NumNamesObj.index)


datosAnorm=pd.concat([datosNum,datosNumInt],axis=1)

from sklearn import preprocessing
datosNorm=preprocessing.scale(datosAnorm)
datosNorm=pd.DataFrame(datosNorm)
datosNorm.columns=datosAnorm.columns

datosNumObj=datosNumObj.astype(np.int64)

datosNumObj.dtypes
print(datosNumObj)

datosfinalNorm=pd.concat([datosNumObj,datosNorm],axis=1)



train_dataN = datosfinalNorm[:3000]
test_dataN = datosfinalNorm[3000:]



# Separamos las variables explicativas y la variable a predecir en train

train_data_X = train_dataN.drop(['Churn'], axis = 1)
train_data_y = train_dataN['Churn']

# Separamos las vaariables explicativas y la variable a predecir en test

test_data_X = test_dataN.drop(['Churn'], axis = 1)
test_data_y = test_dataN['Churn']

# Creamos el modelo

from sklearn.linear_model import LogisticRegression
RegLog = LogisticRegression(C=1.0, class_weight="balanced", dual=False, fit_intercept=True,intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,penalty='l2', random_state=None, solver='liblinear', tol=0.0001,verbose=0, warm_start=False)

# Aplicamos el modelo a los datos de train

RegLog.fit(train_data_X, train_data_y)

# Realizamos la prediccion sobre los datos de test

prediccion = RegLog.predict(test_data_X)

# Calculamos l¡el nivel de acierto en test

from sklearn.metrics import confusion_matrix

confusion_matrix(test_data_y, prediccion)


#OTRO TIPO DE MODELO KNN o K vecinos mas cercanos #

# Una vez que tenemos los datos normalizados y sin normalizar los utilizaremos para los proximos modelos

# En el modelo anterior hemos hecho una particion "a mano"
# Se han dejado los ultimos 333 datos para test.
# En este caso haremos una particion seleccionando datos al azar para test y train


y=datosfinal['Churn']

# Importamos la funcion necesaria.
from sklearn.model_selection import train_test_split

# Realizamos la particion dejando un 75% para entrenar y un 25% de test.
X_train, X_test, y_train, y_test = train_test_split(datosfinal, y, test_size=0.25)

train_data_X = X_train.drop(['Churn'], axis = 1)
train_data_y = X_train['Churn']

test_data_X = X_test.drop(['Churn'], axis = 1)
test_data_y = X_test['Churn']


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=1, n_neighbors=5, p=2, weights='uniform')



knn.fit(train_data_X, train_data_y)

# Realizamos la prediccion sobre los datos de test

prediccion = knn.predict(test_data_X)

# Calculamos l¡el nivel de acierto en train y en test

from sklearn.metrics import confusion_matrix

confusion_matrix(test_data_y, prediccion)

18/(29 + 18)

# Tambien podemos analizar otras metricas
# Accuracy muestra el porcentaje de aciertos del modelo
from sklearn.metrics import accuracy_score
accuracy_score(test_data_y, prediccion)

# la precision mide el porcentaje de aciertos que tiene el modelo cuando predice 0 (no se fuga)
# La sensitividad mide el porcentaje de acierto cuando predice 1 (fuga)

from sklearn import metrics
metrics.precision_score(prediccion, test_data_y)
metrics.recall_score(test_data_y, prediccion)
metrics.precision_score(test_data_y,prediccion)



# A continuacion repetimos el proceso con los datos normalizados.


y=datosfinalNorm['Churn']

# Importamos la funcion necesaria.
from sklearn.model_selection import train_test_split

# Realizamos la particion dejando un 80% para entrenar y un 20% de test.
X_train, X_test, y_train, y_test = train_test_split(datosfinalNorm, y, test_size=0.25)

train_data_X = X_train.drop(['Churn'], axis = 1)
train_data_y = X_train['Churn']

test_data_X = X_test.drop(['Churn'], axis = 1)
test_data_y = X_test['Churn']


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=1, n_neighbors=5, p=2, weights='uniform')



Vecinos=knn.fit(train_data_X, train_data_y)

# Realizamos la prediccion sobre los datos de test

prediccion = knn.predict(test_data_X)

# Calculamos l¡el nivel de acierto en train y en test

from sklearn.metrics import confusion_matrix

confusion_matrix(test_data_y, prediccion)




# Tambien podemos analizar otras metricas
# Accuracy muestra el porcentaje de aciertos del modelo
from sklearn.metrics import accuracy_score
accuracy_score(test_data_y, prediccion)

# la precision mide el porcentaje de aciertos que tiene el modelo cuando predice 0 (no se fuga)
# La sensitividad (recall) midel el porcentaje de acierto cuando predice 1 (fuga)

from sklearn import metrics
metrics.precision_score(test_data_y, prediccion)
metrics.recall_score(test_data_y, prediccion)


# El siguiente modelo es Nayve Bayes
# Una forma de determinar la fiabilidad de los modelos es la validacion cruzada.
# Este sistema hace varias particiones de los datos para fijar train y test.

y = datosfinal['Churn']
X = datosfinal.drop(['Churn'], axis = 1)


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

Bayes=gnb.fit(X, y)
predicciones = gnb.predict(X)
accuracy_score(y, predicciones)

from sklearn.model_selection import KFold
from sklearn import metrics

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
correlaciones = X.corr()
cross_val_score(Bayes, X, y, cv=10)
cross_val_score(Bayes, X, y, cv=10, scoring='precision')
cross_val_score(Bayes, X, y, cv=10, scoring='recall')

# CC
# CX
# XC
# XX

# P(C) * P(C) = 0,5 * 0,5 = 0,25 # INDEPENDIENTES

# SR_SR = 4/40 * 3/39 != P(R) * P(R) 
# SR_NR
# NR_SR
# NR_NR

# Arbol de decision #

from sklearn import tree

modelo=tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=8,max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, random_state=None, splitter='best')

arbol = modelo.fit(X, y)

predicciones = arbol.predict(X)
accuracy_score(y, predicciones)

cross_val_score(arbol, X, y, cv=10).mean()
cross_val_score(arbol, X, y, cv=10, scoring='precision')
cross_val_score(arbol, X, y, cv=10, scoring='recall')

importanciasarbol=pd.DataFrame(arbol.feature_importances_)
importanciasarbol
importanciasarbol.index=(X.columns)
importanciasarbol



os.environ["PATH"] += os.pathsep + r'C:\Users\rportatil112\anaconda3\Library\bin\Graphviz-12.2.0-win64'
from sklearn.tree import export_graphviz
#from sklearn.externals import six  
from IPython.display import Image  
import pydotplus
#from io import StringIO # standard para python
from six import StringIO # solo para python 2 y 3
 
from sklearn import tree
tree.plot_tree(arbol, feature_names = X.columns, class_names=['0','1'], filled=True) 


dot_data = StringIO()

export_graphviz(modelo, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = X.columns,class_names=['0','1'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 

import graphviz
gvz_graph = graphviz.Source(graph.to_string())
gvz_graph

##############################################
############### SEGUNDO METODO ###############
##############################################

import graphviz
os.environ["PATH"] += os.pathsep + r'C:\Users\rportatil112\anaconda3\Library\bin\Graphviz-12.2.0-win64'
dot_data = tree.export_graphviz(arbol, out_file=None, 
                                feature_names=X.columns,  
                                class_names=['0','1'],
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph


# Random Forest#

from sklearn.ensemble import RandomForestClassifier

modelo=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=None, max_features='sqrt', max_leaf_nodes=None,min_impurity_decrease=0.0,min_samples_leaf=1, min_samples_split=2,min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=-1, oob_score=False, random_state=None, verbose=0,warm_start=False)


RF = modelo.fit(X, y)

predicciones = RF.predict(X)
accuracy_score(y, predicciones)

cross_val_score(RF, X, y, cv=10)
cross_val_score(RF, X, y, cv=10, scoring='precision')
cross_val_score(RF, X, y, cv=10, scoring='recall')


importancias=pd.DataFrame(RF.feature_importances_)
importancias.index=(X.columns)
importancias



#XGBoost#
# El primer paso es instalarlo.
# Abrir Anaconda Promt
# Copiar: conda install -c anaconda py-xgboost
# Para mas ayuda: https://stackoverflow.com/questions/35139108/how-to-install-xgboost-in-anaconda-python-windows-platform

import xgboost as xgb

modelo = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=6,min_child_weight=1,missing=1,n_estimators=1000,objective='binary:logistic',reg_alpha=0,reg_lambda=1,scale_pos_weight=1, seed=0, subsample=1)

XGBoost=modelo.fit(X, y)
predicciones = XGBoost.predict(X)
accuracy_score(y, predicciones)

cross_val_score(XGBoost, X, y, cv=10)
cross_val_score(XGBoost, X, y, cv=10, scoring='precision')
cross_val_score(XGBoost, X, y, cv=10, scoring='recall')


importanciasxgb=pd.DataFrame(XGBoost.feature_importances_)
importanciasxgb.index=(X.columns)
importanciasxgb

