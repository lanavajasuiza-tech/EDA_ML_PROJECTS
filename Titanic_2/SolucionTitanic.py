# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 22:20:44 2023

@author: borja
"""

import os
import pandas as pd
os.chdir(r"C:\Users\borja\OneDrive\Documents\Formacion_Python\17-Modelos\Titanic")

data = pd.read_csv("train.csv")

AnalisisNumericas=data.describe()
data.columns

data["Survived"].value_counts()/891
# Eliminamos las variables que no pueden incluirse en el modelo
del(data["PassengerId"])
del(data["Name"])
del(data["Ticket"])


# Anlizamos los valores perdidos.

Perdidos = data.isnull().sum()

# Vemos que la cabina tiene demasiados valores perdidos, por lo que la eliminamos

del(data["Cabin"])


# Con la edad podemos imputarla o eliminarla.
# Probamos las dos opciones a ver cual de mejor

# Primero eliminamos.
datoscomp=data.dropna()


# Binomilizamos las variables
True * 9
datos_completos = pd.get_dummies(datoscomp, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)


# Probamos los modelos vistos en el curso

y=datos_completos['Survived']
X = datos_completos.drop(['Survived'], axis = 1)


# Regresion Logistica

# Definimos el modelo
from sklearn.linear_model import LogisticRegression
RegLog = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,penalty='l2', random_state=None, solver='liblinear', tol=0.0001,verbose=0, warm_start=False)

# Entremos el modelo
RegLog.fit(X, y)

# Realizamos las predicciones y evaluamos

from sklearn.metrics import accuracy_score

# Predecimos sobre train
predicciones = RegLog.predict(X)
accuracy_score(y, predicciones)

# Realizamos la validacion cruzada

from sklearn.model_selection import cross_val_score
Acierto = cross_val_score(RegLog, X, y, cv=10)
Acierto
Acierto.mean()


# KNN

# Cargamos las librerias
from sklearn.neighbors import KNeighborsClassifier

# Definimos el modelo

knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=1, n_neighbors=5, p=2, weights='uniform')

# Entrenamos
knn.fit(X, y)

# Predecimos sobre train
predicciones = knn.predict(X)
accuracy_score(y, predicciones)

# Utilizamos la validacion cruzada
Acierto = cross_val_score(knn, X, y, cv=10)
Acierto
Acierto.mean()


# Nayve Bayes

# Cargamos las librerias

from sklearn.naive_bayes import GaussianNB

# Definimos el modelo
gnb = GaussianNB()

# Entrenamos
gnb.fit(X, y)

# Predecimos sobre train
predicciones = gnb.predict(X)
accuracy_score(y, predicciones)

# Utilizamos la validacion cruzada
Acierto = cross_val_score(gnb, X, y, cv=10)
Acierto
Acierto.mean()


# Arbol de decision

# Cargamos las librerias

from sklearn import tree


# Definimos el modelo
arbol=tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=30,max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, random_state=None, splitter='best')

# Entrenamos
arbol.fit(X, y)

# Predecimos sobre train
predicciones = arbol.predict(X)
accuracy_score(y, predicciones)

# Utilizamos la validacion cruzada
Acierto = cross_val_score(arbol, X, y, cv=10)
Acierto
Acierto.mean()


# Random Forest

# Cargamos las librerias

from sklearn.ensemble import RandomForestClassifier


# Definimos el modelo
RF=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=4, max_features='sqrt', max_leaf_nodes=None,min_impurity_decrease=0.0,min_samples_leaf=1, min_samples_split=2,min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=-1, oob_score=False, random_state=None, verbose=0,warm_start=False)

# Entrenamos
RF.fit(X, y)

# Predecimos sobre train
predicciones = RF.predict(X)
accuracy_score(y, predicciones)

# Utilizamos la validacion cruzada
Acierto = cross_val_score(RF, X, y, cv=10)
Acierto
Acierto.mean()


# XGBoost

# Cargamos las librerias

import xgboost as xgb


# Definimos el modelo
XGB = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=15,min_child_weight=1,missing=1,n_estimators=100,objective='binary:logistic',reg_alpha=0,reg_lambda=1,scale_pos_weight=1, seed=0, subsample=1)

# Entrenamos
XGB.fit(X, y)

# Predecimos sobre train
predicciones = XGB.predict(X)
accuracy_score(y, predicciones)

# Utilizamos la validacion cruzada
Acierto = cross_val_score(XGB, X, y, cv=10)
Acierto
Acierto.mean()



# Estimaos las edades faltantes

import pandas as pd
import numpy as np

datoNum1=data.loc[:, data.dtypes == np.float64]
datoNum2=data.loc[:, data.dtypes == np.int64]

datoNum = pd.concat([datoNum1, datoNum2], axis=1)


datoNoNum=data.loc[:, data.dtypes == np.object]


datoNum=pd.DataFrame(datoNum)
datoNoNum=pd.DataFrame(datoNoNum)

from fancyimpute  import IterativeImputer 
my_imputer = IterativeImputer()
datoNumcomp = my_imputer.fit_transform(datoNum)
datoNumcomp = pd.DataFrame(datoNumcomp)
datoNumcomp.columns = datoNum.columns

datoNumcomp = pd.DataFrame(datoNumcomp)

datos_completos= pd.concat([datoNoNum, datoNumcomp], axis=1)


datos_completos = pd.get_dummies(datos_completos, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)

y=datos_completos['Survived']
X = datos_completos.drop(['Survived'], axis = 1)


# Regresion Logistica

# Definimos el modelo
from sklearn.linear_model import LogisticRegression
RegLog = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,penalty='l2', random_state=None, solver='liblinear', tol=0.0001,verbose=0, warm_start=False)

# Entremos el modelo
RegLog.fit(X, y)

# Realizamos las predicciones y evaluamos

from sklearn.metrics import accuracy_score

# Predecimos sobre train
predicciones = RegLog.predict(X)
accuracy_score(y, predicciones)

# Realizamos la validacion cruzada

from sklearn.model_selection import cross_val_score
Acierto = cross_val_score(RegLog, X, y, cv=10)
Acierto
Acierto.mean()


# KNN

# Cargamos las librerias
from sklearn.neighbors import KNeighborsClassifier

# Definimos el modelo

knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=1, n_neighbors=5, p=2, weights='uniform')

# Entrenamos
knn.fit(X, y)

# Predecimos sobre train
predicciones = knn.predict(X)
accuracy_score(y, predicciones)

# Utilizamos la validacion cruzada
Acierto = cross_val_score(knn, X, y, cv=10)
Acierto
Acierto.mean()


# Nayve Bayes

# Cargamos las librerias

from sklearn.naive_bayes import GaussianNB

# Definimos el modelo
gnb = GaussianNB()

# Entrenamos
gnb.fit(X, y)

# Predecimos sobre train
predicciones = gnb.predict(X)
accuracy_score(y, predicciones)

# Utilizamos la validacion cruzada
Acierto = cross_val_score(gnb, X, y, cv=10)
Acierto
Acierto.mean()


# Arbol de decision

# Cargamos las librerias

from sklearn import tree


# Definimos el modelo
arbol=tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, random_state=None, splitter='best')

# Entrenamos
arbol.fit(X, y)

# Predecimos sobre train
predicciones = arbol.predict(X)
accuracy_score(y, predicciones)

# Utilizamos la validacion cruzada
Acierto = cross_val_score(arbol, X, y, cv=10)
Acierto
Acierto.mean()


# Random Forest

# Cargamos las librerias

from sklearn.ensemble import RandomForestClassifier


# Definimos el modelo
RF=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=7, max_features='sqrt', max_leaf_nodes=None,min_impurity_decrease=0.0,min_samples_leaf=1, min_samples_split=2,min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=-1, oob_score=False, random_state=None, verbose=0,warm_start=False)

# Entrenamos
RF.fit(X, y)

# Predecimos sobre train
predicciones = RF.predict(X)
accuracy_score(y, predicciones)

# Utilizamos la validacion cruzada
Acierto = cross_val_score(RF, X, y, cv=10)
Acierto
Acierto.mean()


# XGBoost

# Cargamos las librerias

import xgboost as xgb


# Definimos el modelo
XGB = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=6,min_child_weight=1,missing=1,n_estimators=100,objective='binary:logistic',reg_alpha=0,reg_lambda=1,scale_pos_weight=1, seed=0, subsample=1)

# Entrenamos
XGB.fit(X, y)

# Predecimos sobre train
predicciones = XGB.predict(X)
accuracy_score(y, predicciones)

# Utilizamos la validacion cruzada
Acierto = cross_val_score(XGB, X, y, cv=10)
Acierto
Acierto.mean()



# El mejor modelo es la RF

# Cargamos los datos de test
test = pd.read_csv("test.csv")


# Eliminamos las variables sobrantes

del(test["PassengerId"])
del(test["Name"])
del(test["Ticket"])
del(test["Cabin"])

# Imputamos los valores perdidos

testNum1=test.loc[:, test.dtypes == np.float64]
testNum2=test.loc[:, test.dtypes == np.int64]

testNum = pd.concat([testNum1, testNum2], axis=1)


testNoNum=test.loc[:, test.dtypes == np.object]


testNum=pd.DataFrame(testNum)
testNoNum=pd.DataFrame(testNoNum)

from fancyimpute  import IterativeImputer 
my_imputer = IterativeImputer()
testNumcomp = my_imputer.fit_transform(testNum)
testNumcomp = pd.DataFrame(testNumcomp)
testNumcomp.columns = testNum.columns

testNumcomp = pd.DataFrame(testNumcomp)

test_completos= pd.concat([testNoNum, testNumcomp], axis=1)

# Binominalizamos las variables categoricas
test_completos = pd.get_dummies(test_completos, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)

# Realizamos las predicciones
test_completos.columns
predicciones = RF.predict(test_completos)

# Cargamos Gender Submision
submit = pd.read_csv("gender_submission.csv")

# Pegamos las predicciones.
submit["Survived"] = predicciones
submit["Survived"] = submit["Survived"].astype(int) 
# Guardamos el CSV
submit.to_csv("Subir.csv", index = False)
