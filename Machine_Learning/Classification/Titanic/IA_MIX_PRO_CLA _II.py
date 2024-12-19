######################################################################################
#                          FLUJO DE TRABAJO PARA CLASIFICACIÓN                       #
######################################################################################
# Este script realiza un análisis end-to-end de un dataset de clasificación:
#
# 1. Carga y exploración inicial del dataset.
# 2. Preprocesamiento:
#    - Manejo de valores faltantes.
#    - Eliminación de columnas irrelevantes.
#    - Transformación de variables categóricas.
# 3. División del dataset en entrenamiento y prueba.
#    - Verificar el balance de clases.
#    - Validacion SMOTE según salida de balance de clases.
# 4. Entrenamiento de múltiples modelos:
#    - Regresión Logística.
#    - KNN.
#    - Naive Bayes.
#    - Árboles de Decisión.
#    - Random Forest.
#    - XGBoost.
# 5. Evaluación de modelos:
#    - Métricas: Accuracy, Precision, Recall.
#    - Matrices de Confusión.
# 6. Validación cruzada para estimar el rendimiento general.
# 7. Comparación automática de modelos con LazyClassifier.
# 8. Visualización de resultados.
######################################################################################

# --- 1. IMPORTACIÓN DE LIBRERÍAS ---
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from lazypredict.Supervised import LazyClassifier
from imblearn.over_sampling import SMOTE

# --- 2. CARGA Y EXPLORACIÓN DE DATOS ---
print("\n--- CARGA Y EXPLORACIÓN ---")
# Ruta al dataset
data_path = r"/ruta/a/tu/dataset/"
os.chdir(data_path)
data = pd.read_csv("dataset.csv")

# Información básica
print(data.head())
print("\nInformación del dataset:")
print(data.info())

# Identifica la columna objetivo
target_column = "Survived"  # Cambiar según dataset
if target_column not in data.columns:
    raise ValueError(f"Columna objetivo '{target_column}' no encontrada.")

# --- 3. LIMPIEZA Y TRANSFORMACIÓN ---
print("\n--- LIMPIEZA DE DATOS ---")

# Manejo de valores faltantes
for col in data.select_dtypes(include=["float", "int"]).columns:
    data[col].fillna(data[col].median(), inplace=True)
for col in data.select_dtypes(include=["object"]).columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Eliminar columnas irrelevantes
cols_to_drop = ["PassengerId", "Name", "Cabin", "Ticket"]
data.drop(columns=[col for col in cols_to_drop if col in data.columns], inplace=True, errors="ignore")

# Transformación de categóricas
data = pd.get_dummies(data, drop_first=True)

# --- 4. DIVISIÓN Y BALANCEO ---
print("\n--- DIVISIÓN EN ENTRENAMIENTO Y PRUEBA ---")
X = data.drop(columns=[target_column])
y = data[target_column]

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Verificar el balance de clases ---
class_distribution = y.value_counts(normalize=True) * 100
print("\nDistribución de clases (%):")
print(class_distribution)

# Obtener la clase minoritaria
minority_class_percentage = class_distribution.min()

# Decidir si aplicar SMOTE
if minority_class_percentage < 30:
    print(f"La clase minoritaria tiene solo el {minority_class_percentage:.2f}% de las muestras. Aplicaremos SMOTE.")
    
    # Aplicar SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
else:
    print(f"La clase minoritaria tiene un {minority_class_percentage:.2f}% de las muestras. No aplicaremos SMOTE.")
    X_train_balanced, y_train_balanced = X_train, y_train


# NOTA: Aplicar SMOTE para balancear las clases en función de la salida anterior
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# --- 5. ENTRENAMIENTO Y EVALUACIÓN ---
def evaluar_modelo(nombre, modelo, X_test, y_test):
    predicciones = modelo.predict(X_test)
    print(f"\n--- {nombre} ---")
    print(f"Accuracy: {accuracy_score(y_test, predicciones):.4f}")
    print(f"Precision: {precision_score(y_test, predicciones):.4f}")
    print(f"Recall: {recall_score(y_test, predicciones):.4f}")
    print(f"F1 Score: {f1_score(y_test, predicciones):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, modelo.predict_proba(X_test)[:, 1]):.4f}")
    sns.heatmap(confusion_matrix(y_test, predicciones), annot=True, fmt="d", cmap="coolwarm")
    plt.title(f"Matriz de Confusión - {nombre}")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

# Modelos a evaluar
modelos = {
    "Logistic Regression": LogisticRegression(max_iter=100, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    evaluar_modelo(nombre, modelo, X_test, y_test)

# --- 6. VALIDACIÓN CRUZADA ---
print("\n--- VALIDACIÓN CRUZADA ---")
for nombre, modelo in modelos.items():
    scores = cross_val_score(modelo, X, y, cv=5, scoring="accuracy")
    print(f"{nombre} - Accuracy promedio: {scores.mean():.4f}")

# --- 7. COMPARACIÓN CON LAZYCLASSIFIER ---
print("\n--- LAZYCLASSIFIER ---")
lazy_clf = LazyClassifier(verbose=0, ignore_warnings=True)
models, predictions = lazy_clf.fit(X_train, X_test, y_train, y_test)
print(models)

plt.figure(figsize=(12, 8))
sns.barplot(x=models.index, y=models["Accuracy"])
plt.title("Comparación de Modelos - LazyClassifier")
plt.xticks(rotation=45, ha="right")
plt.show()
