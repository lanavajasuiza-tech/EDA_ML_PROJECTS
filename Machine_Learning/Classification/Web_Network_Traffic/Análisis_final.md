# Análisis de Tráfico Web y Clasificación de Solicitudes

## Autor: Ana Ndongo

**Fecha:** 20/10/2024
**Dataset:** [Kaggle - Web Firewall Good and Bad Request](https://www.kaggle.com/datasets/rudrakumar96/web-firewall-good-and-bad-request)
**Variable objetivo:** `class`

## Introducción

Este proyecto tiene como objetivo clasificar las solicitudes web como **buenas** o **malas** basándonos en sus características. Para ello, se realizaron varios pasos de análisis y procesamiento del dataset, implementando diferentes modelos de clasificación y evaluando su rendimiento.

---

## 1. **Carga y Exploración de Datos**

### Información general

- **Dimensiones:** 522 filas x 16 columnas.
- **Columnas categóricas y numéricas.**
- **Valores nulos:**
  - La columna `body` contiene un **80% de valores nulos**, que fueron imputados con la cadena `"EMPTY"`.

### Columnas duplicadas

Se identificaron **210 filas duplicadas**, pero fueron consideradas relevantes y no se eliminaron.

### Distribución de valores

- La columna `method` contiene **418 solicitudes GET** y **104 solicitudes POST**.
- Se detectaron rutas con patrones sospechosos como `java.lang.Thread.sleep`, indicando posibles **ataques web**.

---

## 2. **Análisis de Correlación y Transformaciones**

### Matriz de correlación

- **Variables relevantes para la clase objetivo (`class`):** `single_q`, `double_q`, `braces`, `spaces` y `badwords_count`.
- Las columnas `percentages` y `special_chars` no mostraron correlación significativa y fueron eliminadas.

### Extracción de palabras clave

Se crearon nuevas columnas para contar palabras clave sospechosas en las columnas `path` y `body`, basándonos en una lista de términos relacionados con **ataques web**.

---

## 3. **Preparación del Dataset**

### Transformaciones aplicadas

- Se eliminaron columnas irrelevantes.
- Se crearon columnas numéricas basadas en patrones de palabras clave.
- Las columnas categóricas fueron convertidas en variables binarias.

---

## 4. **Balanceo de Clases**

Dado que el tráfico malicioso representa una **minoría**, se utilizó el parámetro `class_weight="balanced"` para asignar mayor peso a las clases minoritarias en los modelos de clasificación.

---

## 5. **Entrenamiento y Evaluación**

### Modelos evaluados

Se entrenaron y evaluaron los siguientes modelos:

- Logistic Regression
- KNN
- Naive Bayes
- Decision Tree
- Random Forest
- XGBoost
- AdaBoost
- Bagging
- Extra Trees

### Métricas de evaluación

| Modelo              | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 0.9333   | 0.9348    | 0.9149 | 0.9247   | 0.9873  |
| KNN                 | 0.9333   | 0.9348    | 0.9149 | 0.9247   | 0.9661  |
| Naive Bayes         | 0.9619   | 0.9388    | 0.9787 | 0.9583   | 0.9736  |
| Decision Tree       | 0.9619   | 0.9388    | 0.9787 | 0.9583   | 0.9635  |
| Random Forest       | 0.9714   | 0.9400    | 1.0000 | 0.9691   | 0.9791  |
| XGBoost             | 0.9524   | 0.9375    | 0.9574 | 0.9474   | 0.9751  |
| AdaBoost            | 0.9524   | 0.9375    | 0.9574 | 0.9474   | 0.9899  |
| Bagging             | 0.9619   | 0.9388    | 0.9787 | 0.9583   | 0.9734  |
| Extra Trees         | 0.9619   | 0.9388    | 0.9787 | 0.9583   | 0.9895  |

**Conclusión:** Random Forest se posicionó como el modelo más robusto, logrando un **recall perfecto (100%)**, lo que es crítico para minimizar falsos negativos.

---

## 6. **Validación Cruzada**

Los modelos fueron evaluados con validación cruzada (5 folds):

| Modelo              | Accuracy Promedio |
| ------------------- | ----------------- |
| Naive Bayes         | 0.9541            |
| XGBoost             | 0.9463            |
| KNN                 | 0.9407            |
| Logistic Regression | 0.9215            |
| Random Forest       | 0.9311            |
| Decision Tree       | 0.9101            |

**Conclusión:** Aunque Random Forest mostró un excelente rendimiento global, **Naive Bayes** lideró la validación cruzada con un **95.41% de accuracy promedio**.

---

## 7. **Comparación con LazyClassifier**

Se utilizó la librería `lazypredict` para evaluar modelos adicionales. El **Extra Tree Classifier** destacó con un **98% de accuracy**, aunque computacionalmente es menos eficiente que Random Forest.

---

## 8. **Conclusiones Generales**

1. **Random Forest** es la mejor opción por su rendimiento y recall perfecto.
2. **Naive Bayes** es una alternativa rápida y efectiva.
3. Este análisis puede ser extendido con más features relevantes o mediante técnicas avanzadas como hyperparameter tuning.

---

**Gracias por revisar este proyecto.**
