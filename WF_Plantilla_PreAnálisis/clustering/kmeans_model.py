"""
Implementación del modelo K-Means para clustering.

Este archivo contiene la configuración y funciones necesarias para entrenar
y aplicar el algoritmo K-Means, un método de agrupamiento no supervisado.
Incluye opciones para configurar parámetros como el número de clusters y
el método de inicialización de centroides.

Responsabilidades:
- Definir y entrenar el modelo K-Means.
- Realizar predicciones y asignar clusters a los datos.
- Extraer información relevante como centros de los clusters.

Funciones clave:
1. `train_kmeans`: Entrena el modelo con datos normalizados.
2. `predict_kmeans`: Predice el cluster para nuevos datos.
3. `get_cluster_centers`: Devuelve los centros de los clusters.
"""
from sklearn.cluster import KMeans
import pandas as pd

def train_kmeans(data, n_clusters=4, random_state=42):
    """
    Entrena un modelo K-Means con los datos proporcionados.
    
    Args:
        data (pd.DataFrame): Datos normalizados para clustering.
        n_clusters (int): Número de clústeres.
        random_state (int): Semilla para reproducibilidad.

    Returns:
        tuple: Modelo K-Means entrenado y etiquetas de clúster.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    return kmeans, labels

def get_cluster_centers(kmeans, feature_names):
    """
    Devuelve los centros de los clústeres como un DataFrame.

    Args:
        kmeans (KMeans): Modelo entrenado de K-Means.
        feature_names (list): Lista de nombres de las características.

    Returns:
        pd.DataFrame: Centros de los clústeres con nombres de columnas.
    """
    centers = kmeans.cluster_centers_
    return pd.DataFrame(centers, columns=feature_names)
