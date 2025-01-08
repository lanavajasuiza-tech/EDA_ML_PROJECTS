import os
print("Archivo 'classified_skills.csv' existe:", os.path.exists("output/classified_skills.csv"))

import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import hdbscan



# Cargar el archivo CSV
skills_df = pd.read_csv("output/classified_skills.csv")

# Instanciar el modelo preentrenado
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convertir habilidades a embeddings
embeddings = model.encode(skills_df["Habilidades no clasificadas"].tolist(), show_progress_bar=True)

# Reducir dimensionalidad con UMAP
reducer = umap.UMAP(n_neighbors=15, n_components=5, metric='cosine')
umap_embeddings = reducer.fit_transform(embeddings)

# Aplicar clustering con HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom')
cluster_labels = clusterer.fit_predict(umap_embeddings)

# Añadir los clusters al DataFrame
skills_df["Cluster"] = cluster_labels

# Guardar el resultado en un nuevo CSV
skills_df.to_csv("output/clustered_skills.csv", index=False)

print("Archivo 'clustered_skills.csv' generado correctamente.")


#-----> Convertimos en diccionario
import pandas as pd

# Cargar el archivo de clusters
skills_df = pd.read_csv("output/clustered_skills.csv")

# Crear un diccionario de categorías
categories = {}

# Recorrer los clusters y asignar habilidades a categorías
for cluster in skills_df["Cluster"].unique():
    if cluster != -1:  # Ignorar ruido (cluster -1 en HDBSCAN)
        category_name = f"Cluster {cluster}"
        categories[category_name] = skills_df[skills_df["Cluster"] == cluster]["Habilidad no clasificada"].tolist()

# Mostrar el diccionario generado
import pprint
pprint.pprint(categories)
