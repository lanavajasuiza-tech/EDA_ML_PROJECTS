# Descripción 

Esta es una plantilla básica para automatizar la visualización de datos de un dataset desconocido para tener un overview a primera vista para analizar la información contenida en el mismo.

# Cómo funciona

1. Mueve tu csv a la carpeta de dataSet 
1. En app.py cambia df = "nombre_de_dataset.csv": 

            # Ruta simplificada al dataset
            df_path = os.path.join(project_root, "dataSet")
            print(f"Ruta del dataset: {df_path}")
            df = "wholesale.csv"

3. Asegura que tienes las rutas agregadas a tu sys.path verficando la ruta en el archvio test-routes.py

            # Ruta del proyecto
            project_root = r"C:\ruta_de_tu_proyecto"

4. Corre el código desde app.py

## Listar Entornos
conda env list

## Crear / activar  entornos
conda create --name env_EDA python=3.9

## generar / actualizar requirements
pip freeze > requirements_ML_.txt