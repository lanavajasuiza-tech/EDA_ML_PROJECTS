"""
Funciones de preprocesamiento de datos.

Este archivo centraliza todas las funciones necesarias para preparar
los datos antes de aplicar los modelos de clustering. Incluye pasos
como normalización, manejo de valores nulos y eliminación de columnas
irrelevantes.

Responsabilidades:
- Limpiar y transformar datos.
- Normalizar características numéricas.
- Preparar datos para su entrada en los modelos.

Funciones clave:
1. `normalize_data`: Normaliza las columnas numéricas.
2. `handle_missing_values`: Rellena o elimina valores nulos.
3. `select_features`: Filtra las columnas relevantes para el análisis.
"""

#=================== CARGAR DATOS ===================#

import os
import pandas as pd




class DataLoader:
    def __init__(self, df_path, df, df_type='csv'):
        """
        Inicializa el DataLoader con la ruta y el nombre del archivo.

        Parámetros:
            df_path (str): Ruta del directorio que contiene el archivo.
            df (str): Nombre del archivo de datos.
            df_type (str): Tipo de archivo ('csv', 'excel', etc.). Por defecto es 'csv'.
        """
        self.df_path = df_path
        self.df = df
        self.df_type = df_type
        self.df_full_path = os.path.join(df_path, df)  # Ruta completa incluido el fichero



    def validate_file(self):
        """
        Verifica si el archivo especificado existe.

        Returns:
            bool: True si el archivo existe, False en caso contrario.
        """
        if not os.path.isfile(self.df_full_path):
            raise FileNotFoundError(f"El archivo '{self.df_full_path}' no existe.")
        return True
    

    def load_data(self):
        """
        Carga los datos desde el archivo especificado.

        Returns:
            pd.DataFrame: DataFrame con los datos cargados.
        """
        self.validate_file()

        # Carga el archivo según su tipo
        if self.df_type == 'csv':
            df = pd.read_csv(self.df_full_path)
        elif self.df_type == 'excel':
            df = pd.read_excel(self.df_full_path)
        else:
            raise ValueError(f"Tipo de archivo no soportado: {self.df_type}")
        
        print(f"Datos cargados exitosamente desde: {self.df_full_path}")
        return df

    def change_file(self, new_df, new_type=None):
        """
        Cambia el archivo de datos y actualiza la ruta completa.

        Parámetros:
            new_df (str): Nombre del nuevo archivo de datos.
            new_type (str, opcional): Nuevo tipo de archivo. Por defecto, mantiene el actual.
        """
        self.df = new_df
        self.df_type = new_type if new_type else self.df_type
        self.df_full_path = os.path.join(self.df_path, new_df)
        print(f"Archivo de datos actualizado a: {self.df_full_path}")


