import pandas as pd


class DataAnalyzer:
    def __init__(self, df):
        """
        Inicializa la clase con un DataFrame.
        
        :param df: DataFrame a analizar
        """
        self.df = df
        self.columns_num = []
        self.columns_cat = []
        self.columns_num = df.select_dtypes(include=['number']).columns.tolist() # Outliners def
        self.columns_cat = df.select_dtypes(include=['object']).columns.tolist() # categorical summary


    def overview(self):
        """
        Muestra una vista general del DataFrame: .describe().T, tipos de datos,
        nombres de columnas, número de filas y columnas.
        """
        print("\n--- Estadísticas descriptivas (transpuesta) ---")
        print(self.df.describe().T)
        print("\n--- Títulos de las columnas ---")
        print(self.df.columns.tolist())
        print("\n--- Número de columnas ---")
        print(len(self.df.columns))
        print("\n--- Número de filas ---")
        print(len(self.df))

    def duplicates_analysis(self):
        """
        Analiza duplicados en filas y columnas del DataFrame.
        """
        # Filas duplicadas
        duplicados_filas = self.df[self.df.index.duplicated(keep=False) & self.df.duplicated(keep=False)]
        print(f"\n--- Número de filas duplicadas ---: {duplicados_filas.shape[0]}")
        if not duplicados_filas.empty:
            print("Filas duplicadas exactas:")
            print(duplicados_filas)
        else:
            print("No se encontraron filas duplicadas exactas.")

        # Columnas duplicadas
        temp_df = self.df.copy()
        temp_df.loc["Columna_Titulo"] = temp_df.columns
        duplicados_columnas = temp_df.T[temp_df.T.duplicated(keep=False)].T
        print(f"\n--- Número de columnas duplicadas ---: {duplicados_columnas.shape[1]}")
        if not duplicados_columnas.empty:
            print("Columnas duplicadas exactas:")
            print(duplicados_columnas.columns.tolist())
        else:
            print("No se encontraron columnas duplicadas exactas.")

    def data_types_analysis(self):
        """
        Muestra un resumen actualizado de las columnas numéricas y categóricas del DataFrame.
        """
        # Recalcular las columnas categóricas y numéricas
        self.columns_num = self.df.select_dtypes(include=['number']).columns.tolist()
        self.columns_cat = self.df.select_dtypes(exclude=['number']).columns.tolist()

        # Validación de columnas presentes
        print("\n--- Validación de columnas actuales en el DataFrame ---")
        print(f"Columnas actuales: {self.df.columns.tolist()}")

        # Mostrar las columnas categóricas y numéricas actualizadas
        print("\n--- Columnas categóricas ---")
        print(self.columns_cat)
        print("\n--- Columnas numéricas ---")
        print(self.columns_num)

        # Crear un resumen de tipos de columnas
        table_type = pd.DataFrame({
            "Column Type": ["Numerical", "Categorical"],
            "Cuantity": [len(self.columns_num), len(self.columns_cat)],
            "Column": [self.columns_num, self.columns_cat],
            "Type": [
                [self.df[col].dtype for col in self.columns_num],
                [self.df[col].dtype for col in self.columns_cat]
            ]
        })
        tabla_exploded = table_type.explode(['Column', 'Type']).reset_index(drop=True)
        print("\n--- Resumen de tipos de columnas ---")
        print(tabla_exploded)

        
    def update_data(self, new_df):
        """
        Actualiza el DataFrame interno de la clase.
        
        :param new_df: Nuevo DataFrame a asignar
        """
        self.df = new_df
        # Actualizar las columnas categóricas y numéricas
        self.columns_num = new_df.select_dtypes(include=['number']).columns.tolist()
        self.columns_cat = new_df.select_dtypes(exclude=['number']).columns.tolist()
        print("DataFrame actualizado exitosamente. Nuevas columnas:")
        print(self.df.columns.tolist())



    def missing_values_analysis(self):
        """
        Analiza los valores perdidos (NaN) en el DataFrame.
        """
        total_nan = self.df.isnull().sum().sum()
        print(f"\n--- Total de valores NaN ---: {total_nan}")
        if total_nan > 0:
            print("Existen valores NaN en la tabla. Es necesario manejarlos.")
        else:
            print("La tabla no contiene valores NaN.")

        # Crear tabla con NaN por columna
        table_type = pd.DataFrame({
            "Column Type": ["Numerical", "Categorical"],  # Cambiado de "Numéricas" y "Categóricas"
            "Count": [len(self.columns_num), len(self.columns_cat)],  # Cambiado de "Cantidad"
            "Columns": [self.columns_num, self.columns_cat],  # Cambiado de "Columnas"
            "Dtype": [
                [self.df[col].dtype for col in self.columns_num],
                [self.df[col].dtype for col in self.columns_cat]
            ]
        })


   # Nuevo método para calcular el porcentaje de valores NaN
    def nan_summary(self):
        """
        Calcula y muestra el porcentaje de valores NaN por columna.
        """
        total_rows = len(self.df)
        nan_values = self.df.isnull().sum()
        nan_percentage = (nan_values / total_rows) * 100

        # Crear un DataFrame resumen
        nan_report = pd.DataFrame({
            'Total NaN': nan_values,
            'Porcentaje NaN': nan_percentage
        }).sort_values(by='Porcentaje NaN', ascending=False)

        # Mostrar el reporte si hay valores NaN
        print("\n--- Resumen de valores NaN por columna ---")
        if nan_report['Total NaN'].sum() > 0:
            print(nan_report)
        else:
            print("No se encontraron valores NaN en el dataset.")
        return nan_report
    
