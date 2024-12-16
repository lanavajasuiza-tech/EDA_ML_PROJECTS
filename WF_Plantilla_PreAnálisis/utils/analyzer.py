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


    def overview(self):
        """
        Muestra una vista general del DataFrame: .describe().T, tipos de datos,
        nombres de columnas, número de filas y columnas.
        """
        print("\n--- Estadísticas descriptivas (transpuesta) ---")
        print(self.df.describe().T)
        print("\n--- Tipos de datos ---")
        print(self.df.dtypes)
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
        Muestra un resumen de las columnas numéricas y categóricas y las almacena.
        """
        self.columns_num = self.df.select_dtypes(include=['number']).columns.tolist()
        self.columns_cat = self.df.select_dtypes(exclude=['number']).columns.tolist()
        print("\n--- Columnas categóricas ---")
        print(self.columns_cat)
        print("\n--- Columnas numéricas ---")
        print(self.columns_num)

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
        tabla_exploded = table_type.explode(['Columns', 'Dtype']).reset_index(drop=True)
        print("\n--- Column Type Summary ---")
        print(tabla_exploded)



    def combined_statistics(self, categorical=None, numerical=None):
        """
        Combina estadísticas descriptivas para columnas categóricas y numéricas.
        
        :param categorical: Lista de columnas categóricas (opcional).
        :param numerical: Lista de columnas numéricas (opcional).
        """
        # Usar las columnas detectadas automáticamente si no se proporcionan
        if categorical is None:
            categorical = self.columns_cat
        if numerical is None:
            numerical = self.columns_num

            if not categorical and not numerical:
                print("\n--- No se encontraron columnas categóricas o numéricas para analizar ---")
                return

            if categorical:
                categ_stats = self.df[categorical].describe().T
                print("\n--- Estadísticas para columnas categóricas ---")
                print(categ_stats)

            if numerical:
                num_stats = self.df[numerical].describe().T
                print("\n--- Estadísticas para columnas numéricas ---")
                print(num_stats)

            if categorical and numerical:
                combined_stats = pd.concat([categ_stats, num_stats], axis=0)
                print("\n--- Estadísticas combinadas ---")
                print(combined_stats)

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