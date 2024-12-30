import pandas as pd
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class DataCleaner:
    def __init__(self, dataframe):
        self.df = dataframe.copy()

    # 1. Limpieza de Datos
    def drop_missing(self, threshold=0.5):
        """Eliminar columnas con porcentaje alto de valores nulos."""
        missing_ratio = self.df.isnull().mean()
        to_drop = missing_ratio[missing_ratio > threshold].index
        self.df.drop(columns=to_drop, inplace=True)
        return self

    def fill_missing(self, strategy='mean', constant=None):
        """Rellenar valores nulos con media, mediana, moda o constante."""
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if strategy == 'mean':
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            elif strategy == 'median':
                self.df[col].fillna(self.df[col].median(), inplace=True)
            elif strategy == 'mode':
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            elif strategy == 'constant' and constant is not None:
                self.df[col].fillna(constant, inplace=True)
        return self

    def convert_column_types(self, column_type_map):
        """Convertir tipos de datos según un diccionario de mapeo."""
        for col, dtype in column_type_map.items():
            self.df[col] = self.df[col].astype(dtype)
        return self

    def drop_irrelevant_columns(self, columns_to_drop):
        """Eliminar columnas irrelevantes."""
        self.df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        return self


    def map_categorical(self, column, mapping_dict):
        """Mapear valores categóricos a numéricos."""
        self.df[column] = self.df[column].map(mapping_dict)
        return self

    def create_features_from_date(self, column, features=['year', 'month', 'day']):
        """Generar nuevas características a partir de una columna de fecha."""
        self.df[column] = pd.to_datetime(self.df[column])
        if 'year' in features:
            self.df[f"{column}_year"] = self.df[column].dt.year
        if 'month' in features:
            self.df[f"{column}_month"] = self.df[column].dt.month
        if 'day' in features:
            self.df[f"{column}_day"] = self.df[column].dt.day
        return self

    def text_to_features(self, column):
        """Convertir texto en características usando conteos de palabras."""
        self.df[f"{column}_word_count"] = self.df[column].apply(lambda x: len(str(x).split()))
        return self

    def calculate_correlations(self):
        """Calcular correlaciones entre variables numéricas."""
        return self.df.corr()

    # 4. Manejo de Datos Desbalanceados
    def check_class_balance(self, target):
        """Verificar el balance de clases."""
        return self.df[target].value_counts(normalize=True)

    def balance_classes(self, target):
        """Aplicar SMOTE para balancear clases."""
        smote = SMOTE()
        X = self.df.drop(columns=[target])
        y = self.df[target]
        X_resampled, y_resampled = smote.fit_resample(X, y)
        self.df = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=[target])], axis=1)
        return self

    def drop_low_variance(self, threshold=0.01):
        """Eliminar columnas con varianza baja."""
        low_variance_cols = self.df.var()[self.df.var() < threshold].index
        self.df.drop(columns=low_variance_cols, inplace=True)
        return self

    # 6. Validación y Reporte
    def quality_report(self):
        """Generar un informe de calidad de datos."""
        report = {
            'missing_values': self.df.isnull().sum(),
            'duplicate_rows': self.df.duplicated().sum(),
            'data_types': self.df.dtypes,
            'outliers': self.detect_outliers()
        }
        return report

    def get_cleaned_data(self):
        """Devolver el dataframe limpio."""
        return self.df

    def handle_missing_values_interactively(self):
            """
            Maneja valores faltantes en el DataFrame de forma interactiva.
            Permite elegir entre eliminar, rellenar o dejar sin cambios.
            """
            print("Resumen inicial de valores faltantes:")
            nan_summary = self.df.isnull().mean().sort_values(ascending=False) * 100
            print(nan_summary)

            for column, nan_percentage in nan_summary.items():
                if nan_percentage > 0:  # Solo procesa columnas con valores faltantes
                    print(f"\nLa columna '{column}' tiene {nan_percentage:.2f}% de valores faltantes.")
                    print("Opciones:")
                    print("1. Eliminar la columna")
                    print("2. Rellenar con 0")
                    print("3. Rellenar con la media (solo columnas numéricas)")
                    print("4. Rellenar con la mediana (solo columnas numéricas)")
                    print("5. Rellenar con la moda (solo columnas categóricas)")
                    print("6. Dejar la columna como está")

                    # Validar la entrada del usuario
                    choice = input(f"¿Qué deseas hacer con '{column}'? (1-6): ")
                    while choice not in {"1", "2", "3", "4", "5", "6"}:
                        print("Opción no válida. Por favor selecciona una opción entre 1 y 6.")
                        choice = input(f"¿Qué deseas hacer con '{column}'? (1-6): ")

                    # Ejecutar la opción seleccionada
                    if choice == "1":  # Eliminar columna
                        print(f"❌ Eliminando la columna '{column}'.")
                        self.df.drop(columns=[column], inplace=True)
                    elif choice == "2":  # Rellenar con 0
                        print(f"✔️ Rellenando valores faltantes de '{column}' con 0.")
                        self.df[column].fillna(0, inplace=True)
                    elif choice == "3" and self.df[column].dtype in ["float64", "int64"]:  # Rellenar con media
                        mean_value = self.df[column].mean()
                        print(f"✔️ Rellenando valores faltantes de '{column}' con la media: {mean_value:.2f}.")
                        self.df[column].fillna(mean_value, inplace=True)
                    elif choice == "4" and self.df[column].dtype in ["float64", "int64"]:  # Rellenar con mediana
                        median_value = self.df[column].median()
                        print(f"✔️ Rellenando valores faltantes de '{column}' con la mediana: {median_value:.2f}.")
                        self.df[column].fillna(median_value, inplace=True)
                    elif choice == "5" and self.df[column].dtype == "object":  # Rellenar con moda
                        mode_value = self.df[column].mode()[0]
                        print(f"✔️ Rellenando valores faltantes de '{column}' con la moda: {mode_value}.")
                        self.df[column].fillna(mode_value, inplace=True)
                    elif choice == "6":  # Dejar la columna como está
                        print(f"✔️ Dejando la columna '{column}' sin cambios.")
                    else:
                        print(f"⚠️ La opción seleccionada no es válida para el tipo de columna '{column}'. No se realizaron cambios.")
                else:
                    print(f"✔️ La columna '{column}' no tiene valores faltantes.")

            print("\nResumen final de valores faltantes:")
            print(self.df.isnull().mean() * 100)
            return self.df
    
    def drop_duplicates_rows_and_columns(self):
        """
        Detecta y elimina filas y columnas duplicadas exactas.
        """
        # --- Detectar y eliminar filas duplicadas ---
        initial_row_count = self.df.shape[0]
        self.df.drop_duplicates(inplace=True)
        final_row_count = self.df.shape[0]
        removed_rows = initial_row_count - final_row_count

        if removed_rows > 0:
            print(f"✔️ Eliminadas {removed_rows} filas duplicadas.")
        else:
            print("✔️ No se encontraron filas duplicadas.")

        # --- Detectar y eliminar columnas duplicadas ---
        duplicated_columns = []
        for i in range(len(self.df.columns)):
            for j in range(i + 1, len(self.df.columns)):
                if self.df.iloc[:, i].equals(self.df.iloc[:, j]):
                    duplicated_columns.append(self.df.columns[j])

        if duplicated_columns:
            print(f"✔️ Columnas duplicadas encontradas y eliminadas: {duplicated_columns}")
            self.df.drop(columns=duplicated_columns, inplace=True)
        else:
            print("✔️ No se encontraron columnas duplicadas.")

        return self

    def get_cleaned_data(self):
        """
        Devuelve el DataFrame limpio.
        """
        return self.df
    
    def preprocess_for_binarization(self):
        """
        Prepara y verifica las columnas categóricas para binarización.
        Realiza comprobaciones previas y permite decidir acciones para cada columna.
        """
        # Identificar columnas categóricas
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns

        if not categorical_columns.any():
            print("✔️ No se encontraron columnas categóricas para binarizar.")
            return self

        print("\n--- Verificación previa a la binarización ---")
        candidates = []
        for column in categorical_columns:
            unique_values = self.df[column].nunique()
            print(f"Columna: '{column}' - Categorías únicas: {unique_values} - Ejemplo: {self.df[column].unique()[:3]}")

            # Excluir columnas binarias ya válidas
            if unique_values == 2:
                print(f"✔️ La columna '{column}' ya es binaria. No se requiere binarización.")
            else:
                candidates.append(column)

        if not candidates:
            print("✔️ No hay columnas categóricas relevantes para binarizar.")
            return self

        # Menú interactivo para cada columna candidata
        for column in candidates:
            print(f"\n--- Opciones para la columna '{column}' ---")
            print("1. Binarizar con todas las categorías")
            print("2. Binarizar eliminando una categoría (drop_first=True)")
            print("3. Agrupar categorías raras y luego binarizar")
            print("4. No realizar ninguna acción")

            choice = input(f"¿Qué deseas hacer con '{column}'? (1-4): ")
            while choice not in {"1", "2", "3", "4"}:
                print("Opción no válida. Por favor selecciona una opción entre 1 y 4.")
                choice = input(f"¿Qué deseas hacer con '{column}'? (1-4): ")

            # Manejo de la opción seleccionada
            if choice == "1":  # Binarizar con todas las categorías
                print(f"✔️ Binarizando '{column}' con todas las categorías.")
                dummies = pd.get_dummies(self.df[column], prefix=column, drop_first=False)
                self.df.drop(columns=[column], inplace=True)
                self.df = pd.concat([self.df, dummies], axis=1)
            elif choice == "2":  # Binarizar eliminando una categoría
                print(f"✔️ Binarizando '{column}' eliminando una categoría (drop_first=True).")
                dummies = pd.get_dummies(self.df[column], prefix=column, drop_first=True)
                self.df.drop(columns=[column], inplace=True)
                self.df = pd.concat([self.df, dummies], axis=1)
            elif choice == "3":  # Agrupar categorías raras antes de binarizar
                threshold_proportion = 0.05
                freq = self.df[column].value_counts(normalize=True)
                rare_categories = freq[freq < threshold_proportion].index
                self.df[column] = self.df[column].apply(lambda x: 'Raro' if x in rare_categories else x)
                print(f"✔️ Categorías raras agrupadas en '{column}': {list(rare_categories)}")
                dummies = pd.get_dummies(self.df[column], prefix=column, drop_first=True)
                self.df.drop(columns=[column], inplace=True)
                self.df = pd.concat([self.df, dummies], axis=1)
            elif choice == "4":  # No realizar ninguna acción
                print(f"✔️ No se realizará ninguna acción en '{column}'.")

        return self

    def get_cleaned_data(self):
        """
        Devuelve el DataFrame limpio.
        """
        return self.df

    def interactive_normalization(self):
        """
        Menú interactivo para decidir qué columnas normalizar y qué estrategia usar.
        """
        # Identificar columnas numéricas
        numerical_columns = self.df.select_dtypes(include=['int', 'float']).columns

        if not numerical_columns.any():
            print("✔️ No se encontraron columnas numéricas para normalizar.")
            return self

        print("\n--- Verificación previa a la normalización ---")
        candidates = []
        for column in numerical_columns:
            print(f"Columna: '{column}' - Valores únicos: {self.df[column].nunique()} - Rango: ({self.df[column].min()}, {self.df[column].max()})")
            candidates.append(column)

        if not candidates:
            print("✔️ No hay columnas relevantes para normalizar.")
            return self

        # Menú interactivo para cada columna candidata
        for column in candidates:
            print(f"\n--- Opciones para la columna '{column}' ---")
            print("1. Normalizar con Min-Max Scaling (0-1)")
            print("2. Estandarizar con Z-Score (media=0, desviación estándar=1)")
            print("3. Escalado robusto (tolerante a valores atípicos)")
            print("4. Aplicar transformación logarítmica")
            print("5. No realizar ninguna acción")

            choice = input(f"¿Qué deseas hacer con '{column}'? (1-5): ")
            while choice not in {"1", "2", "3", "4", "5"}:
                print("Opción no válida. Por favor selecciona una opción entre 1 y 5.")
                choice = input(f"¿Qué deseas hacer con '{column}'? (1-5): ")

            # Manejo de la opción seleccionada
            if choice == "1":  # Min-Max Scaling
                print(f"✔️ Normalizando '{column}' con Min-Max Scaling.")
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                self.df[column] = scaler.fit_transform(self.df[[column]])
            elif choice == "2":  # Z-Score
                print(f"✔️ Estandarizando '{column}' con Z-Score.")
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                self.df[column] = scaler.fit_transform(self.df[[column]])
            elif choice == "3":  # Escalado robusto
                print(f"✔️ Aplicando escalado robusto a '{column}'.")
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                self.df[column] = scaler.fit_transform(self.df[[column]])
            elif choice == "4":  # Transformación logarítmica
                print(f"✔️ Aplicando transformación logarítmica a '{column}'.")
                self.df[column] = self.df[column].apply(lambda x: np.log(x + 1) if x >= 0 else x)  # Manejo de valores negativos
            elif choice == "5":  # No realizar ninguna acción
                print(f"✔️ No se realizará ninguna acción en '{column}'.")

        return self

    def get_cleaned_data(self):
        """
        Devuelve el DataFrame limpio.
        """
        return self.df

    def detect_and_handle_outliers(self):
        """
        Detección y manejo interactivo de outliers en columnas numéricas.
        """
        # Identificar columnas numéricas
        numerical_columns = self.df.select_dtypes(include=['int', 'float']).columns

        if not numerical_columns.any():
            print("✔️ No se encontraron columnas numéricas para detectar outliers.")
            return self

        print("\n--- Detectando outliers ---")
        outliers_summary = {}

        # Detección de outliers por columna
        for column in numerical_columns:
            # Usar IQR por defecto
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
            outlier_count = len(outliers)

            if outlier_count > 0:
                print(f"Columna: '{column}' - Outliers detectados: {outlier_count}")
                outliers_summary[column] = {
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "outlier_count": outlier_count
                }
            else:
                print(f"✔️ No se detectaron outliers en la columna '{column}'.")

        # Menú interactivo para manejar los outliers detectados
        for column, details in outliers_summary.items():
            print(f"\n--- Opciones para manejar outliers en la columna '{column}' ---")
            print(f"Rango permitido: ({details['lower_bound']:.2f}, {details['upper_bound']:.2f})")
            print("1. Eliminar filas con outliers")
            print("2. Reemplazar outliers con la mediana")
            print("3. Reemplazar outliers con el límite más cercano")
            print("4. No realizar ninguna acción")

            choice = input(f"¿Qué deseas hacer con los outliers de '{column}'? (1-4): ")
            while choice not in {"1", "2", "3", "4"}:
                print("Opción no válida. Por favor selecciona una opción entre 1 y 4.")
                choice = input(f"¿Qué deseas hacer con los outliers de '{column}'? (1-4): ")

            # Manejo de la opción seleccionada
            if choice == "1":  # Eliminar filas con outliers
                print(f"✔️ Eliminando filas con outliers en '{column}'.")
                self.df = self.df[(self.df[column] >= details["lower_bound"]) & (self.df[column] <= details["upper_bound"])]
            elif choice == "2":  # Reemplazar con la mediana
                median_value = self.df[column].median()
                print(f"✔️ Reemplazando outliers de '{column}' con la mediana: {median_value:.2f}.")
                self.df[column] = self.df[column].apply(
                    lambda x: median_value if x < details["lower_bound"] or x > details["upper_bound"] else x
                )
            elif choice == "3":  # Reemplazar con el límite más cercano
                print(f"✔️ Reemplazando outliers de '{column}' con los límites más cercanos.")
                self.df[column] = self.df[column].apply(
                    lambda x: details["lower_bound"] if x < details["lower_bound"] else
                    (details["upper_bound"] if x > details["upper_bound"] else x)
                )
            elif choice == "4":  # No realizar ninguna acción
                print(f"✔️ No se realizará ninguna acción en los outliers de '{column}'.")

        return self

    def get_cleaned_data(self):
        """
        Devuelve el DataFrame limpio.
        """
        return self.df

    def reduce_dimensions_interactively(self):
        """
        Reducción de dimensiones con PCA, con opciones interactivas para seleccionar
        el método de escalado, número de componentes y las columnas a incluir.
        """
        # Identificar columnas numéricas
        numerical_columns = self.df.select_dtypes(include=['int', 'float']).columns

        if len(numerical_columns) <= 2:
            print("✔️ No se realizó reducción de dimensiones (menos de 2 columnas numéricas).")
            return self

        print("\n--- Propuesta para Reducción de Dimensiones (PCA) ---")
        print(f"Columnas numéricas detectadas: {list(numerical_columns)}")
        include_columns = input("¿Quieres incluir todas las columnas numéricas? (s/n): ").lower()

        if include_columns == "n":
            # Seleccionar columnas específicas
            selected_columns = input("Introduce las columnas que deseas incluir, separadas por comas: ").split(",")
            selected_columns = [col.strip() for col in selected_columns if col.strip() in numerical_columns]

            if not selected_columns:
                print("⚠️ No se seleccionaron columnas válidas. No se aplicará PCA.")
                return self
        else:
            selected_columns = list(numerical_columns)

        # --- Subopción: Selección de método de escalado ---
        print("\n--- Opciones para Escalar las Columnas ---")
        print("1. Escalar con Z-Score (media=0, desviación estándar=1)")
        print("2. Escalar con Min-Max Scaling (rango 0-1)")
        print("3. Escalado robusto (tolerante a valores atípicos)")
        print("4. No escalar (usar los datos originales)")

        while True:
            scale_choice = input("Selecciona el método de escalado (1-4): ")
            if scale_choice in {"1", "2", "3", "4"}:
                break
            else:
                print("⚠️ Opción no válida. Por favor selecciona una opción entre 1 y 4.")

        # Aplicar el método de escalado seleccionado
        if scale_choice == "1":
            print("✔️ Escalando las columnas seleccionadas con Z-Score.")
            scaler = StandardScaler()
        elif scale_choice == "2":
            print("✔️ Escalando las columnas seleccionadas con Min-Max Scaling.")
            scaler = MinMaxScaler()
        elif scale_choice == "3":
            print("✔️ Aplicando Escalado Robusto.")
            scaler = RobustScaler()
        elif scale_choice == "4":
            print("⚠️ No se aplicará escalado. Los datos originales serán usados para PCA.")
            scaled_data = self.df[selected_columns].values
            scaler = None

        if scaler:
            scaled_data = scaler.fit_transform(self.df[selected_columns])

        # --- Aplicar PCA y mostrar proporción de varianza explicada ---
        print("\n--- Aplicando PCA ---")
        pca = PCA()
        pca.fit(scaled_data)
        explained_variance = pca.explained_variance_ratio_

        print("Proporción de varianza explicada por cada componente:")
        for i, var in enumerate(explained_variance):
            print(f"Componente {i + 1}: {var:.4f}")

        # Elegir el número de componentes
        while True:
            try:
                n_components = int(input("Introduce el número de componentes principales que deseas mantener: "))
                if 1 <= n_components <= len(selected_columns):
                    break
                else:
                    print(f"⚠️ Introduce un número entre 1 y {len(selected_columns)}.")
            except ValueError:
                print("⚠️ Por favor, introduce un número válido.")

        # Reducir dimensiones
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(scaled_data)

        # Crear un DataFrame con los datos reducidos
        pca_columns = [f"PCA_{i + 1}" for i in range(n_components)]
        pca_df = pd.DataFrame(reduced_data, columns=pca_columns, index=self.df.index)

        # Añadir las nuevas columnas al DataFrame original y eliminar las originales si se desea
        self.df = pd.concat([self.df.drop(columns=selected_columns), pca_df], axis=1)

        print(f"✔️ Reducción de dimensiones completada: {n_components} componentes principales añadidos.")
        return self

    def get_cleaned_data(self):
        """
        Devuelve el DataFrame limpio.
        """
        return self.df
