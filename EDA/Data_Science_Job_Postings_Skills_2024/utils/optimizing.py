import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

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

    def drop_duplicates(self, subset=None):
        """Eliminar duplicados basados en todas o algunas columnas."""
        self.df.drop_duplicates(subset=subset, inplace=True)
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

    # 2. Transformación de Datos
    def binarize(self, columns):
        """Convertir variables categóricas en variables dummy."""
        self.df = pd.get_dummies(self.df, columns=columns, drop_first=True)
        return self

    def normalize(self, columns, method='minmax'):
        """Normalizar columnas con Min-Max Scaling o Z-score."""
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'zscore':
            scaler = StandardScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
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

    # 3. Análisis Exploratorio Automático
    def detect_outliers(self, method='iqr', threshold=1.5):
        """Detectar y opcionalmente eliminar outliers."""
        outliers = {}
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers[col] = self.df[(self.df[col] < (Q1 - threshold * IQR)) | (self.df[col] > (Q3 + threshold * IQR))]
            elif method == 'zscore':
                z_scores = (self.df[col] - self.df[col].mean()) / self.df[col].std()
                outliers[col] = self.df[np.abs(z_scores) > threshold]
        return outliers

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

    # 5. Optimización del Dataset
    def reduce_dimensions(self, n_components):
        """Reducir dimensiones usando PCA."""
        pca = PCA(n_components=n_components)
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        pca_data = pca.fit_transform(self.df[numerical_cols])
        self.df = pd.concat([self.df.drop(columns=numerical_cols), pd.DataFrame(pca_data)], axis=1)
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
