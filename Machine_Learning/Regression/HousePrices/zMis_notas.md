### Primeras notas

**Columnas**: 81
**Filas**: 1469
**Tipos**:  
- **Categóricas**: 43  
- **Numéricas**: 38  

**Tabla de columnas:**

| I   | Tipo de columna | Cantidad | Columnas       | Tipo    |
|-----|-----------------|----------|----------------|---------|
| 0   | Numéricas       | 38       | Id             | int64   |
| 1   | Numéricas       | 38       | MSSubClass     | int64   |
| 2   | Numéricas       | 38       | LotFrontage    | float64 |
| 3   | Numéricas       | 38       | LotArea        | int64   |
| 4   | Numéricas       | 38       | OverallQual    | int64   |
| ... | ...             | ...      | ...            | ...     |
| 76  | Categóricas     | 43       | PoolQC         | object  |
| 77  | Categóricas     | 43       | Fence          | object  |
| 78  | Categóricas     | 43       | MiscFeature    | object  |
| 79  | Categóricas     | 43       | SaleType       | object  |
| 80  | Categóricas     | 43       | SaleCondition  | object  |

**[81 rows x 4 columns]**

### **Volumetrías NaN por valor**

| Columna         | Total NaN | Porcentaje NaN | Notas                                                                      |
|------------------|-----------|----------------|----------------------------------------------------------------------------|
| PoolQC          | 1453      | 99.52%         | NaN propuesta de cambio por `'NONE'` (Indica ausencia de piscina).                |
| MiscFeature     | 1406      | 96.30%         | NaN propuesta de cambio por `'NONE'` (Detalles adicionales como cobertizos).      |
| Alley           | 1369      | 93.77%         | NaN propuesta de cambio por `'NONE'` (Acceso por callejón).                       |
| Fence           | 1179      | 80.75%         | NaN propuesta de cambio por `'NONE'` (Indica ausencia de cercas).                 |
| MasVnrType      | 872       | 59.73%         | NaN propuesta de cambio por la moda (Categoría más frecuente de revestimiento).   |
| FireplaceQu     | 690       | 47.26%         | NaN propuesta de cambio por la moda (Categoría más frecuente de calidad).         |
| LotFrontage     | 259       | 17.74%         | NaN propuesta de cambio por la mediana (Metros lineales conectados).              |
| GarageType      | 81        | 5.55%          | NaN propuesta de cambio por `'NONE'` (Indica ausencia de garaje).                 |
| GarageYrBlt     | 81        | 5.55%          | NaN propuesta de cambio por `0` (Año no aplicable por ausencia de garaje).        |
| GarageFinish    | 81        | 5.55%          | NaN propuesta de cambio por `'NONE'` (Indica ausencia de acabado interior).       |
| GarageQual      | 81        | 5.55%          | NaN propuesta de cambio por `'NONE'` (Indica ausencia de calidad de garaje).      |
| GarageCond      | 81        | 5.55%          | NaN propuesta de cambio por `'NONE'` (Indica ausencia de garaje).                 |
| BsmtFinType2    | 38        | 2.60%          | NaN propuesta de cambio por `'NONE'` (Sin tipo de acabado en el sótano).          |
| BsmtExposure    | 38        | 2.60%          | NaN propuesta de cambio por `'NONE'` (Sin exposición en el sótano).               |
| BsmtFinType1    | 37        | 2.53%          | NaN propuesta de cambio por `'NONE'` (Sin tipo de acabado en el sótano).          |
| BsmtCond        | 37        | 2.53%          | NaN propuesta de cambio por `'NONE'` (Sin condición general del sótano).          |
| BsmtQual        | 37        | 2.53%          | NaN propuesta de cambio por `'NONE'` (Sin calidad general del sótano).            |
| MasVnrArea      | 8         | 0.55%          | NaN propuesta de cambio por `0` (Sin área de revestimiento).                      |
| Electrical      | 1         | 0.07%          | NaN propuesta de cambio por la moda (Categoría más frecuente de sistema eléctrico).|


**[Nota: Decide qué columnas eliminar del dataset.]**

---

### **Columnas sobrantes según el análisis:**
Basándome en las volumetrías NaN, el mapa de calor y el gráfico de dispersión, considero que sobran/None las siguientes columnas:

| **Columna**          | **Acción**         | **Método**   | **Notas**                                                                                     |
|-----------------------|--------------------|--------------|-----------------------------------------------------------------------------------------------|
| Id                   | Eliminada          | -            | No aporta información relevante al análisis.                                                 |
| PoolQC               | Rellenada          | `'NONE'`     | Indica si tiene piscina o no. `'NONE'` para propiedades sin piscina.                         |
| MiscFeature          | Rellenada          | `'NONE'`     | Detalles adicionales como cobertizos, elevadores o pistas de tenis.                          |
| Alley                | Rellenada          | `'NONE'`     | Indica si hay acceso por callejón (pavimentado o no). `'NONE'` para propiedades sin acceso.  |
| Fence                | Rellenada          | `'NONE'`     | Indica si hay cercas en la propiedad. `'NONE'` para propiedades sin cerca.                   |
| GarageType           | Rellenada          | `'NONE'`     | Tipo de garaje. `'NONE'` para propiedades sin garaje.                                        |
| GarageFinish         | Rellenada          | `'NONE'`     | Acabado interior del garaje. `'NONE'` para propiedades sin garaje.                          |
| GarageQual           | Rellenada          | `'NONE'`     | Calidad del garaje. `'NONE'` para propiedades sin garaje.                                    |
| GarageCond           | Rellenada          | `'NONE'`     | Condición del garaje. `'NONE'` para propiedades sin garaje.                                  |
| BsmtExposure         | Rellenada          | `'NONE'`     | Exposición del sótano. `'NONE'` para propiedades sin sótano.                                 |
| BsmtFinType1         | Rellenada          | `'NONE'`     | Tipo de acabado del sótano (primera categoría). `'NONE'` para propiedades sin sótano.        |
| BsmtFinType2         | Rellenada          | `'NONE'`     | Tipo de acabado del sótano (segunda categoría). `'NONE'` para propiedades sin sótano.        |
| BsmtQual             | Rellenada          | `'NONE'`     | Calidad general del sótano. `'NONE'` para propiedades sin sótano.                            |
| BsmtCond             | Rellenada          | `'NONE'`     | Condición general del sótano. `'NONE'` para propiedades sin sótano.                          |
| MasVnrType           | Rellenada          | Moda         | Material utilizado para el revestimiento exterior. Se usó la moda por ser categórica.        |
| FireplaceQu          | Rellenada          | Moda         | Calificación de la calidad de la chimenea. Se usó la moda por ser categórica.                |
| Electrical           | Rellenada          | Moda         | Sistema eléctrico de la propiedad. Se usó la moda por ser categórica y tener pocos NaN.      |
| LotFrontage          | Rellenada          | Mediana      | Metros lineales de la calle conectados a la propiedad. Mediana para evitar sesgos por outliers.|
| GarageYrBlt          | Rellenada          | `0`          | Año de construcción del garaje. `0` para propiedades sin garaje.                             |
| MasVnrArea           | Rellenada          | `0`          | Área de revestimiento de mampostería. `0` para propiedades sin revestimiento.                |

# Modelo regresión lineal


# OLS (Ordinary Least Squares) 


# Modelo árbol de decisión


# Modelo RANDOM Forest


# Modelo XGBoost


# Proceso LazyPredict para comparar con otros modelos

### **Resultados de LazyRegressor**
(*Nota: Los resultados de esta tabla están ordenado a través de ChatGPT, Pandas saca el resultado por columnas separadas)
### **Resultados de LazyRegressor**

| Modelo                          | Adjusted R-Squared | R-Squared | RMSE       | Tiempo (s) |
|---------------------------------|--------------------|-----------|------------|------------|
| XGBRegressor                    | 0.88              | 0.91      | 26115.63   | 0.20       |
| PoissonRegressor                | 0.86              | 0.90      | 28018.49   | 0.08       |
| HistGradientBoostingRegressor   | 0.85              | 0.89      | 28745.94   | 1.86       |
| RandomForestRegressor           | 0.85              | 0.89      | 28807.76   | 2.64       |
| GradientBoostingRegressor       | 0.85              | 0.89      | 28830.34   | 0.89       |
| LassoLars                       | 0.85              | 0.89      | 29214.98   | 0.08       |
| LassoLarsIC                     | 0.85              | 0.89      | 29217.84   | 0.10       |
| LGBMRegressor                   | 0.84              | 0.89      | 29452.30   | 0.18       |
| HuberRegressor                  | 0.84              | 0.88      | 29956.08   | 0.27       |
| ExtraTreesRegressor             | 0.83              | 0.88      | 30557.91   | 2.43       |
| BaggingRegressor                | 0.83              | 0.88      | 30650.33   | 0.34       |
| RANSACRegressor                 | 0.82              | 0.87      | 31371.57   | 0.68       |
| KernelRidge                     | 0.82              | 0.87      | 31813.30   | 0.08       |
| Ridge                           | 0.82              | 0.87      | 31819.99   | 0.04       |
| Lasso                           | 0.81              | 0.86      | 32341.42   | 0.28       |
| RidgeCV                         | 0.81              | 0.86      | 32687.20   | 0.07       |
| BayesianRidge                   | 0.81              | 0.86      | 32726.61   | 0.10       |
| GammaRegressor                  | 0.81              | 0.86      | 32728.17   | 0.07       |
| LassoLarsCV                     | 0.81              | 0.86      | 32915.26   | 0.21       |
| LassoCV                         | 0.81              | 0.86      | 32915.83   | 0.56       |
| PassiveAggressiveRegressor      | 0.80              | 0.85      | 33624.64   | 0.10       |
| OrthogonalMatchingPursuit       | 0.79              | 0.84      | 34544.76   | 0.04       |
| OrthogonalMatchingPursuitCV     | 0.79              | 0.84      | 34657.80   | 0.05       |
| LarsCV                          | 0.78              | 0.84      | 35438.81   | 0.29       |
| AdaBoostRegressor               | 0.77              | 0.83      | 36115.19   | 0.50       |
| ElasticNet                      | 0.76              | 0.83      | 36374.70   | 0.04       |
| SGDRegressor                    | 0.74              | 0.81      | 37843.34   | 0.09       |
| TweedieRegressor                | 0.73              | 0.80      | 39029.06   | 0.08       |
| ExtraTreeRegressor              | 0.72              | 0.80      | 39259.37   | 0.06       |
| KNeighborsRegressor             | 0.69              | 0.78      | 41500.05   | 0.06       |
| DecisionTreeRegressor           | 0.69              | 0.77      | 41796.16   | 0.07       |
| LinearRegression                | 0.10              | 0.35      | 70738.18   | 0.05       |
| TransformedTargetRegressor      | 0.10              | 0.35      | 70738.18   | 0.06       |
| ElasticNetCV                    | -0.20             | 0.13      | 81794.94   | 0.26       |
| DummyRegressor                  | -0.37             | -0.00     | 87619.03   | 0.04       |
| NuSVR                           | -0.38             | -0.00     | 87795.50   | 0.23       |
| SVR                             | -0.41             | -0.02     | 88648.75   | 0.34       |
| QuantileRegressor               | -0.41             | -0.02     | 88667.17   | 0.15       |
| LinearSVR                       | -0.49             | -0.08     | 91186.01   | 0.04       |
| MLPRegressor                    | -5.17             | -3.49     | 185631.91  | 1.34       |
| GaussianProcessRegressor        | -5.97             | -4.08     | 197395.64  | 0.27       |

# Resumen del Dataset

## **1. Información clave sobre las propiedades**
- **Tipo de vivienda (`MSSubClass`)**: 
  - Diferencia viviendas según número de pisos, edad y tipo (unifamiliares, dúplex, etc.).
- **Tamaño del lote (`LotArea`)**:
  - Tamaño en pies cuadrados, crucial para evaluar el valor de la propiedad.
- **Zonificación (`MSZoning`)**: 
  - Información sobre el uso de la tierra (residencial, comercial, industrial), importante para segmentar.
- **Condición general (`OverallQual` y `OverallCond`)**:
  - Evaluaciones cualitativas de la calidad y condición del inmueble (escala de 1 a 10).
- **Año de construcción y remodelación (`YearBuilt` y `YearRemodAdd`)**:
  - Relevante para detectar tendencias o cambios en el mercado según la antigüedad.

---

## **2. Aspectos del vecindario y ubicación**
- **Barrio (`Neighborhood`)**:
  - Localización dentro de la ciudad, útil para segmentar propiedades según áreas más o menos deseadas.
- **Condiciones cercanas (`Condition1` y `Condition2`)**:
  - Describen la proximidad a características positivas (parques) o negativas (carreteras, ferrocarriles).

---

## **3. Características constructivas y funcionales**
- **Estilo de construcción (`HouseStyle`)**: 
  - Número de pisos y diseño (nivel dividido, un piso, etc.).
- **Calidad de los materiales exteriores (`ExterQual` y `ExterCond`)**:
  - Información sobre la calidad y el estado actual de los acabados exteriores.
- **Tamaño de áreas clave**:
  - `GrLivArea`: Área habitable por encima del nivel del suelo.
  - `TotalBsmtSF`: Tamaño del sótano.
  - `GarageArea`: Tamaño del garaje.
- **Calidad de áreas específicas**:
  - `KitchenQual` (calidad de la cocina) y `FireplaceQu` (calidad de la chimenea).

---

## **4. Factores adicionales a considerar**
- **Piscinas y cercas (`PoolArea`, `PoolQC`, `Fence`)**:
  - Características que pueden influir en el valor, aunque son menos comunes.
- **Porches y áreas exteriores (`WoodDeckSF`, `OpenPorchSF`)**:
  - Espacios exteriores, importantes según la ubicación y el clima.
- **Mes y año de venta (`MoSold` y `YrSold`)**:
  - Pueden revelar tendencias temporales o estacionales en los precios.

---

## **5. Factores del precio y condiciones de venta**
- **Tipo y condición de venta (`SaleType` y `SaleCondition`)**:
  - Pueden influir en los valores (ventas entre familiares, ejecuciones hipotecarias, etc.).
- **Características adicionales (`MiscFeature` y `MiscVal`)**:
  - Incluyen características raras o adicionales, como canchas de tenis o elevadores.

---

## **Puntos interesantes a investigar**
1. **Relación entre la calidad general (`OverallQual`) y el precio de venta.**
2. **Impacto del vecindario (`Neighborhood`) en los valores de las propiedades.**
3. **Efecto del tamaño de la propiedad (`GrLivArea`, `LotArea`, etc.) en el precio.**
4. **Tendencias temporales con los años de venta (`YrSold`) o estacionalidad con los meses (`MoSold`).**
5. **Influencia de características menos comunes, como `PoolQC` o `MiscFeature`, en propiedades premium.**

---

## **Recomendaciones iniciales para análisis**
- Realiza un análisis exploratorio para detectar valores nulos y características con poca o nula variabilidad.
- Evalúa la distribución de variables continuas como `LotArea`, `GrLivArea` o `SalePrice` (si está incluida en el dataset).
- Usa mapas de calor o gráficas para identificar correlaciones entre variables clave.
