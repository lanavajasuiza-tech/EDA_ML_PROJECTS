# Información del dataSet

| Propiedad                                            | Descripción                                                                     |
| :--------------------------------------------------- | :------------------------------------------------------------------------------- |
| Número de filas                                     | 13644                                                                            |
| Número de columnas                                  | 3                                                                                |
| Nombres de las columnas                              | ['Comment', 'Benchmark_results_of_politeness', 'Benchmark_results_of_sentiment'] |
| Columnas categóricas                                | ['Comment', 'Benchmark_results_of_politeness']                                   |
| Columnas numéricas                                  | ['Benchmark_results_of_sentiment']                                               |
| ¿Hay columnas duplicadas?                           | False                                                                            |
| Número de filas duplicadas                          | 244                                                                              |
| Valores nulos en 'Comment'                           | 34                                                                               |
| Valores únicos en 'Comment'                         | 13391                                                                            |
| Valores únicos en 'Benchmark_results_of_politeness' | 2                                                                                |
| Valores únicos en 'Benchmark_results_of_sentiment'  | 2428                                                                             |

## Análisis y Limpieza

Se analizan los resultados menos obvios:

* **Benchmark_results_of_politeness**: Solo tiene dos valores POLITE / IMPOLITE
* **Benchmark_results_of_sentiment**: Muestra rango de valores en función de la positividad/negatividad del comentrio, susceptible de categorizar
* **Los valores NaN apenas representan un 0.2%** del dataSet por lo que son susceptibles de 'desaparecer'
* **Para los valores duplicados** se observan que tambien hay alguna incongruencia debido al proceso previo del data set original, hay 238 valores exactamente iguales y 6 con alguna diferencia por lo que se decide también borrar estas filas, que **supone un 1.8% que sumado al 0.2 de NaN anterior supone un total del 2% del total del dataSet, cifra bastante asumible.**

Tras el borrado nos queda un total de 13397 filas

## Tramiento de datos númericos a categóricos

Valores de la columna Valores únicos en 'Benchmark_results_of_sentiment':

| Rango de valores | Frecuencia | Categoría   |
| ---------------- | ---------- | ------------ |
| (-1.003, -0.8]   | 8          | Muy negativo |
| (-0.8, -0.6]     | 36         | Muy negativo |
| (-0.6, -0.4]     | 212        | Negativo     |
| (-0.4, -0.2]     | 484        | Negativo     |
| (-0.2, 0.0]      | 5597       | Neutral      |
| (0.0, 0.2]       | 4008       | Neutral      |
| (0.2, 0.4]       | 2071       | Positivo     |
| (0.4, 0.6]       | 792        | Positivo     |
| (0.6, 0.8]       | 353        | Muy positivo |
| (0.8, 1.0]       | 83         | Muy positivo |
