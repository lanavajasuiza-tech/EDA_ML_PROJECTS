# Informe de Análisis: Evolución del Suministro Circulante de BTC y ETH

El presente análisis se centra en las criptomonedas **Bitcoin (BTC)** y **Ethereum (ETH)**, las dos monedas digitales más relevantes en el ecosistema de blockchain. Utilizando datos históricos simulados, se estudia la evolución del suministro circulante (`<span>circulatingSupply</span>`) desde el año 2009 hasta 2023, con el objetivo de identificar tendencias y patrones que puedan ser útiles para la toma de decisiones en el mercado de criptomonedas.

El análisis tiene como finalidad observar cómo ha evolucionado el suministro circulante de BTC y ETH a lo largo de los años, identificando posibles tendencias o comportamientos que puedan influir en las decisiones de inversión y gestión de estas criptomonedas.

### Dataset

Los datos provienen de un dataSet de Kaggle (https://www.kaggle.com/datasets/harshalhonde/coinmarketcap-cryptocurrency-dataset-2023). Los valores de `<span>maxSupply</span>`, `<span>circulatingSupply</span>` y `<span>price</span>` fueron elegidos para reflejar un rango realista basado en valores históricos de criptomonedas. Las monedas seleccionadas (BTC y ETH) representan algunas de las más conocidas y con diferentes características de mercado.

### Proceso de Análisis

1. **Carga de Datos:** Se cargó el dataset y se convirtió la columna `<span>dateAdded</span>` a formato datetime.
2. **Agrupación de Datos:** Se agruparon los datos por año y símbolo (BTC o ETH) y se calculó la mediana del suministro circulante para cada combinación de año y símbolo.
3. **Visualización:** Se generó un gráfico de dispersión con líneas de tendencia para visualizar la evolución del suministro circulante.

## Resultados

### Gráfico de Evolución del Suministro Circulante

El gráfico generado muestra la evolución del suministro circulante (`<span>circulatingSupply</span>`) mediano de BTC y ETH a lo largo de los años.

* **BTC:** El suministro circulante de Bitcoin muestra fluctuaciones significativas en ciertos periodos, con caídas importantes seguidas de recuperaciones graduales.
* **ETH:** Ethereum, por su parte, presenta un comportamiento más estable, aunque también muestra algunas variaciones significativas en ciertos años.

Ambas criptomonedas muestran tendencias al alza en sus respectivos suministros circulantes, lo que es coherente con sus modelos económicos de emisión de monedas.

## Interpretación

Aunque los datos son simulados, las fluctuaciones observadas en el gráfico reflejan posibles eventos reales que podrían influir en el suministro circulante de estas criptomonedas, tales como:

* Cambios en las políticas de emisión de monedas.
* Eventos de reducción a la mitad (halvings) en el caso de Bitcoin.
* Cambios en los protocolos de consenso y actualizaciones de red.

Estas fluctuaciones pueden ser indicativas de periodos de volatilidad en el mercado y podrían influir en las decisiones de los inversores.

 análisis del suministro circulante de BTC y ETH a lo largo de los años permite observar cómo estas criptomonedas han evolucionado en términos de oferta. Identificar patrones y fluctuaciones en los suministros puede ser clave para anticiparse a cambios en el mercado y tomar decisiones de inversión informadas.

Se recomienda continuar con este tipo de análisis utilizando datos reales para obtener conclusiones más precisas y relevantes.
