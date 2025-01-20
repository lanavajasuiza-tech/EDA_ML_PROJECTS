### **Estructura Propuesta del Proyecto**

proyecto_pln/
├── main.py                  # Orquestador principal
├── processing/
│   ├── __init__.py          # Hace que la carpeta sea un módulo
│   ├── cleaning.py          # Limpieza y preprocesamiento de texto
│   ├── corpus.py            # Carga y gestión de corpus
│   ├── embeddings.py        # Generación de representaciones vectoriales (en construcción)
│   ├── frequency_analysis.py # Análisis de frecuencia de palabras
│   ├── lemmatization.py     # Lematización de texto
│   ├── stemming.py          # Stemming de texto
│   ├── tokenization.py      # Tokenización de texto

### **Descripción de Cada Componente**

1. **`main.py`:** Archivo principal donde se orquestan todas las operaciones y módulos. Aquí se importan las funciones de las subcarpetas y se define el flujo de procesamiento.
2. **`/processing`:** Contiene módulos para diferentes tareas de procesamiento de texto:
   * **`cleaning.py`:** Limpieza y normalización (e.g., eliminación de puntuación, stop words).
   * **`tokenization.py`:** Funciones para dividir texto en palabras, frases o caracteres.
   * **`lemmatization.py`:** Stemming y lematización para reducir palabras a su forma base.
   * **`embeddings.py`:** Generación de representaciones vectoriales como Word2Vec o TF-IDF.
3. **`/decorators`:** Decoradores que simplifican tareas comunes, como:
   * **`timing.py`:** Medir tiempos de ejecución para optimizar el rendimiento.
   * **`logging.py`:** Registrar el estado del programa o errores.
4. **`/utils`:** Funciones auxiliares que no pertenecen a un módulo específico:
   * **`file_utils.py`:** Lectura y escritura de archivos, manejo de rutas.
   * **`text_utils.py`:** Funciones comunes de texto (e.g., conteo de palabras, limpieza básica).
   * **`config.py`:** Variables de configuración globales, como rutas de archivos o parámetros del modelo.
5. **`/tests`:** Pruebas unitarias para garantizar que las funciones trabajen como se espera. Cada módulo debería tener un archivo de pruebas asociado.
6. **`/data`:** Almacenamiento de datos:
   * **`raw/`:** Datos sin procesar (e.g., datasets originales).
   * **`processed/`:** Datos procesados listos para análisis.
   * **`embeddings/`:** Almacenamiento de representaciones vectoriales.
7. **`requirements.txt`:** Lista de dependencias necesarias para el proyecto (e.g., NLTK, spaCy, Transformers).
8. **`README.md`:** Documentación básica del proyecto, incluyendo cómo configurarlo y usarlo.


### **Resumen de las Funcionalidades en `010.PNL.py`**

1. **Descarga e Instalación de Librerías y Corpus:**
   * Descarga de `stopwords` y corpus adicionales mediante `nltk.download`.
2. **Carga de un Corpus Externo:**
   * Uso de `CategorizedPlaintextCorpusReader` para cargar un corpus categorizado con revisiones positivas y negativas.
3. **Muestreo y Visualización:**
   * Extracción aleatoria de archivos de cada categoría (`pos` y `neg`) para su visualización.
4. **Análisis Estadístico:**
   * Cuenta de palabras (`'wh' words`) en el corpus `Brown`.
   * Generación de distribuciones de frecuencia.
5. **Extracción desde la Web:**
   * Uso de `BeautifulSoup` para limpiar contenido extraído de HTML.
6. **Limpieza y Preprocesamiento:**
   * Eliminación de puntuación, números, espacios adicionales, palabras vacías (`stopwords`), y conversión a minúsculas.
7. **Lematización y Stemming:**
   * Uso de `WordNetLemmatizer` y `SnowballStemmer` para reducir palabras a sus formas base o raíces.
8. **Lectura de PDFs y Documentos:**
   * Extracción de texto desde archivos PDF y `.docx`.
9. **Tokenización y Sinónimos:**
   * Segmentación en oraciones (`sent_tokenize`) y palabras (`word_tokenize`).
   * Obtención de sinónimos usando `WordNet`.
