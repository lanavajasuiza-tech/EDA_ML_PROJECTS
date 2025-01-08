###### PMV_I_pdfReader_OpenAI

# Pipeline para un Asistente de Empleados Nuevos

## Descripción
Este pipeline está diseñado para asistir a nuevos empleados proporcionando respuestas rápidas y relevantes sobre documentos internos (como manuales o guías) a partir de archivos PDF.

---

## Pipeline Simplificado

1. **Entrada:**
   - Subir un archivo PDF (manual, guía, etc.).

2. **Preprocesamiento:**
   - Extraer texto del PDF.
   - Dividir el texto en fragmentos (*chunks*).
   - Generar representaciones vectoriales (*embeddings*) de cada fragmento.

3. **Almacén:**
   - Guardar estos *embeddings* en un almacén vectorial (como FAISS) para realizar búsquedas rápidas.

4. **Interacción del Usuario:**
   - Permitir que los usuarios hagan preguntas sobre el contenido del PDF.
   - Buscar la respuesta más relevante en el almacén vectorial.

5. **Respuesta:**
   - Generar una respuesta contextual basada en los datos encontrados.

---

## Herramientas Recomendadas

1. **Manejo del PDF:**
   - Librería: `PyPDF2` o `pdfplumber` para extraer texto.

2. **División del Texto (*Text Splitting*):**
   - Librería: `langchain` (por ejemplo, `RecursiveCharacterTextSplitter`).

3. **Generación de *Embeddings*:**
   - Alternativa gratuita: `SentenceTransformers` (disponible en Hugging Face).
   - **Modelo sugerido:** `all-MiniLM-L6-v2` (eficiente y gratuito).

4. **Almacén Vectorial:**
   - Librería: `FAISS` (Facebook AI Similarity Search).
   - Es local, gratuito, y funciona muy bien para búsquedas vectoriales.

5. **Interfaz:**
   - Herramienta: `Streamlit` para construir interfaces de usuario rápidas y sencillas.

---

## Implementación del Pipeline

### Paso 1: Instalación

Ejecuta el siguiente comando para instalar las librerías necesarias:

```bash
pip install -r requirements.txt
```
### Paso 2: Código de la aplicación

Ver código en script app.py

# Explicación del Pipeline

## Subir el PDF:
- Extrae el texto usando `PyPDF2`.

## Preprocesar el Texto:
- Divide el texto en fragmentos para que sea más manejable.

## Generar Embeddings:
- Usa `SentenceTransformers` para convertir texto a vectores.

## Almacén FAISS:
- Guarda los embeddings en un índice de FAISS para búsquedas rápidas.

## Buscar Respuestas:
- Cuando el usuario pregunta, genera un embedding de la consulta, busca los fragmentos más similares en FAISS, y los devuelve.

---

# Ventajas de este Pipeline

## Cero Costos Recurrentes:
- Todo se ejecuta localmente.

## Seguridad:
- Los datos no necesitan salir de la infraestructura.

## Flexibilidad:
- Puedes adaptarlo fácilmente a otros modelos o datos.

## Escalabilidad:
- FAISS y SentenceTransformers manejan cargas grandes eficientemente.

