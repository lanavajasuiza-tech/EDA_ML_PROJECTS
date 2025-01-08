import streamlit as st
import torch
import os
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline
)

# Forzar uso de CPU
device = torch.device("cpu")

# Cargar modelo para QA
def load_qa_model():
    qa_model_name = "deepset/roberta-base-squad2"
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name).to(device)
    return qa_model, qa_tokenizer

# Cargar modelo para Resumen
def load_summarization_model():
    summarization_model_name = "mrm8488/bert2bert_shared-spanish-finetuned-summarization"
    summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)
    summarization_model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name).to(device)
    return summarization_model, summarization_tokenizer

# Cargar modelo para Búsqueda Semántica
def load_feature_extraction_pipeline():
    embedding_pipeline = pipeline("feature-extraction", model="sentence-transformers/LaBSE")
    return embedding_pipeline

# Cargar modelo para Generación de Texto
def load_text_generation_model():
    text_generation_model_name = "mrm8488/GPT2-finetuned-spanish"
    text_generation_tokenizer = AutoTokenizer.from_pretrained(text_generation_model_name)
    text_generation_model = AutoModelForCausalLM.from_pretrained(text_generation_model_name).to(device)
    return text_generation_model, text_generation_tokenizer

# Cargar los modelos
qa_model, qa_tokenizer = load_qa_model()
summarization_model, summarization_tokenizer = load_summarization_model()
embedding_pipeline = load_feature_extraction_pipeline()
text_generation_model, text_generation_tokenizer = load_text_generation_model()

def generate_qa_response(question, context):
    inputs = qa_tokenizer(question, context, return_tensors="pt").to(device)
    outputs = qa_model(**inputs)
    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits)
    answer = qa_tokenizer.decode(inputs["input_ids"][0][start_index:end_index+1])
    return answer

def generate_summary(text):
    inputs = summarization_tokenizer(text, return_tensors="pt", max_length=512).to(device)
    outputs = summarization_model.generate(**inputs)
    summary = summarization_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def generate_text_response(prompt):
    inputs = text_generation_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = text_generation_model.generate(**inputs, max_new_tokens=256)
    response = text_generation_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    st.title("PMV-GPT-Local en Español")
    task = st.selectbox(
        "Selecciona la tarea que deseas realizar:",
        ("Pregunta y Respuesta", "Resumen", "Búsqueda Semántica", "Generación de Texto")
    )

    if task == "Pregunta y Respuesta":
        context = st.text_area("Contexto:")
        question = st.text_input("Pregunta:")
        if st.button("Generar Respuesta"):
            response = generate_qa_response(question, context)
            st.write(response)

    elif task == "Resumen":
        text = st.text_area("Texto a resumir:")
        if st.button("Generar Resumen"):
            summary = generate_summary(text)
            st.write(summary)

    elif task == "Búsqueda Semántica":
        text = st.text_area("Texto para embeddings:")
        if st.button("Generar Embeddings"):
            embeddings = embedding_pipeline(text)
            st.write(embeddings)

    elif task == "Generación de Texto":
        prompt = st.text_input("Introduce un prompt para generar texto:")
        if st.button("Generar Texto"):
            response = generate_text_response(prompt)
            st.write(response)

if __name__ == "__main__":
    main()

# Sidebar content
with st.sidebar:
    st.title('PMV-GPT-Local')
    st.markdown('''
    ## Acerca de este MVP:
    - Esta es una aplicación de lectura de PDF
    - Se pretende facilitar el aterrizaje de nuevas incorporaciones
    - Usa un LLM local para garantizar la privacidad de la información
    ''')
    add_vertical_space(5)
    st.write('MVP 1.0 realizado por LNS, ENE-2025')

# Procesamiento de PDFs
def process_pdf():
    st.header('Consultor de Documentación Interna')

    # Subir un archivo PDF
    pdf = st.file_uploader('Sube un PDF con la documentación', type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        # Extraer texto del PDF
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Dividir texto en fragmentos
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Crear nombre para el índice
        store_name = pdf.name[:-4]
        index_path = f'{store_name}_index'

        try:
            # Verificar si ya existe un índice guardado
            if os.path.exists(index_path):
                VectorStore = Chroma.load_local(
                    index_path, 
                    allow_dangerous_deserialization=True
                )
                st.write('Embeddings cargados desde el disco.')
            else:
                # Calcular embeddings y crear un índice Chroma
                embeddings = chunks  # Simplificado para GGML
                VectorStore = Chroma.from_texts(chunks, embeddings)

                # Guardar el índice Chroma
                VectorStore.save_local(index_path)
                st.write('Embeddings calculados y guardados.')

            # Aceptar preguntas del usuario
            query = st.text_input('Pregunta sobre este documento PDF:')
            if query:
                docs = VectorStore.similarity_search(query=query, k=3)
                response = generate_text_response(query)
                st.write(response)

        except Exception as e:
            st.error(f"Error al cargar o guardar el índice: {e}")

if __name__ == '__main__':
    process_pdf()
