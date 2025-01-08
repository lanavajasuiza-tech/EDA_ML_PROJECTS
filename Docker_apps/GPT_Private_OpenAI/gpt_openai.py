import streamlit as st
import os
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Sidebar content
with st.sidebar:
    st.title('CAU Talio - UD')
    st.markdown('''
    ## Acerca de este MVP:
    - Esta es una aplicación de lectura de PDF
    - Se pretende facilitar el aterrizaje de nuevas incorporaciones
    - Usa modelo pre-entrenado de OpenAI
    ''')
    add_vertical_space(5)
    st.write('MVP 1.0 realizado por LNS, DIC-2024')


def main():
    st.header('Habla conmigo')

    # Cargar las variables de entorno
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("No se encontró la clave de API en las variables de entorno.")
        return

    # Subir un archivo PDF
    pdf = st.file_uploader('Sube un PDF', type='pdf')

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
                VectorStore = FAISS.load_local(
                    index_path, 
                    OpenAIEmbeddings(openai_api_key=api_key),
                    allow_dangerous_deserialization=True  # Permitir deserialización
                )
                st.write('Embeddings cargados desde el disco.')
            else:
                # Calcular embeddings y crear un índice FAISS
                embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                VectorStore = FAISS.from_texts(chunks, embeddings)

                # Guardar el índice FAISS
                VectorStore.save_local(index_path)
                st.write('Embeddings calculados y guardados.')

            # Aceptar preguntas del usuario
            query = st.text_input('Pregunta sobre este documento PDF:')
            if query:
                docs = VectorStore.similarity_search(query=query, k=3)
                
                llm=OpenAI(temperature=0,)
                chain=load_qa_chain(llm=llm, chain_type='stuff')
                response = chain.run(input_documents=docs, question=query)
                st.write(response)
                #st.write("Resultados relevantes:")
                #for doc in docs:
                    #st.write(doc.page_content)

        except Exception as e:
            st.error(f"Error al cargar o guardar el índice: {e}")


if __name__ == '__main__':
    main()
