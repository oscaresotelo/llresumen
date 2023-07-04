# import streamlit as st
# from langchain.docstore.document import Document
# from langchain.chains.summarize import load_summarize_chain
# import os
# from dotenv import load_dotenv, find_dotenv
# from langchain import HuggingFaceHub
# from langchain import PromptTemplate, LLMChain, OpenAI
# from langchain.text_splitter import RecursiveCharacterTextSplitter,  CharacterTextSplitter
# from langchain.chains.summarize import load_summarize_chain
# from langchain.document_loaders import YoutubeLoader
# import textwrap
# from textblob import TextBlob


# def generate_response(txt):
#     # Instanciar el modelo LLM
#     load_dotenv()
#     HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
#     repo_id = "tiiuae/falcon-7b"
#     llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, 
#                          repo_id=repo_id, 
#                          model_kwargs={"temperature":0.7, "max_new_tokens":4000})
#     # Dividir el texto
#     text_splitter = CharacterTextSplitter()
#     texts = text_splitter.split_text(txt)
#     # Crear múltiples documentos
#     docs = [Document(page_content=t) for t in texts]
#     # Resumen del texto
#     chain = load_summarize_chain(llm, chain_type='map_reduce')
#     return chain.run(docs)

# # Configuración de la página
# st.set_page_config(page_title='🦜🔗 App de Resumen')
# st.title('🦜🔗 App de Resumen')

# # Entrada de texto
# txt_input = st.text_area('Ingrese texto', '', height=200)

# # Formulario para aceptar la entrada de texto del usuario para resumirlo
# result = []
# with st.form('summarize_form', clear_on_submit=True):
    
#     submitted = st.form_submit_button('Enviar')
#     if submitted:
#         with st.spinner('Procesando...'):
#             response = generate_response(txt_input)

#             result.append(response)

# if len(result):
#    # st.info(response)
#    blob = TextBlob(response)
#    st.info(str(blob.translate(from_lang="en",to='es'))) 



import streamlit as st
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import os
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter,  CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
import textwrap
from textblob import TextBlob

load_dotenv()


def generate_response(txt):
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
    repo_id = "tiiuae/falcon-7b"
    llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, 
                            repo_id=repo_id, 
                            model_kwargs={"temperature":0.7, "max_new_tokens":4000})
    # Instanciar el modelo LLM
   
    # Dividir el texto
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)

    # Crear múltiples documentos
    docs = [Document(page_content=t) for t in texts]

    # Resumen del texto
    chain = load_summarize_chain(llm, chain_type = 'map_reduce')
   
    return chain.run(docs)
    
# Configuración de la página
st.set_page_config(page_title='🦜🔗 App de Resumen')
st.title('🦜🔗 App de Resumen')

# Entrada de texto
# txt_input = st.text_area('Ingrese texto', '', height=200)
file = st.file_uploader("Cargar Archivo txt")

# Formulario para aceptar la entrada de texto del usuario para resumirlo
result = []
with st.form('summarize_form', clear_on_submit=True):
    
    submitted = st.form_submit_button('Enviar')
    if submitted:
        if file is not None:
            with st.spinner('Procesando...'):

                text = file.read().decode()
                
                response = generate_response(text)

                result.append(response)
            
               
if len(result):
   st.info(response)
   # blob = TextBlob(response)
   # st.info(str(blob.translate(from_lang="en",to='es'))) 