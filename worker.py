import os
import torch
import logging

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from langchain_core.prompts import PromptTemplate  # Importación actualizada según el aviso de deprecación
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceInstructEmbeddings  # Nueva ruta de importación
from langchain_community.document_loaders import PyPDFLoader  # Nueva ruta de importación
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # Nueva ruta de importación
from langchain_ibm import WatsonxLLM

# Verificar la disponibilidad de GPU y establecer el dispositivo adecuado para el cálculo.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Variables globales
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None

# Función para inicializar el modelo de lenguaje y sus embeddings
def init_llm():
    global llm_hub, embeddings

    logger.info("Inicializando WatsonxLLM y embeddings...")

    # Configuración del modelo Llama
    MODEL_ID = "meta-llama/llama-3-3-70b-instruct"
    WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
    PROJECT_ID = "skills-network"

    # Usar los mismos parámetros que antes:
    #   MAX_NEW_TOKENS: 256, TEMPERATURE: 0.1
    model_parameters = {
        # "decoding_method": "greedy",
        "max_new_tokens": 256,
        "temperature": 0.1,
    }

    # Inicializar Llama LLM utilizando la API actualizada de WatsonxLLM
    llm_hub = WatsonxLLM(
        model_id=MODEL_ID,
        url=WATSONX_URL,
        project_id=PROJECT_ID,
        params=model_parameters
    )
    logger.debug("WatsonxLLM inicializado: %s", llm_hub)

    # Inicializar embeddings utilizando un modelo preentrenado para representar los datos de texto.
    ### --> si estás utilizando la API de huggingFace:
    # Configurar la variable de entorno para HuggingFace e inicializar el modelo deseado, y cargar el modelo en HuggingFaceHub
    # no olvides eliminar llm_hub para watsonX

    # os.environ["HUGGINGFACEHUB_API_TOKEN"] = "TU CLAVE API"
    # model_id = "tiiuae/falcon-7b-instruct"
    #llm_hub = HuggingFaceHub(repo_id=model_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 600, "max_length": 600})

    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={"device": DEVICE}
    )
    logger.debug("Embeddings inicializados con el dispositivo del modelo: %s", DEVICE)

# Función para procesar un documento PDF
def process_document(document_path):
    global conversation_retrieval_chain

    logger.info("Cargando documento desde la ruta: %s", document_path)
    # Cargar el documento
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    logger.debug("Cargados %d documento(s)", len(documents))

    # Dividir el documento en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)
    logger.debug("Documento dividido en %d fragmentos de texto", len(texts))

    # Crear una base de datos de embeddings usando Chroma a partir de los fragmentos de texto divididos.
    logger.info("Inicializando la tienda de vectores Chroma a partir de documentos...")
    db = Chroma.from_documents(texts, embedding=embeddings)
    logger.debug("Tienda de vectores Chroma inicializada.")

    # Opcional: Registrar colecciones disponibles si son accesibles (esto puede ser API interna)
    try:
        collections = db._client.list_collections()  # _client es interno; ajustar si es necesario
        logger.debug("Colecciones disponibles en Chroma: %s", collections)
    except Exception as e:
        logger.warning("No se pudieron recuperar colecciones de Chroma: %s", e)

    # Construir la cadena de QA, que utiliza el LLM y el recuperador para responder preguntas. 
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
        return_source_documents=False,
        input_key="question"
        # chain_type_kwargs={"prompt": prompt}  # si estás utilizando una plantilla de aviso, descomenta esta parte
    )
    logger.info("Cadena RetrievalQA creada con éxito.")

# Función para procesar un aviso del usuario
def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history

    logger.info("Procesando aviso: %s", prompt)
    # Consultar el modelo utilizando el nuevo método .invoke()
    output = conversation_retrieval_chain.invoke({"question": prompt, "chat_history": chat_history})
    answer = output["result"]
    logger.debug("Respuesta del modelo: %s", answer)

    # Actualizar el historial de chat
    chat_history.append((prompt, answer))
    logger.debug("Historial de chat actualizado. Total de intercambios: %d", len(chat_history))

    # Devolver la respuesta del modelo
    return answer

# Inicializar el modelo de lenguaje
init_llm()
logger.info("Inicialización de LLM y embeddings completa.")