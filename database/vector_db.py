from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import tempfile

data_path = './database/file_stores'
vector_db_path = './database/vector_stores'
model_path = './models'

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)


def create_db_from_text(raw_text, name_vector):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)

    # Embeddings
    embedding_model = GPT4AllEmbeddings(model_path=model_path)

    # Kiem tra name_vector ton tai chua
    i=-1
    vector_path = f"{vector_db_path}/{name_vector}"
    vector_path_check = vector_path
    while(os.path.exists(vector_path_check)):
        i+=1
        vector_path_check = f"{vector_db_path}/{name_vector}_{i}"
    vector_path = vector_path_check

    # Chia nho van ban tu cac file ra thanh cac doan
    chunks = text_splitter.split_text(raw_text)

    # Embedding cac doan van ban
    db = FAISS.from_texts(chunks, embedding_model)
    db.save_local(vector_path)
    print("Success")
    return db

def create_db_from_PDF(folder_path = data_path, name_vector = "vectorDB"):
    # embedding_model = GPT4AllEmbeddings(model_path=model_path)
    embedding_model = GPT4AllEmbeddings()
    # model_name = "vinai/phobert-base-v2"
    # model_kwargs = {'device': 'cuda'}
    # encode_kwargs = {'normalize_embeddings': True}
    # embedding_model = HuggingFaceEmbeddings(
    #     model_name=model_name,
    #     model_kwargs=model_kwargs,
    #     encode_kwargs=encode_kwargs
    # )

    # Load file PDF
    loader= DirectoryLoader(folder_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(documents)
    if(len(documents)==0):
        pass

    # Kiem tra name_vector ton tai chua
    i=-1
    vector_path = f"{vector_db_path}/{name_vector}"
    vector_path_check = vector_path
    while(os.path.exists(vector_path_check)):
        i+=1
        vector_path_check = f"{vector_db_path}/{name_vector}_{i}"
    vector_path = vector_path_check

    # Chia nho van ban tu cac file ra thanh cac doan
    chunks = text_splitter.split_documents(documents)

    # Embedding cac doan van ban
    db = FAISS.from_documents(chunks, embedding_model)
    print(db)
    db.save_local(vector_path)
    return db

def create_db_from_uploaded_PDF(uploaded_files, name_vector = "vectorDB"):
    embedding_model = GPT4AllEmbeddings()
    documents = []

    for pdf_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete = False) as temp_pdf:
            temp_pdf.write(pdf_file.getbuffer())
            temp_pdf.flush()
            loader = PyPDFLoader(temp_pdf.name)
            documents.extend(loader.load())
        os.remove(temp_pdf.name)

    # Check if vector path exists
    i = -1
    vector_path = f"{vector_db_path}/{name_vector}"
    vector_path_check = vector_path
    while os.path.exists(vector_path_check):
        i += 1
        vector_path_check = f"{vector_db_path}/{name_vector}_{i}"
    vector_path = vector_path_check

    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)

    # Embed chunks and create FAISS index
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_path)
    return db

