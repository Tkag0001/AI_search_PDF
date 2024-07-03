import time
import os
import logging
import shutil
import requests
import torch
from database import vector_db
from models import model
import streamlit as st
from database import vector_db

# To avoid duplicate library errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.cuda.empty_cache()

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up the folder vector database path
db_path = os.path.abspath("./database/vector_stores")

# Initialize session state for the response queue and other variables
if 'queue' not in st.session_state:
    st.session_state.queue = ""
if 'gpu_layers' not in st.session_state:
    st.session_state.gpu_layers = 0
if 'database' not in st.session_state:
    st.session_state.database = ""
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'vector_dbs' not in st.session_state:
    st.session_state.vector_dbs = os.listdir(db_path)
if 'model_instance' not in st.session_state:
    st.session_state.model_instance = None

#warning no database exist
if len(os.listdir(db_path))==0:
    st.sidebar.warning("Don't have any vector databases to init model!",  icon="⚠️")

# Streamlit app layout
st.write("# AI Search PDF Tool")
st.sidebar.header("Option")
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
vector_db_name = st.sidebar.text_input("Vector database name:")
create_db = st.sidebar.button("Create database")
selected_database = st.sidebar.selectbox("Database", st.session_state.vector_dbs)
selected_gpu_layers = st.sidebar.number_input("GPU layers (Select 0 if you don't have GPU)", 0, 50, value=0)
adapted_model = st.sidebar.button("Update model")
delete_db = st.sidebar.button("Delete database")

# Function to handle answering the question
def handle_answer(question):
    try:
        if question.strip() != "":
            st.session_state.queue += f"Question: {question}\n"
            start_time = time.time()
            answer = st.session_state.model_instance.answer(question)
            answer = answer["result"]
            logging.info(f"Answer received: {answer}\nTime: {time.time() - start_time}")
            st.session_state.queue += f"Answer: {answer}\nTime: {time.time() - start_time}"
        st.experimental_rerun()
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please check your internet connection and try again.")
        logging.error("Request timed out.")
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
        logging.error(f"An error occurred: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logging.error(f"An unexpected error occurred: {e}")

# Function to load model
def get_model(database, gpu_layers):
    return model.Model(os.path.join(db_path, database), gpu_layers)

# Update model when "Update model" button is pressed
if adapted_model:
    if selected_database == None:
        st.sidebar.warning("Don't have any vector databases",  icon="⚠️")
    elif not os.path.exists(os.path.join(db_path, selected_database)):
        st.sidebar.warning("This vector database is not exist, you can reload to solve",  icon="⚠️")
    else:
        st.session_state.model_instance = get_model(selected_database, selected_gpu_layers)
        st.session_state.database = selected_database
        st.session_state.gpu_layers = selected_gpu_layers

# Display response and question text areas
response = st.text_area("Response", value=st.session_state.queue, height=300)
question = st.text_area("Question", height=80)

# Define the button action
if st.button("Answer"):
    logging.info(f"Question: {question}")
    handle_answer(question)

if create_db:
    if vector_db_name.strip() == "" or vector_db_name == None:
        st.sidebar.warning('This vector database name is empty!', icon="⚠️")
    elif len(uploaded_files) == 0:
        st.sidebar.warning("Can't detect any PDF files to create!",  icon="⚠️")
    else:
        vector_db.create_db_from_uploaded_PDF(uploaded_files, vector_db_name)
        vector_path = os.path.join(db_path, vector_db_name)
        print(vector_path)
        if os.path.exists(vector_path):
            st.session_state.vector_dbs = os.listdir(db_path)
            st.success(f"Vector database created and saved to {vector_path}")
        else:
            st.error(f"Failed to create vector database from uploaded PDF files")

if delete_db:
    if len(st.session_state.vector_dbs) == 0:
        st.sidebar.warning("Don't have any databases to delete!")
    else:
        vector_path_to_delete = os.path.join(db_path, selected_database)
        shutil.rmtree(vector_path_to_delete)
        st.session_state.vector_dbs = os.listdir(db_path)
        if not os.path.exists(vector_path_to_delete):
            st.sidebar.success(f"Delete successful!")
        else:
            st.sidebar.error(f"Failed to delete")

