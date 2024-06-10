from langchain_community.llms import CTransformers
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
import time

model_file_path = "vinallama-7b-chat_q5_0.gguf"
vector_db_path = "../database/vector_stores/vectorDB"
# Tạo prompt
template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
def load_model(model_file = model_file_path):
    config = {'max_new_tokens': 512, 'repetition_penalty': 1.1, 'context_length': 1000, 'temperature':0.1, 'gpu_layers':30}
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        config=config
    )
    return llm

def create_prompt(template):
    prompt = PromptTemplate(template = template, input_variables = ["context", "question"])
    return prompt

def create_qa_chain(prompt, llm, db):
    llm_chain= RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= 'stuff',
        retriever = db.as_retriever(search_kwags = {"k": 3}),
        return_source_documents = False,
        chain_type_kwargs= {'prompt': prompt}
    )
    return llm_chain

def read_vectors_db(vector_db = vector_db_path):
    embedding_model = GPT4AllEmbeddings(model_file = "all-MiniLM-L6-v2.gguf2.f16.gguf")
    db = FAISS.load_local(vector_db, embedding_model, allow_dangerous_deserialization=True)
    return db


class Model:
    def __init__(self, vecto_DB_path):
        db = read_vectors_db(vecto_DB_path)
        llm = load_model()
        prompt = create_prompt(template)
        self.llm_chain = create_qa_chain(prompt, llm, db)

    def answer(self, question):
        response = self.llm_chain.invoke({'query': question})
        return response

question = "Tác giả đã đưa ra điều gì?"
model = Model("../database/vector_stores/test_text")

start_time = time.time()
print(model.answer(question))
print("End time %ss"%(time.time()-start_time))