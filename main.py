import time
import os
from database import vector_db
from models import model
import torch

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.cuda.empty_cache()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("""
    Main task:
    1. Create vector database
    2. Q & A chat""")
    task = int(input("Choose your task (1, 2):"))
    if(task == 1):
        print("Create vector database.")
        folder_path = input("Link folder have files: ")
        vector_db_name = input("Name vector database: ")
        db = vector_db.create_db_from_PDF(folder_path=folder_path, name_vector= vector_db_name)
        if(db == None):
            print("Create failed!")
        else:
            print("Create sucessful!")
    else:
        db_path = os.path.abspath("./database/vector_stores")
        vector_dbs = os.listdir(db_path)
        for i,v in enumerate(vector_dbs):
            print(f"{i}: {v}")
        # print(vector_dbs)
        i = int(input("Choose vector_dbs: "))
        vector_db_path = os.path.join(db_path, vector_dbs[i])

        gpu_layers = int(input("GPU layers (0 if you don't have gpu): "))
        model_search = model.Model(vector_db_path, gpu_layers)

        continue_state = 1
        while continue_state!=0:
            question = input("Type your question: ")
            print("Start")
            start_time = time.time()
            answer = model_search.answer(question)
            end_time = time.time()
            print_hi(f'Answer: {answer}\nTime: {end_time - start_time}')
            print(answer['result'])
            continue_state = int(input("Do you want to continue: "))

