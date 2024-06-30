import time
import os
from database import vector_db
from models import model

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
#     raw_text = """ Transformer attention thông thường thực hiện attention trên toàn bộ
# feature map nó dẫn đến độ phức tạp của thuật toán tăng cao khi spatial size của feature map tăng.
# Tác giả đưa ra một kiểu attention mới mà chỉ attend
# vào một số sample locations (sample locations này cũng không cố định mà được học trong
# quá trình training tương tự như trong deformable convolution)
# qua đó giúp giảm độ phức tạp của thuật toán và làm giảm thời gian training mô hình. """
    # vector_db.create_db_from_PDF("D:\\AI_search_PDF\\database\\file_stores", "test_24_06_2024")
    # vector_db.create_db_from_text(raw_text, 'test_text')

    db_path = os.path.abspath("./database/vector_stores")
    vector_dbs = os.listdir(db_path)
    vector_db_path = os.path.join(db_path, vector_dbs[1])

    model_search = model.Model(os.path.join(db_path, vector_dbs[1]), 10)
    question = "Tô Hoài là"
    print("Start")
    start_time = time.time()
    answer = model_search.answer(question)
    end_time = time.time()
    print_hi(f'Answer: {answer}\nTime: {end_time - start_time}')

