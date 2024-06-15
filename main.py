from database import vector_db
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
    vector_db.create_db_from_PDF("D:\\AI_search_PDF\\database\\file_stores", "test_11_06_2024")
    # vector_db.create_db_from_text(raw_text, 'test_text')
    print_hi('PyCharm')

