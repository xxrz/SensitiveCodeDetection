
# with open('New.py','r') as f:
#     for line in f.readlines():
#         if '\xef' in line:
#             print("True")
#             break
#     else:
#         print("False")

# f_posdict = open("New.py", encoding='utf-8')
# posdict = f_posdict.read().split('\n')
# posdict = [x.encode('utf-8').decode("utf-8-sig") for x in posdict]
# with open("test.py","a",encoding='utf-8') as f:
#     for i in posdict:
#         f.write(i)
#         f.write("\n")

import chardet


# 获取文件编码类型
def get_encoding(file):
    # 二进制方式读取，获取字节数据，检测类型
    with open(file, 'rb') as f:
        data = f.read()
        return chardet.detect(data)['encoding']

file_name = 'test.py'
encoding = get_encoding(file_name)
print(encoding)

