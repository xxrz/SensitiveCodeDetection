"""
Interface to VulDeePecker project
"""
import sys
import os
import uuid

import pandas
from clean_gadget import clean_gadget
from vectorize_gadget import GadgetVectorizer
from blstm import BLSTM


"""
Parses gadget file to find individual gadgets
Yields each gadget as list of strings, where each element is code line
Has to ignore first line of each gadget, which starts as integer+space
At the end of each code gadget is binary value
    This indicates whether or not there is vulnerability in that gadget
"""
# def parse_file(dir):
#     # files = os.listdir(dir)
#     # for file in files:
#     #     path = dir+"\\"+file
#     with open(dir, "r", encoding="utf8") as file:
#         gadget = []
#         gadget_val = 0
#         for line in file:
#             stripped = line.strip()
#             if not stripped:
#                 continue
#             if "-" * 30 in line and gadget:
#             # if gadget:
#                 yield clean_gadget(gadget), gadget_val
#                 gadget = []
#             elif stripped.split()[0].isdigit():
#                 if gadget:
#                     # Code line could start with number (somehow)
#                     if stripped.isdigit():
#                         gadget_val = int(stripped)
#                     else:
#                         gadget.append(stripped)
#             else:
#                 gadget.append(stripped)

def parse_file(dir):
    files = os.listdir(dir)
    for file in files:
        path = dir+"\\"+file
        with open(path, "r", encoding="utf8") as file:
            gadget = []
            gadget_val = 0
            for line in file:
                stripped = line.strip()
                if not stripped:
                    continue
                if "-" * 30 in line and gadget:
                    # if gadget:
                    yield clean_gadget(gadget), gadget_val
                    gadget = []
                elif stripped.split()[0].isdigit():
                    if gadget:
                        # Code line could start with number (somehow)
                        if stripped.isdigit():
                            gadget_val = int(stripped)
                        else:
                            gadget.append(stripped)
                else:
                    gadget.append(stripped)


"""
Uses gadget file parser to get gadgets and vulnerability indicators
Assuming all gadgets can fit in memory, build list of gadget dictionaries
    Dictionary contains gadgets and vulnerability indicator
    Add each gadget to GadgetVectorizer
Train GadgetVectorizer model, prepare for vectorization
Loop again through list of gadgets
    Vectorize each gadget and put vector into new list
Convert list of dictionaries to dataframe when all gadgets are processed
"""
def get_vectors_df(filename, vector_length=100):
    gadgets = []
    count = 0
    vectorizer = GadgetVectorizer(vector_length)
    for gadget, val in parse_file(filename):#gadget是每个gadget的每一行经过clean的列表
        count += 1
        print("Collecting gadgets...", count, end="\r")
        vectorizer.add_gadget(gadget)#分词添加到vectorizer中的gadgets（这个不同于gadgets,是分了词的）
        row = {"gadget" : gadget, "val" : val}
        #写入文件
        x = row['gadget']
        # filedir = r"D:\XRZ\Data\sard\C\joern切片\transform"+"\\"+filename.split("\\")[-2]+"\\"+filename.split("\\")[-1]
        filedir = r"D:\XRZ\Data\joern\github\SensitiveOperations\testData\4transform" + "\\" + filename.split("\\")[-2] + "\\" + \
                  filename.split("\\")[-1]
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        file=filedir+"\\"+str(uuid.uuid1())+".c"
        with open(file,"a",encoding='utf-8') as f:
            for i in x:
                f.write(i+"\n")

        gadgets.append(row)
    # print('Found {} forward slices and {} backward slices'
    #       .format(vectorizer.forward_slices, vectorizer.backward_slices))
    # print()
    # print("Training model...", end="\r")
    # #训练完了所有的词向量
    # vectorizer.train_model()
    # print()
    # vectors = []
    # count = 0
    # for gadget in gadgets:
    #     count += 1
    #     print("Processing gadgets...", count, end="\r")
    #     vector = vectorizer.vectorize(gadget["gadget"])
    #     row = {"vector" : vector, "val" : gadget["val"]}
    #     vectors.append(row)
    # print()
    # df = pandas.DataFrame(vectors)
    # return df
            
"""
Gets filename, either loads vector DataFrame if exists or creates it if not
Instantiate neural network, pass data to it, train, test, print accuracy
"""
def main():
    # if len(sys.argv) != 2:
    #     print("Usage: python vuldeepecker.py [filename]")
    #     exit()
    # filename = r"D:\XRZ\Data\sard\C\joern切片\CWE-200\bad"
    filename = r"D:\XRZ\Data\joern\github\SensitiveOperations\testData\3deal\pm\bad"
    parse_file(filename)
    base = os.path.splitext(os.path.basename(filename))[0]
    vector_filename = base + "_gadget_vectors.pkl"
    vector_length = 50
    if os.path.exists(vector_filename):
        df = pandas.read_pickle(vector_filename)
    else:
        df = get_vectors_df(filename, vector_length)
        # df.to_pickle(vector_filename)
    # blstm = BLSTM(df,name=base)

    # blstm.train()
    # blstm.test()
    # print(filename)

if __name__ == "__main__":
    main()
    print("文件预处理结束")