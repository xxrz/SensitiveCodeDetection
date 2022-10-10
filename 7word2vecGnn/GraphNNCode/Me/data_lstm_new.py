

"""
Interface to VulDeePecker project
"""

import numpy as np

import os
import pandas


from Me.clean_gadget import clean_gadget

from Me.vectorize_gadget import GadgetVectorizer

"""
Parses gadget file to find individual gadgets
Yields each gadget as list of strings, where each element is code line
Has to ignore first line of each gadget, which starts as integer+space
At the end of each code gadget is binary value
    This indicates whether or not there is vulnerability in that gadget
"""

def parse_file(filename):
    gadget_graph_all=[]
    # filenames = []
    for root, dirs, files in os.walk(filename):  # os.walk()返回一个三元组,
        np.random.shuffle(files)
        for file in files:
            try:
                length = 0
                label = 0
                # value_all = 0
                flag=0
                # filenames.append(file)
                with open(os.path.join(root + "/" + file), "r") as f:
                    gadget = []
                    gadget_graph=[]
                    gadget_val = 0
                    for line in f.readlines():
                        if flag ==0:
                            stripped = line.strip()
                            if not stripped:
                                continue
                            if "^" * 10 in line and gadget:
                            # if "*" * 20 in line and gadget:
                                yield clean_gadget(gadget), gadget_val, file
                                gadget = []
                                flag = 1
                            elif stripped.split()[0].isdigit():
                                if gadget:
                                    # Code line could start with number (somehow)
                                    if stripped.isdigit():
                                        gadget_val = int(stripped)
                                    else:
                                        gadget.append(stripped)
                            else:
                                gadget.append(stripped)
                        elif flag ==1:
                            stripped = line.strip()
                            if not stripped:
                                continue
                            # if "-" * 36 in line and gadget:
                            # if "-" * 20 in line and gadget_graph:
                            if  (line.strip() == "0" or line.strip() == '1'):
                                for line in gadget_graph:
                                    # a=root.split("\\")
                                    line = line.strip().split(",")

                                    max = int(line[0]) if int(line[0]) > int(line[1]) else int(
                                        line[1])  # 获取file中数组的最大编号
                                    length = int(max) if int(max) > length else length

                                s = [[0 for j in range(0, length + 1)] for i in range(0, length + 1)]
                                # a=np.array(s)
                                f.seek(0)
                                for line in gadget_graph:
                                    if not (line.strip() == "0" or line.strip() == '1'):
                                        line = line.strip().split(",")
                                        l = int(line[0])
                                        m = int(line[1])
                                        value = int(line[2])
                                        if value ==3:
                                            value=4
                                        s[l][m] += value
                                s = np.array(s)
                                if len(s) < 100:
                                    zero = np.zeros((len(s), 100), dtype=np.int)  # 行处理统一到300

                                    s = np.column_stack((s, zero))  # 拼到300列
                                    s = s[:, 0:100]
                                    zero = np.zeros((100, 100))
                                    s = np.row_stack((s, zero))
                                    s = s[0:100, :]
                                elif len(s) > 100:
                                    s = s[:100, :100]
                                gadget_graph_all.append(np.array(s))
                                gadget_graph = []
                                flag = 0
                            elif stripped.split()[0].isdigit():
                                if gadget_graph:
                                    # Code line could start with number (somehow)
                                    if stripped.isdigit():
                                        gadget_val = int(stripped)
                                    else:
                                        gadget_graph.append(stripped)
                            else:
                                gadget_graph.append(stripped)
            except:
                print(file)
                os.remove(filename+'/'+file)
                continue
    np.save(os.path.splitext(os.path.basename(filename))[0]+r'_CWE-191_graph.npy', gadget_graph_all)






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


def get_vectors_df(filename, vector_length):
    len_list = []
    gadgets = []
    count = 0
    num_100 = 0
    num_200 = 0
    num_300 = 0
    num_400 = 0
    num_500 = 0
    num_max = 0
    vectorizer = GadgetVectorizer(vector_length)
    for gadget, val, file in parse_file(filename):
        count += 1
        print("Collecting gadgets...", count, end="\r")
        len = vectorizer.add_gadget(gadget)
        # row = {"gadget": gadget, "val": val}
        row = {"gadget": gadget, "val": val, "filename":file}
        gadgets.append(row)
        len_list.append(len)
    for i in len_list:
        if i < 100:
            num_100 = num_100 + 1
        elif i > 100 & i < 200:
            num_200 = num_200 + 1
        elif i > 200 & i < 300:
            num_300 = num_300 + 1
        elif i > 300 & i < 400:
            num_400 = num_400 + 1
        elif i > 400 & i < 500:
            num_500 = num_500 + 1
        elif i > 500:
            num_max = num_max + 1
    print("小于100：", num_100)
    print("小于200：", num_200)
    print("小于300：", num_300)
    print("小于400：", num_400)
    print("小于500：", num_500)
    print("大于500：", num_max)
    print('Found {} forward slices and {} backward slices'
          .format(vectorizer.forward_slices, vectorizer.backward_slices))
    print()
    print("Training model...", end="\r")
    vectorizer.train_model()
    print()
    vectors = []
    count = 0
    for gadget in gadgets:
        count += 1
        print("Processing gadgets...", count, end="\r")
        vector = vectorizer.vectorize(gadget["gadget"])
        # row = {"vector": vector, "val": gadget["val"]}
        row = {"vector": vector, "val": gadget["val"], "filename":gadget["filename"]}
        vectors.append(row)
    # print()
    df = pandas.DataFrame(vectors,columns=["val", "vector", "filename"])
    return df


"""
Gets filename, either loads vector DataFrame if exists or creates it if not
Instantiate neural network, pass data to it, train, test, print accuracy
"""
def data(file_name):
    filename = file_name
    parse_file(filename)

    base = os.path.splitext(os.path.basename(filename))[0]
    vector_filename = base + "_gadget_vectors.pkl"
    vector_length = 100  # 代表多少维字典
    if os.path.exists(vector_filename):
        df = pandas.read_pickle(vector_filename)
    else:
        df = get_vectors_df(filename, vector_length)
    df.to_pickle(vector_filename)
    vectors = np.stack(df.iloc[:, 1].values)

    labels = df.iloc[:, 0].values
    positive_idxs = np.where(labels == 1)[0]
    negative_idxs = np.where(labels == 0)[0]
    #
    undersampled_negative_idxs = np.random.choice(negative_idxs, int(len(positive_idxs)), replace=True)
    resampled_idxs = np.concatenate([positive_idxs, undersampled_negative_idxs])
    # vectors = vectors[resampled_idxs,]
    # labels = labels[resampled_idxs]
    #reshape graph
    graph = np.load((os.path.basename(filename))+r'_CWE-191_graph.npy')
    data_all=[]
    label=[]
    for i in labels.tolist():
        if i == 1:
            label.append([0,1])
        elif i == 0:
            label.append([1,0])
    data_all.append(vectors)
    data_all.append(graph)
    data_all.append(np.array(label))

    data_train=[]
    data_val=[]
    data_test=[]
    for i in range(3):
        #测试集：0.8
        data_single = data_all[i][:int(0.8 * len(data_all[0]))]
        data_train.append(data_single)
    for i in range(3):
        #验证集：0.8
        data_single = data_all[i][int(0.8 * len(data_all[0])):int(0.9 * len(data_all[0]))]
        data_val.append(data_single)
    for i in range(3):
        data_single = data_all[i][int(0.9 * len(data_all[0])):]
        data_test.append(data_single)

    # os.remove(vector_filename)
    # 删除每次产生的图
    # os.remove("CWE-191_graph.npy")

    return data_train,data_val,data_test


def data_train(file_name):
    filename = file_name
    parse_file(filename)

    base = os.path.splitext(os.path.basename(filename))[0]
    vector_filename = base + "_gadget_vectors.pkl"
    vector_length = 100  # 代表多少维字典
    if os.path.exists(vector_filename):
        df = pandas.read_pickle(vector_filename)
    else:
        df = get_vectors_df(filename, vector_length)
    df.to_pickle(vector_filename)
    vectors = np.stack(df.iloc[:, 1].values)

    labels = df.iloc[:, 0].values
    positive_idxs = np.where(labels == 1)[0]
    negative_idxs = np.where(labels == 0)[0]
    #
    undersampled_negative_idxs = np.random.choice(negative_idxs, int(len(positive_idxs)), replace=True)
    resampled_idxs = np.concatenate([positive_idxs, undersampled_negative_idxs])
    # vectors = vectors[resampled_idxs,]
    # labels = labels[resampled_idxs]
    #reshape graph
    graph = np.load(base+r'_CWE-191_graph.npy')
    data_all=[]
    label=[]
    for i in labels.tolist():
        if i == 1:
            label.append([0,1])
        elif i == 0:
            label.append([1,0])
    data_all.append(vectors)
    data_all.append(graph)
    data_all.append(np.array(label))

    data_train=[]
    data_val=[]
    data_test=[]
    for i in range(3):
        data_single = data_all[i][:int(0.8 * len(data_all[0]))]
        data_train.append(data_single)
    for i in range(3):
        data_single = data_all[i][int(0.8 * len(data_all[0])):]
        data_val.append(data_single)

    #删除产生的向量
    # os.remove(vector_filename)

    return data_train,data_val

def data_test(file_name):
    filename = file_name
    parse_file(filename)

    base = os.path.splitext(os.path.basename(filename))[0]
    vector_filename = base + "_gadget_vectors.pkl"
    vector_length = 100  # 代表多少维字典
    if os.path.exists(vector_filename):
        df = pandas.read_pickle(vector_filename)
    else:
        df = get_vectors_df(filename, vector_length)
    df.to_pickle(vector_filename)
    vectors = np.stack(df.iloc[:, 1].values)

    labels = df.iloc[:, 0].values
    positive_idxs = np.where(labels == 1)[0]
    negative_idxs = np.where(labels == 0)[0]
    #
    undersampled_negative_idxs = np.random.choice(negative_idxs, int(len(positive_idxs)), replace=True)
    resampled_idxs = np.concatenate([positive_idxs, undersampled_negative_idxs])
    vectors = vectors[resampled_idxs,]
    labels = labels[resampled_idxs]
    #reshape graph
    graph = np.load(base+r'_CWE-191_graph.npy')
    data_all=[]
    label=[]
    for i in labels.tolist():
        if i == 1:
            label.append([0,1])
        elif i == 0:
            label.append([1,0])
    data_all.append(vectors)
    data_all.append(graph[resampled_idxs])
    data_all.append(np.array(label))

    data_train=[]
    data_val=[]
    data_test=[]
    for i in range(3):
        data_single = data_all[i][:]
        data_test.append(data_single)

    # os.remove(vector_filename)
    # 删除每次产生的图
    # os.remove("CWE-191_graph.npy")

    return data_test

def data_select(file_name):
    filename = file_name
    parse_file(filename)

    base = os.path.splitext(os.path.basename(filename))[0]
    vector_filename = base + "_gadget_vectors_select.pkl"
    vector_length = 100  # 代表多少维字典
    if os.path.exists(vector_filename):
        df = pandas.read_pickle(vector_filename)
    else:
        df = get_vectors_df(filename, vector_length)
    df.to_pickle(vector_filename)
    vectors = np.stack(df.iloc[:, 1].values)
    filenames = df.iloc[:, 2].values
    filenames=np.array(filenames)
    labels = df.iloc[:, 0].values
    positive_idxs = np.where(labels == 1)[0]
    negative_idxs = np.where(labels == 0)[0]
    #
    undersampled_negative_idxs = np.random.choice(negative_idxs, int(len(positive_idxs)), replace=True)
    resampled_idxs = np.concatenate([positive_idxs, undersampled_negative_idxs])
    # vectors = vectors[resampled_idxs,]
    # labels = labels[resampled_idxs]
    #reshape graph
    graph = np.load(base+r'_CWE-191_graph_select.npy')
    data_all=[]
    label=[]
    for i in labels.tolist():
        if i == 1:
            label.append([1])
        elif i == 0:
            label.append([0])
    data_all.append(vectors)
    data_all.append(graph)
    data_all.append(np.array(label))
    data_all.append(filenames)
    data_train=[]
    data_val=[]
    data_test=[]
    for i in range(4):
        data_single = data_all[i][:]
        data_test.append(data_single)

    # os.remove(vector_filename)
    # 删除每次产生的图
    # os.remove("CWE-191_graph.npy")

    return data_test
# data(r'F:\Github\CCS19\Curre\GNN\GraphNNCode\GraphNNCode\benchmark\detect0')
# blstm = BLSTM(df,name=base)
# blstm.train()
# blstm.test()

# if __name__ == "__main__":
#     main()