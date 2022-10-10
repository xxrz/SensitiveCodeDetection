

"""
Interface to VulDeePecker project
"""

import numpy as np
import sys
import os
import pandas
from keras import Sequential, Model
from keras.layers import Convolution2D, ConvLSTM2D
from sklearn.cross_validation import train_test_split
from clean_gadget import clean_gadget
from vectorize_gadget import GadgetVectorizer


"""
Parses gadget file to find individual gadgets
Yields each gadget as list of strings, where each element is code line
Has to ignore first line of each gadget, which starts as integer+space
At the end of each code gadget is binary value
    This indicates whether or not there is vulnerability in that gadget
"""

def parse_file(filename):
    gadget_graph_all=[]
    for root, dirs, files in os.walk(filename):  # os.walk()返回一个三元组,
        np.random.shuffle(files)
        for file in files:
            try:
                length = 0
                label = 0
                value_all = 0
                flag=0

                with open(os.path.join(root + "/" + file), "r") as f:
                    gadget = []
                    gadget_graph=[]
                    gadget_val = 0
                    for line in f.readlines():
                        if flag ==0:
                            stripped = line.strip()
                            if not stripped:
                                continue
                            if "*" * 10 in line and gadget:
                            # if "*" * 20 in line and gadget:
                                yield clean_gadget(gadget), gadget_val
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
                                        value_all += value
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

    np.save(r'F:\Github\CCS19\Curre\GNN\GraphNNCode\GraphNNCode\graph.npy', gadget_graph_all)






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
    for gadget, val in parse_file(filename):
        count += 1
        print("Collecting gadgets...", count, end="\r")
        len = vectorizer.add_gadget(gadget)
        row = {"gadget": gadget, "val": val}
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
        row = {"vector": vector, "val": gadget["val"]}
        vectors.append(row)
    print()
    df = pandas.DataFrame(vectors)
    return df


"""
Gets filename, either loads vector DataFrame if exists or creates it if not
Instantiate neural network, pass data to it, train, test, print accuracy
"""
def data(file_name):
    filename = file_name
    # if len(sys.argv) != 2:
    #     print("Usage: python vuldeepecker.py [filename]")
    #     exit()
    # filename = sys.argv[1]
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
    # data = vectors[resampled_idxs,]
    # label = labels[resampled_idxs]
    #reshape lstm
    # vector = vectors.reshape(1,vectors.shape[0],vectors.shape[1] , vectors.shape[2] ,1)  # TFchannel last theano (1,28,28)
    # labels = labels.astype('float32')
    #
    # model = Sequential()
    # input_shape = (vector.shape[1], vector.shape[2], vector.shape[3], 1)
    # model.add(ConvLSTM2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same', input_shape=input_shape,
    #                      return_sequences=True))
    #
    # intermediate_layer_model = Model(inputs=model.input,
    #                                  outputs=model.get_layer(index=0).output)
    # intermediate_output = intermediate_layer_model.predict(vector)  # a=model.layers[0].output
    # vectors=intermediate_output.reshape(vector.shape[1], vector.shape[2], vector.shape[3])

    vector = vectors.reshape(vectors.shape[0], vectors.shape[1], vectors.shape[2], 1)  # TFchannel last theano (1,28,28)
    labels = labels.astype('float32')

    model = Sequential()
    model.add(Convolution2D(
        filters=1,
        kernel_size=[1, 1],
        padding='same',
        input_shape=(vector.shape[1], vector.shape[2], 1)
    ))
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(index=0).output)
    intermediate_output = intermediate_layer_model.predict(vector)  # a=model.layers[0].output
    vectors = intermediate_output.reshape(vector.shape[0], vector.shape[1], vector.shape[2])
    #reshape graph
    graph = np.load(r'F:\Github\CCS19\Curre\GNN\GraphNNCode\GraphNNCode\graph.npy')
    #
    vector = graph.reshape(graph.shape[0], graph.shape[1], graph.shape[2],1)  # TFchannel last theano (1,28,28)

    labels = labels.astype('float32')

    model = Sequential()
    model.add(Convolution2D(
        filters=1,
        kernel_size=[1, 1],
        padding='same',
        input_shape=(vector.shape[1], vector.shape[2], 1)
    ))
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(index=0).output)
    intermediate_output = intermediate_layer_model.predict(vector)  # a=model.layers[0].output
    graph=intermediate_output.reshape(vector.shape[0], vector.shape[1], vector.shape[2])



    data_single=[]
    data_all=[]
    label=[]
    for i in labels.tolist():
        if i == 1:
            label.append([1,0])
        elif i == 0:
            label.append([0,1])
    data_all.append(vectors)
    data_all.append(graph)
    data_all.append(np.array(label))
    # for i in resampled_idxs:
    #     if flag==0:
    #         data_single=vectors[i].tolist()
    #         data_single.append(graph[i])
    #         if labels[i] == 1:
    #             data_single.append([1, 0])
    #         elif labels[i] == 0:
    #             data_single.append([0, 1])
    #         flag = 1
    #     elif flag == 1:
    #         data_single[1].append(vectors[i])
    #         data_single[2].append(graph[i])
    #         if labels[i] == 1:
    #             data_single[3].append([1, 0])
    #         elif labels[i] == 0:
    #             data_single[3].append([0, 1])

        # data_all.append(data_single)
        # data_single = []
    data_train=[]
    data_val=[]

    for i in range(3):
        data_single = data_all[i][:int(0.8 * len(data_all[0]))]
        data_train.append(data_single)
    for i in range(3):
        data_single = data_all[i][int(0.8 * len(data_all[0])):]
        data_val.append(data_single)

    return data_train,data_val


"""
Parses gadget file to find individual gadgets
Yields each gadget as list of strings, where each element is code line
Has to ignore first line of each gadget, which starts as integer+space
At the end of each code gadget is binary value
    This indicates whether or not there is vulnerability in that gadget
"""

def data_test(file_name):
    filename = file_name
    # if len(sys.argv) != 2:
    #     print("Usage: python vuldeepecker.py [filename]")
    #     exit()
    # filename = sys.argv[1]
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

    vector = vectors.reshape(vectors.shape[0], vectors.shape[1], vectors.shape[2], 1)  # TFchannel last theano (1,28,28)
    labels = labels.astype('float32')

    model = Sequential()
    model.add(Convolution2D(
        filters=1,
        kernel_size=[1, 1],
        padding='same',
        input_shape=(vector.shape[1], vector.shape[2], 1)
    ))
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(index=0).output)
    intermediate_output = intermediate_layer_model.predict(vector)  # a=model.layers[0].output
    vectors = intermediate_output.reshape(vector.shape[0], vector.shape[1], vector.shape[2])
    # reshape graph
    graph = np.load(r'F:\Github\CCS19\Curre\GNN\GraphNNCode\GraphNNCode\graph.npy')
    #
    vector = graph.reshape(graph.shape[0], graph.shape[1], graph.shape[2], 1)  # TFchannel last theano (1,28,28)

    labels = labels.astype('float32')

    model = Sequential()
    model.add(Convolution2D(
        filters=1,
        kernel_size=[1, 1],
        padding='same',
        input_shape=(vector.shape[1], vector.shape[2], 1)
    ))
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(index=0).output)
    intermediate_output = intermediate_layer_model.predict(vector)  # a=model.layers[0].output
    graph = intermediate_output.reshape(vector.shape[0], vector.shape[1], vector.shape[2])

    data_single = []
    data_all = []
    label = []
    for i in labels.tolist():
        if i == 1:
            label.append([1, 0])
        elif i == 0:
            label.append([0, 1])
    data_all.append(vectors)
    data_all.append(graph)
    data_all.append(np.array(label))

    data_test = []

    for i in range(3):
        data_single = data_all[i][:]
        data_test.append(data_single)

    return data_test
