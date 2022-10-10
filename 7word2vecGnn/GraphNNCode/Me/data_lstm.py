"""
Interface to VulDeePecker project
"""
from keras import Sequential, Model
from keras.layers import Convolution2D
from numpy import random
import numpy
import sys
import os
import pandas
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
    with open(filename, "r", encoding="utf8") as file:
        gadget = []
        gadget_val = 0
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            # if "-" * 36 in line and gadget:
            if "-" * 15 in line and gadget:
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
    a = parse_file(filename)
    base = os.path.splitext(os.path.basename(filename))[0]
    vector_filename = base + "_gadget_vectors.pkl"
    vector_length = 100  # 代表多少维字典
    if os.path.exists(vector_filename):
        df = pandas.read_pickle(vector_filename)
    else:
        df = get_vectors_df(filename, vector_length)
    df.to_pickle(vector_filename)
    vectors = numpy.stack(df.iloc[:, 1].values)
    labels = df.iloc[:, 0].values
    positive_idxs = numpy.where(labels == 1)[0]
    negative_idxs = numpy.where(labels == 0)[0]
    #
    a = int(len(positive_idxs))

    undersampled_negative_idxs = random.choice(negative_idxs, a, replace=True)
    resampled_idxs = numpy.concatenate([positive_idxs, undersampled_negative_idxs])
    # data = vectors[resampled_idxs,]
    # label = labels[resampled_idxs]
    X_train, X_test, y_train, y_test = train_test_split(vectors[resampled_idxs,], labels[resampled_idxs],
                                                        test_size=0.2, stratify=labels[resampled_idxs])
    X_train, X_val, y_train, y_val = train_test_split(vectors[resampled_idxs,], labels[resampled_idxs],
                                                        test_size=0.2, stratify=labels[resampled_idxs])



    data_train=[]
    data_val = []
    data_test = []
    y_train_new=[]
    y_test_new=[]
    y_val_new=[]
    for i in y_train:
        if i==1:
            y_train_new.append([1, 0])
        elif i==0:
            y_train_new.append([0, 1])
    y_train_new=numpy.array(y_train_new)
    for i in y_test:
        if i==1:
            y_test_new.append([1, 0])
        elif i==0:
            y_test_new.append([0, 1])
    y_test_new=numpy.array(y_test_new)
    for i in y_val:
        if i==1:
            y_val_new.append([1, 0])
        elif i==0:
            y_val_new.append([0, 1])
    y_val_new=numpy.array(y_val_new)



    data_train.append(X_train)
    data_train.append(X_train)
    data_train.append(y_train_new)
    data_test.append(X_test)
    data_test.append(X_test)
    data_test.append(y_test_new)
    data_val.append(X_val)
    data_val.append(X_val)
    data_val.append(y_val_new)
    return data_train,data_val,data_test
# data('cwe119_cgd.txt')
# blstm = BLSTM(df,name=base)
# blstm.train()
# blstm.test()

# if __name__ == "__main__":
#     main()