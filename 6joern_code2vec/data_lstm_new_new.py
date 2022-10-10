import random
import warnings

import gensim
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

warnings.filterwarnings("ignore")
from gensim.models import word2vec
from gensim.models import Word2Vec
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
import re
import os
import numpy as np
import sys
import os
import pandas
from keras import Sequential, Model

from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import VarianceThreshold

from Me.clean_gadget import clean_gadget
from Me import vectorize_gadget
from Me.vectorize_gadget import GadgetVectorizer

def w2v(filename):
    words = []
    # filenames = []
    for root, dirs, files in os.walk(filename):  # os.walk()返回一个三元组,
        np.random.shuffle(files)
        for file in files:
            flag_vec=0
            word=[]
            with open(os.path.join(root + "/" + file), "r") as f:
                for line in f.readlines():
                    if "-" * 10 in line:
                        flag_vec=1

                    if flag_vec == 1:#处理特征矩阵
                        if "^" * 10 in line:
                            for i in word:
                                words.append(i)
                            # continue
                        if re.search('(?<=,).*', line):
                            # print(a.group().split(')')[0])
                            # 去除空的情况
                            # if a.group().split(')')[0] != '':
                            word.append(re.search('(?<=,).*', line).group().split(')')[0].split(" "))#将token加入数组
    model = Word2Vec(words, min_count=1, size=100, sg=1, window=5,
                            negative=3, sample=0.001, hs=1, workers=4)

    # a=model.wv.vocab
    model.save("./" + "word2vec_test_191.pkl")
    print("训练好字典了")

# w2v(r'F:\Github\CCS19\Curre\GNN\GraphNNCode\GraphNNCode\New\result\result\test')
# print("a")

def parse_file(filename):
    gadget_graph_all=[]
    # filenames = []
    for root, dirs, files in os.walk(filename):  # os.walk()返回一个三元组,
        np.random.shuffle(files)
        adj=[]#邻接矩阵
        label=[]#标签
        feature=[]#特征矩阵
        all_graph=[]
        k = 0
        kk = 0
        for file in files:

            gadget_graph = [] #图
            adj_single=np.zeros([100,100])
            flag_vec=0
            flag_label=0
            word=[]

            with open(os.path.join(root + "/" + file), "r") as f:
                for line in f.readlines():

                    if flag_vec==0:
                        if "-" * 10 in line:
                            adj.append(adj_single)
                            flag_vec=1

                        else:
                            threedot = line.split('\n')[0].split('(')[1].split(')')[0].split(',') #分割成三元组
                            try:
                                if int(threedot[2])==0:#AST
                                    adj_single[int(threedot[0])-1,int(threedot[1])-1]=1
                                    continue
                                if int(threedot[2])==1:#CFG
                                    adj_single[int(threedot[0])-1,int(threedot[1])-1]=2
                                    continue
                                if int(threedot[2])==2:#PDG
                                    adj_single[int(threedot[0])-1,int(threedot[1])-1]=3
                                    continue
                            except:
                                continue #大于100直接跳过

                    if flag_vec == 1:#处理特征矩阵
                        if "^" * 10 in line:
                            flag_label=1
                            vectors = np.zeros((100, 100))
                            #
                            model = word2vec.Word2Vec.load(r'word2vec_test_191.pkl')
                            # a = model.wv.vocab
                            for i in range(min(len(word), 100)):
                                vector=[]
                                word_single=word[i].split(" ")
                                # for j in word_single:
                                try:
                                    vector=model.wv[word_single]#查词
                                except:
                                    # vector.append(np.random.rand(100))
                                    print(i)
                                    continue
                                vector = np.array(vector)
                                if np.size(vector) > 100:  # 降维
                                    # vector=np.array(vector).sum(axis=0)
                                    vector = vector.T
                                    gauss_proj = GaussianRandomProjection(n_components=1)
                                    vector = gauss_proj.fit_transform(vector).T
                                vector=np.array(vector)
                                vectors[i]=vector
                            feature.append(vectors)

                            # continue
                        if re.search('(?<=,).*', line):
                            # print(a.group().split(')')[0])
                            # 去除空的情况
                            # if a.group().split(')')[0] != '':
                            word.append(re.search('(?<=,).*', line).group().split(')')[0])#将token加入数组
                        if line.split()[0].isdigit():
                                label.append(int(line))
            # print("一个文件搞完了")

        # all_graph.append(np.array(feature))
        # all_graph.append(np.array(adj))
        # all_graph.append(label)
        feature=np.array(feature)
        adj=np.array(adj)
        np.save(os.path.splitext(os.path.basename(filename))[0]+r'_feature.npy', feature)
        np.save(os.path.splitext(os.path.basename(filename))[0] + r'_adj.npy', adj)
        np.save(os.path.splitext(os.path.basename(filename))[0] + r'_label.npy', label)
        # print("a")
        return  feature,adj,label

def data(file_name):
    filename = file_name
    base = os.path.splitext(os.path.basename(filename))[0]
    #如果没有字典就运行w2v,有就注释
    # w2v(filename)
    vector_filename = base + "_feature.npy"
    if os.path.exists(vector_filename):
        df_feature=np.load(base + "_feature.npy",allow_pickle = True)
        df_adj = np.load(base + "_adj.npy", allow_pickle=True)
        df_label = np.load(base + "_label.npy", allow_pickle=True)
        feature = df_feature
        adj = df_adj
        label = df_label
    else:
        feature, adj, label =parse_file(file_name)


    label=np.array(label)
    positive_idxs = np.where(label == 1)[0]
    negative_idxs = np.where(label == 0)[0]
    #

    undersampled_negative_idxs = np.random.choice(negative_idxs, int(len(positive_idxs)), replace=True)
    resampled_idxs = np.concatenate([positive_idxs, undersampled_negative_idxs])
    random.shuffle(resampled_idxs)
    feature = feature[resampled_idxs]
    adj = adj[resampled_idxs]
    label = label[resampled_idxs]
    data_all = []
    labels = []
    for i in label.tolist():
        if i == 1:
            labels.append([0,1])
        elif i == 0:
            labels.append([1,0])
    data_all.append(feature)
    data_all.append(adj)
    data_all.append(np.array(labels))
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

    return data_train,data_val,data_test

data(r'result/test')