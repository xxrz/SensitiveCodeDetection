# 存放一些工具文件
import os
import random

import numpy as np
import json
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from keras.utils import to_categorical

# 设置类别标签，有bug的标注为[0,1]，修复后的标注为[1,0]

TRUE = [0, 1]
FALSE = [1, 0]


# BUG = 1
# CLEAN = -1

# 从指定文件夹下，获取所有文件名字
def get_file_name(data_dir):
    file_name = []
    file_true = []
    file_true_sard = []
    file_true_nvd = []
    file_false_sard = []
    file_false_nvd = []
    for childDir in os.listdir(data_dir):
        if ("trueg" not in childDir) and ("falseg" not in childDir):
        # if (childDir != "trueg") and (childDir != "falseg"):
            continue
        for file in os.listdir(os.path.join(data_dir, childDir)):
            if "trueg" in childDir:
                # file_true.append(os.path.join(os.path.join(data_dir, childDir), file))
                if file.startswith('CWE'):
                    file_true_sard.append(os.path.join(os.path.join(data_dir, childDir), file))
                else:
                    file_true_nvd.append(os.path.join(os.path.join(data_dir, childDir), file))
            elif "falseg" in childDir:
                if file.startswith('CWE'):
                    file_false_sard.append(os.path.join(os.path.join(data_dir, childDir), file))
                else:
                    file_false_nvd.append(os.path.join(os.path.join(data_dir, childDir), file))
            # file_name.append(os.path.join(os.path.join(data_dir, childDir), file))
    #min_graph_number = len(file_true)
    #if len(file_false_nvd) + len(file_false_sard) < min_graph_number:
    #    min_graph_number = len(file_false_sard) + len(file_false_nvd)
    #random.shuffle(file_true)
    random.shuffle(file_false_nvd)
    random.shuffle(file_false_sard)
    file_name.extend(file_true_sard)
#    file_name.extend(file_true_nvd)
#   file_name.extend(file_false_nvd[:len(file_true_nvd)])
    file_name.extend(file_false_sard[:len(file_true_sard)])
    return file_name


# 使用graph类存储文件中的Graph数据
class graph(object):
    def __init__(self, node_num=0, label=None, name=None):
        # 节点数目
        self.node_num = node_num
        # 标签
        self.label = label
        self.name = name
        # 特征矩阵
        self.features = []
        # 存储各节点的后继节点
        self.succs = []
        # 存储各节点的前趋节点
        self.preds = []
        # 按节点数目，对矩阵初始化
        if (node_num > 0):
            for i in range(node_num):
                self.features.append([])
                self.succs.append([])
                self.preds.append([])

    # 添加节点与此节点的特征向量
    def add_node(self, feature=[]):
        self.node_num += 1
        self.features.append(feature)
        self.succs.append([])
        self.preds.append([])

    # 添加两个节点之间的边，在后继/前趋矩阵中反映两节点的连接
    def add_edge(self, u, v):
        self.succs[u].append(v)
        self.preds[v].append(u)

    # graph对象的字符串表示
    def toString(self):
        ret = '{} {}\n'.format(self.node_num, self.label)
        for u in range(self.node_num):
            for fea in self.features[u]:
                ret += '{} '.format(fea)
            ret += str(len(self.succs[u]))
            for succ in self.succs[u]:
                ret += ' {}'.format(succ)
            ret += '\n'
        return ret


# 从存储Graph数据的文件中读取graph并返回graph对象列表与类别
def read_graph(F_NAME, Use_Self_Fea):
    graphs = []
    classes = [[], []]
    for f_name in F_NAME:
        with open(f_name) as inf:
            for line in inf:
                g_info = json.loads(line.strip())
                if "trueg" in f_name:
                    label = TRUE
                    classes[0].append(len(graphs))
                elif "falseg" in f_name:
                    label = FALSE
                    classes[1].append(len(graphs))
                cur_graph = graph(g_info['n_num'], label, f_name)
                for u in range(g_info['n_num']):
                    if Use_Self_Fea:
                        cur_graph.features[u] = np.array(g_info['featureDims'][u])
                    else:
                        cur_graph.features[u] = np.array(g_info['features'][u])
                    for v in g_info['succs'][u]:
                        cur_graph.add_edge(u, v)
                graphs.append(cur_graph)
    return graphs, classes
    # graphs 是图，有顶点数目、feature等；
    # classes 包括各个 label (-1, 1) 的graph在 graphs中的索引


# 将graph们分为三部分：训练集、验证集、测试集，
def split_data_my(Gs, classes):
    Cs = len(classes)
    min_graph_number = len(classes[0])
    if len(classes[1]) < min_graph_number:
        min_graph_number = len(classes[1])
    ret = []
    Gs_train = []
    classes_train = []
    Gs_dev = []
    classes_dev = []
    Gs_test = []
    classes_test = []

    for cls in range(Cs):
        cur_g = []
        cur_c = []
        graph_number = min_graph_number
        print(graph_number)
        stt = 0.0
        edt = stt + 0.8 * graph_number
        edd = edt + 0.1 * graph_number
        edte = edd + 0.1 * graph_number
        random.shuffle(classes[cls])
        for i in range(graph_number):
            gi = classes[cls][i]
            if i < edt:
                Gs_train.append(Gs[gi])
                classes_train.append([cls])
            elif i < edd:
                Gs_dev.append(Gs[gi])
                classes_dev.append([cls])
            elif i < edte:
                Gs_test.append(Gs[gi])
                classes_test.append([cls])
            cur_g.append(Gs[gi])
            cur_c.append([cls])
    random.shuffle(Gs_train)
    random.shuffle(Gs_dev)
    random.shuffle(Gs_test)
    ret.append(Gs_train)
    ret.append(classes_train)
    ret.append(Gs_dev)
    ret.append(classes_dev)
    ret.append(Gs_test)
    ret.append(classes_test)
    return ret


# eval时，取两类数据数目相等
def split_data_eval(Gs, classes):
    Cs = len(classes)
    min_graph_number = len(classes[0])
    if len(classes[1]) < min_graph_number:
        min_graph_number = len(classes[1])
    ret = []
    Gs_eval = []
    classes_eval = []
    for cls in range(Cs):
        cur_g = []
        cur_c = []
        graph_number = min_graph_number
        print(graph_number)
        random.shuffle(classes[cls])
        for i in range(graph_number):
            gi = classes[cls][i]
            if i < graph_number:
                Gs_eval.append(Gs[gi])
                classes_eval.append([cls])
            cur_g.append(Gs[gi])
            cur_c.append([cls])
    random.shuffle(Gs_eval)
    ret.append(Gs_eval)
    ret.append(classes_eval)
    return ret


# 对于给定的Graph数据集（Gs），按照Batch Size（M）进行分割
def generate_epoch_valid(Gs, M):
    epoch_data = []
    st = 0
    while st < len(Gs[0]):
        # 从数据集Gs中获得从索引位置st开始，长度为Batch Size的部分数据
        X =Gs[0][st:st+M,:]
        # mask= Gs[1][0,st:st+M]
        mask = Gs[1][st:st+M,:]
        y = Gs[2][st:st+M, :]
        # X, mask, y = get_pair_my(Gs, M, st=st)  # X 是特征矩阵，m 是邻接矩阵，y 是label
        epoch_data.append((X, mask, y))
        st += M
    return epoch_data


# 从数据集Gs中获得从索引位置st开始，长度为Batch Size（M）的部分数据
# 返回三个数组，每个数组长度都为Batch Size（M），分别存放特征矩阵、邻接矩阵、类别
def get_pair_my(Gs, M, st=-1, output_id=False, load_id=None):
    if load_id is None:
        if (st + M > len(Gs)):
            M = len(Gs) - st
        ed = st + M
        ids = []
        for g_id in range(st, ed):
            ids.append(g_id)
    else:
        ids = load_id[0]

    max_node_number = 0
    for id in ids:
        max_node_number = max(max_node_number, Gs[id].node_num)

    feature_dim = len(Gs[0].features[0])
    X_input = np.zeros((M, max_node_number, feature_dim))
    node_mask = np.zeros((M, max_node_number, max_node_number))
    y_input = np.zeros((M, 2))

    for i in range(M):
        graph = Gs[ids[i]]
        y_input[i] = graph.label
        # y_input[i] = graph.label
        for u in range(graph.node_num):
            X_input[i, u, :] = np.array(graph.features[u])
            for v in graph.succs[u]:
                node_mask[i, u, v] = 1

    if output_id:
        return X_input, node_mask, y_input, ids
    else:
        return X_input, node_mask, y_input


# 训练单个epoch，返回此epoch训练后的loss值
def train_epoch(model, batch_size, load_data=None):

    epoch_data = load_data
    # 对当前epoch要训练的数据，进行随机化
    perm = np.random.permutation(len(epoch_data))  # Random shuffle
    cum_loss = 0.0
    for index in perm:
        cur_data = epoch_data[index]
        X, m, y = cur_data
        # 每次都用Batch Size大小的数据训练模型，获取训练后的loss值
        loss = model.train(X, m, y)
        cum_loss += loss
    return cum_loss / len(perm)


# 计算模型的准确度
def get_auc_epoch_test(model, graphs, batch_size, load_data=None):
    if load_data is None:
        epoch_data = generate_epoch_valid(graphs, batch_size)
    else:
        epoch_data = load_data
    results = []
    labels = []
    for cur_data in epoch_data:
        X, m, y = cur_data
        result, label = model.get_result_array(X, m, y)
        # a_embed=model.get_embed(X, m)
        #y和a_embed
        # a_result_1=model.calc_result(X, m)
        results.extend(result)
        labels.extend(label)
    val_targ = to_categorical(labels, num_classes=2)
    val_predict = to_categorical(results, num_classes=2)
    _val_accuracy = accuracy_score(val_targ, val_predict)
    _val_f1 = f1_score(val_targ, val_predict, average=None)
    _val_recall = recall_score(val_targ, val_predict, average=None)
    _val_precision = precision_score(val_targ, val_predict, average=None)
    return _val_accuracy, _val_f1, _val_recall, _val_precision


def get_auc_epoch_my(model, graphs, batch_size, load_data=None):
    if load_data is None:
        epoch_data = generate_epoch_valid(graphs, batch_size)
    else:
        epoch_data = load_data
    all = 0.0
    for cur_data in epoch_data:
        X, m, y = cur_data
        accuracy = model.calc_accuracy(X, m, y)
        all += accuracy
    model_auc = all / len(epoch_data)
    return model_auc
