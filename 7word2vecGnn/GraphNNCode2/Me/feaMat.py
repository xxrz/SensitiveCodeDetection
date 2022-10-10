import os
import random

import numpy as np
import json

# def createX(path):
#     file_path = path
#     if not os.path.isfile(file_path):
#         raise TypeError(file_path + " does not exist")
#
#     lines = []
#     label = False
#
#     # 找到属性字段下面一行，按照分号分割成属性字符串
#     with open(file_path, 'r') as f1:
#         for line in f1.readlines():
#             if(line.find("-----attribute-----")>=0):
#                 label = True
#                 continue
#             if (label):
#                 lines = line.split(';')
#
#     # 分割会出现空字符，剔除空字符
#     for x in lines:
#         if (x == ""):
#             lines.remove(x)
#     nodeNum = len(lines)
#     # print(lines)
#     # 打开映射表，将节点映射成向量，构建特征矩阵
#     with open("./our_map_all.txt", "r") as f:
#         s = f.read()
#         mapping = json.loads(json.dumps(eval(s)))
#     # 构建一个（节点数，100）的全零矩阵
#     X_input = np.zeros((nodeNum, 100))
#     # 为全零矩阵添加每一个节点的向量，构成特征矩阵
#     for u in range(nodeNum):
#         X_input[u, :] = np.array(mapping[lines[u]]);
#     # print(X_input.shape)
#     return X_input

def trans_adj(adj_who,nodeNum,adj_all):
    adj = np.zeros((nodeNum, nodeNum))
    flag_deal = 0
    for i in adj_who:
        for j in i:
            j = int(j)
            if flag_deal == 0:
                flag_deal = 1
                k = j
                continue
            else:
                adj[k-1][j-1] = 1
                adj_all[k-1][j-1]=1
                flag_deal = 0
    return adj,adj_all
def next_adj(adj_who,nodeNum,adj_all):
    adj = np.zeros((nodeNum, nodeNum))
    flag_deal=0
    for i in adj_who[0]:
        i=int(i)
        if flag_deal==0:
            flag_deal=1
            j=i
            continue
        else:
            adj[j-1][i-1]=1
            adj_all[j-1][i-1] = 1
            j = i
    return adj,adj_all

def createadj(path):
    file_path = path
    if not os.path.isfile(file_path):
        raise TypeError(file_path + " does not exist")

    lines = []


    label_label=False
    label_child=False
    label_from=False
    label_next=False
    label_by=False
    label_negation=False
    label_att=False
    label_use=False
    label_jump=False
    ### label_single 标签  adj_child： children边

    adj_child=[]
    adj_from=[]
    adj_next=[]
    adj_by=[]
    adj_negation=[]
    adj_att=[]
    adj_use=[]
    adj_jump=[]
    # 找到属性字段下面一行，按照分号分割成属性字符串
    with open(file_path, 'r') as f1:
        for line in f1.readlines():
            if (line.find("-----ast_node-----")>=0):
                break

            if (line.find("-----label-----") >= 0):
                label_label = True
                continue
            if (label_label):
                label_single = int(line.split('\n')[0].split(',')[0])
                label_label=False
                continue

            if (line.find("-----children-----") >= 0):
                label_child = True
                continue
            if (label_child):
                if line.find("-----nextToken-----") >= 0:
                    label_child=False
                else:
                    adj_child.append(line.split('\n')[0].split(','))

            if (line.find("-----nextToken-----") >= 0):
                label_next = True
                continue
            if (label_next):
                if line.find("-----computeFrom-----") >= 0:
                    label_next=False
                else:
                    adj_next.append(line.split('\n')[0].split(','))

            if (line.find("-----computeFrom-----") >= 0):
                label_from = True
                continue
            if (label_from):
                if line.find("-----guardedBy-----") >= 0:
                    label_from=False
                else:
                    adj_from.append(line.split('\n')[0].split(','))

            if (line.find("-----guardedBy-----") >= 0):
                label_by = True
                continue
            if (label_by):
                if line.find("-----guardedByNegation-----") >= 0:
                    label_by=False
                else:
                    adj_by.append(line.split('\n')[0].split(','))

            if (line.find("-----guardedByNegation-----") >= 0):
                label_negation = True
                continue
            if (label_negation):
                if line.find("-----lastLexicalUse-----") >= 0:
                    label_negation=False
                else:
                    adj_negation.append(line.split('\n')[0].split(','))

            if (line.find("-----lastLexicalUse-----") >= 0):
                label_use = True
                continue
            if (label_use):
                if line.find("-----jump-----") >= 0:
                    label_use=False
                else:
                    adj_use.append(line.split('\n')[0].split(','))

            if (line.find("-----jump-----") >= 0):
                label_jump = True
                continue
            if (label_jump):
                if line.find("-----attribute-----") >= 0:
                    label_jump=False
                else:
                    adj_jump.append(line.split('\n')[0].split(','))

            if (line.find("-----attribute-----") >= 0):
                label_att = True
                continue
            if (label_att):
                if line.find("-----attribute-----") >= 0:
                    label_att=False
                else:
                    num = line.split(';')
                    for x in num:
                        if (x == ""or x=='\n'):
                            num.remove(x)
                    nodeNum = len(num)
                    label_adj=False

        # #特征矩阵
        with open("F:\Github\CCS19\Curre\GNN\GraphNNCode\GraphNNCode\Me\our_map_all.txt", "r") as f:
            s = f.read()
            mapping = json.loads(json.dumps(eval(s)))
            # 构建一个（节点数，100）的全零矩阵
        adj_att = np.zeros((nodeNum, 100))
        # 为全零矩阵添加每一个节点的向量，构成特征矩阵
        for u in range(nodeNum):
            adj_att[u, :] = np.array(mapping[num[u]])
        # print(X_input.shape)

        #七条边各自的矩阵和邻接矩阵
        #邻接矩阵
        adj_all=np.zeros((nodeNum, nodeNum))
        #邻接矩阵求和
        adj_tem=np.zeros((nodeNum, nodeNum))
        if adj_child != []:
            adj_child,adj_all = trans_adj(adj_child,nodeNum,adj_all)
            adj_tem=adj_tem+adj_child
        if adj_from != []:
            adj_from,adj_all = trans_adj(adj_from, nodeNum,adj_all)
            adj_tem=adj_tem+adj_from
        if adj_negation != []:
            adj_negation,adj_all = trans_adj(adj_negation, nodeNum,adj_all)
            adj_tem=adj_tem+adj_negation
        if adj_use != []:
            adj_use,adj_all = trans_adj(adj_use, nodeNum,adj_all)
            adj_tem=adj_tem+adj_use
        if adj_by != []:
            adj_by,adj_all = trans_adj(adj_by, nodeNum,adj_all)
            adj_tem=adj_tem+adj_by
        if adj_jump != []:
            adj_jump,adj_all = trans_adj(adj_jump, nodeNum,adj_all)
            adj_tem=adj_tem+adj_jump
        if adj_next != []:
            adj_next,adj_all = next_adj(adj_next, nodeNum,adj_all)
            adj_tem=adj_tem+adj_next

        #特征矩阵
        adj_att=np.dot(adj_tem, adj_att)

        # #裁剪
        add=np.zeros((nodeNum,100))
        adj_att=np.hstack((adj_att,add))[:,0:100]
        adj_all = np.hstack((adj_all, add))[:, 0:100]
        add=np.zeros((100,100))
        adj_att = np.vstack((adj_att, add))[0:100, :]
        adj_all = np.vstack((adj_all, add))[0:100, :]

    return adj_att,adj_all,label_single
            # if (line.find("-----nextToken-----") >= 0):
            #     label_feature = True
            #     continue
            # if (label_feature):
            #     lines = line.split('\n')[0].split(',')
            #     label_feature=False
            #
            # if (line.find("-----attribute-----") >= 0):
            #     label_adj = True
            #     continue
            # if (label_adj):
            #     num = line.split(';')
            #     for x in num:
            #         if (x == ""):
            #             num.remove(x)
            #     nodeNum = len(num)
            #     label_adj=False
    # #特征矩阵
    # with open("F:\Github\CCS19\Curre\GNN\GraphNNCode\GraphNNCode\Me\our_map_all.txt", "r") as f:
    #     s = f.read()
    #     mapping = json.loads(json.dumps(eval(s)))
    #     # 构建一个（节点数，100）的全零矩阵
    # X_input = np.zeros((nodeNum, 100))
    # # 为全零矩阵添加每一个节点的向量，构成特征矩阵
    # for u in range(nodeNum):
    #     X_input[u, :] = np.array(mapping[num[u]]);
    # # print(X_input.shape)
    # #邻接矩阵
    # adj = np.zeros((nodeNum, nodeNum))
    # flag_deal=0
    # for i in lines:
    #     i=int(i)
    #     if flag_deal==0:
    #         flag_deal=1
    #         j=i
    #         continue
    #     else:
    #         adj[j][i]=1
    #         j = i
    # #裁剪
    # add=np.zeros((nodeNum,100))
    # X_input=np.hstack((X_input,add))[:,0:100]
    # adj = np.hstack((adj, add))[:, 0:100]
    # add=np.zeros((100,100))
    # X_input = np.vstack((add, X_input))[0:100, :]
    # adj = np.vstack((adj, X_input))[0:100, :]

    # return X_input,adj,label_single
# createadj(r'F:\Github\CCS19\Curre\GNN\GraphNNCode\GraphNNCode\New\featureMatrix\featureMatrix\out\test\0c296c8a-ccca-4940-8b49-fdc973c52f3e.txt')

# X_input,adj,label_single=createadj(r'F:\Github\CCS19\Curre\GNN\GraphNNCode\GraphNNCode\New\featureMatrix\featureMatrix\out\out\0bb2485a-3ddc-468e-ab36-59200b5605f4.txt')
# print("a")
def data(file_name):


    X_input=[]
    adj=[]
    label=[]
    labels=[]
    data_all=[]
    for root, dirs, files in os.walk(file_name):  # os.walk()返回一个三元组,
        np.random.shuffle(files)
    for file in files:
        try:
            str=os.path.join(root + "\\" + file)
            X_input_single, adj_single, label_single=createadj(str)

            X_input.append(X_input_single)
            adj.append(adj_single)
            label.append(label_single)
        except:
            print(file)

    positive_idxs = np.where(np.array(label) == 1)[0]
    negative_idxs = np.where(np.array(label) == 0)[0]
    #
    undersampled_negative_idxs = np.random.choice(negative_idxs, int(len(positive_idxs)), replace=True)
    resampled_idxs = np.concatenate([positive_idxs, undersampled_negative_idxs])
    random.shuffle(resampled_idxs)
    X_input = np.array(X_input)[resampled_idxs,]
    adj = np.array(adj)[resampled_idxs]
    label = np.array(label)[resampled_idxs]
    for i in label:
        if i == 1:
            labels.append([0, 1])
        elif i == 0:
            labels.append([1, 0])

    np.save(r'476_X_input.npy', X_input)
    np.save(r'476_adj.npy', adj)
    np.save(r'476_labels.npy', labels)

    # X_input=np.load("120_X_input.npy")
    # adj = np.load("120_adj.npy")
    # labels = np.load("120_labels.npy")

    data_all.append(np.array(X_input))
    data_all.append(np.array(adj))
    data_all.append(np.array(labels))
    # np.save("filename.npy", a)
    # b = np.load("filename.npy")

    data_train = []
    data_val = []
    data_test = []
    for i in range(3):
        # 测试集：0.8
        data_single = data_all[i][:int(0.8 * len(data_all[0]))]
        data_train.append(data_single)
    for i in range(3):
        # 验证集：0.8
        data_single = data_all[i][int(0.8 * len(data_all[0])):int(0.9 * len(data_all[0]))]
        data_val.append(data_single)
    for i in range(3):
        data_single = data_all[i][int(0.9 * len(data_all[0])):]
        data_test.append(data_single)

    return data_train,data_val,data_test
    # print("a")
# data(r'F:\Github\CCS19\Curre\GNN\GraphNNCode\GraphNNCode\New\featureMatrix\featureMatrix\out\test')
# def get_file_name(dir_path):
#     import os
#     name_numb = [] #相对数字  258
#     name_file = [] #绝对路径  dir/258.json
#     if os.path.isdir(dir_path): #isdir确定文件夹
#         for filename in os.listdir(dir_path):  # 获取当前路径下的文件名
#             # name_numb.append( ( os.path.splitext(filename)[0] ) )
#             name_numb.append(filename)
#             name_file.append(" ")
#         for i in range(len(name_numb)):
#             name_file[i] =dir_path + str(name_numb[i])
#     return name_file
# def main():
#     file_name = get_file_name("./out/")
#     print(len(file_name))
#     for name in file_name:
#         print(createX(name))
#
# if __name__ == "__main__":
#     main()