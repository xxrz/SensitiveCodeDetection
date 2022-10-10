import re
import numpy as np
import os


def graphRelation(path,pathdir,tag):
    nodeRelation = []
    nodeInformation = []

    with open(path, 'r', encoding='utf-8') as f:
        lines = str(f.read())
        # for line in lines:
        alllist = lines.split("),List(")
        # 判断函数是不是空
        node1 = []
        node2 = []
        relation = []
        if (alllist[1] != ""):
            filename = alllist[0].split("/")[-1]
            # 添加边
            nodeRelation.append(alllist[1])
            nodeRelation.append(alllist[3])
            nodeRelation.append(alllist[5])
            # 添加顶点
            nodeInformation.append(alllist[2])
            nodeInformation.append(alllist[4])
            nodeInformation.append(alllist[6])
            # 正则处理
            nodeRelation = re.findall(r"\(\d*,\d*,\d*\)", str(nodeRelation))
            nodeInformation = re.findall(r"\(\d*,.*?\)", str(nodeInformation))
            # 去除重复的顶点
            nodeInformation = list(set(nodeInformation))

            # 提取每一列内容成list=>批量处理
            nodeRelation = ' '.join(nodeRelation)
            b = re.findall('\d+', nodeRelation)
            for i in range(0, len(b), 3):
                node1.append(b[i])
                node2.append(b[i + 1])
                relation.append(b[i + 2])
            # relation_matrix = np.vstack([node1, node2, relation]).T

            nodes = []
            means = []
            for i in nodeInformation:
                node = re.search('\d+(?=,)', i)
                mean = re.search('(?<=,).*', i)
                nodes.append(node.group())
                means.append(mean.group())
            # feature_matrix = np.vstack([nodes,means]).T

            # 替换数字
            new_node1 = []
            new_node2 = []
            new_nodes = list(range(0, len(nodes)))
            for x in node1:
                for i in range(len(nodes)):
                    if x == nodes[i]:
                        # new_node1.append(x.replace(str(x), str(i)))
                        new_node1.append(str(i))
                        break
            for x in node2:
                for i in range(len(nodes)):
                    if x == nodes[i]:
                        # new_node2.append(x.replace(str(x), str(i)))
                        new_node2.append(str(i))
                        break

            # 拼接成矩阵
            relation_matrix = np.vstack([new_node1, new_node2, relation]).T
            # print(relation_matrix[0])
            feature_matrix = np.vstack([new_nodes, means]).T

            # 写入文件
            if os.path.exists(pathdir) == False:
                os.makedirs(pathdir)
            with open(pathdir + filename + ".txt", 'w', encoding='utf-8') as f1:
                for x, y, z in zip(new_node1, new_node2, relation):
                    # for x, y, z in zip(node1, node2, relation):
                    f1.write('(' + x + ',' + y + ',' + z + ')')
                    # f1.write(x + ',' + y + ',' + z)
                    f1.write("\n")
                f1.write("-----------------------------------")
                f1.write("\n")
                for x, y in zip(new_nodes, means):
                    # for x, y in zip(nodes, means):
                    f1.write('(' + str(x) + ',' + y)
                    # f1.write(str(x) + ',' + y)
                    f1.write(("\n"))
                f1.write('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                f1.write("\n")
                if tag == 'good':
                    f1.write('0')
                elif tag == 'bad':
                    f1.write('1')

path = r"1ayy22z4o35bohoss6kh.txt"
# path = r"13artvbfi2xt51f59pjz.txt"
pathdir = r"test/"
tag = 'bad'
graphRelation(path,pathdir,tag)