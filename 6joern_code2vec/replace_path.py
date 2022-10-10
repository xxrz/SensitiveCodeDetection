import re
import pandas as pd
import time
import numpy as np

def replaceToken(waitingPath,mapPath,targetPath):
    try:
        with open(waitingPath, 'r', encoding='utf-8') as f1:
            lines = f1.readlines()
            for line in lines:
                node1 = []
                path = []
                node2 = []
                new_node1 = []
                new_node2 = []
                func = line.split(" ")[0]
                nums = re.findall('\d+', line)
                if len(nums) % 3 == 0:
                    # 转成列表
                    for i in range(0, len(nums), 3):
                        node1.append(nums[i])
                        path.append(nums[i + 1])
                        node2.append(nums[i + 2])
                    #读取映射关系
                    id = list(pd.read_csv(filepath_or_buffer=mapPath)["id"].values)
                    token = list(pd.read_csv(filepath_or_buffer=mapPath)["token"].values)
                    #添加到新token
                    for i in node1:
                        for j in range(len(id)):
                            if i == str(id[j]):
                                new_node1.append(token[j])
                                break
                    for i in node2:
                        for j in range(len(id)):
                            if i == str(id[j]):
                                new_node2.append(token[j])
                                break
                    #确保长度相同(如果不相同就会报错)
                    paths_matrix = np.vstack([new_node1, path, new_node2]).T
                    #写入文件
                    with open(targetPath, "a", encoding='utf-8') as f2:
                        f2.write(func + " ")
                        count = 0
                        for t1, p, t2 in zip(new_node1, path, new_node2):
                            if count!=len(path)-1:
                                f2.write(str(t1) + "," + str(p) + "," + str(t2) + " ")
                            else:
                                f2.write(str(t1) + "," + str(p) + "," + str(t2))
                            count += 1
                        f2.write("\n")
                else:
                    print("数目有问题")
    except Exception:
        print(Exception)

if __name__ == '__main__':
    # 待处理路径
    t1 = time.time()
    waitingPath = r"path_contexts.csv"
    # id-token的映射
    mapPath = r"tokens.csv"
    # 最后存放路径
    targetPath = r"path_contexts_new.csv"
    replaceToken(waitingPath,mapPath,targetPath)
    t2 = time.time()
    print(t2-t1)
