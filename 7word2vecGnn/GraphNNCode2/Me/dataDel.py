
import os
import numpy as np

def data(path,length_want):
    # path = r".\trueg"  # file路径
    data_single1=[]
    data_single2 = []
    label_all=[]
    data_all=[]
    label_single=[]
    label_neg=[]
    label_pos=[]
    all_value=[]
    len_all=[]


    for root, dirs, files in os.walk(path):  # os.walk()返回一个三元组,
        for file in files:
            # try:
                length = 0
                label = 0
                value_all = 0

                with open(os.path.join(root + "/" + file), "r") as f:
                    for line in f.readlines():
                        # a=root.split("\\")
                        if line.strip() == "0" or line.strip() == "1":
                            label = line
                        else:
                            line = line.strip().split(",")

                            max = int(line[0]) if int(line[0]) > int(line[1]) else int(line[1])  # 获取file中数组的最大编号
                            length = int(max) if int(max) > length else length

                    s = [[0 for j in range(0, length + 1)] for i in range(0, length + 1)]
                    # a=np.array(s)
                    f.seek(0)
                    for line in f.readlines():
                        if not (line.strip() == "0" or line.strip() == '1'):
                            line = line.strip().split(",")
                            l = int(line[0])
                            m = int(line[1])
                            value = int(line[2])
                            value_all +=value

                            s[l][m] += value
                # print(s)  #矩阵
                # print(label)  # 标记
                ####矩阵维度统一
                len_all.append(len(s))
                all_value.append(value_all)
                s = np.array(s)


                if len(s) < length_want:
                    zero = np.zeros((len(s), length_want),dtype=np.int)  # 行处理统一到300

                    s = np.column_stack((s, zero))  # 拼到300列
                    s = s[:, 0:length_want]
                    zero = np.zeros((length_want, length_want))
                    s = np.row_stack((s, zero))
                    s = s[0:length_want, :]
                elif len(s) > length_want:
                    s = s[:length_want, :length_want]
                data_single1.append(np.array(s))
                data_single2.append(np.array(s))
                if int(label)==0:
                    label_single.append(np.array([int(label),1]))
                    label_neg.append(len(label_single))
                elif int(label)==1:
                    label_single.append(np.array([int(label), 0]))
                    label_pos.append(len(label_single))

                # data_all.append(data)
                # data=[]
                # print("a")


            # except :
            #     print(file)

    # 均值
    mean=np.mean(all_value)
    mean_len=np.mean(len_all)
    # 中位数
    mid=np.median(all_value)
    mid_len = np.median(len_all)
    # print(mean)
    # print(mid)
    # print(mean_len)
    # print(mid_len)
    data_all.append(np.array(data_single1))
    data_all.append(np.array(data_single2))
    data_all.append(np.array(label_single))
    label_all.append(label_neg)
    label_all.append(label_pos)
    # label_all=np.array(label_all)

    return data_all,label_all
# print(list(data_all[0]))
# print(data_all)
# f=open(r"C:\Users\Administrator\Desktop\me.txt","a+")
# f.write(str(list(data_all[0])))
# print("\n\n")
# print(label_all)

# print("over")
# a=data(r"F:\Github\CCS19\Curre\GNN\GraphNNCode\GraphNNCode\benchmark\ours\saveData\output\train\trueg",200)