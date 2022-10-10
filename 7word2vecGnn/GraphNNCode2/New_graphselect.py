# 训练图网络模型
import tensorflow as tf
# print(tf.__version__)
import numpy as np
from datetime import datetime
from graphnnSiamese_my_graphs import graphnn
from utils_my_graphs_select import *
# from utils_my_graphs_cpp_tvt_br import *
# from utils_my_graphs import *
import os
import argparse
import json
import LogUtil
from Me import data_lstm
from Me import data_lstm_newa
import os

import os


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


# 定义要创建的目录
mkpath = "d:\\qttc\\web\\"
# 调用函数
mkdir(mkpath)

# 超参数设置
# 节点特征维度
Feature_Dim = 100
# Feature_Dim = 100
# 学习率
# Learning_Rate = 1e-2
Learning_Rate = 1e-3
# 嵌入深度
# Embed_Depth = 2
Embed_Depth = 16
# Embed_Depth = 64
# 输出向量维度
Output_Dim = 2
# Output_Dim = md
Batch_Size = 128
# Batch_Size = 5
# EPOCH = 50000
#记得改回来，是50
EPOCH = 50
CAT = "ours/saveData"
# DATA_DIR = '/media/common/88C4F9BDC4F9AE16/kwx/cgtvt/'
# DATA_DIR = '/home/common/kwx/C/cg/'
DATA_DIR = '../GraphNNCode/benchmark/'
LOG_DIR = "logs/"
LOG_FILE = "LOG_ctest2_" + str(random.randint(0, 100)) + str(random.randint(0, 100)) + str(random.randint(0, 100)) + str(random.randint(0, 100)) + ".txt"
USE_SELF_FEA = True
# 在命令行界面运行时，可以额外指定超参数设置，不设置则使用默认超参数
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0',
        help='visible gpu device')
parser.add_argument('--fea_dim', type=int, default=Feature_Dim,
        help='feature dimension')
parser.add_argument('--embed_dim', type=int, default=64,
        help='embedding dimension')
parser.add_argument('--embed_depth', type=int, default=Embed_Depth,
        help='embedding network depth')
parser.add_argument('--output_dim', type=int, default=Output_Dim,
        help='output layer dimension')
parser.add_argument('--iter_level', type=int, default=5,
        help='iteration times')
parser.add_argument('--lr', type=float, default=Learning_Rate,
        help='learning rate')
parser.add_argument('--epoch', type=int, default=EPOCH,
        help='epoch number')
parser.add_argument('--batch_size', type=int, default=Batch_Size,
        help='batch size')
parser.add_argument('--load_path', type=str,
                    default=None,
                    # default='./saved_model/graphnn-model_best',
        help='path for model loading, "#LATEST#" for the latest checkpoint')
parser.add_argument('--save_path', type=str,
        default='./saved_model/' + LOG_FILE, help='path for model saving')
        # default='./saved_model/graphnn-model', help='path for model saving')
parser.add_argument('--log_path', type=str, default=None,
        help='path for training log')
parser.add_argument('--log', type=str, default=LOG_FILE,
        help='path for training log')
parser.add_argument('--dataset', type=str, default=DATA_DIR,
        help='path for training log')
parser.add_argument('--cat', type=str, default=CAT,
        help='Web Vul Category')
parser.add_argument('--self_fea', type=bool, default=USE_SELF_FEA,
                    help='Use ourselves feature')


if __name__ == '__main__':
    # 解析命令行参数
    args = parser.parse_args()
    args.dtype = tf.float32

    # 日志文件设置
    ID = args.log + "_" + str(random.randint(0, 100)) + str(random.randint(0, 100)) + str(random.randint(0, 100)) + str(random.randint(0, 100)) + ".txt"
    ID = "_cgtvt_" + ID
    Log_file = LOG_DIR + ID
    logger = LogUtil.LogConfig(Log_file).getLog()

    logger.info("=================================")
    logger.info("\nargs = %s\n", args)
    logger.info("\n=================================")

    # 将超参数赋值到下列变量中
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    Dtype = args.dtype  # tf.float32
    NODE_FEATURE_DIM = args.fea_dim  # 7
    EMBED_DIM = args.embed_dim  # 64
    EMBED_DEPTH = args.embed_depth  # 2
    OUTPUT_DIM = args.output_dim  # 2
    ITERATION_LEVEL = args.iter_level  # 5
    LEARNING_RATE = args.lr  # 1e-4
    MAX_EPOCH = args.epoch  # 100
    BATCH_SIZE = args.batch_size  # 5
    LOAD_PATH = args.load_path  # None
    SAVE_PATH = './saved_model/' + ID  # './saved_model/graphnn-model'
    #SAVE_PATH = args.save_path  # './saved_model/graphnn-model'
    LOG_PATH = args.log_path  # None
    # Graph数据集存在的目录
    CAT = args.cat
    DATA_DIR = args.dataset + "ours/saveData/train/"
    val_DIR = args.dataset + "ours/saveData/val" + "/"
    test_DIR = args.dataset + "ours/saveData/test" + "/"
    Use_Self_Fea = args.self_fea
    # 打印输出结果的频率
    SHOW_FREQ = 1
    # 测试模型的频率
    TEST_FREQ = 1
    # 保存模型的频率
    SAVE_FREQ = 10
    # 从目录中读取Graph文件名称
    logger.info("get file name")

    logger.info("read graph")

    #分开划分训练集验证集+测试集
    train_str=r'F:\Github\CCS19\Curre\GNN\GraphNNCode\GraphNNCode\GNN-Data\data\CWE-191\sard\merge'
    test_str=r'F:\Github\CCS19\Curre\GNN\GraphNNCode\GraphNNCode\GNN-Data\data\CWE-191\github\分类\备份数据\all'
    dist_bad_str=r"F:\Github\CCS19\Curre\GNN\GraphNNCode\GraphNNCode\GNN-Data\data\CWE-191\github\分类\备份数据\select_bad"
    dist_good_str = r"F:\Github\CCS19\Curre\GNN\GraphNNCode\GraphNNCode\GNN-Data\data\CWE-191\github\分类\备份数据\select_good"
    mkdir(dist_bad_str)
    mkdir(dist_good_str)
    Gs_train, Gs_valid = data_lstm_newa.data_train(train_str)
    Gs_test = data_lstm_newa.data_test(test_str)
    print("over")

    gnn = graphnn(
            NODE_FEATURE_DIM=NODE_FEATURE_DIM,
            Dtype=Dtype,
            N_embed=EMBED_DIM,
            depth_embed=EMBED_DEPTH,
            N_output=OUTPUT_DIM,
            ITER_LEVEL=ITERATION_LEVEL,
            lr=LEARNING_RATE)
    gnn.init(LOAD_PATH, LOG_PATH)
    # 模型的最佳准确度
    best_f1 = 0
    # 从验证集中生成验证每个epoch所需的数据
    # valid_epoch_data=Gs_train
    valid_epoch_data = generate_epoch_valid(Gs_valid, BATCH_SIZE)
    # 从验证集中生成训练每个epoch所需的数据
    train_epoch_data=generate_epoch_valid(Gs_train, BATCH_SIZE)
    best_f1=0
    best_auc=0
    best_recall=0
    best_precision=0
    # train_epoch_data = generate_epoch_valid(Gs_train, BATCH_SIZE)   # 是一个数组，每个element都是一个元组 (Graph, class)
    # 迭代MAX_EPOCH此的训练、验证过程
    select_num=0
    for i in range(1, MAX_EPOCH+1):
        # 训练epoch，获取此epoch训练后的loss值
        loss = train_epoch(gnn, BATCH_SIZE, train_epoch_data)
        # logger.info("EPOCH %d/%d, loss = %s @ %s", i, MAX_EPOCH, loss, datetime.now())
        print("EPOCH ",i,"/",MAX_EPOCH, "loss =",loss,"Time =",datetime.now())
        # 测试此epoch后的模型
        if i % TEST_FREQ == 0:
            # 获得在训练集上的准确率
            train_accuracy = get_auc_epoch_my(gnn, Gs_train, BATCH_SIZE, load_data=train_epoch_data)
            print("\nTrain model: accuracy =", train_accuracy)

            # 获得在验证集上的准确率===============================================================================
            _val_accuracy, _val_f1, _val_recall, _val_precision = \
                get_auc_epoch_test(gnn, Gs_valid, BATCH_SIZE, test_str,dist_bad_str,dist_good_str,
                                   load_data=valid_epoch_data,valortest = False)
            #====================================================================================================

            print("\nValid model:")
            print("_val_accuracy ：", _val_accuracy)


            valid_accuracy = _val_accuracy
            valid_f1 = _val_f1[1]
            valid_recall = _val_recall[1]
            valid_precision = _val_precision[1]


            if  valid_precision!=0 and valid_recall!=0 and valid_accuracy >= 0.4 and select_num<1:
                select_num=select_num+1
                path = gnn.save(SAVE_PATH+'_best_'+ str(i))

                test_epoch_data = generate_epoch_valid(Gs_test, BATCH_SIZE)
                _val_accuracy, _val_f1, _val_recall, _val_precision = \
                    get_auc_epoch_test(gnn, Gs_test, BATCH_SIZE, test_str,dist_bad_str,dist_good_str
                                       ,load_data=test_epoch_data,valortest = True)
                # logger.info("\nTest model:")
                # logger.info("\n_Test_accuracy ：=%s\n_Test_f1       ：=%s\n_Test_recall   : =%s\n_Test_precision: =%s",
                #             _val_accuracy, _val_f1, _val_recall, _val_precision)
                # print("\nTest model:")
            if select_num>1:
                print("overrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
                break
            # logger.info("Model saved in %s", path)
    print("*********************************************************************")
    if select_num==0:
         print("失败了，需要调节 valid_accuracy 小一点")
    print("*********************************************************************")
