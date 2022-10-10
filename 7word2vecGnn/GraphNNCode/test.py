# coding:utf-8
from datetime import datetime
from graphnnSiamese_my_graphs import graphnn
from utils_my_graphs_newer import *
import argparse
import LogUtil
import data_lstm_new_new

Feature_Dim = 100

Learning_Rate = 1e-3
Embed_Depth = 3
Output_Dim = 2
Batch_Size = 128
EPOCH = 2
CAT = "ours/saveData"
DATA_DIR = '../GraphNNCode/benchmark/'
LOG_DIR = "logs/"
LOG_FILE = "LOG_ctest2_" + str(random.randint(0, 100)) + str(random.randint(0, 100)) + str(random.randint(0, 100)) + str(random.randint(0, 100)) + ".txt"
USE_SELF_FEA = True
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
parser.add_argument('--iter_level', type=int, default=6,
        help='iteration times')
parser.add_argument('--lr', type=float, default=Learning_Rate,
        help='learning rate')
parser.add_argument('--epoch', type=int, default=EPOCH,
        help='epoch number')
parser.add_argument('--batch_size', type=int, default=Batch_Size,
        help='batch size')
parser.add_argument('--load_path', type=str,
                    default=None,
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
    args = parser.parse_args()
    args.dtype = tf.float32
    ID = args.log + "_" + str(random.randint(0, 100)) + str(random.randint(0, 100)) + str(random.randint(0, 100)) + str(random.randint(0, 100)) + ".txt"
    ID = "_cgtvt_" + ID
    Log_file = LOG_DIR + ID
    logger = LogUtil.LogConfig(Log_file).getLog()

    logger.info("=================================")
    logger.info("\nargs = %s\n", args)
    logger.info("\n=================================")

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
    LOG_PATH = args.log_path  # None
    SHOW_FREQ = 1
    TEST_FREQ = 1
    SAVE_FREQ = 10
    logger.info("get file name")
    logger.info("read graph")
    Gs_train, Gs_valid, Gs_test = data_lstm_new_new.data(r'D:\XRZ\Data\joern\github\SensitiveOperations\result\auth\auth')
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

    valid_epoch_data = generate_epoch_valid(Gs_valid, BATCH_SIZE)

    train_epoch_data=generate_epoch_valid(Gs_train, BATCH_SIZE)
    best_f1=0
    best_auc=0
    best_recall=0
    best_precision=0
    best_val_f1 = 0
    best_val_auc = 0
    best_val_recall = 0
    best_val_precision = 0

    for i in range(1,MAX_EPOCH+1):
        loss = train_epoch(gnn, BATCH_SIZE, train_epoch_data)
        # logger.info("EPOCH %d/%d, loss = %s @ %s", i, MAX_EPOCH, loss, datetime.now())
        print("EPOCH ",i,"/",MAX_EPOCH, "loss =",loss,"Time =",datetime.now())

        if i % TEST_FREQ == 0:

            train_accuracy = get_auc_epoch_my(gnn, Gs_train, BATCH_SIZE, load_data=train_epoch_data)
            print("\nTrain model: accuracy =", train_accuracy)

            # ===============================================================================
            _val_accuracy, _val_f1, _val_recall, _val_precision = \
                get_auc_epoch_test(gnn, Gs_valid, BATCH_SIZE, load_data=valid_epoch_data)
            #====================================================================================================
            # #excel
            # _val_accuracy, _val_f1, _val_recall, _val_precision = \
            #     get_auc_epoch_test_new(gnn, Gs_valid, BATCH_SIZE, load_data=valid_epoch_data)

            print("\nValid model:")
            print("_val_accuracy ：", _val_accuracy)
            print("_val_recall   ：", _val_recall)
            print("_val_precision：", _val_precision)
            print("_val_f1       ：", _val_f1)

            valid_accuracy = _val_accuracy
            valid_f1 = _val_f1[1]
            valid_recall = _val_recall[1]
            valid_precision = _val_precision[1]

            if _val_f1[1]>best_val_f1:
                best_val_auc = _val_accuracy
                best_val_f1 = _val_f1[1]
                best_val_recall = _val_recall[1]
                best_val_precision = _val_precision[1]

            test_epoch_data = generate_epoch_valid(Gs_test, BATCH_SIZE)
            _test_accuracy, _test_f1, _test_recall, _test_precision = \
                get_auc_epoch_test(gnn, Gs_test, BATCH_SIZE, load_data=test_epoch_data)
            print("\nTest model:")
            print("_Test_accuracy ：", _test_accuracy)
            print("_Test_recall   ：", _test_recall)
            print("_Test_precision：", _test_precision)
            print("_Test_f1       ：", _test_f1)
            if _test_f1[1]>best_f1:
                best_auc = _test_accuracy
                best_f1 = _test_f1[1]
                best_recall = _test_recall[1]
                best_precision = _test_precision[1]
        print("*********************************************************************")
        print("best_val_acu      :", best_val_auc)
        print("best_val_recall   ：", best_val_recall)
        print("best_val_precision：", best_val_precision)
        print("best_val_f1       ：", best_val_f1)
        print("*********************************************************************")

        print("*********************************************************************")
        print("best_test_acu      :" , best_auc)
        print("best_test_recall   ：", best_recall)
        print("best_test_precision：", best_precision)
        print("best_test_f1       ：", best_f1)
        print("*********************************************************************")


