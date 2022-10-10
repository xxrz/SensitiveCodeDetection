# 图网络模型及相关操作函数
import tensorflow as tf
# import tensorflow.compat.v1 as tf
import numpy as np
import datetime
# from sklearn.metrics import roc_auc_score
# def weight_bias(W_shape, b_shape, bias_init=0.1):
#     W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1), name='weight')
#     b = tf.Variable(tf.constant(bias_init, shape=b_shape), name='bias')
#     return W, b
#
# def dense_layer(x, W_shape, b_shape, activation):
#     W, b = weight_bias(W_shape, b_shape)
#     return activation(tf.matmul(x, W) + b)
#
# def conv2d_layer(x, W_shape, b_shape, strides, padding):
#     W, b = weight_bias(W_shape, b_shape)
#     return tf.nn.relu(tf.nn.conv2d(x, W, strides, padding) + b)
#
# def highway_conv2d_layer(x, W_shape, b_shape, strides, padding, carry_bias=-1.0):
#     W, b = weight_bias(W_shape, b_shape, carry_bias)
#     W_T, b_T = weight_bias(W_shape, b_shape)
#     H = tf.nn.relu(tf.nn.conv2d(x, W, strides, padding) + b, name='activation')
#     T = tf.sigmoid(tf.nn.conv2d(x, W_T, strides, padding) + b_T, name='transform_gate')
#     C = tf.sub(1.0, T, name="carry_gate")
#     return tf.add(tf.mul(H, T), tf.mul(x, C), 'y') # y = (H * T) + (x * C)

# def graph_embed(X, msg_mask, N_x, N_embed, iter_level, Wnode, Wembed, W_output, b_output):
#     #X -- affine(W1) -- ReLU -- (Message -- affine(W2) -- add (with aff W1)
#     # -- ReLU -- )* MessageAll  --  output
#     #X*W
#     node_val = tf.reshape(tf.matmul( tf.reshape(X, [-1, N_x]) , Wnode),
#             [tf.shape(X)[0], -1, N_embed])
#
#     # cur_msg = tf.nn.relu(node_val)   #[batch, node_num, embed_dim]
#     D_sqrt= tf.sqrt(tf.reduce_sum(msg_mask, axis=0))
#     adj=tf.matmul(D_sqrt,msg_mask,D_sqrt)
#     for t in range(iter_level):
#         #Message convey
#         Li_t = tf.matmul(msg_mask, node_val)  #[batch, node_num, embed_dim]
#         #Complex Function
#         cur_info = tf.reshape(Li_t, [-1, N_embed])
#         for Wi in Wembed:
#             if (Wi == Wembed[-1]):
#                 cur_info = tf.matmul(cur_info, Wi)
#             else:
#                 cur_info = tf.nn.relu(tf.matmul(cur_info, Wi))
#         cur_msg = tf.reshape(cur_info, tf.shape(Li_t))
#
#
#     g_embed = tf.reduce_sum(cur_msg, 1)   #[batch, embed_dim]
#     output = tf.matmul(g_embed, W_output) + b_output
#
#     return output

# 给定graph的特征矩阵X，邻接矩阵msg_mask，返回graph的Embedding
# def graph_embed(X, msg_mask, N_x, N_embed, iter_level, Wnode, Wembed, W_output, b_output):
#     #X -- affine(W1) -- ReLU -- (Message -- affine(W2) -- add (with aff W1)
#     # -- ReLU -- )* MessageAll  --  output
#     node_val = tf.reshape(tf.matmul( tf.reshape(X, [-1, N_x]) , Wnode),
#             [tf.shape(X)[0], -1, N_embed])
#
#     cur_msg = tf.nn.relu(node_val)   #[batch, node_num, embed_dim]
#     for t in range(iter_level):
#         #Message convey
#         Li_t = tf.matmul(msg_mask, cur_msg)  #[batch, node_num, embed_dim]
#         #Complex Function
#         cur_info = tf.reshape(Li_t, [-1, N_embed])
#         for Wi in Wembed:
#             if (Wi == Wembed[-1]):
#                 cur_info = tf.matmul(cur_info, Wi)
#             else:
#                 cur_info = tf.nn.relu(tf.matmul(cur_info, Wi))
#         neigh_val_t = tf.reshape(cur_info, tf.shape(Li_t))
#         #Adding
#         tot_val_t = node_val + neigh_val_t
#         #Nonlinearity
#         tot_msg_t = tf.nn.tanh(tot_val_t)
#         cur_msg = tot_msg_t   #[batch, node_num, embed_dim]
#
#     g_embed = tf.reduce_sum(cur_msg, 1)   #[batch, embed_dim]
#     output = tf.matmul(g_embed, W_output) + b_output
#
#     return output


# 图网络模型对象
from tensorflow_core.contrib.layers.python.layers.initializers import xavier_initializer


class graphnn(object):
    def __init__(self,
                 # 节点特征向量的维度
                 NODE_FEATURE_DIM,
                 # TensorFlow数据类型
                 Dtype,
                 # graph Embedding的维度
                 N_embed,
                 # 嵌入深度
                 depth_embed,
                 # 输出的维度
                 N_output,
                 # 迭代次数
                 ITER_LEVEL,
                 # 学习率
                 lr,
                 device='/gpu:0',
                 cpu_device='/cpu:0',
                 ):

        tf.reset_default_graph()
        # with tf.device(device):   # 由于本地计算机没有GPU，所以，注释掉此行，但又保证后面的代码可以执行，因此open gitignore文件
        # with open(".gitignore", 'r') as f:
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
        self.Wnode = tf.Variable(tf.truncated_normal(
            shape=[NODE_FEATURE_DIM, N_embed], stddev=0.1, dtype=Dtype))
        # self.Wnode = tf.get_variable('W99', [NODE_FEATURE_DIM, N_embed], tf.float32, xavier_initializer())
        # self.Wnode = tf.get_variable('W99', [NODE_FEATURE_DIM, N_embed], tf.float32, tf.variance_scaling_initializer())


        self.Wembed = []
        for i in range(depth_embed):
            self.Wembed.append(tf.Variable(tf.truncated_normal(
                shape=[N_embed, N_embed], stddev=0.1, dtype=Dtype)))
            # self.Wembed.append(tf.get_variable('W'+str(i), [N_embed, N_embed], tf.float32, xavier_initializer()))
            # self.Wembed.append(tf.get_variable('W' + str(i), [N_embed, N_embed], tf.float32, tf.variance_scaling_initializer()))
        self.W_output = tf.Variable(tf.truncated_normal(
            shape=[N_embed, N_output], stddev=0.1, dtype=Dtype))
        # self.W_output = tf.get_variable('W98', [N_embed, N_output], tf.float32, xavier_initializer())
        # self.W_output = tf.get_variable('W98', [N_embed, N_output], tf.float32, tf.variance_scaling_initializer())

        self.b_output = tf.Variable(tf.constant(0, shape=[N_output], dtype=Dtype))
        # 不可用 self.b_output = tf.get_variable('W97', [N_output], tf.float32, xavier_initializer())
        # 特征矩阵，需要从数据集中输入
        self.X1 = tf.placeholder(Dtype, [None, None, NODE_FEATURE_DIM])  # [Batch_size, N_node, N_x]
        # self.X1 = tf.layers.dropout(self.X1, rate=0.5, training=True)
        # 邻接矩阵，需要从数据集中输入
        self.msg1_mask = tf.placeholder(Dtype, [None, None, None])  # [Batch_size, N_node, N_node]
        # self.msg1_mask = tf.layers.dropout(self.msg1_mask, rate=0.5, training=True)
        # X -- affine(W1) -- ReLU -- (Message -- affine(W2) -- add (with aff W1)
        # -- ReLU -- )* MessageAll  --  output
        self.node_val = tf.reshape(tf.matmul(tf.reshape(self.X1, [-1, NODE_FEATURE_DIM]), self.Wnode),
                                   [tf.shape(self.X1)[0], -1, N_embed])
        self.cur_msg = tf.nn.relu(self.node_val)   # [batch, node_num, embed_dim]
        # self.cur_msg = tf.layers.dropout(self.cur_msg, rate=0.5,training=True)
        for t in range(ITER_LEVEL):
            # Message convey
            self.Li_t = tf.matmul(self.msg1_mask, self.cur_msg)  # [batch, node_num, embed_dim]

            # Complex Function
            self.cur_info = tf.reshape(self.Li_t, [-1, N_embed])
            for Wi in self.Wembed:
                if Wi == self.Wembed[-1]:
                    self.cur_info = tf.matmul(self.cur_info, Wi)

                else:
                    self.cur_info = tf.nn.relu(tf.matmul(self.cur_info, Wi))
            self.neigh_val_t = tf.reshape(self.cur_info, tf.shape(self.Li_t))
            # self.neigh_val_t = tf.layers.dropout(self.neigh_val_t, rate=0.5,training=True)
            # Adding
            self.tot_val_t = self.node_val + self.neigh_val_t
            # NonLinearity
            self.tot_msg_t = tf.nn.relu(self.tot_val_t)
            # self.tot_msg_t = tf.layers.dropout(self.tot_msg_t, rate=0.5,training=True)

            self.x= self.cur_msg
            self.cur_msg = self.tot_msg_t  # [batch, node_num, embed_dim]
            self.T = tf.nn.sigmoid(self.node_val)
            self.C = tf.subtract(1.0, self.T, name="carry_gate")
            self.cur_msg = tf.add(tf.multiply(self.cur_msg, self.T), tf.multiply(self.x, self.T), 'y')

        self.g_embed = tf.reduce_sum(self.cur_msg, 1)   # [batch, embed_dim]
        # 训练后的图Embedding，设置为二维，以和标签类别相对应
        self.embed1 = tf.matmul(self.g_embed, self.W_output) + self.b_output


        # self.embed1 = graph_embed(self.X1, self.msg1_mask, NODE_FEATURE_DIM, N_embed, ITER_LEVEL,
        #                           self.Wnode, self.Wembed, self.W_output, self.b_output)
        # 图的类别标签，二维，需从数据集中输入
        self.label = tf.placeholder(Dtype, [None, 2])  # Bug: [0, 1]; Good:[1, 0]
        # 图Embedding二维的，作为图的类别
        # self.result = self.embed1

        # self.dense = tf.layers.dense(self.embed1,units=300)
        #########++++
        # self.dense = tf.layers.dropout(self.dense, rate=0.5)
        # self.dense = tf.layers.dense(self.dense, 300)
        # self.dense = tf.layers.dropout(self.dense, rate=0.5)
        #
        # self.dense = tf.layers.dense(self.embed1, 2)
        # self.dense = tf.layers.dropout(self.dense, rate=0.5)
        #############
        # self.result = self.dense
        # self.result = tf.cast(tf.nn.softmax(self.embed1), tf.float32)
        self.result=tf.cast(tf.nn.softmax(self.embed1), tf.float32)

        # 比较给定的标签lable和训练后的结果，得到正确训练的数目
        self.correct_prediction = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.result, 1))
        # 计算此次训练的准确率
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        # 计算交叉熵
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.result, name='cross_entropy')
        # 计算loss值
        self.loss = tf.reduce_mean(self.cross_entropy, name="loss")
        # 设置优化器
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

    # 输出函数
    def say(self, string):
        print (string)
        if self.log_file != None:
            self.log_file.write(string+'\n')

    # 模型初始化
    def init(self, LOAD_PATH, LOG_PATH):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        saver = tf.train.Saver(max_to_keep=1000)
        self.sess = sess
        self.saver = saver
        self.log_file = None
        if (LOAD_PATH is not None):
            if LOAD_PATH == '#LATEST#':
                checkpoint_path = tf.train.latest_checkpoint('./')
            else:
                checkpoint_path = LOAD_PATH
            saver.restore(sess, checkpoint_path)
            if LOG_PATH != None:
                self.log_file = open(LOG_PATH, 'a+')
            self.say('{}, model loaded from file: {}'.format(
                datetime.datetime.now(), checkpoint_path))
        else:
            sess.run(tf.global_variables_initializer())
            if LOG_PATH != None:
                self.log_file = open(LOG_PATH, 'w')
            self.say('Training start @ {}'.format(datetime.datetime.now()))

    # 获取训练后的图Embedding
    def get_embed(self, X1, mask1):
        vec, = self.sess.run(fetches=[self.embed1],
                feed_dict={self.X1:X1, self.msg1_mask:mask1})
        return vec

    # 计算loss值
    def calc_loss(self, X1, X2, mask1, mask2, y):
        cur_loss, = self.sess.run(fetches=[self.loss], feed_dict={self.X1:X1,
            self.X2:X2,self.msg1_mask:mask1,self.msg2_mask:mask2,self.label:y})
        return cur_loss

    # 在相似度预测论文中，用于技术两个Embedding的差异
    def calc_diff(self, X1, X2, mask1, mask2):
        diff, = self.sess.run(fetches=[self.diff], feed_dict={self.X1:X1,
            self.X2:X2, self.msg1_mask:mask1, self.msg2_mask:mask2})
        return diff

    # 计算训练结果（类别）
    def calc_result(self, X1, mask1):
        result, = self.sess.run(fetches=[self.result], feed_dict={self.X1:X1,
            self.msg1_mask:mask1})
        return result

    # 训练结果转为1维列表（类别）
    def get_result_array(self, X1, mask1, y):
        result, label = self.sess.run(fetches=[self.result, self.label], feed_dict={self.X1:X1,
            self.msg1_mask:mask1, self.label:y})
        return tf.argmax(result, 1).eval(session=self.sess).tolist(), tf.argmax(label, 1).eval(session=self.sess).tolist()

    # 标签转为1维列表（类别）
    def get_label_array(self, label):
        return tf.argmax(label, 1).eval(session=self.sess).tolist()

    # 计算准确度
    def calc_accuracy(self, X1, mask1, y):
        accuracy, = self.sess.run(fetches=[self.accuracy], feed_dict={self.X1:X1,
            self.msg1_mask:mask1, self.label:y})
        return accuracy

    # 输入特征矩阵、邻接矩阵、类别，进行训练，输出loss值
    def train(self, X, m, y):
        loss,_ = self.sess.run([self.loss,self.optimizer],feed_dict={self.X1:X,self.msg1_mask:m,self.label:y})
        return loss

    # 保存模型
    def save(self, path, epoch=None):
        checkpoint_path = self.saver.save(self.sess, path, global_step=epoch)
        return checkpoint_path
