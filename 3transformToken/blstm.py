from __future__ import print_function

import warnings

from keras import Model
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, LeakyReLU
from keras.optimizers import Adamax

from sklearn.model_selection import train_test_split
import xlsxwriter


"""
Bidirectional LSTM neural network
Structure consists of two hidden layers and a BLSTM layer
Parameters, as from the VulDeePecker paper:
    Nodes: 300
    Dropout: 0.5
    Optimizer: Adamax
    Batch size: 64
    Epochs: 4
"""



class BLSTM:
    def __init__(self, data, name="", batch_size=64):
        vectors = np.stack(data.iloc[:, 0].values)
        labels = data.iloc[:, 1].values
        positive_idxs = np.where(labels == 1)[0]
        negative_idxs = np.where(labels == 0)[0]
        undersampled_negative_idxs = np.random.choice(negative_idxs, len(positive_idxs), replace=True)
        resampled_idxs = np.concatenate([positive_idxs, undersampled_negative_idxs])

        X_train, X_test, y_train, y_test = train_test_split(vectors[resampled_idxs, ], labels[resampled_idxs],
                                                            test_size=0.2, stratify=labels[resampled_idxs])
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)
        self.name = name
        self.batch_size = batch_size
        self.class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=labels)
        model = Sequential()
        model.add(Bidirectional(LSTM(300), input_shape=(vectors.shape[1], vectors.shape[2])))
        model.add(Dense(300))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(Dense(300))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        #=========================================================================================================================
        aintermediate_layer_model = Model(inputs=model.input,
                                          outputs=model.get_layer(index=6).output)
        aintermediate_output = aintermediate_layer_model.predict(vectors)  # a=model.layers[0].output
        # 下列变为excel

        l=list(labels)
        pca = PCA(n_components=2)  # 降到2维
        # pca.fit(aintermediate_output)  # 训练
        newdata= pca.fit_transform(aintermediate_output)  # 降维后的数据
        # PCA(copy=True, n_components=2, whiten=False)

        x=list(newdata[:,0])
        y=list(newdata[:,1])

        workbook = xlsxwriter.Workbook('githubVul.xlsx')  # 创建一个Excel文件
        worksheet = workbook.add_worksheet()
        headings = ['x', 'y', 'label']  # 设置表头
        worksheet.write_row(0, 0,headings)
        worksheet.write_column(1, 0, x)
        worksheet.write_column(1, 1, y)
        worksheet.write_column(1, 2, l)
        workbook.close()
        #=========================================================================================================================
        # Lower learning rate to prevent divergence
        adamax = Adamax(lr=0.002)
        model.compile(adamax, 'categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    """
    Trains model based on training data
    """
    def train(self):
        self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, epochs=4, class_weight=self.class_weight)
        self.model.save_weights(self.name + "_model.h5")

    """
    Tests accuracy of model based on test data
    Loads weights from file if no weights are attached to model object
    """
    def test(self):
        self.model.load_weights(self.name + "_model.h5")
        values = self.model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size)
        print("Accuracy is...", values[1])
        predictions = (self.model.predict(self.X_test, batch_size=self.batch_size)).round()

        tn, fp, fn, tp = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(predictions, axis=1)).ravel()
        print('False positive rate is...', fp / (fp + tn))
        print('False negative rate is...', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('True positive rate is...', recall)
        precision = tp / (tp + fp)
        print('Precision is...', precision)
        print('F1 score is...', (2 * precision * recall) / (precision + recall))