from scipy.linalg import svd
from sklearn import datasets, manifold
import time

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import scale
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

import numpy as np

# digits = datasets.load_digits(n_class=5)
# X = digits.data
# y = digits.target

#目标 5*100->1*100

x = np.random.rand(5, 100).T

# t-sne(听说是最好的办法)
t1 = time.time()
tsne = manifold.TSNE(n_components=1, init='pca', random_state=0)
X_tsne = tsne.fit_transform(x).T
t2 =time.time()
print("tsne",t2-t1,X_tsne.shape)

# PCA 计算代价高昂，只适用于特征向量间存在线性相关的环境下
t1 = time.time()
pca = PCA(n_components=1)
newdata= pca.fit_transform(x).T
t2 = time.time()
print("PCA",t2-t1,newdata.shape)

#核PCA 针对非线性数据集进行降维
t1 = time.time()
kpca = KernelPCA(kernel='rbf',gamma=10,n_components=1)
x_kpca = kpca.fit_transform(x).T
t2 = time.time()
print("核PCA",t2-t1,x_kpca.shape)

#SVD
t1 = time.time()
x_s = scale(x, with_mean=True, with_std=False, axis=0)
# 没必要缩放数据，full_matrices=False是一定要有的
U, S, V = svd(x_s, full_matrices=False)
# 选择最前两个奇异值来近似原始的矩阵
x_t = U[:, :1].T
t2 = time.time()
print("SVD",t2-t1,x_t.shape)

#高斯随机映射 速度快，利用数据间的距离来降低维度
t1 = time.time()
gauss_proj = GaussianRandomProjection(n_components=1)
vector_t = gauss_proj.fit_transform(x).T
t2 = time.time()
print("高斯随机映射",t2-t1,vector_t.shape)

#稀疏矩阵随机映射，比高斯随机映射质量更好（内存使用量和效率相对差）
t1 = time.time()
transformer = SparseRandomProjection(n_components=1)
X_new = transformer.fit_transform(x).T
t2 = time.time()
print("稀疏矩阵随机映射",t2-t1,X_new.shape)
