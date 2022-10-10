import numpy as np
file_path = "E:\GraphNNCode\g4_128.npy"
dataset = np.load(open(file_path, 'rb+'))
X, y = np.array(dataset['X']), np.array(dataset['y'], dtype=np.int)
print("End")
