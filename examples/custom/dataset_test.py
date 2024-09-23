import numpy as np
import scipy.io as sio

# 读取 .mat 文件
data = sio.loadmat('../data/arrhythmia.mat')
# data = sio.loadmat('../data/cardio.mat')
# data = sio.loadmat('../data/ionosphere.mat')
# data = sio.loadmat('../data/letter.mat')
# data = sio.loadmat('../data/pima.mat')
X, y = data['X'], data['y']

# 使用 numpy.unique 找到唯一值
unique_values = np.unique(y)
# 计算唯一值的数量
num_unique_values = len(unique_values)
print("y属性不同的标签数量：", num_unique_values)