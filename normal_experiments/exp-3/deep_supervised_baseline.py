"""
(半) 监督离群值检测算法对ugly outliers的检测能力
"""
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from pyod.models.xgbod import XGBOD

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=np.inf)

# section 标准数据集处理

file_path = "../datasets/real_outlier/Cardiotocography.csv"
# file_path = "../datasets/real_outlier/annthyroid.csv"
# file_path = "../datasets/real_outlier/optdigits.csv"
# file_path = "../datasets/real_outlier/PageBlocks.csv"
# file_path = "../datasets/real_outlier/pendigits.csv"
# file_path = "../datasets/real_outlier/satellite.csv"
# file_path = "../datasets/real_outlier/shuttle.csv"
# file_path = "../datasets/real_outlier/yeast.csv"

data = pd.read_csv(file_path)

# 如果数据量超过20000行，就随机采样到20000行
if len(data) > 20000:
    data = data.sample(n=20000, random_state=42)

enc = LabelEncoder()
label_name = data.columns[-1]

# 原始数据集D对应的Dataframe
data[label_name] = enc.fit_transform(data[label_name])

# 检测非数值列
non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns

# 为每个非数值列创建一个 LabelEncoder 实例
encoders = {}
for column in non_numeric_columns:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column])
    encoders[column] = encoder  # 保存每个列的编码器，以便将来可能需要解码

X = data.values[:, :-1]
y = data.values[:, -1]

# 统计不同值及其数量
unique_values, counts = np.unique(y, return_counts=True)

# 输出结果
for value, count in zip(unique_values, counts):
    print(f"标签: {value}, 数量: {count}")

# 找到最小标签的数量
min_count = counts.min()
total_count = counts.sum()

# 计算比例
proportion = min_count / total_count
print(f"较少标签占据的比例: {proportion:.4f}")
min_count_index = np.argmin(counts)  # 找到最小数量的索引
min_label = unique_values[min_count_index]  # 对应的标签值

# section 数据特征缩放以及添加噪声

# 对不同维度进行标准化
X = StandardScaler().fit_transform(X)
# 记录原始索引
original_indices = np.arange(len(X))
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, original_indices, test_size=0.3, random_state=1)
# 加入随机噪声的比例
noise_level = 0.2
# 计算噪声数量
n_samples = X.shape[0]
n_noise = int(noise_level * n_samples)
# 随机选择要添加噪声的样本
noise_indices = np.random.choice(n_samples, n_noise, replace=False)
# 添加高斯噪声到特征
X_copy = np.copy(X)
X_copy[noise_indices] += np.random.normal(0, 1, (n_noise, X.shape[1]))
# 从加噪数据中生成加噪训练数据和加噪测试数据
X_train_copy = X_copy[train_indices]
X_test_copy = X_copy[test_indices]
feature_names = data.columns.values.tolist()
combined_array = np.hstack((X_copy, y.reshape(-1, 1)))  # 将 y 重新调整为列向量并合并
# 添加噪声后的数据集D'对应的Dataframe
data_copy = pd.DataFrame(combined_array, columns=feature_names)
# 训练集中添加了高斯噪声的样本在原始数据集D中的索引
train_noise = np.intersect1d(train_indices, noise_indices)
# 测试集中添加了高斯噪声的样本在原始数据集D中的索引
test_noise = np.intersect1d(test_indices, noise_indices)

# section 选择监督异常检测器

# subsection 确定参数以及少数标签的索引

epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42
hidden_dims = 20
epoch_steps = 20
batch_size = 256
lr = 1e-5

# 设置弱监督训练样本
# 找到所有标签为 1 的样本索引
semi_label_ratio = 0.1  # 设置已知的异常标签比例
positive_indices = np.where(y_train == min_label)[0]
# 随机选择 10% 的正样本
n_positive_to_keep = int(len(positive_indices) * semi_label_ratio)
selected_positive_indices = np.random.choice(positive_indices, n_positive_to_keep, replace=False)
# 创建用于异常检测器的训练标签
y_semi = np.zeros_like(y_train)  # 默认全为 0
y_semi[selected_positive_indices] = 1  # 设置选中的正样本为 1
# 创建用于异常检测器的测试标签
y_semi_test = np.zeros_like(y_test)
test_positive_indices = np.where(y_test == min_label)[0]
y_semi_test[test_positive_indices] = 1

# choice DevNet异常检测器
out_clf = XGBOD()
out_clf.fit(X_train, y_semi)
out_clf_noise = XGBOD()
out_clf_noise.fit(X_train_copy, y_semi)

# choice RoSAS异常检测器
# out_clf = RoSAS(epochs=epochs, hidden_dims=hidden_dims, device=device, random_state=random_state)
# out_clf.fit(X_train, y_semi)
# out_clf_noise = RoSAS(epochs=epochs, hidden_dims=hidden_dims, device=device, random_state=random_state)
# out_clf_noise.fit(X_train_copy, y_semi)

# choice PReNeT异常检测器
# out_clf = PReNet(epochs=epochs, device=device, random_state=random_state)
# out_clf.fit(X_train, y_semi)
# out_clf_noise = PReNet(epochs=epochs, device=device, random_state=random_state)
# out_clf_noise.fit(X_train_copy, y_semi)

# SECTION 借助异常检测器，在训练集上进行异常值检测。
#  经过检验，加入高斯噪声会影响异常值判别

# subsection 从原始训练集中检测出异常值索引

print("*"*100)
train_scores = out_clf.decision_function(X_train)
train_pred_labels = out_clf.predict(X_train)
# print("训练集中异常值判定阈值为：", out_clf.threshold_)
train_outliers_index = []
print("训练集样本数：", len(X_train))
for i in range(len(X_train)):
    if train_pred_labels[i] == 1:
        train_outliers_index.append(i)
# 训练样本中的异常值索引
print("训练集中异常值索引：", train_outliers_index)
print("训练集中的异常值数量：", len(train_outliers_index))
print("训练集中的异常值比例：", len(train_outliers_index)/len(X_train))

# subsection 从原始测试集中检测出异常值索引

test_scores = out_clf.decision_function(X_test)
test_pred_labels = out_clf.predict(X_test)
# print("测试集中异常值判定阈值为：", out_clf.threshold_)
test_outliers_index = []
print("测试集样本数：", len(X_test))
for i in range(len(X_test)):
    if test_pred_labels[i] == 1:
        test_outliers_index.append(i)
# 训练样本中的异常值索引
print("测试集中异常值索引：", test_outliers_index)
print("测试集中的异常值数量：", len(test_outliers_index))
print("测试集中的异常值比例：", len(test_outliers_index)/len(X_test))

"""Accuracy指标"""
print("*" * 100)
print("半监督异常检测器在原始测试集中的分类准确度：" + str(accuracy_score(y_test, test_pred_labels)))

# subsection 从全部数据中检测出异常值索引

print("*"*100)
scorese = out_clf.decision_function(X)
pred_labels = out_clf.predict(X)
outliers_index = []
for i in range(len(X)):
    if pred_labels[i] == 1:
        outliers_index.append(i)
print("全部数据中的异常值数量：", len(outliers_index))

# section 从加噪数据集的训练集和测试集中检测出的异常值

# subsection 从加噪训练集中检测出异常值索引

train_scores_noise = out_clf_noise.decision_function(X_train_copy)
train_pred_labels_noise = out_clf_noise.predict(X_train_copy)
# print("加噪训练集中异常值判定阈值为：", out_clf_noise.threshold_)
train_outliers_index_noise = []
print("加噪训练集样本数：", len(X_train_copy))
for i in range(len(X_train_copy)):
    if train_pred_labels_noise[i] == 1:
        train_outliers_index_noise.append(i)
# 训练样本中的异常值索引
print("加噪训练集中异常值索引：", train_outliers_index_noise)
print("加噪训练集中的异常值数量：", len(train_outliers_index_noise))
print("加噪训练集中的异常值比例：", len(train_outliers_index_noise)/len(X_train_copy))

# subsection 从加噪测试集中检测出异常值索引

test_scores_noise = out_clf_noise.decision_function(X_test_copy)
test_pred_labels_noise = out_clf_noise.predict(X_test_copy)
# print("加噪测试集中异常值判定阈值为：", out_clf_noise.threshold_)
test_outliers_index_noise = []
print("加噪测试集样本数：", len(X_test_copy))
for i in range(len(X_test_copy)):
    if test_pred_labels_noise[i] == 1:
        test_outliers_index_noise.append(i)
# 训练样本中的异常值索引
print("加噪测试集中异常值索引：", test_outliers_index_noise)
print("加噪测试集中的异常值数量：", len(test_outliers_index_noise))
print("加噪测试集中的异常值比例：", len(test_outliers_index_noise)/len(X_test_copy))

# subsection 从全部数据中检测出异常值索引

print("*"*100)
scores_noise = out_clf_noise.decision_function(X_copy)
pred_labels_noise = out_clf_noise.predict(X_copy)
outliers_index_noise = []
for i in range(len(X_copy)):
    if pred_labels_noise[i] == 1:
        outliers_index_noise.append(i)
print("加噪数据中的异常值数量：", len(outliers_index_noise))

# section 训练下游任务的SVM模型

# subsection 原始数据集上训练的SVM模型在训练集和测试集中分错的样本比例

print("*" * 100)
svm_model = svm.SVC(kernel='linear', C=1.0, probability=True, class_weight='balanced')
svm_model.fit(X_train, y_train)
train_label_pred = svm_model.predict(X_train)

# 训练样本中被SVM模型错误分类的样本
wrong_classified_train_indices = np.where(y_train != svm_model.predict(X_train))[0]
print("训练样本中被SVM模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices)/len(y_train))

# 测试样本中被SVM模型错误分类的样本
wrong_classified_test_indices = np.where(y_test != svm_model.predict(X_test))[0]
print("测试样本中被SVM模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices)/len(y_test))

# 整体数据集D中被SVM模型错误分类的样本
print("完整数据集D中被SVM模型错误分类的样本占总完整数据的比例：", (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))

# subsection 加噪数据集上训练的SVM模型在训练集和测试集中分错的样本比例

print("*" * 100)
svm_model_noise = svm.SVC(kernel='linear', C=1.0, probability=True, class_weight='balanced')
svm_model_noise.fit(X_train_copy, y_train)
train_label_pred_noise = svm_model_noise.predict(X_train_copy)

# 加噪训练样本中被SVM模型错误分类的样本
wrong_classified_train_indices_noise = np.where(y_train != svm_model_noise.predict(X_train_copy))[0]
print("加噪训练样本中被SVM模型错误分类的样本占总加噪训练样本的比例：", len(wrong_classified_train_indices_noise)/len(y_train))

# 加噪测试样本中被SVM模型错误分类的样本
wrong_classified_test_indices_noise = np.where(y_test != svm_model_noise.predict(X_test_copy))[0]
print("加噪测试样本中被SVM模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices_noise)/len(y_test))

# 整体加噪数据集D中被SVM模型错误分类的样本
print("完整数据集D中被SVM模型错误分类的样本占总完整数据的比例：", (len(wrong_classified_train_indices_noise) + len(wrong_classified_test_indices_noise))/(len(y_train) + len(y_test)))

# section 计算加噪数据中的交叉熵损失

# 获取决策值
decision_values = svm_model_noise.decision_function(X_copy)
# 将决策值转换为适用于 Softmax 的二维数组
decision_values_reshaped = decision_values.reshape(-1, 1)  # 变成 (n_samples, 1)
# 应用 Softmax 函数（可以手动实现或使用 scipy）
y_pred = softmax(np.hstack((decision_values_reshaped, -decision_values_reshaped)), axis=1)
# 创建 OneHotEncoder 实例
encoder = OneHotEncoder(sparse=False)
# 拟合并转换 y_test
y_true = encoder.fit_transform(y.reshape(-1, 1))
# 计算每个样本的损失
loss_per_sample = -np.sum(y_true * np.log(y_pred + 1e-12), axis=1)
# 计算测试集平均多分类交叉熵损失
average_loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-12), axis=1))
bad_samples_noise = np.where(loss_per_sample > average_loss)[0]
good_samples_noise = np.where(loss_per_sample <= average_loss)[0]
print("损失不超过阈值的样本数量：", len(good_samples_noise))
print("损失大于阈值的样本数量：", len(bad_samples_noise))

# section 全部加噪数据中被SVM分类器误分类的数量
label_pred = svm_model_noise.predict(X_copy)
wrong_classify_indices = []
for i in range(len(X_copy)):
    if y[i] != label_pred[i]:
        wrong_classify_indices.append(i)
print("被误分类的样本数量：", len(wrong_classify_indices))

# section 检测ugly outliers的召回率
ugly_found_by_detector = list(set(outliers_index_noise) & set(wrong_classify_indices))
print("召回的ugly outliers的数量：", len(ugly_found_by_detector))
print("ugly outliers的召回率为：", len(ugly_found_by_detector)/len(wrong_classify_indices))
