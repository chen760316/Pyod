"""
无监督离群值检测算法在传统异常检测领域的效果
"""
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
from pyod.models.abod import ABOD
from pyod.models.cof import COF
from pyod.models.copod import COPOD
from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.sod import SOD


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=np.inf)

# section 标准数据集处理

# choice 选取数据集
# file_path = "../datasets/real_outlier/Cardiotocography.csv"
# file_path = "../datasets/real_outlier/annthyroid.csv"
file_path = "../datasets/real_outlier/optdigits.csv"
# file_path = "../datasets/real_outlier/PageBlocks.csv"
# file_path = "../datasets/real_outlier/pendigits.csv"
# file_path = "../datasets/real_outlier/satellite.csv"
# file_path = "../datasets/real_outlier/shuttle.csv"
# file_path = "../datasets/real_outlier/yeast.csv"

def run(file_path):

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

    # SECTION 无监督异常检测器的检测精度测试
    epochs = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_trans = 64
    random_state = 42

    # choice ABOD异常检测器
    out_clf = ABOD()
    out_clf.fit(X_train)
    out_clf_noise = ABOD()
    out_clf_noise.fit(X_train_copy)

    # # choice COF异常检测器
    # out_clf = COF()
    # out_clf.fit(X_train)
    # out_clf_noise = COF()
    # out_clf_noise.fit(X_train_copy)
    #
    # # choice COPOD异常检测器
    # out_clf = COPOD()
    # out_clf.fit(X_train)
    # out_clf_noise = COPOD()
    # out_clf_noise.fit(X_train_copy)
    #
    # # choice ECOD异常检测器
    # out_clf = ECOD()
    # out_clf.fit(X_train)
    # out_clf_noise = ECOD()
    # out_clf_noise.fit(X_train_copy)
    #
    # # choice IForest异常检测器
    # out_clf = IForest()
    # out_clf.fit(X_train)
    # out_clf_noise = IForest()
    # out_clf_noise.fit(X_train_copy)
    #
    # # choice LODA异常检测器
    # out_clf = LODA()
    # out_clf.fit(X_train)
    # out_clf_noise = LODA()
    # out_clf_noise.fit(X_train_copy)
    #
    # # choice LOF异常检测器
    # out_clf = LOF()
    # out_clf.fit(X_train)
    # out_clf_noise = LOF()
    # out_clf_noise.fit(X_train_copy)
    #
    # # choice OCSVM异常检测器
    # out_clf = OCSVM()
    # out_clf.fit(X_train)
    # out_clf_noise = OCSVM()
    # out_clf_noise.fit(X_train_copy)
    #
    # # choice SOD异常检测器
    # out_clf = SOD()
    # out_clf.fit(X_train)
    # out_clf_noise = SOD()
    # out_clf_noise.fit(X_train_copy)

    # SECTION 借助异常检测器，在训练集上进行异常值检测。
    #  经过检验，加入高斯噪声会影响异常值判别

    # subsection 从原始训练集中检测出异常值索引

    print("*"*100)
    train_scores = out_clf.decision_function(X_train)
    train_pred_labels, train_confidence = out_clf.predict(X_train, return_confidence=True)
    print("训练集中异常值判定阈值为：", out_clf.threshold_)
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
    test_pred_labels, test_confidence = out_clf.predict(X_test, return_confidence=True)
    print("测试集中异常值判定阈值为：", out_clf.threshold_)
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
    print("无监督异常检测器在原始测试集中的分类准确度：" + str(accuracy_score(y_test, test_pred_labels)))

    """Precision/Recall/F1指标"""
    print("*" * 100)

    # average='micro': 全局计算 F1 分数，适用于处理类别不平衡的情况。
    # average='macro': 类别 F1 分数的简单平均，适用于需要均衡考虑每个类别的情况。
    # average='weighted': 加权 F1 分数，适用于类别不平衡的情况，考虑了每个类别的样本量。
    # average=None: 返回每个类别的 F1 分数，适用于详细分析每个类别的表现。

    print("无监督异常检测器在原始测试集中的分类精确度：" + str(precision_score(y_test, test_pred_labels, average='weighted')))
    print("无监督异常检测器在原始测试集中的分类召回率：" + str(recall_score(y_test, test_pred_labels, average='weighted')))
    print("无监督异常检测器在原始测试集中的分类F1分数：" + str(f1_score(y_test, test_pred_labels, average='weighted')))

    """ROC-AUC指标"""
    try:
        print("*" * 100)
        y_test_prob = 1 / (1 + np.exp(-test_scores))
        roc_auc_test = roc_auc_score(y_test, y_test_prob, multi_class='ovr')  # 一对多方式
        print("无监督异常检测器在原始测试集中的ROC-AUC分数：" + str(roc_auc_test))

        """PR AUC指标"""
        print("*" * 100)
        # 计算预测概率
        y_scores = 1 / (1 + np.exp(-test_scores))
        # 计算 Precision 和 Recall
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        # 计算 PR AUC
        pr_auc = auc(recall, precision)
        print("无监督异常检测器在原始测试集中的PR AUC 分数:", pr_auc)

        """AP指标"""
        print("*" * 100)
        # 计算预测概率
        y_scores = 1 / (1 + np.exp(-test_scores))
        # 计算 Average Precision
        ap_score = average_precision_score(y_test, y_scores)
        print("无监督异常检测器在原始测试集中的AP分数:", ap_score)
    except:
        pass

    # section 从加噪数据集的训练集和测试集中检测出的异常值

    # subsection 从加噪训练集中检测出异常值索引

    train_scores_noise = out_clf_noise.decision_function(X_train_copy)
    train_pred_labels_noise, train_confidence_noise = out_clf_noise.predict(X_train_copy, return_confidence=True)
    print("加噪训练集中异常值判定阈值为：", out_clf_noise.threshold_)
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
    test_pred_labels_noise, test_confidence_noise = out_clf_noise.predict(X_test_copy, return_confidence=True)
    print("加噪测试集中异常值判定阈值为：", out_clf_noise.threshold_)
    test_outliers_index_noise = []
    print("加噪测试集样本数：", len(X_test_copy))
    for i in range(len(X_test_copy)):
        if test_pred_labels_noise[i] == 1:
            test_outliers_index_noise.append(i)
    # 训练样本中的异常值索引
    print("加噪测试集中异常值索引：", test_outliers_index_noise)
    print("加噪测试集中的异常值数量：", len(test_outliers_index_noise))
    print("加噪测试集中的异常值比例：", len(test_outliers_index_noise)/len(X_test_copy))

    """Accuracy指标"""
    print("*" * 100)
    print("无监督异常检测器在加噪测试集中的分类准确度：" + str(accuracy_score(y_test, test_pred_labels_noise)))
    r_acc = str(accuracy_score(y_test, test_pred_labels_noise))

    """Precision/Recall/F1指标"""
    print("*" * 100)

    # average='micro': 全局计算 F1 分数，适用于处理类别不平衡的情况。
    # average='macro': 类别 F1 分数的简单平均，适用于需要均衡考虑每个类别的情况。
    # average='weighted': 加权 F1 分数，适用于类别不平衡的情况，考虑了每个类别的样本量。
    # average=None: 返回每个类别的 F1 分数，适用于详细分析每个类别的表现。

    print("无监督异常检测器在加噪测试集中的分类精确度：" + str(precision_score(y_test, test_pred_labels_noise, average='weighted')))
    r_recall = str(recall_score(y_test, test_pred_labels_noise, average='weighted'))
    print("无监督异常检测器在加噪测试集中的分类召回率：" + str(recall_score(y_test, test_pred_labels_noise, average='weighted')))
    print("无监督异常检测器在加噪测试集中的分类F1分数：" + str(f1_score(y_test, test_pred_labels_noise, average='weighted')))

    try:
        """ROC-AUC指标"""
        print("*" * 100)
        y_test_prob_noise = 1 / (1 + np.exp(-test_scores_noise))
        roc_auc_test = roc_auc_score(y_test, y_test_prob_noise, multi_class='ovr')  # 一对多方式
        print("无监督异常检测器在加噪测试集中的ROC-AUC分数：" + str(roc_auc_test))
        r_roc_auc = str(roc_auc_test)

        """PR AUC指标"""
        print("*" * 100)
        # 计算预测概率
        y_scores = 1 / (1 + np.exp(-test_scores_noise))
        # 计算 Precision 和 Recall
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        # 计算 PR AUC
        pr_auc = auc(recall, precision)
        print("无监督异常检测器在加噪测试集中的PR AUC 分数:", pr_auc)

        """AP指标"""
        print("*" * 100)
        # 计算预测概率
        y_scores = 1 / (1 + np.exp(-test_scores_noise))
        # 计算 Average Precision
        ap_score = average_precision_score(y_test, y_scores)
        print("无监督异常检测器在加噪测试集中的AP分数:", ap_score)
    except:
        r_roc_auc = "NAN"
        pass

    return r_recall, r_acc, r_roc_auc

if __name__ == '__main__':
    paths = [
        "../datasets/real_outlier/Cardiotocography.csv",
        "../datasets/real_outlier/annthyroid.csv",
        "../datasets/real_outlier/optdigits.csv",
        "../datasets/real_outlier/PageBlocks.csv",
        "../datasets/real_outlier/pendigits.csv",
        "../datasets/real_outlier/satellite.csv",
        "../datasets/real_outlier/shuttle.csv",
        "../datasets/real_outlier/yeast.csv"
    ]
    res_list = [[],[],[]]
    for file_path in paths:
        recall, acc, roc_auc = run(file_path)
        # res_list[0].append(recall)
        res_list[1].append(acc)
        # res_list[2].append(roc_auc)
    res_new = res_list[0] + res_list[1] + res_list[2]
    print(" ".join(res_new))