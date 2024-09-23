# -*- coding: utf-8 -*-
"""Example of using Angle-base outlier detection (ABOD) for outlier detection
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.models.iforest import IForest
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score

if __name__ == "__main__":
    """
    0指代pyod合成函数合成的数据集
    1 指代datasets/real_outlier下的数据集
    2 指代datasets/multi_class_to_outlier下的数据集
    3 指代datasets/real_outlier下的数据集
    4 指代datasets/real_outlier_varying_ratios下的数据集
    5 指代datasets/synthetic_outlier下的数据集
    """
    data_type = 1

    if data_type == 0:
        contamination = 0.1  # percentage of outliers
        n_train = 200  # number of training points
        n_test = 100  # number of testing points
        # Generate sample data
        X_train, X_test, y_train, y_test = \
            generate_data(n_train=n_train,
                          n_test=n_test,
                          n_features=2,
                          contamination=contamination,
                          random_state=42)

        # train ABOD detector
        clf_name = 'IForest'
        clf = IForest()
        clf.fit(X_train)

        # get the prediction labels and outlier scores of the training data
        y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        y_train_scores = clf.decision_scores_  # raw outlier scores

        # get the prediction on the test data
        y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
        y_test_scores = clf.decision_function(X_test)  # outlier scores

        # evaluate and print the results
        print("\nOn Training Data:")
        evaluate_print(clf_name, y_train, y_train_scores)
        print("\nOn Test Data:")
        evaluate_print(clf_name, y_test, y_test_scores)

        # visualize the results
        visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
                  y_test_pred, show_figure=True, save_figure=False)
    elif data_type == 1:
        data = pd.read_csv('../datasets/real_outlier/pendigits.csv')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # train LOF detector
        clf_name = 'IForest'
        clf = IForest()
        clf.fit(X_train)
        # get the prediction labels and outlier scores of the training data
        y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        y_train_scores = clf.decision_scores_  # raw outlier scores

        # 确保 X_test 是 NumPy 数组
        X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        # get the prediction on the test data
        y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
        y_test_scores = clf.decision_function(X_test)  # outlier scores

        print("\nOn Test Data:")
        # 返回ROC AUC分数和precision @ rank n分数
        evaluate_print(clf_name, y_test, y_test_scores)

        """Accuracy指标"""
        print("*" * 100)
        print("LOF在测试集中的分类准确度：" + str(accuracy_score(y_test, y_test_pred)))

        """Precision/Recall/F1指标"""
        print("*" * 100)
        # average='micro': 全局计算 F1 分数，适用于处理类别不平衡的情况。
        # average='macro': 类别 F1 分数的简单平均，适用于需要均衡考虑每个类别的情况。
        # average='weighted': 加权 F1 分数，适用于类别不平衡的情况，考虑了每个类别的样本量。
        # average=None: 返回每个类别的 F1 分数，适用于详细分析每个类别的表现。
        print("LOF在测试集中的分类精确度：" + str(precision_score(y_test, y_test_pred, average='weighted')))
        print("LOF在测试集中的分类召回率：" + str(recall_score(y_test, y_test_pred, average='weighted')))
        print("LOF在测试集中的分类F1分数：" + str(f1_score(y_test, y_test_pred, average='weighted')))

        """ROC-AUC指标"""
        print("*" * 100)
        roc_auc_test = roc_auc_score(y_test, y_test_pred, multi_class='ovr')  # 一对多方式
        print("LOF在测试集中的ROC-AUC分数：" + str(roc_auc_test))

        """PR AUC指标"""
        print("*" * 100)
        # 预测概率
        y_scores = clf.predict_proba(X_test)[:, 1]  # 取正类的概率
        # 计算 Precision 和 Recall
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        # 计算 PR AUC
        pr_auc = auc(recall, precision)
        print("PR AUC 分数:", pr_auc)

        """AP指标"""
        print("*" * 100)
        # 预测概率
        y_scores = clf.predict_proba(X_test)[:, 1]  # 取正类的概率
        # 计算 Average Precision
        ap_score = average_precision_score(y_test, y_scores)
        print("AP分数:", ap_score)

        """Rank Power指标"""
        # print("*" * 100)
        # # 获取异常得分
        # y_scores = clf.predict_proba(X_test)[:, 1]  # 取正类的概率
        # # 生成排名
        # y_test_numpy = np.array(y_test)
        # ranked_indices = np.argsort(y_scores)[::-1]  # 从高到低排序
        # ranked_relevance = [y_test_numpy[i] for i in ranked_indices]  # 根据排序获取真实相关性
        # # 计算 Rank Power
        # cumulative_relevance = np.cumsum(ranked_relevance)
        # rank_power = cumulative_relevance / np.arange(1, len(ranked_relevance) + 1)
        # # 输出 Rank Power
        # print("Rank Power 值:")
        # for i, power in enumerate(rank_power):
        #     print(f"排名 {i + 1}: {power}")

        """其他指标（时间，内存占用，鲁棒性等）"""