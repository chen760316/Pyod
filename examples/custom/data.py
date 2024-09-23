# -*- coding: utf-8 -*-
"""Utility functions for manipulating data
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# Author: Yahya Almardeny <almardeny@gmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

from warnings import warn

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import check_X_y
from sklearn.utils import check_consistent_length
from sklearn.utils import check_random_state
from sklearn.utils import column_or_1d

from examples.custom.utility import check_parameter
from examples.custom.utility import precision_n_scores

MAX_INT = np.iinfo(np.int32).max


def _generate_data(n_inliers, n_outliers, n_features, coef, offset,
                   random_state, n_nan=0, n_inf=0):
    """Internal function to generate data samples.

    Parameters
    ----------
    n_inliers : int
        The number of inliers.

    n_outliers : int
        The number of outliers.

    n_features : int
        The number of features (dimensions).

    coef : float in range [0,1)+0.001
        The coefficient of data generation.

    offset : int
        Adjust the value range of Gaussian and Uniform.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    n_nan : int
        The number of values that are missing (np.nan). Defaults to zero.

    n_inf : int
        The number of values that are infinite. (np.inf). Defaults to zero.

    Returns
    -------
    X : numpy array of shape (n_train, n_features)
        Data.

    y : numpy array of shape (n_train,)
        Ground truth.
    """

    inliers = coef * random_state.randn(n_inliers, n_features) + offset
    outliers = random_state.uniform(low=-1 * offset, high=offset,
                                    size=(n_outliers, n_features))
    X = np.r_[inliers, outliers]

    y = np.r_[np.zeros((n_inliers,)), np.ones((n_outliers,))]

    if n_nan > 0:
        X = np.r_[X, np.full((n_nan, n_features), np.nan)]
        y = np.r_[y, np.full((n_nan), np.nan)]

    if n_inf > 0:
        X = np.r_[X, np.full((n_inf, n_features), np.inf)]
        y = np.r_[y, np.full((n_inf), np.inf)]

    return X, y


def get_outliers_inliers(X, y):
    """Internal method to separate inliers from outliers.

    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features)
        The input samples

    y : list or array of shape (n_samples,)
        The ground truth of input samples.

    Returns
    -------
    X_outliers : numpy array of shape (n_samples, n_features)
        Outliers.

    X_inliers : numpy array of shape (n_samples, n_features)
        Inliers.

    """
    X_outliers = X[np.where(y == 1)]
    X_inliers = X[np.where(y == 0)]
    return X_outliers, X_inliers


def generate_data(n_train=1000, n_test=500, n_features=2, contamination=0.1,
                  train_only=False, offset=10, behaviour='new',
                  random_state=None, n_nan=0, n_inf=0):
    """Utility function to generate synthesized data.
    Normal data is generated by a multivariate Gaussian distribution and
    outliers are generated by a uniform distribution.
    "X_train, X_test, y_train, y_test" are returned.

    Parameters
    ----------
    n_train : int, (default=1000)
        The number of training points to generate.

    n_test : int, (default=500)
        The number of test points to generate.

    n_features : int, optional (default=2)
        The number of features (dimensions).

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    train_only : bool, optional (default=False)
        If true, generate train data only.

    offset : int, optional (default=10)
        Adjust the value range of Gaussian and Uniform.

    behaviour : str, default='new'
        Behaviour of the returned datasets which can be either 'old' or
        'new'. Passing ``behaviour='new'`` returns
        "X_train, X_test, y_train, y_test", while passing ``behaviour='old'``
        returns "X_train, y_train, X_test, y_test".

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    n_nan : int
        The number of values that are missing (np.nan). Defaults to zero.

    n_inf : int
        The number of values that are infinite. (np.inf). Defaults to zero.

    Returns
    -------
    X_train : numpy array of shape (n_train, n_features)
        Training data.

    X_test : numpy array of shape (n_test, n_features)
        Test data.

    y_train : numpy array of shape (n_train,)
        Training ground truth.

    y_test : numpy array of shape (n_test,)
        Test ground truth.

    """

    # initialize a random state and seeds for the instance
    random_state = check_random_state(random_state)
    offset_ = random_state.randint(low=offset)
    coef_ = random_state.random_sample() + 0.001  # in case of underflow

    if isinstance(contamination, (float, int)):
        n_outliers_train = int(n_train * contamination)
    else:
        contamination = 0.1
        n_outliers_train = int(n_train * contamination)

    n_inliers_train = int(n_train - n_outliers_train)

    X_train, y_train = _generate_data(n_inliers_train, n_outliers_train,
                                      n_features, coef_, offset_, random_state,
                                      n_nan, n_inf)

    if train_only:
        return X_train, y_train

    n_outliers_test = int(n_test * contamination)
    n_inliers_test = int(n_test - n_outliers_test)

    X_test, y_test = _generate_data(n_inliers_test, n_outliers_test,
                                    n_features, coef_, offset_, random_state,
                                    n_nan, n_inf)

    if behaviour == 'old':
        warn('behaviour="old" is deprecated and will be removed '
             'in version 0.9.0. Please use behaviour="new", which '
             'makes the returned datasets in the order of '
             'X_train, X_test, y_train, y_test.',
             FutureWarning)
        return X_train, y_train, X_test, y_test

    else:
        return X_train, X_test, y_train, y_test


def check_consistent_shape(X_train, y_train, X_test, y_test, y_train_pred,
                           y_test_pred):
    """Internal shape to check input data shapes are consistent.

    Parameters
    ----------
    X_train : numpy array of shape (n_samples, n_features)
        The training samples.

    y_train : list or array of shape (n_samples,)
        The ground truth of training samples.

    X_test : numpy array of shape (n_samples, n_features)
        The test samples.

    y_test : list or array of shape (n_samples,)
        The ground truth of test samples.

    y_train_pred : numpy array of shape (n_samples, n_features)
        The predicted binary labels of the training samples.

    y_test_pred : numpy array of shape (n_samples, n_features)
        The predicted binary labels of the test samples.

    Returns
    -------
    X_train : numpy array of shape (n_samples, n_features)
        The training samples.

    y_train : list or array of shape (n_samples,)
        The ground truth of training samples.

    X_test : numpy array of shape (n_samples, n_features)
        The test samples.

    y_test : list or array of shape (n_samples,)
        The ground truth of test samples.

    y_train_pred : numpy array of shape (n_samples, n_features)
        The predicted binary labels of the training samples.

    y_test_pred : numpy array of shape (n_samples, n_features)
        The predicted binary labels of the test samples.
    """

    # check input data shapes are consistent
    X_train, y_train = check_X_y(X_train, y_train)
    X_test, y_test = check_X_y(X_test, y_test)

    y_test_pred = column_or_1d(y_test_pred)
    y_train_pred = column_or_1d(y_train_pred)

    check_consistent_length(y_train, y_train_pred)
    check_consistent_length(y_test, y_test_pred)

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("X_train {0} and X_test {1} have different number "
                         "of features.".format(X_train.shape, X_test.shape))

    return X_train, y_train, X_test, y_test, y_train_pred, y_test_pred


def evaluate_print(clf_name, y, y_pred):
    """Utility function for evaluating and printing the results for examples.
    Default metrics include ROC and Precision @ n

    Parameters
    ----------
    clf_name : str
        The name of the detector.

    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).

    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.

    """

    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)
    check_consistent_length(y, y_pred)

    print('{clf_name} ROC:{roc}, precision @ rank n:{prn}'.format(
        clf_name=clf_name,
        roc=np.round(roc_auc_score(y, y_pred), decimals=4),
        prn=np.round(precision_n_scores(y, y_pred), decimals=4)))


def generate_data_clusters(n_train=1000, n_test=500, n_clusters=2,
                           n_features=2, contamination=0.1, size='same',
                           density='same', dist=0.25, random_state=None,
                           return_in_clusters=False):
    """Utility function to generate synthesized data in clusters.
       Generated data can involve the low density pattern problem and global
       outliers which are considered as difficult tasks for outliers detection
       algorithms.

    Parameters
    ----------
    n_train : int, (default=1000)
        The number of training points to generate.

    n_test : int, (default=500)
        The number of test points to generate.

    n_clusters : int, optional (default=2)
       The number of centers (i.e. clusters) to generate.

    n_features : int, optional (default=2)
       The number of features for each sample.

    contamination : float in (0., 0.5), optional (default=0.1)
       The amount of contamination of the data set, i.e.
       the proportion of outliers in the data set.

    size : str, optional (default='same')
       Size of each cluster: 'same' generates clusters with same size,
       'different' generate clusters with different sizes.

    density : str, optional (default='same')
       Density of each cluster: 'same' generates clusters with same density,
       'different' generate clusters with different densities.

    dist: float, optional (default=0.25)
       Distance between clusters. Should be between 0. and 1.0
       It is used to avoid clusters overlapping as much as possible.
       However, if number of samples and number of clusters are too high,
       it is unlikely to separate them fully even if ``dist`` set to 1.0

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    return_in_clusters : bool, optional (default=False)
        If True, the function returns x_train, y_train, x_test, y_test each as
        a list of numpy arrays where each index represents a cluster.
        If False, it returns x_train, y_train, x_test, y_test each as numpy
        array after joining the sequence of clusters arrays,

    Returns
    -------
    X_train : numpy array of shape (n_train, n_features)
        Training data.

    y_train : numpy array of shape (n_train,)
        Training ground truth.

    X_test : numpy array of shape (n_test, n_features)
        Test data.

    y_test : numpy array of shape (n_test,)
        Test ground truth.
    """
    # initialize a random state and seeds for the instance
    random_state = check_random_state(random_state)

    if isinstance(n_clusters, int):
        check_parameter(n_clusters, low=1, param_name='n_clusters')
    else:
        raise ValueError("n_clusters should be int, got %s" % n_clusters)

    if isinstance(n_features, int):
        check_parameter(n_features, low=1, param_name='n_features')
    else:
        raise ValueError("n_features should be int, got %s" % n_features)

    if isinstance(contamination, (float, int)):
        check_parameter(contamination, low=0, high=0.5,
                        param_name='contamination')
    else:
        raise ValueError(
            "contamination should be float, got %s" % contamination)

    if isinstance(dist, float):
        check_parameter(dist, low=0, high=1.0, param_name='dist')
    else:
        raise ValueError("dist should be float, got %s" % dist)

    if not isinstance(return_in_clusters, bool):
        raise ValueError("return_in_clusters should be of type bool, "
                         "got %s" % return_in_clusters)

    # find the required number of outliers and inliers
    n_samples = n_train + n_test
    n_outliers = int(n_samples * contamination)
    n_inliers = n_samples - n_outliers

    if size == 'same':
        a_ = [int(n_inliers / n_clusters)] * (n_clusters - 1)
        clusters_size = a_ + [int(n_inliers - sum(a_))]
    elif size == 'different':
        if (n_clusters * 10) > n_samples:
            raise ValueError('number of samples should be at least 10 times of'
                             'the number of clusters')
        if (n_clusters * 10) > n_inliers:
            raise ValueError('contamination ratio is too high, try to increase'
                             ' number of samples or decrease the contamination')
        _r = 1. / n_clusters
        _offset = random_state.uniform(_r * 0.2, _r * 0.4,
                                       size=(int(n_clusters / 2),)).tolist()
        _offset += [i * -1. for i in _offset]
        clusters_size = np.round(
            np.multiply(n_inliers, np.add(_r, _offset))).astype(int)
        if n_clusters % 2 == 0:  # if it is even number
            clusters_size[n_clusters - 1] += n_inliers - sum(clusters_size)
        else:
            clusters_size = np.append(clusters_size,
                                      n_inliers - sum(clusters_size))
    else:
        raise ValueError(
            'size should be a string of value \'same\' or \'different\'')

    # check for clusters densities and apply split accordingly
    if density == 'same':
        clusters_density = random_state.uniform(low=0.1, high=0.5, size=(
            1,)).tolist() * n_clusters
    elif density == 'different':
        clusters_density = random_state.uniform(low=0.1, high=0.5,
                                                size=(n_clusters,))
    else:
        raise ValueError(
            'density should be a string of value \'same\' or \'different\'')

    # calculate number of outliers for every cluster
    n_outliers_ = []
    for i in range(n_clusters):
        n_outliers_.append(int(round(clusters_size[i] * contamination)))
    _diff = int((n_outliers - sum(n_outliers_)) / n_clusters)
    for i in range(n_clusters - 1):
        n_outliers_[i] += _diff
    n_outliers_[n_clusters - 1] += n_outliers - sum(n_outliers_)
    random_state.shuffle(n_outliers_)

    # generate data
    X_clusters, y_clusters = [], []
    X, y = np.zeros([n_samples, n_features]), np.zeros([n_samples, ])

    center_box = list(filter(lambda a: a != 0, np.linspace(
        -np.power(n_samples * n_clusters, dist),
        np.power(n_samples * n_clusters, dist),
        n_clusters + 2)))

    # index tracker for value assignment
    tracker_idx = 0

    for i in range(n_clusters):
        inliers, outliers = [], []
        _blob, _y = make_blobs(n_samples=clusters_size[i], centers=1,
                               cluster_std=clusters_density[i],
                               center_box=(center_box[i], center_box[i + 1]),
                               n_features=n_features,
                               random_state=random_state)

        inliers.append(_blob)

        center_box_l = center_box[i] * (1.2 + dist + clusters_density[i])
        center_box_r = center_box[i + 1] * (1.2 + dist + clusters_density[i])

        outliers.append(make_blobs(n_samples=n_outliers_[i], centers=1,
                                   cluster_std=random_state.uniform(
                                       clusters_density[i] * 3.5,
                                       clusters_density[i] * 4.,
                                       size=(1,)[0]),
                                   center_box=(center_box_l, center_box_r),
                                   n_features=n_features,
                                   random_state=random_state)[0])
        _y = np.append(_y, [1] * int(n_outliers_[i]))

        # generate X
        if np.array(outliers).ravel().shape[0] > 0:
            stacked_X_temp = np.vstack(
                (np.concatenate(inliers), np.concatenate(outliers)))
            X_clusters.append(stacked_X_temp)
            tracker_idx_new = tracker_idx + stacked_X_temp.shape[0]
            X[tracker_idx:tracker_idx_new, :] = stacked_X_temp
        else:
            X_clusters.append(np.concatenate(inliers))

        # generate Y
        y_clusters.append(_y)
        y[tracker_idx:tracker_idx_new, ] = _y

        tracker_idx = tracker_idx_new

    if return_in_clusters:
        return X_clusters, y_clusters

    # return X_train, X_test, y_train, y_test
    else:
        return train_test_split(X, y, test_size=n_test,
                                random_state=random_state)


def generate_data_categorical(n_train=1000, n_test=500, n_features=2,
                              n_informative=2, n_category_in=2,
                              n_category_out=2, contamination=0.1,
                              shuffle=True, random_state=None):
    """Utility function to generate synthesized categorical data.

    Parameters
    ----------
    n_train : int, (default=1000)
        The number of training points to generate.

    n_test : int, (default=500)
        The number of test points to generate.

    n_features : int, optional (default=2)
       The number of features for each sample.

    n_informative : int in (1, n_features), optional (default=2)
       The number of informative features in the outlier points.
       The higher the easier the outlier detection should be.
       Note that n_informative should not be less than or
       equal n_features.

    n_category_in : int in (1, n_inliers), optional (default=2)
       The number of categories in the inlier points.

    n_category_out : int in (1, n_outliers), optional (default=2)
       The number of categories in the outlier points.

    contamination : float in (0., 0.5), optional (default=0.1)
       The amount of contamination of the data set, i.e.
       the proportion of outliers in the data set.

    shuffle: bool, optional(default=True)
        If True, inliers will be shuffled which makes more noisy distribution.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.


    Returns
    -------
    X_train : numpy array of shape (n_train, n_features)
        Training data.

    y_train : numpy array of shape (n_train,)
        Training ground truth.

    X_test : numpy array of shape (n_test, n_features)
        Test data.

    y_test : numpy array of shape (n_test,)
        Test ground truth.
    """

    # initialize a random state and seeds for the instance
    random_state = check_random_state(random_state)

    if isinstance(n_train, int):
        check_parameter(n_train, low=1, param_name='n_train')
    else:
        raise ValueError("n_train should be int, got %s" % n_train)

    if isinstance(n_test, int):
        check_parameter(n_test, low=0, param_name='n_test')
    else:
        raise ValueError("n_test should be int, got %s" % n_test)

    if isinstance(n_features, int):
        check_parameter(n_features, low=0, param_name='n_features')
    else:
        raise ValueError("n_features should be int, got %s" % n_features)

    if isinstance(n_informative, int):
        check_parameter(n_informative, low=0, high=n_features + 1, param_name='n_informative')
    else:
        raise ValueError("n_informative should be int, got %s" % n_informative)

    if isinstance(contamination, (float, int)):
        check_parameter(contamination, low=0, high=0.5,
                        param_name='contamination')
    else:
        raise ValueError("contamination should be float, got %s" % contamination)

    if not isinstance(shuffle, bool):
        raise ValueError("shuffle should be bool, got %s" % shuffle)

    # find the required number of outliers and inliers
    n_samples = n_train + n_test
    n_outliers = int(n_samples * contamination)
    n_inliers = n_samples - n_outliers

    if isinstance(n_category_in, int):
        check_parameter(n_category_in, low=0, high=n_inliers + 1, param_name='n_category_in')
    else:
        raise ValueError("n_category_in should be int, got %s" % n_category_in)

    if isinstance(n_category_out, int):
        check_parameter(n_category_out, low=0, high=n_outliers + 1, param_name='n_category_out')
    else:
        raise ValueError("n_category_out should be int, got %s" % n_category_out)

    # Encapsulated functions to generate features
    def __f(f):
        quot, rem = divmod(f - 1, 26)
        return __f(quot) + chr(rem + ord('A')) if f != 0 else ''

    # generate pool of features to be the base for naming the data points
    features = []
    for i in range(1, n_features + 1):
        features.append(__f(i))

    # find the required distributions of categories over inliers and outliers
    temp_ = [int(n_inliers / n_category_in)] * (n_category_in - 1)
    dist_in = temp_ + [int(n_inliers - sum(temp_))]
    temp_ = [int(n_outliers / n_category_out)] * (n_category_out - 1)
    dist_out = temp_ + [int(n_outliers - sum(temp_))]

    # generate categorical data
    X = []
    count = 0
    for f in features:
        inliers = np.hstack([[f + str(i)] * dist_in[i] for i in range(n_category_in)])
        if shuffle:
            random_state.shuffle(inliers)
        if count < n_informative:
            outliers = list(np.hstack(
                [[f + str((n_category_in * 2) + i)] * dist_out[i] for i in range(n_category_out)]))
        else:
            outliers = list(inliers[random_state.randint(0, len(inliers), size=n_outliers)])
        count += 1

        X.append(list(inliers) + outliers)

    return train_test_split(np.array(X).T,
                            np.array(([0] * n_inliers) + ([1] * n_outliers)),
                            test_size=n_test,
                            random_state=random_state)
