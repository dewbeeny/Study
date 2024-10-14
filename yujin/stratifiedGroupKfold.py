import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from typing import List, Tuple

def stratifiedGroupKFold(x: np.ndarray, y: np.ndarray, group: np.ndarray, n_splits: int = 10) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    
    fold_x_train = []
    fold_y_train = []
    fold_g_train = []
    fold_x_test = []
    fold_y_test = []
    fold_g_test = []

    for (train_index, test_index) in sgkf.split(x, y, group):
        fold_x_train.append(x[train_index])
        fold_y_train.append(y[train_index])
        fold_g_train.append(group[train_index])
        fold_x_test.append(x[test_index])
        fold_y_test.append(y[test_index])
        fold_g_test.append(group[test_index])

    return fold_x_train, fold_y_train, fold_g_train, fold_x_test, fold_y_test, fold_g_test


"""
    * 본 연구에서 group = subject다.
    * fold별 train/test dataset, label, group 식별번호를 return한다.

    parameters
    x: 전체 dataset. chunk를 요소로 가짐
    y: x의 chunk별 label
    group: x의 chunk별 group 식별번호
    n_splits: fold의 개수

    returns
    fold_x_train
    ex) fold_x_train[0][1] = fold 0의 train datasest[1]
    fold_y_train
    ex) fold_y_train[0][1] = fold 0의 train datasest[1]의 label
    fold_g_train
    ex) fold_g_train[0][1] = fold 0의 train datasest[1]의 group 식별번호
    fold_x_test
    ex) fold_x_test[0][1] = fold 0의 test datasest[1]
    fold_y_test
    ex) fold_y_test[0][1] = fold 0의 test datasest[1]의 label
    fold_g_test
    ex) fold_g_test[0][1] = fold 0의 test datasest[1]의 group 식별번호
    """
