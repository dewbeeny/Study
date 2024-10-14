import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from typing import List, Tuple

import os
import numpy as np
import scipy.io as sio
import numpy as np
from scipy.signal import welch

from keras import metrics
from load_data import load_data
from preprocess_data import preprocess_data
from keras import models, layers
from model import model 

def main():
    #y = x의 label List
    #s = x의 subject List

    #x = time x chnnel의 eeg signal(2d ndarray)를 요소로 갖는 List
    x_ADHD, y_ADHD, s_ADHD = load_data("/home/unixuser/cdproject/public_data/ADHD", 1)
    x_HC, y_HC, s_HC = load_data("/home/unixuser/cdproject/public_data/HC", 0)

    #f = psd를 측정한 frequency의 1d ndarray
    #x = frequency x channel의 psd(2d ndarray)를 요소로 갖는 List
    f, x_ADHD, y_ADHD, s_ADHD = preprocess_data(x_ADHD, y_ADHD, s_ADHD)
    f, x_HC, y_HC, s_HC = preprocess_data(x_HC, y_HC, s_HC)

    #x = frequecy x channel의 psd(2d ndarray)를 요소로 갖는 List(ADHD + HC)
    x: np.ndarray = np.concatenate((x_ADHD, x_HC), axis=0) 
    y: np.ndarray = np.concatenate((y_ADHD, y_HC), axis=0) 
    s: np.ndarray = np.concatenate((s_ADHD, s_HC), axis=0)

    cnn = model((len(f), len(x[0][0]), 1))
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', metrics.Precision(), metrics.Recall(), metrics.AUC()])
    
    sgkf = StratifiedGroupKFold(n_splits=10)
    i = 1
    for (train_index, test_index) in sgkf.split(x, y, s):
        x_train = x[train_index]
        y_train = y[train_index]
        s_train = s[train_index]

        x_test = x[test_index]
        y_test = y[test_index]
        s_test = s[test_index]

        print(f"fold {i}")
        history = cnn.fit(x_train, y_train)
        for key in history.history.keys():
            print(f"{key}: {history.history[key]}")
        metric = cnn.evaluate(x_test, y_test)
        print(metric)
        print("\n")
        i += 1
        #아래는 작성 예정
        #fold별 정확도, 그래프, 혼동행렬, t-sne 등 지표 출력
        #fold 종합 평균 지표 출력
        #할 수 있다면! 모델별로 오판한 subject도 출력해서 직접 수동 비교...