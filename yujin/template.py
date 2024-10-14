import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from typing import List, Tuple

def main():
    #전처리
  
    #x = input tensor를 요소로 갖는 List
    #y = x의 label List
    #s = x의 subject List
    #아래 x, y, s에 대입한 값은 임의. 수정 가능
    x: np.ndarray = np.concatenate((x_ADHD, x_HC), axis=0) 
    y: np.ndarray = np.concatenate((y_ADHD, y_HC), axis=0) 
    s: np.ndarray = np.concatenate((s_ADHD, s_HC), axis=0)
    
    sgkf = StratifiedGroupKFold(n_splits=10)
    for (train_index, test_index) in sgkf.split(x, y, s):
        x_train = x[train_index]
        y_train = y[train_index]
        s_train = s[train_index]

        x_test = x[test_index]
        y_test = y[test_index]
        s_test = s[test_index]

        #모델 fit 및 evaluate
      
        #아래는 작성 예정
        #fold별 정확도, 그래프, 혼동행렬, t-sne 등 지표 출력
        #fold 종합 평균 지표 출력
        #할 수 있다면! 모델별로 오판한 subject도 출력해서 직접 수동 비교...
