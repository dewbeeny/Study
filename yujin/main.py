import numpy as np

from keras import metrics

from sklearn.model_selection import StratifiedGroupKFold

from load_data import load_data 
from preprocess_data import preprocess_data 
from cnn import cnn

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

cnn = cnn((len(f), len(x[0][0]), 1))
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', metrics.Precision(), metrics.Recall(), metrics.AUC()])

i = 1
history_list = []
metric_list = []
sgkf = StratifiedGroupKFold(n_splits=10)
for (train_index, test_index) in sgkf.split(x, y, s):
    x_train = x[train_index]
    y_train = y[train_index]
    s_train = s[train_index]

    x_test = x[test_index]
    y_test = y[test_index]
    s_test = s[test_index]

    print(f"fold {i}")
    history = cnn.fit(x_train, y_train, epochs=100, batch_size=5, verbose=0)
    history_list.append(history.history)
    metric = cnn.evaluate(x_test, y_test, verbose=0)
    metric_list.append(metric)

    # 각 폴드의 평균 메트릭 출력
    print(f"Fold {i} Metrics:")
    print(f"Loss: {metric[0]}")
    print(f"Accuracy: {metric[1]}")
    print(f"Precision: {metric[2]}")
    print(f"Recall: {metric[3]}")
    print(f"AUC: {metric[4]}")
    print("\n")
    
    i += 1

# 평균 히스토리 계산
average_history = {}
for key in history_list[0].keys():
    average_history[key] = np.mean([history[key] for history in history_list], axis=0)

# 평균 매트릭 계산
average_metrics = np.mean(metric_list, axis=0)

# 출력
print("\nAverage History:")
for key, value in average_history.items():
    print(f"{key}: {value}")

print("\nAverage Metrics:")
print(f"Loss: {average_metrics[0]}")
print(f"Accuracy: {average_metrics[1]}")
print(f"Precision: {average_metrics[2]}")
print(f"Recall: {average_metrics[3]}")
print(f"AUC: {average_metrics[4]}")
