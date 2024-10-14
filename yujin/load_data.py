import os
import numpy as np
import scipy.io as sio

def load_data(directory_path: str, label: int):
    x = []
    s = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.mat'):
            file_path = os.path.join(directory_path, file_name)
            subject = os.path.splitext(file_name)[0]

            mat_data = sio.loadmat(file_path)
            x.append(mat_data[subject])
            s.append(subject)
    return x, [label] * len(x), s