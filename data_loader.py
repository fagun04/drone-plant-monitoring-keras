import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

CLASS_MAP = {'dry':0, 'healthy':1, 'overwatered':2}

def load_data(data_dir, img_size=(256,256)):
    imgs, labels = [], []
    for cls, idx in CLASS_MAP.items():
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            path = os.path.join(cls_dir, fname)
            img = cv2.imread(path)
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
            labels.append(idx)
    X = np.array(imgs, dtype='float32') / 255.0
    y = to_categorical(labels, num_classes=len(CLASS_MAP))
    return X, y
