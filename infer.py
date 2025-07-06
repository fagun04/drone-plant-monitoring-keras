import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

CLASS_MAP = ['dry', 'healthy', 'overwatered']

def preprocess(img_path, size=(256,256)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype('float32') / 255.0

def infer(model_path, img_path):
    model = load_model(model_path)
    img = preprocess(img_path)
    prob = model.predict(np.expand_dims(img,0))[0]
    cls_idx = prob.argmax()
    cls_name = CLASS_MAP[cls_idx]
    print(f"Predicted: {cls_name} (confidence: {prob[cls_idx]*100:.2f}%)")
    return cls_name, prob

if __name__ == "__main__":
    import sys
    mp, ip = sys.argv[1], sys.argv[2]
    infer(mp, ip)
