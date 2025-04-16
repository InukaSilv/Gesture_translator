import numpy as np
import os
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = []
labels = []

dataset_path = "data/asl_dataset"

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    for file in folder_path:
        file_path = os.path.join(folder_path, file)

        img = cv2.imread(file_path)
        img = cv2.resize(img, (64, 64))

        data.append(img)
        labels.append(folder)

data = np.array(data)
labels = np.array(labels)

data = data / 255.0