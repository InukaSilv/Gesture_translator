import numpy as np
import os
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_images_from_folders(base_dir):
    data = []
    labels = []
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                img = cv2.imread(file_path)
                img = cv2.resize(img, (64, 64))
                data.append(img)
                labels.append(folder)
    return np.array(data), np.array(labels)


def load_loose_images(base_dir):
    data = []
    filenames = []
    for file in os.listdir(base_dir):
        file_path = os.path.join(base_dir, file)
        img = cv2.imread(file_path)
        img = cv2.resize(img, (64, 64))
        data.append(img)
        filenames.append(file)
    return np.array(data), filenames

train_dir = "data/asl_dataset/train"
X_train, y_train = load_images_from_folders(train_dir)

test_dir = "data/asl_dataset/test"
X_test, test_filenames = load_loose_images(test_dir)

X_train = X_train / 255.0
X_test = X_test / 255.0

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'), 
    Dense(26, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)

model.save('asl_model.h5')

print("Test filenames:", test_filenames)