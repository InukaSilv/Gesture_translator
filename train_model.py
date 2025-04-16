import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def load_images_from_folders(base_dir):
    data = []
    labels = []
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                img = cv2.imread(file_path)
                if img is not None:
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
        if img is not None:
            img = cv2.resize(img, (64, 64))
            data.append(img)
            filenames.append(file)
    return np.array(data), filenames

# Load data
train_dir = "data/asl_dataset/train"
X_train, y_train = load_images_from_folders(train_dir)
X_train = X_train / 255.0

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

test_dir = "data/asl_dataset/test"
X_test, test_filenames = load_loose_images(test_dir)
X_test = X_test / 255.0

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),  # Randomly disable 25% of neurons
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # 50% dropout before final layer
    Dense(26, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train with validation
history = model.fit(X_train, y_train, epochs=10)

# Save and evaluate
model.save('asl_model.keras')

try:
    y_test = [filename.split('_')[0] for filename in test_filenames]
    y_test = label_encoder.transform(y_test)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
except Exception as e:
    print("\nCould not evaluate test accuracy. Reason:", str(e))
    print("Test filenames sample:", test_filenames[:5])

# Plot class distribution
plt.bar(label_encoder.classes_, np.bincount(y_train))
plt.title("Training Class Distribution")
plt.xticks(rotation=45)
plt.show()