import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv(r'E:\Complete_Project2\DP\training folder\WHOLE_DATA\fingernail\metadataFingernail.csv')


image_size = (128, 128)
image_folder = r'E:\Complete_Project2\DP\training folder\WHOLE_DATA\fingernail\DATA'


def load_images(data, image_folder, image_size):
    images = []
    labels = []
    for index, row in data.iterrows():
        img_path = os.path.join(image_folder, row['F_name'])
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.resize(image, image_size)
            image = image / 255.0  
            images.append(image)
            labels.append(row['anemic'])
    return np.array(images), np.array(labels)

images, labels = load_images(data, image_folder, image_size)


X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)


input_shape = (image_size[0], image_size[1], 3)


base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
base_model.trainable = False  # Freeze the base model


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)  # Binary classification


model = Model(inputs=base_model.input, outputs=output)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))


model.predict(X_val)


model.save(filepath=r"E:\Complete_Project2\DP\training folder\fingernail_densenet.h5")

train_loss, train_accuracy = model.evaluate(X_train, y_train)
print(f"\nTraining Loss: {train_loss:.4f}")
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

# ✅ Evaluate model on test data
test_loss, test_accuracy = model.evaluate(X_val, y_val)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

y_pred_probs = model.predict(X_val)
y_pred = (y_pred_probs > 0.5).astype(int)

# ✅ Classification report & confusion matrix
print("\nClassification Report:\n", classification_report(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))