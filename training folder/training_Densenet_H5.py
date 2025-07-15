### PKL DENSENET stored 
'''
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Paths to trained models
model_paths = {
    "Conjunctiva": r"E:\Complete_Project2\DP\training folder\conjunctiva_densenet.h5",
    "Palm": r"E:\Complete_Project2\DP\training folder\palm_densenet.h5",
    "Fingernail": r"E:\Complete_Project2\DP\training folder\fingernail_densenet.h5"
}

# Paths to datasets
datasets = {
    "Conjunctiva": {
        "csv_path": r"E:\Complete_Project2\DP\training folder\WHOLE_DATA\conjunctiva\metadataConjuctiva.csv",
        "image_folder": r"E:\Complete_Project2\DP\training folder\WHOLE_DATA\conjunctiva\DATA",
    },
    "Fingernail": {
        "csv_path": r"E:\Complete_Project2\DP\training folder\WHOLE_DATA\fingernail\metadataFingernail.csv",
        "image_folder": r"E:\Complete_Project2\DP\training folder\WHOLE_DATA\fingernail\DATA",
    },
    "Palm": {
        "csv_path": r"E:\Complete_Project2\DP\training folder\WHOLE_DATA\palm\metadataPalm.csv",
        "image_folder": r"E:\Complete_Project2\DP\training folder\WHOLE_DATA\palm\DATA",
    }
}

# Image size and test split ratio
image_size = (128, 128)
test_size = 0.2
epochs = 10
batch_size = 32

# Function to load and preprocess images
def load_images(csv_path, image_folder, image_size):
    data = pd.read_csv(csv_path)
    images, labels = [], []
    for _, row in data.iterrows():
        img_path = os.path.join(image_folder, row['F_name'])
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.resize(image, image_size) / 255.0
            images.append(image)
            labels.append(row['anemic'])
    return np.array(images), np.array(labels)

# Dictionary to store training history
history_dict = {}

for model_name, model_path in model_paths.items():
    print(f"\nTraining and evaluating {model_name} model...")
    
    # Load dataset
    images, labels = load_images(datasets[model_name]["csv_path"], datasets[model_name]["image_folder"], image_size)
    
    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=test_size, random_state=42)
    
    # Load model
    model = load_model(model_path)
    
    # Train model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Save history
    history_dict[model_name] = history.history
    
    # Evaluate model
    y_pred_probs = model.predict(X_val, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int)
    conf_matrix = confusion_matrix(y_val, y_pred)
    
    # Plot Accuracy vs Epoch
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy", color="blue")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Training and Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Plot Loss vs Epoch
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss", color="blue")
    plt.plot(history.history["val_loss"], label="Validation Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Plot Confusion Matrix
    plt.figure(figsize=(5, 5))
    plt.imshow(conf_matrix, cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(conf_matrix[i, j]), ha="center", va="center", color="red")
    plt.show()

# Save history to pkl file
with open(r"E:\Complete_Project2\DP\training folder\history.pkl", "wb") as f:
    pickle.dump(history_dict, f)

print("Training histories saved to history.pkl")

'''

#### Resnet and VGG16 model saved in h5 file and history stored and pkl for comparision with desnet (only dfor Conjunctiva)

'''

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, ResNet50, VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv(r'E:\Complete_Project2\DP\training folder\WHOLE_DATA\conjunctiva\metadataConjuctiva.csv')

# Paths
image_size = (128, 128)
image_folder = r'E:\Complete_Project2\DP\training folder\WHOLE_DATA\conjunctiva\DATA'
densenet_path = r"E:\Complete_Project2\DP\training folder\conjunctiva_densenet.h5"

# Load DenseNet model
densenet_model = load_model(densenet_path)

# Load ResNet model
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x = GlobalAveragePooling2D()(resnet_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(1, activation='sigmoid')(x)
resnet_model = Model(inputs=resnet_model.input, outputs=x)
resnet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load VGG16 model
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x = GlobalAveragePooling2D()(vgg16_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(1, activation='sigmoid')(x)
vgg16_model = Model(inputs=vgg16_model.input, outputs=x)
vgg16_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load images
def load_images(data, image_folder, image_size):
    images, labels = [], []
    for _, row in data.iterrows():
        img_path = os.path.join(image_folder, row['F_name'])
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.resize(image, image_size) / 255.0
            images.append(image)
            labels.append(row['anemic'])
    return np.array(images), np.array(labels)

images, labels = load_images(data, image_folder, image_size)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# ✅ Train Deep Learning Models
history_densenet = densenet_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
history_resnet = resnet_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
history_vgg16 = vgg16_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

resnet_model.save(r"E:\Complete_Project2\DP\training folder\resnet_model.h5")
vgg16_model.save(r"E:\Complete_Project2\DP\training folder\vgg16_model.h5")

with open(r"E:\Complete_Project2\DP\training folder\resnethistory.pkl", "wb") as f:
    pickle.dump(history_resnet.history, f)

with open(r"E:\Complete_Project2\DP\training folder\VGG16_history.pkl", "wb") as f:
    pickle.dump(history_vgg16.history, f)

# ✅ Feature Extraction for Naïve Bayes
feature_extractor = Model(inputs=densenet_model.input, outputs=densenet_model.layers[-2].output)
X_train_features = feature_extractor.predict(X_train)
X_test_features = feature_extractor.predict(X_test)

# ✅ Evaluate Deep Learning Models
densenet_train_acc = densenet_model.evaluate(X_train, y_train)[1]
densenet_test_acc = densenet_model.evaluate(X_test, y_test)[1]

resnet_train_acc = resnet_model.evaluate(X_train, y_train)[1]
resnet_test_acc = resnet_model.evaluate(X_test, y_test)[1]

vgg16_train_acc = vgg16_model.evaluate(X_train, y_train)[1]
vgg16_test_acc = vgg16_model.evaluate(X_test, y_test)[1]

# ✅ Plot Grouped Line Chart for Accuracy Comparison
models = ["DenseNet121", "ResNet50", "VGG16", "Naïve Bayes"]
train_accuracies = [densenet_train_acc, resnet_train_acc, vgg16_train_acc]
test_accuracies = [densenet_test_acc, resnet_test_acc, vgg16_test_acc]

plt.figure(figsize=(10, 5))
plt.plot(models, train_accuracies, marker='o', linestyle='dashed', color='blue', label="Training Accuracy")
plt.plot(models, test_accuracies, marker='s', linestyle='solid', color='green', label="Testing Accuracy")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("Conjunctiva Training vs Testing Accuracy for Different Models")
plt.legend()
plt.grid(True)
plt.show()

# ✅ Line Graph for Training & Validation Accuracy Over Epochs
plt.figure(figsize=(10, 5))
plt.plot(history_densenet.history['accuracy'], label="DenseNet121 Training Accuracy", linestyle='dashed')
plt.plot(history_densenet.history['val_accuracy'], label="DenseNet121 Testing Accuracy")
plt.plot(history_resnet.history['accuracy'], label="ResNet50 Training Accuracy", linestyle='dashed')
plt.plot(history_resnet.history['val_accuracy'], label="ResNet50 Testing Accuracy")
plt.plot(history_vgg16.history['accuracy'], label="VGG16 Training Accuracy", linestyle='dashed')
plt.plot(history_vgg16.history['val_accuracy'], label="VGG16 Testing Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Conjunctiva Training & Testing Accuracy Over Epochs")
plt.grid(True)
plt.show()
'''

### it will take time to give graph so stored pkl file used to provide graph for comparision between models


'''
import pickle
import matplotlib.pyplot as plt

# Function to load history from a .pkl file
def load_history(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

# Load history data from .pkl files
resnet_history = load_history("E:\Complete_Project2\DP\training folder\resnethistory.pkl")
conjunctiva_history = load_history("E:\Complete_Project2\DP\training folder\history.pkl")
vgg_history = load_history("E:\Complete_Project2\DP\training folder\VGG16_history.pkl")

# Extract accuracy values
epochs = range(1, len(resnet_history["accuracy"]) + 1)

resnet_acc = resnet_history["accuracy"]
resnet_val_acc = resnet_history["val_accuracy"]

conjunctiva_acc = conjunctiva_history["Conjunctiva"]["accuracy"]
conjunctiva_val_acc = conjunctiva_history["Conjunctiva"]["val_accuracy"]

vgg_acc = vgg_history["accuracy"]
vgg_val_acc = vgg_history["val_accuracy"]

# Plot Training and Validation Accuracy in One Graph
plt.figure(figsize=(14, 8))  # Bigger figure for better readability

# ResNet
plt.plot(epochs, resnet_acc, "r-o", markersize=8, linewidth=2.5, label="ResNet Training Accuracy")
plt.plot(epochs, resnet_val_acc, "r--o", markersize=8, linewidth=2.5, label="ResNet Validation Accuracy")

# Conjunctiva
plt.plot(epochs, conjunctiva_acc, "g-s", markersize=8, linewidth=2.5, label="DenseNet-121 Training Accuracy")
plt.plot(epochs, conjunctiva_val_acc, "g--s", markersize=8, linewidth=2.5, label="DenseNet-121 Validation Accuracy")

# VGG16
plt.plot(epochs, vgg_acc, "b-^", markersize=8, linewidth=2.5, label="VGG16 Training Accuracy")
plt.plot(epochs, vgg_val_acc, "b--^", markersize=8, linewidth=2.5, label="VGG16 Validation Accuracy")

# Configure plot aesthetics
plt.xlabel("Epochs", fontsize=18, fontweight="bold")
plt.ylabel("Accuracy", fontsize=18, fontweight="bold")
plt.title("Training & Validation Accuracy Comparison", fontsize=20, fontweight="bold")

plt.xticks(fontsize=14)  # Increase x-axis font size
plt.yticks(fontsize=14)  # Increase y-axis font size
plt.legend(fontsize=14, loc="lower right")  # Better legend positioning
plt.grid(True, linestyle="--", linewidth=0.7)  # Subtle grid for readability

# Save the figure (for IEEE report)
plt.savefig("accuracy_comparison.png", dpi=300, bbox_inches="tight")

# Show the plot
plt.show()
'''


