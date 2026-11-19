import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

print("Loading local MNIST dataset...")

# Load dataset from local CSV files
train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

# Extract labels (y)
y_train = train_data['label'].values
y_test = test_data['label'].values

# Extract features (X), reshape to 28x28x1 for the CNN, and normalize
# The '1' adds the channel dimension required by Conv2D layers
X_train = train_data.drop('label', axis=1).values.reshape(-1, 28, 28, 1) / 255.0
X_test = test_data.drop('label', axis=1).values.reshape(-1, 28, 28, 1) / 255.0

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# --- Basic Data Augmentation (Safe for digits) ---
data_augmentation = models.Sequential([
    layers.RandomRotation(0.1, input_shape=(28, 28, 1)),
    layers.RandomZoom(0.1)
], name="data_augmentation")

# --- 1. Build the CNN Model ---
cnn_model = models.Sequential([
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
], name="CNN_Model")
cnn_model.summary()

cnn_model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

# --- 2. Build the MLP (Fully Connected) Model ---
mlp_model = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)), # Flatten the 2D images
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
], name="MLP_Model")
mlp_model.summary()

mlp_model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

epochs = 10
batch_size = 128

print("\n--- Training MLP Model ---")
history_mlp = mlp_model.fit(X_train, y_train, epochs=epochs, 
                            validation_data=(X_test, y_test), 
                            batch_size=batch_size, verbose=1)

print("\n--- Training CNN Model (with Augmentation) ---")
history_cnn = cnn_model.fit(X_train, y_train, epochs=epochs, 
                            validation_data=(X_test, y_test), 
                            batch_size=batch_size, verbose=1)

plt.figure(figsize=(14, 5))

# Plot Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_mlp.history['val_accuracy'], label='MLP Val Accuracy', linestyle='--', color='red')
plt.plot(history_cnn.history['val_accuracy'], label='CNN Val Accuracy', color='blue', linewidth=2)
plt.title('Validation Accuracy: CNN vs MLP')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# Plot Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history_mlp.history['val_loss'], label='MLP Val Loss', linestyle='--', color='red')
plt.plot(history_cnn.history['val_loss'], label='CNN Val Loss', color='blue', linewidth=2)
plt.title('Validation Loss: CNN vs MLP')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()