import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD AND PREPROCESS DATA 
# ==========================================
print("Loading Fashion-MNIST dataset locally...")

# Load dataset from local CSV files
train_data = pd.read_csv('fashion-mnist_train.csv')
test_data = pd.read_csv('fashion-mnist_test.csv')

# Extract labels (y) and pixel values (X)
y_train = train_data['label'].values
X_train = train_data.drop('label', axis=1).values

y_test = test_data['label'].values
X_test = test_data.drop('label', axis=1).values

# Reshape from flat 784 pixels to 28x28 images to match the Flatten layer
X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# ==========================================
# 2. BUILD THE MODELS
# ==========================================

# --- Model 1: Baseline (No Regularization) ---
model_baseline = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model_baseline.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- Model 2: L2 Regularized ---
model_l2 = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(10, activation='softmax')
])
model_l2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- Model 3: Dropout Regularized ---
model_dropout = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])
model_dropout.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ==========================================
# 3. TRAIN THE MODELS
# ==========================================
epochs = 20
batch_size = 64

print("\n--- Training Baseline Model ---")
history_baseline = model_baseline.fit(X_train, y_train, epochs=epochs,
                                      validation_split=0.2, batch_size=batch_size, verbose=0)

print("\n--- Training L2 Regularized Model ---")
history_l2 = model_l2.fit(X_train, y_train, epochs=epochs,
                          validation_split=0.2, batch_size=batch_size, verbose=0)

print("\n--- Training Dropout Model ---")
history_dropout = model_dropout.fit(X_train, y_train, epochs=epochs,
                                    validation_split=0.2, batch_size=batch_size, verbose=0)

# ==========================================
# 4. VISUALIZE AND COMPARE PERFORMANCE
# ==========================================
plt.figure(figsize=(16, 6))

# --- Plot Validation Loss ---
plt.subplot(1, 2, 1)
plt.plot(history_baseline.history['val_loss'], label='Baseline (Overfitting)', linestyle='--', color='red')
plt.plot(history_l2.history['val_loss'], label='L2 Regularization', color='blue', linewidth=2)
plt.plot(history_dropout.history['val_loss'], label='Dropout (30%)', color='green', linewidth=2)
plt.title('Validation Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# --- Plot Validation Accuracy ---
plt.subplot(1, 2, 2)
plt.plot(history_baseline.history['val_accuracy'], label='Baseline', linestyle='--', color='red')
plt.plot(history_l2.history['val_accuracy'], label='L2 Regularization', color='blue', linewidth=2)
plt.plot(history_dropout.history['val_accuracy'], label='Dropout (30%)', color='green', linewidth=2)
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()