import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# Load dataset from local CSV files
train_data = pd.read_csv('fashion-mnist_train.csv')
test_data = pd.read_csv('fashion-mnist_test.csv')

# Extract labels (Y) and pixel values (X)
Y_train = train_data['label'].values
X_train = train_data.drop('label', axis=1).values

Y_test = test_data['label'].values
X_test = test_data.drop('label', axis=1).values

# Reshape from flat 784 pixels to 28x28 images to match your Flatten layer
X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(64, activation = 'tanh'),
    layers.Dense(10, activation = 'softmax')
])
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

epochs = 20
training = model.fit(X_train, Y_train, epochs=epochs, validation_split=0.2, batch_size=32)

test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
print(f"\n Final Test accuracy: {test_acc*100}%")
print(f"Final Test Loss: {test_loss:.4f}")

plt.figure(figsize=(12, 5))

# Accuracy Curve
plt.subplot(1, 2, 1)
plt.plot(training.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
plt.plot(training.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

# Loss Curve
plt.subplot(1, 2, 2)
plt.plot(training.history['loss'], label='Training Loss', color='red', linewidth=2)
plt.plot(training.history['val_loss'], label='Validation Loss', color='green', linewidth=2)
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()