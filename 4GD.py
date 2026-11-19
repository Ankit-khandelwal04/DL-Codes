import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

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

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Create tensors for the custom gradient tracker callback
x_val_sample = tf.convert_to_tensor(X_test[:500], dtype=tf.float32)
y_val_sample = tf.convert_to_tensor(y_test[:500], dtype=tf.int32)


class GradientTracker(tf.keras.callbacks.Callback):
    def __init__(self, x_sample, y_sample):
        super().__init__()
        self.x_sample = x_sample
        self.y_sample = y_sample
        self.grad_norms = []

    def on_epoch_end(self, epoch, logs=None):
        # Record the gradients of the loss with respect to the first hidden layer
        with tf.GradientTape() as tape:
            predictions = self.model(self.x_sample)
            # Use sparse categorical crossentropy manually for the tape
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
            loss = loss_fn(self.y_sample, predictions)
            
        gradients = tape.gradient(loss, self.model.trainable_weights)
        # gradients[0] contains the weights of the first Dense layer
        avg_grad = np.mean(np.abs(gradients[0].numpy()))
        self.grad_norms.append(avg_grad)


# --- Model 1: Sigmoid ---
model_sigmoid = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='sigmoid'),
    layers.Dense(64, activation='sigmoid'),
    layers.Dense(10, activation='softmax')
])
model_sigmoid.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- Model 2: Tanh ---
model_tanh = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='tanh'),
    layers.Dense(64, activation='tanh'),
    layers.Dense(10, activation='softmax')
])
model_tanh.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- Model 3: ReLU ---
model_relu = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model_relu.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


epochs = 15
batch_size = 64

# Create trackers
tracker_sigmoid = GradientTracker(x_val_sample, y_val_sample)
tracker_tanh = GradientTracker(x_val_sample, y_val_sample)
tracker_relu = GradientTracker(x_val_sample, y_val_sample)

print("\n--- Training Sigmoid Model ---")
hist_sigmoid = model_sigmoid.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                                 validation_data=(X_test, y_test), callbacks=[tracker_sigmoid], verbose=0)

print("\n--- Training Tanh Model ---")
hist_tanh = model_tanh.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                           validation_data=(X_test, y_test), callbacks=[tracker_tanh], verbose=0)

print("\n--- Training ReLU Model ---")
hist_relu = model_relu.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                           validation_data=(X_test, y_test), callbacks=[tracker_relu], verbose=0)


plt.figure(figsize=(18, 5))

# Plot 1: Training Speed (Loss Curve)
plt.subplot(1, 3, 1)
plt.plot(hist_sigmoid.history['loss'], label='Sigmoid', color='red')
plt.plot(hist_tanh.history['loss'], label='Tanh', color='blue')
plt.plot(hist_relu.history['loss'], label='ReLU', color='green')
plt.title('Training Loss (Speed of Convergence)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Plot 2: Validation Accuracy
plt.subplot(1, 3, 2)
plt.plot(hist_sigmoid.history['val_accuracy'], label='Sigmoid', color='red')
plt.plot(hist_tanh.history['val_accuracy'], label='Tanh', color='blue')
plt.plot(hist_relu.history['val_accuracy'], label='ReLU', color='green')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Plot 3: Gradient Flow
plt.subplot(1, 3, 3)
plt.plot(tracker_sigmoid.grad_norms, label='Sigmoid', color='red')
plt.plot(tracker_tanh.grad_norms, label='Tanh', color='blue')
plt.plot(tracker_relu.grad_norms, label='ReLU', color='green')
plt.title('Gradient Flow (Avg Grad Size in Layer 1)')
plt.xlabel('Epochs')
plt.ylabel('Average Absolute Gradient')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()