import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Update these paths to where you extracted the CIFAR-10 image folders on your PC
train_dir = 'path/to/cifar10/train' 
test_dir = 'path/to/cifar10/test'

batch_size = 64
img_height = 32
img_width = 32

print("Loading local CIFAR-10 image dataset...")

# Load training data directly from the folder structure
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Load validation/test data directly from the folder structure
val_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# --- Data Augmentation (Safe for objects/animals) ---
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal", input_shape=(32, 32, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
], name="data_augmentation")

# --- 1. Build the CNN Model ---
cnn_model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./255), # Normalize RGB values to [0, 1]
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
], name="CNN_Model")
cnn_model.summary()

cnn_model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

# --- 2. Build the MLP (Fully Connected) Model ---
mlp_model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Rescaling(1./255), # Normalize RGB values to [0, 1]
    layers.Flatten(),         # Flatten the 32x32x3 image into 3072 features
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
], name="MLP_Model")
mlp_model.summary()

mlp_model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

epochs = 15

print("\n--- Training MLP Model ---")
history_mlp = mlp_model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=epochs, 
    verbose=1
)

print("\n--- Training CNN Model (with Augmentation) ---")
history_cnn = cnn_model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=epochs, 
    verbose=1
)

# --- 4. Visualize and Compare Performance ---
plt.figure(figsize=(14, 5))

# Plot Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_mlp.history['val_accuracy'], label='MLP Val Accuracy', linestyle='--', color='red')
plt.plot(history_cnn.history['val_accuracy'], label='CNN Val Accuracy', color='blue', linewidth=2)
plt.title('Validation Accuracy: CNN vs MLP on CIFAR-10')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# Plot Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history_mlp.history['val_loss'], label='MLP Val Loss', linestyle='--', color='red')
plt.plot(history_cnn.history['val_loss'], label='CNN Val Loss', color='blue', linewidth=2)
plt.title('Validation Loss: CNN vs MLP on CIFAR-10')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()