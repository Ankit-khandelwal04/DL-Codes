import os
import urllib.request
import zipfile
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt

# ==========================================
# 1. DOWNLOAD & EXTRACT
# ==========================================
print("Downloading Cats vs Dogs dataset into Colab...")
url = "https://download.mlcc.google.com/mledu-datasets/cats_and_dogs_filtered.zip"
zip_path = "/tmp/cats_and_dogs_filtered.zip"

# Download the zip file directly from Google's secure servers
urllib.request.urlretrieve(url, zip_path)

print("Extracting dataset...")
# Extract the zip file to the /tmp/ directory
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("/tmp/")

# Set the correct absolute paths for Colab
DATA_DIR = '/tmp/cats_and_dogs_filtered'
train_dir = os.path.join(DATA_DIR, 'train')
validation_dir = os.path.join(DATA_DIR, 'validation')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("\nLoading images from directory...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

# Normalize pixel values to 0-1
normalization_layer = layers.Rescaling(1./255)
train_batches = train_dataset.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(1)
val_batches = val_dataset.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(1)

# ==========================================
# 2. FEATURE EXTRACTION (FROZEN LAYERS)
# ==========================================
print("\n--- Building VGG16 Model ---")
# Load VGG16 without the final dense classification layers
vgg_base = VGG16(input_shape=(224, 224, 3),
                 include_top=False,
                 weights='imagenet')

# FREEZE the base model to protect ImageNet weights
vgg_base.trainable = False

# Attach custom classification head for Cats (0) vs Dogs (1)
model = models.Sequential([
    vgg_base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Training Frozen VGG16 (Feature Extraction)...")
history_frozen = model.fit(train_batches, epochs=5, validation_data=val_batches)

# ==========================================
# 3. EXPERIMENT: FINE-TUNING
# ==========================================
print("\n--- Fine-Tuning VGG16 ---")
# UNFREEZE the base model
vgg_base.trainable = True

# Freeze all layers up to layer 15 to only fine-tune the final convolution block
for layer in vgg_base.layers[:15]:
    layer.trainable = False

# Recompile with a VERY SMALL learning rate to preserve pre-trained weights
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Training Unfrozen VGG16 (Fine-Tuning)...")
history_finetuned = model.fit(train_batches, epochs=3, validation_data=val_batches)

# ==========================================
# 4. VISUALIZATION
# ==========================================
plt.figure(figsize=(10, 5))

# Combine the history of both training phases
total_val_acc = history_frozen.history['val_accuracy'] + history_finetuned.history['val_accuracy']
total_train_acc = history_frozen.history['accuracy'] + history_finetuned.history['accuracy']

plt.plot(total_train_acc, label='Training Accuracy', color='blue')
plt.plot(total_val_acc, label='Validation Accuracy', color='orange')
plt.axvline(x=4, color='black', linestyle='--', label='Unfrozen Here (Start Fine-Tuning)')

plt.title('VGG16 Transfer Learning: Feature Extraction to Fine-Tuning')
plt.xlabel('Total Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()