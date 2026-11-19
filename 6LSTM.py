import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
 
print("Loading local IMDB dataset...")

# Load dataset from local CSV file (Update the filename if yours is different)
df = pd.read_csv('IMDB Dataset.csv')

# Convert string labels ('positive', 'negative') to binary (1, 0)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Extract text and labels
reviews = df['review'].values
labels = df['sentiment'].values

# Split into 80% training and 20% testing
X_train_text, X_test_text, y_train, y_test = train_test_split(
    reviews, labels, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train_text)}")
print(f"Testing samples: {len(X_test_text)}")

# Hyperparameters for preprocessing
vocab_size = 10000  # Only keep the top 10,000 most frequent words
max_length = 200    # Cut reviews after 200 words or pad if shorter

# Initialize and fit the tokenizer on training text
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

# Convert raw text into integer sequences
X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

# Pad sequences to ensure uniform input size for the LSTM
X_train = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")

# Build the LSTM model
model = models.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length),
    layers.LSTM(64, return_sequences=False),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid') # Binary classification output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
epochs = 5
batch_size = 64

print("\n--- Training LSTM Model ---")
history = model.fit(
    X_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=(X_test, y_test),
    verbose=1
)

# --- 1. Plot Accuracy and Loss Curves ---
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

# --- 2. Visualize Example Predictions ---
print("\n--- Example Predictions ---")

# Get model predictions on a few test samples
predictions = model.predict(X_test[:5])

# Reverse dictionary to convert tokens back to words for readability
reverse_word_index = dict([(value, key) for (key, value) in tokenizer.word_index.items()])

def decode_review(encoded_text):
    return ' '.join([reverse_word_index.get(i, '?') for i in encoded_text if i != 0])

# Print out 5 test samples alongside their predictions
for i in range(5):
    text = decode_review(X_test[i])
    true_label = "Positive" if y_test[i] == 1 else "Negative"
    predicted_label = "Positive" if predictions[i][0] > 0.5 else "Negative"
    confidence = predictions[i][0] if predicted_label == "Positive" else 1 - predictions[i][0]
    
    print(f"\nReview {i+1}: {text[:150]}...") # Print first 150 characters
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {predicted_label} (Confidence: {confidence:.2f})")