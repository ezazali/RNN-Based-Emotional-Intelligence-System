import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pickle
import json
import os
from datetime import datetime

 
try:
    from IPython.display import FileLink
    from google.colab import files
    COLAB_ENV = True
except ImportError:
    COLAB_ENV = False

print("TensorFlow version:", tf.__version__)
print("Environment: Colab" if COLAB_ENV else "Local/Kaggle")

 
try:
    df = pd.read_csv("/kaggle/input/emotions-in-text/Emotion_final.csv")
    print(f"Loaded dataset with shape: {df.shape}")
except:
    print("Creating sample dataset...")
    # Create sample data for testing
    texts = [
        "I am so happy today", "This is amazing", "I love this so much",
        "I am sad and upset", "This makes me cry", "I feel terrible",
        "I am angry about this", "This is frustrating me", "I hate this",
        "I am scared of this", "This is frightening", "I feel anxious",
        "What a surprise this is", "I didn't expect this", "That's shocking"
    ] * 200  # Create larger dataset
    
    emotions = ["happy", "happy", "happy", "sad", "sad", "sad",
               "angry", "angry", "angry", "fear", "fear", "fear",
               "surprise", "surprise", "surprise"] * 200
    
    df = pd.DataFrame({'Text': texts, 'Emotion': emotions})

print(f"Dataset shape: {df.shape}")
print("Sample data:")
print(df.head())
print("\nEmotion distribution:")
print(df['Emotion'].value_counts())

# ----------------------------
# STEP 2: Define Parameters FIRST
# ----------------------------
MAX_WORDS = 5000      # Vocabulary size
MAX_LEN = 50          # Maximum sequence length  
EMBEDDING_DIM = 100   # Embedding dimensions

print(f"\nModel parameters:")
print(f"MAX_WORDS: {MAX_WORDS}")
print(f"MAX_LEN: {MAX_LEN}")
print(f"EMBEDDING_DIM: {EMBEDDING_DIM}")

# ----------------------------
# STEP 3: Preprocess Data
# ----------------------------
# Clean text (basic cleaning)
df['Text'] = df['Text'].str.lower()
df['Text'] = df['Text'].str.replace('[^a-zA-Z0-9\s]', '', regex=True)

# Encode labels
label_encoder = LabelEncoder()
df['Emotion_encoded'] = label_encoder.fit_transform(df['Emotion'])
NUM_CLASSES = len(label_encoder.classes_)

print(f"Number of classes: {NUM_CLASSES}")
print(f"Classes: {label_encoder.classes_}")

# Tokenize and pad
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df['Text'])

X = tokenizer.texts_to_sequences(df['Text'])
X = pad_sequences(X, maxlen=MAX_LEN)

# Convert labels to categorical
y = to_categorical(df['Emotion_encoded'], num_classes=NUM_CLASSES)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# ----------------------------
# STEP 4: Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['Emotion_encoded']
)

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")

# ----------------------------
# STEP 5: Create Model (Simple & Working)
# ----------------------------
print("\n" + "="*50)
print("CREATING MODEL...")
print("="*50)

model = Sequential([
    Embedding(input_dim=MAX_WORDS, 
             output_dim=EMBEDDING_DIM, 
             input_length=MAX_LEN),
    
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dropout(0.3),
    
    Dense(NUM_CLASSES, activation='softmax')
])

# ----------------------------
# STEP 6: Compile Model
# ----------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled successfully!")

# ----------------------------
# STEP 7: Build Model and Show Summary
# ----------------------------
print("\n" + "="*50)
print("BUILDING MODEL...")
print("="*50)

# Build the model explicitly
model.build(input_shape=(None, MAX_LEN))

print("✅ Model built successfully!")

# Show summary
print("\nMODEL SUMMARY:")
print("="*30)
model.summary()

# Count parameters
total_params = model.count_params()
print(f"\n✅ Total parameters: {total_params:,}")

# ----------------------------
# STEP 8: Training with Model Saving
# ----------------------------
print("\n" + "="*50)
print("TRAINING MODEL...")
print("="*50)

try:
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    print("✅ Training completed successfully!")
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"❌ Training failed: {str(e)}")
    print("Check your data and model configuration")
    test_acc = 0.0
    history = None

# ----------------------------
# STEP 9: Save Model and Components
# ----------------------------
print("\n" + "="*50)
print("SAVING MODEL AND COMPONENTS...")
print("="*50)

# Create timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"emotion_rnn_model_{timestamp}"

try:
    # 1. Save the trained model
    model_filename = f"{model_name}.h5"
    model.save(model_filename)
    print(f"✅ Model saved as: {model_filename}")
    
    # 2. Save the tokenizer
    tokenizer_filename = f"tokenizer_{timestamp}.pkl"
    with open(tokenizer_filename, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"✅ Tokenizer saved as: {tokenizer_filename}")
    
    # 3. Save the label encoder
    label_encoder_filename = f"label_encoder_{timestamp}.pkl"
    with open(label_encoder_filename, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✅ Label encoder saved as: {label_encoder_filename}")
    
    # 4. Save model configuration and training history
    config_data = {
        'model_parameters': {
            'MAX_WORDS': MAX_WORDS,
            'MAX_LEN': MAX_LEN,
            'EMBEDDING_DIM': EMBEDDING_DIM,
            'NUM_CLASSES': NUM_CLASSES
        },
        'training_results': {
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss) if 'test_loss' in locals() else None,
            'total_parameters': int(total_params),
            'epochs_trained': 10
        },
        'classes': label_encoder.classes_.tolist(),
        'training_history': {
            'accuracy': [float(x) for x in history.history['accuracy']] if history else [],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']] if history else [],
            'loss': [float(x) for x in history.history['loss']] if history else [],
            'val_loss': [float(x) for x in history.history['val_loss']] if history else []
        } if history else {}
    }
    
    config_filename = f"model_config_{timestamp}.json"
    with open(config_filename, 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"✅ Configuration saved as: {config_filename}")
    
    # 5. Create a prediction script
    prediction_script = f'''
import tensorflow as tf
import pickle
import json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Load the saved components
print("Loading model and components...")

# Load model
model = tf.keras.models.load_model('{model_filename}')
print("✅ Model loaded")

# Load tokenizer
with open('{tokenizer_filename}', 'rb') as f:
    tokenizer = pickle.load(f)
print("✅ Tokenizer loaded")

# Load label encoder  
with open('{label_encoder_filename}', 'rb') as f:
    label_encoder = pickle.load(f)
print("✅ Label encoder loaded")

# Load configuration
with open('{config_filename}', 'r') as f:
    config = json.load(f)

MAX_LEN = config['model_parameters']['MAX_LEN']
classes = config['classes']

print(f"Model ready! Classes: {{classes}}")

def preprocess_text(text):
    """Preprocess text for prediction"""
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9\\s]', '', text)
    return text

def predict_emotion(text):
    """Predict emotion for given text"""
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Convert to sequence and pad
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)
    
    # Make prediction
    prediction = model.predict(padded_sequence, verbose=0)
    predicted_class_idx = np.argmax(prediction)
    predicted_emotion = label_encoder.classes_[predicted_class_idx]
    confidence = float(np.max(prediction))
    
    return predicted_emotion, confidence, prediction[0]

# Example usage
if __name__ == "__main__":
    test_texts = [
        "I am so happy today!",
        "This makes me really sad",
        "I am angry about this situation",
        "This is quite surprising",
        "I feel scared about the future"
    ]
    
    print("\\n" + "="*50)
    print("TESTING PREDICTIONS:")
    print("="*50)
    
    for text in test_texts:
        emotion, confidence, all_probs = predict_emotion(text)
        print(f"Text: '{{text}}'")
        print(f"Predicted Emotion: {{emotion}} (Confidence: {{confidence:.3f}})")
        print(f"All probabilities: {{dict(zip(classes, all_probs))}}")
        print("-" * 30)
 
