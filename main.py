# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import librosa
import numpy as np
import os

# Function to extract audio features using librosa
def extract_features(audio_path):
    audio, _ = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
    return mfccs

# Data paths
data_dir = "D:\surah_audio\wav"

# Collect audio files and labels
audio_files = []
labels = []

for title in os.listdir(data_dir):
    title_path = os.path.join(data_dir, title)
    if os.path.isdir(title_path):
        #print(f"Label: {title}")
        for audio_file in os.listdir(title_path):
            audio_path = os.path.join(title_path, audio_file)
            if audio_file.endswith(".wav"):
                audio_files.append(audio_path)
                labels.append(title)
                duration = librosa.get_duration(filename=audio_path)
                #print(f"  Audio File: {audio_file}, Duration: {duration:.2f} seconds")

# Encode labels
label_mapping = {label: idx for idx, label in enumerate(set(labels))}
encoded_labels = [label_mapping[label] for label in labels]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(audio_files, encoded_labels, test_size=0.2, random_state=42)

# Function to load and preprocess audio data
def load_and_preprocess(audio_paths, labels):
    features = []
    max_length = 0
    for path, label in zip(audio_paths, labels):
        feature = extract_features(path)
        # Update max_length if the current feature length is greater
        max_length = max(max_length, feature.shape[1])
        features.append((feature, label))
    return features

# Load and preprocess training and testing data
train_data = load_and_preprocess(X_train, y_train)
test_data = load_and_preprocess(X_test, y_test)

# Separate features and labels
X_train, y_train = zip(*train_data)
X_test, y_test = zip(*test_data)

# Pad sequences to the length of the longest sequence
max_length = max(len(feature[0]) for feature in X_train)
X_train = np.array([librosa.util.fix_length(x[0], size=max_length, axis=0) for x in X_train])
X_test = np.array([librosa.util.fix_length(x[0], size=max_length, axis=0) for x in X_test])

# Convert to NumPy arrays
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

# Add an extra dimension for the LSTM input
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Build the neural network model with an LSTM layer
model = Sequential()
model.add(LSTM(64, input_shape=(None, X_train.shape[2])))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(set(encoded_labels)), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# Predict labels for the test set
predictions = model.predict(X_test)

# Decode label indices back to original labels
predicted_labels = [list(label_mapping.keys())[list(label_mapping.values()).index(np.argmax(prediction))] for prediction in predictions]
true_labels = [list(label_mapping.keys())[list(label_mapping.values()).index(label)] for label in y_test]

# Print the predicted and true labels for each test sample
for i in range(len(X_test)):
    print(f"Audio File: {X_test[i]}, Predicted: {predicted_labels[i]}, True: {true_labels[i]}")

# Save the model for future use
#model.save("/path/to/save/your/model")
