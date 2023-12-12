# Import necessary libraries
import tensorflow as tf
from keras.src.utils import pad_sequences
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt

# Function to extract audio features using librosa and plot the spectrogram
def extract_features_and_plot(audio_path, label):
    audio, _ = librosa.load(audio_path, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128, hop_length=512)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_spectrogram, sr=16000, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram for class: {label}')
    plt.show()

    return log_spectrogram

# Data paths
data_dir = "D:\surah_audio\wav"

# Collect audio files and labels
audio_files = []
labels = []

for title in os.listdir(data_dir):
    title_path = os.path.join(data_dir, title)
    if os.path.isdir(title_path):
        print(f"Label: {title}")
        for audio_file in os.listdir(title_path):
            audio_path = os.path.join(title_path, audio_file)
            if audio_file.endswith(".wav"):
                audio_files.append(audio_path)
                labels.append(title)
                duration = librosa.get_duration(path=audio_path)
                print(f"  Audio File: {audio_file}, Duration: {duration:.2f} seconds")

# Encode labels
label_mapping = {label: idx for idx, label in enumerate(set(labels))}
encoded_labels = [label_mapping[label] for label in labels]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(audio_files, encoded_labels, test_size=0.2, random_state=42)

# Function to load and preprocess audio data
def load_and_preprocess(audio_paths, labels):
    features = []
    for path, label in zip(audio_paths, labels):
        feature = extract_features_and_plot(path,label)
        features.append((feature, label))
    return features

# Load and preprocess training and testing data
train_data = load_and_preprocess(X_train, y_train)
test_data = load_and_preprocess(X_test, y_test)

# Separate features and labels
X_train, y_train = zip(*train_data)
X_test, y_test = zip(*test_data)

# Find the maximum shape of a spectrogram in your dataset
max_shape = max(feature.shape for feature, _ in train_data)

# Pad or truncate each spectrogram to the maximum shape
# Pad or truncate each spectrogram to the maximum shape
X_train = np.array([np.pad(feature, ((0, max_shape[0] - feature.shape[0]), (0, max_shape[1] - feature.shape[1])), mode='constant') for feature in X_train])
X_test = np.array([np.pad(feature, ((0, max_shape[0] - feature.shape[0]), (0, max_shape[1] - feature.shape[1])), mode='constant') for feature in X_test])


# Convert to NumPy arrays
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

# Add an extra dimension for the LSTM input
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Build the neural network model with a stacked LSTM architecture
model = Sequential()
model.add(LSTM(64, input_shape=(None, X_train.shape[2]), return_sequences=True))
model.add(LSTM(64))  # You can add more LSTM layers if needed
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))  # Add dropout for regularization
model.add(Dense(len(set(encoded_labels)), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# Predict labels for the test set
predictions = model.predict(X_test)

# Decode label indices back to original labels
predicted_labels = [list(label_mapping.keys())[list(label_mapping.values()).index(np.argmax(prediction))] for prediction in predictions]
true_labels = [list(label_mapping.keys())[list(label_mapping.values()).index(label)] for label in y_test]

# Print the predicted and true labels along with the corresponding audio file names and MFCC shapes
for i in range(len(X_test)):
    print(f"Audio File: {X_test[i]}, Predicted: {predicted_labels[i]}, True: {true_labels[i]}, MFCC Shape: {X_test[i].shape}")

# Convert one-hot encoded predictions to labels
predicted_labels = np.argmax(predictions, axis=1)

# Plot the spectrograms for the first few examples in each class
for label in set(labels):
    class_examples = [audio_path for audio_path, class_label in zip(audio_files, labels) if class_label == label]
    for example in class_examples[:3]:
        extract_features_and_plot(example, label)


# Print classification report
print(classification_report(y_test, predicted_labels, target_names=label_mapping.keys()))
model.summary()

