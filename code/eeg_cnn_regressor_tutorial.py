# -*- coding: utf-8 -*-
"""
EEG CNN Regressor for Music Reconstruction from EEG Data
This script trains a CNN model to reconstruct music mel spectrograms from EEG data.
"""

# Import Dependencies
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
import librosa.display

# Custom activation functions
def mish(x):
    """Mish activation function: x * tanh(softplus(x))"""
    return x * keras.backend.tanh(keras.backend.softplus(x))

def swish(x, beta=1):
    """Swish activation function: x * sigmoid(beta * x)"""
    from keras.backend import sigmoid
    return x * sigmoid(beta * x)

# Register custom activation functions
get_custom_objects().update({'mish': Activation(mish)})

def load_data(path):
    """Load and prepare training and testing data"""
    # Load EEG data (Power Spectrum Density Representation)
    X_train = np.load(os.path.join(path, "x_trainp.npy"))
    X_test = np.load(os.path.join(path, "x_testp.npy"))
    
    # Load Mel Spectrograms of Music Stimuli
    y_train = np.load(os.path.join(path, "y_train3.npy"))
    y_test = np.load(os.path.join(path, "y_test3.npy"))
    
    print(f"EEG data shapes: {X_train.shape}, {X_test.shape}")
    print(f"Mel spectrogram shapes: {y_train.shape}, {y_test.shape}")
    
    # Shuffle training data to stabilize training
    train_idx = np.random.permutation(X_train.shape[0])
    X_train = np.take(X_train, train_idx, axis=0)
    y_train = np.take(y_train, train_idx, axis=0)
    
    return X_train, X_test, y_train, y_test

def visualize_eeg(X_train, idx=9):
    """Visualize EEG data as a heatmap"""
    font = {'family': 'DejaVu Sans', 'color': 'black', 'size': 13}
    
    y = np.reshape(X_train[idx], (63, 125))
    y = 20 * np.log10(y)
    plt.figure(dpi=100)
    plt.imshow(y, cmap='jet')
    plt.ylabel('Frequency (Hz)', fontdict=font, labelpad=16)
    plt.xlabel('Channels', fontdict=font, labelpad=16)
    plt.xticks([0.5, 25.5, 50.5, 75.5, 100.5, 124.5], ['1', '25', '50', '75', '100', '125'])
    plt.ylim([0, 63])
    plt.title("EEG Power Spectrum Density")
    plt.show()

def visualize_mel_spectrogram(mel_data, idx=9, title="Mel Spectrogram"):
    """Visualize mel spectrogram"""
    SR = 22050
    
    fig, ax = plt.subplots(dpi=100)
    img = librosa.display.specshow(
        mel_data[idx].T, 
        cmap='magma', 
        x_axis='time', 
        y_axis='mel', 
        ax=ax, 
        sr=SR, 
        hop_length=512
    )
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.set(title=title)
    plt.show()

def create_model(input_shape, output_shape, batch_size=60):
    """Create CNN model for EEG to mel spectrogram regression"""
    # Define hyperparameters
    kernel_init = keras.initializers.he_uniform(seed=1369)
    kernel_reg = keras.regularizers.l2(0.000001)
    act_reg = keras.regularizers.l2(0.0000001)
    k = 4  # Kernel Size
    p = 2  # Pool Size
    f = 8  # Number of Filters
    my_act = 'relu'  # Activation function
    
    # Define model architecture
    input_layer = keras.Input(shape=input_shape, batch_size=batch_size)
    
    # Convolutional layers
    x = layers.ZeroPadding2D()(input_layer)
    x = layers.Conv2D(kernel_size=(k,k), filters=f, activation=my_act, 
                      kernel_initializer=kernel_init, padding="same")(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv2D(kernel_size=(k,k), filters=f*2, activation=my_act, 
                      kernel_initializer=kernel_init, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv2D(kernel_size=(k,k), filters=f*4, activation=my_act, 
                      kernel_initializer=kernel_init, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv2D(kernel_size=(k,k), filters=f*8, activation=my_act, 
                      kernel_initializer=kernel_init, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv2D(kernel_size=(k,k), strides=(2,2), filters=f*16, activation=my_act, 
                      kernel_initializer=kernel_init, padding="same")(x)
    x = layers.ZeroPadding2D()(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.MaxPooling2D(pool_size=(p,p))(x)
    x = layers.BatchNormalization()(x)
    
    # Fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Dense(128, activation=my_act, kernel_initializer=kernel_init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)
    
    x = layers.Dense(128, activation=my_act, kernel_initializer=kernel_init)(x)
    x = layers.BatchNormalization()(x)
    
    # Output layer
    rows, cols = output_shape
    x = layers.Dense(rows*cols, activation='linear')(x)
    x = layers.Reshape((rows, cols))(x)
    
    model = Model(input_layer, x, name='EEG2MelCNN')
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, batch_size=60, epochs=200):
    """Train the model and plot training history"""
    # Compile model
    opt = keras.optimizers.Adam(learning_rate=0.0015)
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=opt,
        metrics=tf.keras.metrics.MeanSquaredError()
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_test, y_test),
        batch_size=batch_size
    )
    
    # Plot training history
    plt.figure(dpi=100)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
    return model, history

def evaluate_model(model, X_test, y_test, idx=20):
    """Evaluate model by comparing original and reconstructed spectrograms"""
    # Generate predictions
    predictions = model.predict(X_test)
    print(f"Predictions shape: {predictions.shape}")
    
    # Visualize original mel spectrogram
    visualize_mel_spectrogram(y_test, idx=idx, title="Original Mel Spectrogram")
    
    # Visualize reconstructed mel spectrogram
    visualize_mel_spectrogram(predictions, idx=idx, title="Reconstructed Mel Spectrogram")
    
    # Convert spectrograms to audio and visualize waveforms
    SR = 22050
    max_db = 100
    ref_db = 46
    
    # Original audio waveform
    mel_orig = y_test[idx].T
    mel_orig = (np.clip(mel_orig, 0, 1) * max_db) - max_db + ref_db
    audio_orig = librosa.feature.inverse.mel_to_audio(mel_orig)
    
    plt.figure(dpi=100)
    plt.plot(audio_orig)
    plt.title("Original Audio Waveform")
    plt.show()
    
    # Reconstructed audio waveform
    mel_recon = predictions[idx].T
    mel_recon = (np.clip(mel_recon, 0, 1) * max_db) - max_db + ref_db
    audio_recon = librosa.feature.inverse.mel_to_audio(mel_recon)
    
    plt.figure(dpi=100)
    plt.plot(audio_recon)
    plt.title("Reconstructed Audio Waveform")
    plt.show()
    
    return predictions, audio_orig, audio_recon

def main():
    """Main function to run the EEG to music reconstruction pipeline"""
    # Set the data path
    data_path = '/MyPath/'  # Change this to your actual data path
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(data_path)
    
    # Reshape data for CNN input (add channel dimension)
    if len(X_train.shape) == 3:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    
    # Visualize sample data
    visualize_eeg(X_train)
    visualize_mel_spectrogram(y_train)
    
    # Create and train model
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    output_shape = (y_train.shape[1], y_train.shape[2])
    
    model = create_model(input_shape, output_shape)
    model.summary()
    
    model, history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluate model
    predictions, audio_orig, audio_recon = evaluate_model(model, X_test, y_test)
    
    # Optional: Save model
    # model.save('eeg_to_music_model.h5')
    
    return model, predictions

if __name__ == "__main__":
    main()

