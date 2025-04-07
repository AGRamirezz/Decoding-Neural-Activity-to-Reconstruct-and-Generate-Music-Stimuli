#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectrum Feature Extraction

This script processes EEG data to extract spectral features using power spectral density.
It demonstrates the transformation from time-domain EEG signals to frequency-domain 
representations that can be used for classification tasks.

Original notebook: https://colab.research.google.com/drive/1LBarIhUAW2o3xnBkVErA0_95BCHu13j_
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal
import argparse
from tensorflow.keras import layers, activations
from tensorflow import keras
import tensorflow as tf


def load_data(data_path):
    """
    Load training and testing data from numpy files.
    
    Args:
        data_path (str): Path to the directory containing data files
        
    Returns:
        tuple: (X_train, X_test) - Training and testing data arrays
    """
    print(f"Loading data from {data_path}...")
    
    # Load training and testing data
    X_train = np.load(os.path.join(data_path, "x_traing.npy"))
    X_test = np.load(os.path.join(data_path, "x_testg.npy"))
    
    print(f"Data loaded: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    return X_train, X_test


def plot_grayscale_representation(X_train, sample_idx=100):
    """
    Plot an example of grayscale input representation of EEG data.
    
    Args:
        X_train (numpy.ndarray): Training data array
        sample_idx (int): Index of the sample to plot
    """
    print("Plotting grayscale representation example...")
    
    font = {'family': 'Verdana', 'color': 'black', 'size': 13}
    
    plt.figure(dpi=150)
    
    # Reshape the sample to 2D for visualization
    rep1 = X_train[sample_idx]
    rep1 = np.reshape(rep1, (125, 125))
    
    # Set axis labels
    plt.yticks([0.5, 25.5, 50.5, 75.5, 100.5, 124.5], ['125', '100', '75', '50', '25', '1'])
    plt.xticks([0.5, 25.5, 50.5, 75.5, 100.5, 124.5], ['1', '25', '50', '75', '100', '125'])
    
    # Plot the image
    plt.imshow(rep1.T, cmap='Greys', interpolation='nearest')
    plt.ylabel('Channels', fontdict=font, labelpad=16)
    plt.xlabel('Samples', fontdict=font, labelpad=16)
    plt.title('Grayscale Representation of EEG Data')
    plt.tight_layout()
    plt.savefig('plots/grayscale_representation.png')
    plt.show()


def plot_channel_periodogram(X_train, sample_idx=10, channel_idx=25):
    """
    Plot the periodogram of a specific channel from a sample.
    
    Args:
        X_train (numpy.ndarray): Training data array
        sample_idx (int): Index of the sample to use
        channel_idx (int): Index of the channel to plot
    """
    print(f"Plotting periodogram for sample {sample_idx}, channel {channel_idx}...")
    
    # Extract the channel data
    y = X_train[sample_idx, :, channel_idx]
    y = np.squeeze(y, axis=1)
    
    # Calculate periodogram
    f, Pxx_den = signal.periodogram(y, fs=125)
    
    # Plot
    plt.figure(dpi=150)
    plt.plot(f, Pxx_den)
    plt.title(f'Periodogram of Sample {sample_idx}, Channel {channel_idx}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/channel_periodogram.png')
    plt.show()


def compute_psd_features(X_data, num_samples=10, method='periodogram'):
    """
    Compute power spectral density features for the data.
    
    Args:
        X_data (numpy.ndarray): Input data array
        num_samples (int): Number of samples to process
        method (str): Method to use for PSD calculation ('periodogram' or 'welch')
        
    Returns:
        numpy.ndarray: PSD features array
    """
    print(f"Computing PSD features for {num_samples} samples using {method} method...")
    
    psd_features = []
    
    # Process each sample
    for i in range(num_samples):
        x = []
        # Process each channel
        for j in range(125):
            y = X_data[i, :, j]
            y = np.squeeze(y, axis=1)
            
            # Calculate PSD using specified method
            if method == 'welch':
                f, Pxx_den = signal.welch(y, fs=125, nperseg=125)
            else:  # default to periodogram
                f, Pxx_den = signal.periodogram(y, fs=125)
                
            x.append(Pxx_den)
        
        # Convert to numpy array and append to results
        x = np.array(x)
        psd_features.append(x.T)
    
    # Convert to numpy array and reshape
    psd_features = np.array(psd_features)
    psd_features = np.reshape(psd_features, 
                             (psd_features.shape[0], psd_features.shape[1], 
                              psd_features.shape[2], 1))
    
    print(f"PSD features computed. Shape: {psd_features.shape}")
    
    return psd_features


def plot_psd_representation(psd_features, sample_idx=5):
    """
    Plot power spectral density representation of a sample.
    
    Args:
        psd_features (numpy.ndarray): PSD features array
        sample_idx (int): Index of the sample to plot
    """
    print(f"Plotting PSD representation for sample {sample_idx}...")
    
    font = {'family': 'Verdana', 'color': 'black', 'size': 13}
    
    # Extract the sample and convert to dB scale
    y = np.squeeze(psd_features[sample_idx], axis=2)
    y = 20 * np.log10(y)
    
    # Plot
    plt.figure(dpi=150)
    plt.imshow(y, cmap='jet')
    plt.ylabel('Frequency (Hz)', fontdict=font, labelpad=16)
    plt.xlabel('Channels', fontdict=font, labelpad=16)
    plt.xticks([0.5, 25.5, 50.5, 75.5, 100.5, 124.5], ['1', '25', '50', '75', '100', '125'])
    plt.ylim([0, 63])
    plt.title('Power Spectral Density Representation')
    plt.colorbar(label='Power (dB)')
    plt.tight_layout()
    plt.savefig('plots/psd_representation.png')
    plt.show()


def save_psd_features(psd_features, output_path, prefix='train'):
    """
    Save PSD features to a numpy file.
    
    Args:
        psd_features (numpy.ndarray): PSD features array
        output_path (str): Path to save the features
        prefix (str): Prefix for the filename (train/test)
    """
    # Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save the features
    output_file = os.path.join(output_path, f"{prefix}_psd_features.npy")
    np.save(output_file, psd_features)
    print(f"PSD features saved to {output_file}")


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='EEG Spectrum Feature Extraction')
    parser.add_argument('--data-path', type=str, default='/content/drive/MyPath/',
                        help='Path to the directory containing data files')
    parser.add_argument('--output-path', type=str, default='features',
                        help='Path to save the extracted features')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to process for PSD features')
    parser.add_argument('--psd-method', type=str, choices=['periodogram', 'welch'], 
                        default='periodogram',
                        help='Method to use for PSD calculation')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip generating plots')
    return parser.parse_args()


def main():
    """
    Main function to run the spectrum feature extraction pipeline.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create plots directory if needed
    if not args.skip_plots:
        os.makedirs('plots', exist_ok=True)
    
    # Load data
    X_train, X_test = load_data(args.data_path)
    
    # Plot examples if not skipped
    if not args.skip_plots:
        plot_grayscale_representation(X_train)
        plot_channel_periodogram(X_train)
    
    # Compute PSD features
    train_psd = compute_psd_features(X_train, num_samples=args.num_samples, 
                                    method=args.psd_method)
    
    # Plot PSD representation if not skipped
    if not args.skip_plots:
        plot_psd_representation(train_psd)
    
    # Save features
    save_psd_features(train_psd, args.output_path, prefix='train')
    
    # Process test data if needed
    # Uncomment the following lines to process test data
    # test_psd = compute_psd_features(X_test, num_samples=args.num_samples, 
    #                                method=args.psd_method)
    # save_psd_features(test_psd, args.output_path, prefix='test')
    
    print("Spectrum feature extraction completed successfully!")


if __name__ == "__main__":
    main()

