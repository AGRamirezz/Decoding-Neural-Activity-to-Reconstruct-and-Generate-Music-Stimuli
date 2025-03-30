#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Music to Spectra Analysis

This script processes audio files to generate and visualize different spectral representations,
including linear magnitude spectrograms, mel spectrograms, and power spectral density.

Usage:
    1. Set the directory path containing music files
    2. Run the script to generate spectral visualizations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from typing import List, Tuple


def load_audio_files(directory_path: str, sample_rate: int = 22050) -> Tuple[List[str], List[Tuple[np.ndarray, int]]]:
    """
    Load all audio files from the specified directory.
    
    Args:
        directory_path: Path to directory containing audio files
        sample_rate: Target sample rate for audio files
        
    Returns:
        Tuple containing:
            - List of file names
            - List of (audio_data, sample_rate) tuples
    """
    os.chdir(directory_path)
    
    # Get sorted list of audio files
    file_list = [item for item in os.listdir(os.getcwd())]
    file_list.sort()
    
    # Load audio files
    audio_data = [librosa.load(file_list[i], sr=sample_rate) for i in range(len(file_list))]
    
    return file_list, audio_data


def extract_waveforms(audio_data: List[Tuple[np.ndarray, int]], duration_mins: int = 4) -> np.ndarray:
    """
    Extract waveforms from audio data and trim to specified duration.
    
    Args:
        audio_data: List of (audio_data, sample_rate) tuples
        duration_mins: Duration in minutes to trim audio to
        
    Returns:
        Array of trimmed waveforms
    """
    # Extract waveforms from audio data
    waveforms = [data[0] for data in audio_data]
    
    # Calculate sample length for specified duration
    sample_rate = audio_data[0][1]
    sample_length = sample_rate * 60 * duration_mins
    
    # Trim waveforms to consistent length
    trimmed_waveforms = [waveform[0:sample_length] for waveform in waveforms]
    
    return np.array(trimmed_waveforms)


def create_linear_spectrogram(y: np.ndarray, sr: int = 22050) -> np.ndarray:
    """
    Create a linear magnitude spectrogram from audio data.
    
    Args:
        y: Audio time series
        sr: Sample rate
        
    Returns:
        Magnitude spectrogram in dB
    """
    # Compute short-time Fourier transform
    linear = librosa.stft(y=y)
    
    # Convert to magnitude and then to dB
    mag = np.abs(linear)
    mag_db = 20 * np.log10(np.maximum(1e-5, mag))
    
    return mag_db


def create_mel_spectrogram(y: np.ndarray, sr: int = 22050) -> np.ndarray:
    """
    Create a mel spectrogram from audio data.
    
    Args:
        y: Audio time series
        sr: Sample rate
        
    Returns:
        Mel spectrogram in dB
    """
    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    
    # Convert to dB
    mel_db = 20 * np.log10(np.maximum(1e-5, mel))
    
    return mel_db


def compute_power_spectral_density(y: np.ndarray) -> np.ndarray:
    """
    Compute power spectral density from audio data.
    
    Args:
        y: Audio time series
        
    Returns:
        Power spectral density in dB
    """
    # Compute STFT
    linear = librosa.stft(y=y)
    
    # Compute power
    mag = np.abs(linear)
    power = np.mean(mag**2, axis=1)  # squared power mean
    
    # Convert to dB
    power_db = 20 * np.log10(np.maximum(1e-5, power))
    
    return power_db


def plot_spectrogram(spectrogram: np.ndarray, title: str, y_axis: str = 'linear', sr: int = 22050):
    """
    Plot a spectrogram with appropriate formatting.
    
    Args:
        spectrogram: Spectrogram data to plot
        title: Title for the plot
        y_axis: Type of y-axis ('linear' or 'mel')
        sr: Sample rate
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        spectrogram, 
        cmap='magma', 
        x_axis='time', 
        y_axis=y_axis, 
        ax=ax, 
        sr=sr, 
        hop_length=512
    )
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.set(title=title)
    plt.tight_layout()
    plt.show()


def main():
    """Main function to execute the audio processing pipeline."""
    # Define constants
    SAMPLE_RATE = 22050
    MUSIC_DIR = '/path/to/your/music/files'  # Update this path
    
    # Load audio files
    print("Loading audio files...")
    file_list, audio_data = load_audio_files(MUSIC_DIR, SAMPLE_RATE)
    print(f"Loaded {len(file_list)} files: {file_list}")
    
    # Extract and trim waveforms
    waveforms = extract_waveforms(audio_data, duration_mins=4)
    print(f"Waveform array shape: {waveforms.shape}")
    
    # Plot example waveform
    plt.figure(figsize=(10, 4))
    plt.plot(waveforms[0])
    plt.title("Example Waveform")
    plt.tight_layout()
    plt.show()
    
    # Select first audio file for spectral analysis
    y = waveforms[0]
    
    # Create and plot linear spectrogram
    mag_db = create_linear_spectrogram(y, SAMPLE_RATE)
    print(f"Linear spectrogram shape: {mag_db.shape}")
    plot_spectrogram(mag_db, "Linear Spectrogram", 'linear', SAMPLE_RATE)
    
    # Create and plot mel spectrogram
    mel_db = create_mel_spectrogram(y, SAMPLE_RATE)
    print(f"Mel spectrogram shape: {mel_db.shape}")
    plot_spectrogram(mel_db, "Mel Spectrogram", 'mel', SAMPLE_RATE)
    
    # Compute and plot power spectral density
    power_db = compute_power_spectral_density(y)
    print(f"Power spectral density shape: {power_db.shape}")
    
    plt.figure(figsize=(10, 4))
    plt.plot(power_db)
    plt.title("Power Spectral Density")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

