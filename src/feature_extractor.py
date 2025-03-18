import os
import sys
import numpy as np
from numpy.typing import ArrayLike
from typing import Union, List
from collections.abc import Iterable
import torch
import matplotlib.pyplot as plt
import librosa
import librosa.display as ldisp
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import AudioArrayClip

# Append project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.append(project_root)

from src.audio_manipulator import AudioManipulator  # noqa: E402

class FeatureExtractor:
    def __init__(self) -> None:
        self.sr = 22050
        self.n_mfccs = 13
        self.n_mels = 128
        self.manipulator = AudioManipulator()
        self.audio_exts = ["wav", "mp3", "m4a"]
        self.video_exts = ["mp4", "mov", "avi"]


    def find_mfccs(self, signal: ArrayLike, n_fft: int = 2048, hop_length: int = 512) -> ArrayLike:
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=signal, n_mfcc=self.n_mfccs, sr=self.sr, n_fft=n_fft, hop_length=hop_length)

        return mfccs
    

    def find_deltas(self, signal: ArrayLike, order: int, n_fft: int = 2048, hop_length: int = 512) -> ArrayLike:
        return librosa.feature.delta(self.find_mfccs(signal, n_fft=n_fft, hop_length=hop_length), order=order)
    

    def show_mfccs(self, signal: ArrayLike, hop_length: int) -> None:
        # Calculate time frames to create appropriate time axis on the plot
        frames = range(signal.shape[1])
        times = librosa.frames_to_time(frames, sr=self.sr, hop_length=hop_length)

        # Plot
        plt.figure(figsize=(10, 5))
        ldisp.specshow(signal, x_coords=times, x_axis="s", sr=self.sr, hop_length=hop_length)
        plt.ylabel(f"Coefficients (1-{self.n_mfccs})")
        plt.title("Variation of MFCCs over time")
        plt.colorbar(format="%+2f")
        plt.show()

        
    def find_snr(self, noisy_signal, sr):
        # Clean the signal
        clean_signal = self.manipulator.reduce_noise(noisy_signal, sr)
    
        # Calculate signal power 
        signal_power = np.mean(clean_signal**2)

        # Calculate noise power
        noise = noisy_signal - clean_signal
        noise_power = np.mean(noise**2)

        snr = 10 * np.log10(signal_power / noise_power)

        return snr


    def get_duration(self, audio: Union[str, Iterable]) -> float:
        if isinstance(audio, str):
            # Treat as the relative path to the audio file
            return AudioFileClip(audio).duration
        elif isinstance(audio, Iterable):
            # Treat as an audio time-series array
            # Check that the array is 2D, if not reshape
            if audio.ndim == 1:
                audio = audio.reshape(-1, 1)
            return AudioArrayClip(audio, fps=22050).duration


    def feature_extraction(self, filepath: str, scaler):
        # Check whether the path is a video or audio file
        ext = filepath.split(".")[-1]
        if ext in self.audio_exts:
            # Load audio
            audio_signal = self.manipulator.load_audio(filepath)
        elif ext in self.video_exts:
            # Extract audio
            audio_signal = self.manipulator.extract_audio(filepath)

        # Improve SNR by reducing noise
        cleaned_signal = self.manipulator.reduce_noise(audio_signal, sr=22050)

        # Find and prepare MFCCs
        mfccs = self.find_mfccs(cleaned_signal, hop_length=22050)

        # Transpose matrix
        mfccs = mfccs.T

        # Scale the data
        scaled_mfccs = scaler.fit_transform(mfccs)

        # Transform into tensor
        mfccs_tensor = torch.tensor(scaled_mfccs, dtype=torch.float32)

        return mfccs_tensor


    def detect_pauses(self, audio_signal: ArrayLike, sampling_rate: int, amplitude_threshold: float, time_frame_threshold: float) -> tuple[List[tuple[float, float]], ArrayLike]:
        """
        Detects the pauses within a speech signal by determining whether the absolute amplitude of the audio signal is below the defined threshold.
        If a section is below the defined threshold, it is marked as a section where this is a pause with an array of 0's (no pause) and 1's (pause).
        The start and end times for each section is stored in a list of tuples.

        Arg(s):
        - audio (ArrayLike): The audio signal for which to detect pauses
        - sampling_rate (int): The sampling rate of the audio signal
        - amplitude threshold (float): The threshold beneath which a signal will be said to have a pause
        - time_frame_threshold (float): The minimum length a section must be to be considered a pause (in seconds)

        Returns:
        - tuple[List[(float, float)], ArrayLike]: The list contains the start and end times for each pause. The ArrayLike object contains an array of 0's and 1's
        """
        # Get absolute value of signal
        amplitude = np.abs(audio_signal)

        # Create binary mask to know which sections are below the defined threshold
        low_amplitude_mask = amplitude < amplitude_threshold

        # Make sure section of low amplitude is a minimum length to be considered a pause
        num_samples_threshold = int(sampling_rate * time_frame_threshold)

        # Instantiate arrays to store results
        low_amplitude_sections = []
        pauses = np.zeros((len(audio_signal), 1))

        # Flag when looping through list
        start = None

        for i in range(len(low_amplitude_mask)):
            if low_amplitude_mask[i]:
                if start is None: # The start of a low amplitude section
                    # Get the index of the start of the low amplitude section
                    start = i
            else:
                # Has been going over a low amplitude section and is bigger than the threshold length to be considered a pause
                if start is not None and (i - start) >= num_samples_threshold:
                    # Append indices to list to visualise later
                    low_amplitude_sections.append((start/sampling_rate, i/sampling_rate))

                    # Mark those time frames as there is a pause
                    for i in range(start, i+1):
                        pauses[i] = 1
                start = None

        return low_amplitude_sections, pauses