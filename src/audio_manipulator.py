from moviepy import VideoFileClip, AudioFileClip
from moviepy.audio.AudioClip import AudioArrayClip
from numpy.typing import ArrayLike
from typing import List
import noisereduce as nr
import numpy as np
import pyloudnorm as pyln
import librosa
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Possible clipped samples in output.")

class AudioManipulator():
    def __init__(self) -> None:
        self.sr = 22050


    def extract_audio(self, filepath: str) -> ArrayLike:
        """
        Extracts an audio from video with the specified filepath.
        The extracted audio is also subclipped to make the duration of the audio an exact integer in seconds.

        Arg(s):
        - filepath (str): The video filepath to extract an audio from

        Returns:
        - ArrayLike: The time-series audio array
        """
        # Extract audio
        video = VideoFileClip(filename=filepath)
        audio = video.audio

        # Get duration of audio and clip the audio
        duration = int(audio.duration)
        audio = audio.subclipped(0, duration)

        # Resample when converting to sound array
        audio_array = audio.to_soundarray(fps=self.sr)

        # Convert from two-channel stereo to mono
        audio_array = np.mean(audio_array, axis=1)

        # Close audio and video to prevent any leakage
        audio.close()
        video.close()

        return audio_array


    def load_audio(self, filepath: str) -> ArrayLike:
        """
        Loads an audio from a file with the specified filepath.

        Arg(s):
        - filepath (str): The audio filepath to load

        Returns:
        - ArrayLike: The time-series audio array
        """
        audio = AudioFileClip(filepath)

        # Get the duration of the audio and clip the audio
        duration = int(audio.duration)
        audio = audio.subclipped(0, duration)

        # Resample when converting to sound array
        audio_array = audio.to_soundarray(fps=self.sr)

        # Convert from two-channel stereo to mono
        audio_array = np.mean(audio_array, axis=1)

        # Close audio to prevent any leakage
        audio.close()

        return audio_array


    def reduce_noise(self, signal: ArrayLike, sr: int) -> ArrayLike:
        clean_signal = nr.reduce_noise(y=signal, sr=sr)
        return clean_signal
    

    def add_noise(self, signal: ArrayLike, level: int) -> ArrayLike:
        noise_factor = 0.001
        noise = np.random.normal(0, noise_factor * level, signal.shape)
        return signal + noise


    def create_noise_levels(self, signal: ArrayLike, levels: ArrayLike, save: bool = False) -> List[ArrayLike]:
        noisy_signals = []
        for i in range(len(levels)):
            noisy_signal = self.add_noise(signal, levels[i])
            noisy_signals.append(noisy_signal)

        # Saves the noisy signals in individual files
        if save:
            for i in range(len(noisy_signals)):
                # Ensure the data is in the correct format (float32 for MoviePy)
                audio_data = noisy_signals[i].astype(np.float32)

                # Reshape into 2D array to be mono for the AudioArrayClip
                audio_data = audio_data.reshape(-1, 1)

                # Create an AudioArrayClip
                audio_clip = AudioArrayClip(audio_data, fps=self.sr)

                # Write the audio to a file
                audio_clip.write_audiofile(f"local/output_noise_data/output_noise_level_{levels[i]}.wav", codec='pcm_s16le') # codec for WAV file

        return noisy_signals
        

    def apply_limiter(self, audio, threshold_db):
        # Convert threshold from dB to linear scale
        threshold = 10 ** (threshold_db / 20)

        # Create a copy of the audio to apply limiting
        limited_audio = np.copy(audio)

        # Find samples that exceed the threshold
        peaks = np.abs(limited_audio) > threshold

        # Apply limiting: scale down the peaks
        if np.any(peaks):
            # Calculate the gain reduction factor
            peak_values = np.abs(limited_audio[peaks])
            reduction_factor = threshold / peak_values

            # Apply the reduction factor to the limited audio
            limited_audio[peaks] *= reduction_factor

        return limited_audio


    def normalise_loudness(self, audio_signal: np.ndarray, sampling_rate: int, compression_strength: int = 255, target_loudness: int = -23):
        # Use mu-law compression
        compressed_audio = librosa.mu_compress(audio_signal, mu=compression_strength, quantize=False)

        # Get loudness of the compressed audio signal
        meter = pyln.Meter(sampling_rate)
        loudness = meter.integrated_loudness(compressed_audio)

        # Normalise the loudness of the compressed audio signal
        normalised_audio = pyln.normalize.loudness(compressed_audio, loudness, target_loudness)

        # Clip to ensure values fall within the range of -1 to 1
        normalised_audio = np.clip(normalised_audio, -1, 1)

        return normalised_audio