# advanced_phonk_vocal_processor.py

import librosa
import soundfile as sf
from pydub import AudioSegment
import numpy as np

# --- Configuration ---
# You must have these libraries installed:
# 1. librosa: 'pip install librosa'
# 2. soundfile: 'pip install soundfile'
# 3. pydub: 'pip install pydub'
# 4. ffmpeg: A standalone program. Must be in your system's PATH.

# Path to the input audio file.
# We will use your vocals.mp3 as the source.
VOCAL_PATH = "vocals.mp3"

# Output file path for the processed vocals.
OUTPUT_PATH = "processed_vocals.wav"

# --- Main Parameters for Phonk Sound ---
# Target BPM for the final track. Phonk is often in the 60-75 BPM range.
# We'll aim for a solid, hard-hitting 65 BPM.
TARGET_BPM = 65.0

# Number of semitones to shift the pitch down.
# A negative value lowers the pitch. A shift of -5 is a good starting point for a dark sound.
PITCH_SHIFT_SEMITONES = -5

def process_vocals_for_phonk(input_path: str, output_path: str, target_bpm: float, pitch_shift: int):
    """
    Processes a vocal track for a phonk song by time-stretching and pitch-shifting it.

    Args:
        input_path: Path to the input audio file.
        output_path: Path to save the processed audio file.
        target_bpm: The desired beats per minute for the output.
        pitch_shift: The number of semitones to shift the pitch.
    """
    print("Starting vocal processing for phonk...")
    
    try:
        # Load the audio file using librosa. It handles the format conversion.
        # This will return the audio time series (y) and the sampling rate (sr).
        y, sr = librosa.load(input_path, sr=None)
        print(f"Loaded audio file: {input_path} with a sample rate of {sr} Hz.")

        # --- Step 1: Analyze and Adjust Tempo ---
        # First, we'll estimate the original tempo of the track.
        # This helps us determine the correct stretch ratio.
        original_tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        print(f"Estimated original tempo: {original_tempo:.2f} BPM.")

        # Calculate the stretch ratio.
        # A ratio less than 1 will slow the audio down.
        stretch_ratio = original_tempo / target_bpm
        
        # Time-stretch the audio without changing the pitch.
        # This is the key difference from the old script.
        # This is a high-quality process that will not sound "glitchy."
        y_stretched = librosa.effects.time_stretch(y=y, rate=stretch_ratio)
        print(f"Time-stretched audio to approximately {target_bpm} BPM.")
        
        # --- Step 2: Pitch-Shift the Vocals ---
        # Now we'll lower the pitch of the stretched audio.
        # This gives it a deep, distorted, menacing feel.
        y_shifted = librosa.effects.pitch_shift(y=y_stretched, sr=sr, n_steps=pitch_shift)
        print(f"Pitch-shifted audio by {pitch_shift} semitones.")
        
        # --- Step 3: Normalize and Export ---
        # Normalize the audio to prevent clipping and ensure consistent volume.
        # We'll use pydub for the final export as it's great for writing files.
        # First, convert the numpy array back to an AudioSegment.
        y_shifted_int16 = (y_shifted * 32767).astype(np.int16)
        
        # Create a pydub AudioSegment from the numpy array.
        processed_audio = AudioSegment(
            y_shifted_int16.tobytes(),
            frame_rate=sr,
            sample_width=y_shifted_int16.dtype.itemsize,
            channels=1 # librosa loads as mono by default
        )
        
        # Normalize to prevent clipping.
        processed_audio = processed_audio.normalize()
        
        # Export the final track.
        print(f"Exporting final processed vocal track to {output_path}...")
        processed_audio.export(output_path, format="wav")
        
        print("Vocal processing complete! Check the new file.")

    except FileNotFoundError:
        print(f"Error: The audio file at '{input_path}' was not found. Please ensure it exists.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    process_vocals_for_phonk(VOCAL_PATH, OUTPUT_PATH, TARGET_BPM, PITCH_SHIFT_SEMITONES)
