# step 1: install mirdata, in terminal -> pip install mirdata

# if some import errors occur, run the following command in terminal
# pip install librosa (as an example)

import mirdata
import json
import pandas as pd
import librosa 
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from IPython.display import Audio

saraga = None

def create_dataset(data_home):
    # Load the dataset
    global saraga
    saraga = mirdata.initialize("saraga_carnatic", data_home=data_home)

    return saraga


def get_metadata(track_id):
    """
    For <track_id>, return a dataframe of associated metadata
    """
    track = saraga.track(track_id)

    with open(track.metadata_path, 'r') as f:
        json_data = json.load(f)

    json_data["track_id"] = track.track_id

    formatted_metadata = json.dumps(json_data, indent=4)


    return print(formatted_metadata)


def get_performer(track_id):
    """
    For <track_id>, return the performer
    """
    track = saraga.track(track_id)

    with open(track.metadata_path, 'r') as f:
        json_data = json.load(f)

    performer = json_data.get("artists")
    return performer

def get_performance(track_id):
    """
    For <track_id>, return the performance name
    """
    track = saraga.track(track_id)

    with open(track.metadata_path, 'r') as f:
        json_data = json.load(f)

    performance = json_data.get("concert")
    return performance

def get_raga(track_id):
    """
    For <track_id>, return the raga name
    """
    track = saraga.track(track_id)

    with open(track.metadata_path, 'r') as f:
        json_data = json.load(f)

    raga = json_data.get("raaga")
    return raga


def get_tonic(track_id):
    """
    For <track_id>, return the tonic in hertz
    """
    track = saraga.track(track_id)

    with open(track.ctonic_path, 'r') as f:
        tonic = float(f.read().strip())

    return tonic


def get_track_info(track_id):

    track = saraga.track(track_id)

    # Extract metadata using functions
    raga = get_raga(track_id)
    performer = get_performer(track_id)[0] if get_performer(track_id) else None  # Handle cases with no artist info
    performance = get_performance(track_id)
    multitrack = hasattr(track, "audio_vocal_path")

    return {
        "track_id": track_id,
        "raga": raga,
        "performer": performer,
        "performance": performance,
        "multitrack": multitrack
    }


def load_mixed_audio(track_id):
    """
    For <track_id>, return the loaded audio
    """
    track = saraga.track(track_id)
    audio_array, sr = librosa.load(track.audio_path, sr=44100)
    return audio_array

def load_violin_audio(track_id, saraga):
    """
    For <track_id>, return the isolated violin track
    """

    track = saraga.track(track_id)

    if track.audio_violin_path is None:
            print(f"Warning: No violin audio for track {track_id}")
            return None

    audio_array, sr = librosa.load(track.audio_violin_path, sr=44100)
    return audio_array

def load_voice_audio(track_id):
    """
    For <track_id>, return the isolated voice track
    """
    track = saraga.track(track_id)

    if track.audio_vocal_path is None:
            print(f"Warning: No voice audio for track {track_id}")
            return None

    audio_array, sr = librosa.load(track.audio_vocal_path, sr=44100)
    return audio_array

def load_mridangam_left_audio(track_id):
    """
    For <track_id>, return the isolated mridangam track
    """
    track = saraga.track(track_id)

    if track.audio_mridangam_left_path is None:
            print(f"Warning: No left mridangam audio for track {track_id}")
            return None

    audio_array, sr = librosa.load(track.audio_mridangam_left_path, sr=44100)

    return audio_array

def load_mridangam_right_audio(track_id):
    """
    For <track_id>, return the isolated mridangam track
    """
    track = saraga.track(track_id)

    if track.audio_mridangam_right_path is None:
            print(f"Warning: No right mridangam audio for track {track_id}")
            return None

    audio_array, sr = librosa.load(track.audio_mridangam_right_path, sr=44100)

    return audio_array


def plot_waveform(audio_array, sr=44100):
    """
    Plot waveform for <audio_array> using matplotlib.pyplot
    """
    # Create time axis
    time = np.arange(len(audio_array)) / sr

    # Plot the waveform
    plt.plot(time, audio_array)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")
    plt.grid(True)
    plt.show()

def play_audio(audio_array, sr=44100, file_name='output.wav'):
    """
    Save and play audio using soundfile.
    """
    
    # Write the file
    sf.write(file_name, audio_array, sr)
    print(f"Audio has been saved as '{file_name}'.")