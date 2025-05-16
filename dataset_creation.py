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
from spleeter.separator import Separator
import uuid
import os
import soundfile as sf


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

def separate_voice(audio_path, isolated_audio_output_path):
    """
    Apply spleeter source separation to input audio
    """
    # Load spleeter model for voice separation
    separator = Separator('spleeter:2stems')

    # Perform separation
    separator.separate_to_file(audio_path, isolated_audio_output_path)

def detect_silence(audio_array, top_db=20):
    """
    Return array of 0 and 1 (is silent/is not silent) for input <audio_array>.
    Returned array should be equal in length to input array.
    """
    # Detect non-silent intervals
    non_silent_intervals = librosa.effects.split(audio_array, top_db=top_db)

    # Create an array of zeros with the same length as the audio array
    is_silent = np.zeros(len(audio_array))

    # Mark non-silent regions as 1
    for start, end in non_silent_intervals:
        is_silent[start:end] = 1

    return is_silent

def split_audio_into_chunks(audio_array, chunk_size_seconds, sr=44100):

    chunk_size_samples = int(chunk_size_seconds * sr)
    num_chunks = len(audio_array) // chunk_size_samples
    chunks = [audio_array[i * chunk_size_samples:(i + 1) * chunk_size_samples]
            for i in range(num_chunks)]
    return chunks

def chunk_contains_instrument(instrument_silence_array, chunk_start_sample, chunk_end_sample):

    chunk_slice = instrument_silence_array[chunk_start_sample:chunk_end_sample]
    return np.any(chunk_slice)  # Check if any sample in the chunk is non-silent

def save_audio_chunk(audio_chunk, output_dir, chunk_index, sr=44100):

    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    file_path = os.path.join(output_dir, f"chunk_{chunk_index}.wav")
    sf.write(file_path, audio_chunk, sr)



def process_tracks_and_chunks(
    saraga,
    track_ids,
    chunk_size_seconds=5,
    sr=44100,
    output_directory="G:/.shortcut-targets-by-id/17yphSXB2IgKWLJF-VDo9xJDWWM2e6mkH/S104/dataset/audio_chunks",
    metadata_csv_path="G:/.shortcut-targets-by-id/17yphSXB2IgKWLJF-VDo9xJDWWM2e6mkH/S104/dataset/metadata.csv"
):
    """
    Process multiple tracks, split into chunks, save audio, and build metadata DataFrame.
    """
    metadata_df = pd.DataFrame(columns=[
        "unique_chunk_id", "track_id", "raga", "performer", "performance",
        "chunk_index", "contains_violin", "contains_vocal", "contains_mridangam"
    ])
    os.makedirs(output_directory, exist_ok=True)

    chunk_global_index = 0

    for track_id in track_ids:
        try:
            # Load all audio arrays
            mix_array, vocal_array, violin_array, mridangam_left_array, mridangam_right_array = load_all_audio(track_id)

            # Detect silence
            violin_silence = detect_silence(violin_array)
            vocal_silence = detect_silence(vocal_array)
            mridangam_silence = np.logical_or(
                detect_silence(mridangam_left_array),
                detect_silence(mridangam_right_array)
            ).astype(int)

            # Split mixed audio into chunks
            mix_audio_chunks = split_audio_into_chunks(mix_array, chunk_size_seconds, sr=sr)
            num_chunks = len(mix_audio_chunks)

            # For each chunk, annotate and save
            for i, chunk in enumerate(mix_audio_chunks):
                chunk_start_sample = i * int(chunk_size_seconds * sr)
                chunk_end_sample = (i + 1) * int(chunk_size_seconds * sr)

                contains_violin = chunk_contains_instrument(violin_silence, chunk_start_sample, chunk_end_sample)
                contains_vocal = chunk_contains_instrument(vocal_silence, chunk_start_sample, chunk_end_sample)
                contains_mridangam = chunk_contains_instrument(mridangam_silence, chunk_start_sample, chunk_end_sample)

                # Generate unique chunk id
                unique_chunk_id = str(uuid.uuid4())

                # Save audio chunk
                file_path = os.path.join(output_directory, f"chunk_{unique_chunk_id}.wav")
                sf.write(file_path, chunk, sr)

                # Get track metadata
                track_info = get_track_info(track_id)

                # Build row
                row_data = {
                    "unique_chunk_id": unique_chunk_id,
                    "track_id": track_id,
                    "raga": track_info["raga"],
                    "performer": track_info["performer"],
                    "performance": track_info["performance"],
                    "chunk_index": chunk_global_index,
                    "contains_violin": contains_violin,
                    "contains_vocal": contains_vocal,
                    "contains_mridangam": contains_mridangam
                }
                metadata_df = pd.concat([metadata_df, pd.DataFrame([row_data])], ignore_index=True)
                chunk_global_index += 1

        except Exception as e:
            print(f"Error processing track {track_id}: {e}")

    # Save metadata
    metadata_df.to_csv(metadata_csv_path, index=False)
    print(f"Saved metadata to {metadata_csv_path} and audio chunks to {output_directory}")


def load_sample(index, metadata_path="G:/.shortcut-targets-by-id/17yphSXB2IgKWLJF-VDo9xJDWWM2e6mkH/S104/dataset/metadata.csv", audio_dir="G:/.shortcut-targets-by-id/17yphSXB2IgKWLJF-VDo9xJDWWM2e6mkH/S104/dataset/audio_chunks"):

    metadata_df = pd.read_csv(metadata_path)

    sample_row = metadata_df[metadata_df["chunk_index"] == index].iloc[0]

    audio_file_path = os.path.join(audio_dir, f"chunk_{index}.wav")

    audio_array, sr = librosa.load(audio_file_path, sr=44100)

    return audio_array, sr

def get_metadata(index, metadata_path="G:/.shortcut-targets-by-id/17yphSXB2IgKWLJF-VDo9xJDWWM2e6mkH/S104/dataset/metadata.csv"):
    metadata_df = pd.read_csv(metadata_path)

    metadata = metadata_df[metadata_df["chunk_index"] == index].iloc[0]

    return metadata

