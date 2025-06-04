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
import os
import soundfile as sf


saraga = None

def create_dataset(data_home):
    # Load the dataset
    global saraga
    saraga = mirdata.initialize("saraga_carnatic", data_home=data_home)

    return saraga

def get_number_of_tracks(n=10):
    list_of_track_id = []
    attempts = 0
    max_attempts = n * 10  # Prevent infinite loop if not enough valid tracks

    while len(list_of_track_id) < n and attempts < max_attempts:
        print(f"Attempt {attempts + 1} to find a valid track...")
        track = saraga.choice_track()
        print(f"Selected track ID: {track.track_id}")
        track_id = track.track_id
        # Check all required audio paths
        if (track.audio_violin_path is not None and
            track.audio_vocal_path is not None and
            track.audio_mridangam_left_path is not None and
            track.audio_mridangam_right_path is not None and
            track_id not in list_of_track_id):
            list_of_track_id.append(track_id)
            print(f"Track {track_id} added to the list.")
        attempts += 1

    if len(list_of_track_id) < n:
        print(f"Warning: Only found {len(list_of_track_id)} tracks with all required audio stems.")
    return list_of_track_id


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

def load_violin_audio(track_id):
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

""""
def separate_voice(audio_path, isolated_audio_output_path):
    
    Apply spleeter source separation to input audio
    
    # Load spleeter model for voice separation
    separator = Separator('spleeter:2stems')

    # Perform separation
    separator.separate_to_file(audio_path, isolated_audio_output_path)
"""

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

def load_all_audio(track_id):

    mix_array = load_mixed_audio(track_id)
    vocal_array = load_voice_audio(track_id)
    violin_array = load_violin_audio(track_id)
    mridangam_left_array = load_mridangam_left_audio(track_id)
    mridangam_right_array = load_mridangam_right_audio(track_id)

    return mix_array, vocal_array, violin_array, mridangam_left_array, mridangam_right_array


def process_tracks_and_chunks(
    track_ids,
    chunk_size_seconds=0.25,
    sr=44100
):
    """
    Process multiple tracks, split into chunks, save audio, and build metadata DataFrame.
    """
    metadata_df = pd.DataFrame(columns=[
        "song", "t1", "t2",
     "contains_violin", "contains_vocal", "contains_mridangam"
    ])

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
           

            # For each chunk, annotate and save
            for i, chunk in enumerate(mix_audio_chunks):
                chunk_start_sample = i * int(chunk_size_seconds*sr)
                chunk_end_sample = (i + 1) * int(chunk_size_seconds*sr)

                contains_violin = chunk_contains_instrument(violin_silence, chunk_start_sample, chunk_end_sample)
                contains_vocal = chunk_contains_instrument(vocal_silence, chunk_start_sample, chunk_end_sample)
                contains_mridangam = chunk_contains_instrument(mridangam_silence, chunk_start_sample, chunk_end_sample)

                # Get track metadata
                song = track_id
                
                # Build row
                row_data = {
                    "song": song,
                    "t1": chunk_start_sample,
                    "t2": chunk_end_sample,
                    "contains_violin": int(contains_violin),
                    "contains_vocal": int(contains_vocal),
                    "contains_mridangam": int(contains_mridangam)
                }
                metadata_df = pd.concat([metadata_df, pd.DataFrame([row_data])], ignore_index=True)
                chunk_global_index += 1

        except Exception as e:
            print(f"Error processing track {track_id}: {e}")

    # Save metadata
    return metadata_df

def select_90_chunks_per_track(metadata_df):
    """
    For each track (performance), select up to 30 unique chunks for each instrument (violin, vocal, mridangam),
    maximizing the number of unique (non-overlapping) chunks per instrument. Returns a new DataFrame with up to 90 chunks per track.
    """
    selected_rows = []
    for song, group in metadata_df.groupby('song'):
        group = group.copy()
        # Find chunks for each instrument
        violin_chunks = group[group['contains_violin'] == 1]
        vocal_chunks = group[group['contains_vocal'] == 1]
        mridangam_chunks = group[group['contains_mridangam'] == 1]

        # Select up to 30 unique chunks for each instrument, avoiding overlap if possible
        used_indices = set()
        def pick_chunks(df, n, used):
            # Prefer chunks not already used
            unused = df[~df.index.isin(used)]
            pick = unused.sample(min(n, len(unused)), random_state=42)
            used.update(pick.index)
            # If not enough, allow overlap
            if len(pick) < n:
                needed = n - len(pick)
                overlap = df[df.index.isin(used)]
                if not overlap.empty:
                    pick = pd.concat([pick, overlap.sample(min(needed, len(overlap)), random_state=42)])
            return pick

        violin_sel = pick_chunks(violin_chunks, 500, used_indices)
        vocal_sel = pick_chunks(vocal_chunks, 500, used_indices)
        mridangam_sel = pick_chunks(mridangam_chunks, 500, used_indices)

        selected_rows.append(pd.concat([violin_sel, vocal_sel, mridangam_sel]))

    result_df = pd.concat(selected_rows).sort_values(['song', 't1']).reset_index(drop=True)
    return result_df




