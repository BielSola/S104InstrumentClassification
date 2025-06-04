import dataset_creation
import librosa
import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
from IPython.display import Audio
import numpy as np
import pandas as pd


def load_audio(audio_path):
    """
    Load an audio file and return the audio time series and sample rate.
    """
    y, sr = librosa.load(audio_path, sr=None)
    return y, sr

def get_audio_duration(y, sr):
    """
    Get the duration of an audio time series in seconds.
    """
    return librosa.get_duration(y=y, sr=sr)

def plot_waveform(y, sr):
    """
    Plot the waveform of an audio time series.
    """
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

def plot_sample_waveform(y, sr, start_time=0, duration=5):
    """
    Plot a sample of the waveform of an audio time series.
    """
    start_sample = int(start_time * sr)
    end_sample = int((start_time + duration) * sr)
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y[start_sample:end_sample], sr=sr)
    plt.title('Waveform (Sample)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

def plot_amplitude_envelope(y, start_sample, end_sample, sr, frame_size=1024):
    sample = y[start_sample:end_sample]
    frames = np.array_split(sample, len(sample) // frame_size)
    
    # Get amplitude envelope
    amplitude_envelope = [np.max(np.abs(frame)) for frame in frames]
    frame_times = np.linspace(0, len(sample) / sr, num=len(amplitude_envelope))
    
    # Plotting
    plt.figure(figsize=(14, 4))
    plt.plot(frame_times, amplitude_envelope)
    plt.title("Amplitude Envelope")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

def compute_rms(y, sr):
    """
    Compute the RMS energy of an audio signal using librosa.
    """
    rms = librosa.feature.rms(y=y)[0]
    frames = range(len(rms))
    times = librosa.frames_to_time(frames, sr=sr)
    return rms.mean(), times

def compute_zcr(y, sr):
    """
    Compute the zero-crossing rate of an audio signal using librosa.
    """
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    frames = range(len(zcr))
    times = librosa.frames_to_time(frames, sr=sr)
    return zcr.mean(), times

def compute_spectral_centroid(y, sr):
    """
    Compute the spectral centroid of an audio signal using librosa.
    """
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    frames = range(len(spectral_centroid))
    times = librosa.frames_to_time(frames, sr=sr)
    return spectral_centroid.mean(), times

def compute_bandwidth(y, sr):
    """
    Compute the spectral bandwidth of an audio signal using librosa.
    """
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    frames = range(len(bandwidth))
    times = librosa.frames_to_time(frames, sr=sr)
    return bandwidth.mean(), times

def compute_mfcc(y, sr):
    """
    Compute the MFCCs of an audio signal using librosa.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    frames = range(mfccs.shape[1])
    times = librosa.frames_to_time(frames, sr=sr)
    return mfccs.mean(axis=1), times



##------

def get_features(list_tracks_id, chunk_size=0.25, sr=44100):
    """
    Get features for a list of tracks.
    """
    features_list = []  # Collect features in a list
    for track_id in list_tracks_id:
        # Load the audio file
        try:
            y = dataset_creation.load_mixed_audio(track_id)
        except:
            continue
        # Get the duration of the audio file
        duration = get_audio_duration(y, sr)
        # Calculate the number of chunks
        num_chunks = int(duration / chunk_size)
        # Loop through each chunk and extract features
        for i in range(num_chunks):
            print(f"Processing chunk {i+1}/{num_chunks} for track")
            start_time = i * chunk_size
            end_time = (i + 1) * chunk_size
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            chunk = y[start_sample:end_sample]
            print("Compute features for chunk")
            rms, _ = compute_rms(chunk, sr)
            zcr, _ = compute_zcr(chunk, sr)
            spectral_centroid, _ = compute_spectral_centroid(chunk, sr)
            bandwidth, _ = compute_bandwidth(chunk, sr)
            mfccs, _ = compute_mfcc(chunk, sr)
            print("Features computed for chunk")
            mfcc_d = {f"mfcc{i+1}":float(mfccs[i]) for i in range(len(mfccs))}
            features = {
                'song': track_id,
                't1': start_sample,
                't2': end_sample,
                'rms': float(rms),
                'zcr': float(zcr),
                'spectral_centroid': float(spectral_centroid),
                'bandwidth': float(bandwidth),
            }
            features.update(mfcc_d)
            print("Features dictionary created for chunk")
            features_list.append(features)  # Add to list
    
    # Create DataFrame ONCE from the complete list
    metadata_df = pd.DataFrame(features_list)
    return metadata_df

def get_features2(path, chunk_size=0.25, sr=44100):
    # Load the audio file
    features_list = []  # Collect features in a list
    y, sr = load_audio(path)  # Unpack tuple to y and sr
    # Get the duration of the audio file
    duration = get_audio_duration(y, sr)
    # Calculate the number of chunks
    num_chunks = int(duration / chunk_size)
    # Loop through each chunk and extract features
    for i in range(num_chunks):
        print(f"Processing chunk {i+1}/{num_chunks} for track")
        start_time = i * chunk_size
        end_time = (i + 1) * chunk_size
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        chunk = y[start_sample:end_sample]
        print("Compute features for chunk")
        rms, _ = compute_rms(chunk, sr)
        zcr, _ = compute_zcr(chunk, sr)
        spectral_centroid, _ = compute_spectral_centroid(chunk, sr)
        bandwidth, _ = compute_bandwidth(chunk, sr)
        mfccs, _ = compute_mfcc(chunk, sr)
        print("Features computed for chunk")
        mfcc_d = {f"mfcc{i+1}":float(mfccs[i]) for i in range(len(mfccs))}
        features = {
            'song': path,
            't1': start_sample,
            't2': end_sample,
            'rms': float(rms),
            'zcr': float(zcr),
            'spectral_centroid': float(spectral_centroid),
            'bandwidth': float(bandwidth),
            }
        features.update(mfcc_d)
        print("Features dictionary created for chunk")
        features_list.append(features)  # Add to list
    
    # Create DataFrame ONCE from the complete list
    metadata_df = pd.DataFrame(features_list)
    return metadata_df