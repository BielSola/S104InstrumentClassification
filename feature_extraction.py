import dataset_creation
import librosa
import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
from IPython.display import Audio
import numpy as np


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
    rms = librosa.feature.rms(y)[0]
    frames = range(len(rms))
    times = librosa.frames_to_time(frames, sr=sr)
    return rms, times


def compute_zcr(y, sr):
    """
    Compute the zero-crossing rate of an audio signal using librosa.
    """
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    frames = range(len(zcr))
    times = librosa.frames_to_time(frames, sr=sr)
    return zcr, times

def compute_spectral_centroid(y, sr):
    """
    Compute the spectral centroid of an audio signal using librosa.
    """
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    frames = range(len(spectral_centroid))
    times = librosa.frames_to_time(frames, sr=sr)
    return spectral_centroid, times

def compute_bandwidth(y, sr):
    """
    Compute the spectral bandwidth of an audio signal using librosa.
    """
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    frames = range(len(bandwidth))
    times = librosa.frames_to_time(frames, sr=sr)
    return bandwidth, times

def compute_mfcc(y, sr):
    """
    Compute the MFCCs of an audio signal using librosa.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    frames = range(mfccs.shape[1])
    times = librosa.frames_to_time(frames, sr=sr)
    return mfccs, times

def get_features(list_tracks_id, chunk_size=0.5, sr=44100):
    """
    Get features for a list of tracks.
    """
    features = []
    for track_id in list_tracks_id:
        # Load the audio file
        y, sr = load_audio(track_id)
        # Get the duration of the audio file
        duration = get_audio_duration(y, sr)
        # Calculate the number of chunks
        num_chunks = int(duration / chunk_size)
        # Loop through each chunk and extract features
        for i in range(num_chunks):
            start_time = i * chunk_size
            end_time = (i + 1) * chunk_size
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            chunk = y[start_sample:end_sample]
            rms, rms_times = compute_rms(chunk, sr)
            zcr, zcr_times = compute_zcr(chunk, sr)
            spectral_centroid, centroid_times = compute_spectral_centroid(chunk, sr)
            bandwidth, bandwidth_times = compute_bandwidth(chunk, sr)
            mfccs, mfcc_times = compute_mfcc(chunk, sr)
            features.append({
                'track_id': track_id,
                'start_time': start_time,
                'end_time': end_time,
                'rms': rms,
                'zcr': zcr,
                'spectral_centroid': spectral_centroid,
                'bandwidth': bandwidth,
                'mfccs': mfccs
            })
    return features