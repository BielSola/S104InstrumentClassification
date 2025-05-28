import dataset_creation
import feature_extraction
import pandas as pd
    
def combine_features_and_metadata(track_ids, chunk_size=0.25, sr=44100):
    """
    Combines the DataFrames from get_features and process_tracks_and_chunks
    for the given track_ids.
    Returns a merged DataFrame. Optionally saves to CSV if csv_path is provided.
    """
    # Get features DataFrame
    features_df = feature_extraction.get_features(track_ids, chunk_size=chunk_size, sr=sr)
   
    # Get metadata DataFrame
    metadata_df = dataset_creation.process_tracks_and_chunks(track_ids, chunk_size_seconds=chunk_size, sr=sr)
    chunked_metadata_df = dataset_creation.select_90_chunks_per_track(metadata_df)
    
    # Merge on performance, t1, t2 (adjust if you have a better unique key)
    combined_df = pd.merge(features_df, chunked_metadata_df, on=['song', 't1', 't2'], how='inner')
    return combined_df