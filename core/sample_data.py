import core.dataset_creation as dataset_creation
import core.feature_extraction as feature_extraction
import pandas as pd
    
def combine_features_and_metadata(track_ids, chunk_size=0.25, sr=44100, save_csv_path="SampleData.csv"):
    """
    Combines the DataFrames from get_features and process_tracks_and_chunks
    for the given track_ids.
    Returns a merged DataFrame. Optionally saves to CSV if csv_path is provided.
    """
    # Get features DataFrame
    print("Extracting features for tracks")
    features_df = feature_extraction.get_features(track_ids, chunk_size=chunk_size, sr=sr)
    print("Features extraction completed")
    # Get metadata DataFrame
    print("Processing tracks and chunks")
    metadata_df = dataset_creation.process_tracks_and_chunks(track_ids, chunk_size_seconds=chunk_size, sr=sr)
    print("Metadata processing completed")
   
    
    # Merge on performance, t1, t2 (adjust if you have a better unique key)
    print("Merging features and metadata")
    combined_df = pd.merge(features_df, metadata_df, on=['song', 't1', 't2'], how='inner')
    print("Merging completed")
    
    # Optionally save to CSV
    
    combined_df.to_csv(save_csv_path, index=False)
    print(f"Combined DataFrame saved to {save_csv_path}")
    
    return combined_df