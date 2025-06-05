import feature_extraction
import pickle
import pandas as pd


def convert(result1, result2, result3):
    list_of_results = []
    for i in range((len(result1))):
        list_of_results.append({
            'time': i * 0.25,  # Assuming chunk size is 0.25 seconds
            'contains_violin': result1[i],
            'contains_vocal': result2[i],
            'contains_mridangam': result3[i]
        })
        
    df = pd.DataFrame(list_of_results)
    
    return df
        
def run(audio_path, save_csv_path=None):
    features = feature_extraction.get_features2(audio_path, chunk_size=0.25, sr=44100)

    # Load violin model
    with open('violin_model.pkl', 'rb') as f:
        violin_model = pickle.load(f)
    
    with open('vocal_model.pkl', 'rb') as f:
        vocal_model = pickle.load(f)
    
    with open('mridangam_model.pkl', 'rb') as f:
        mridangam_model = pickle.load(f)
    
    # Remove columns not used in training
    features_for_violin_model = features.drop(columns=['song', 't1', 't2',"zcr", "spectral_centroid", "bandwidth"], errors='ignore')
    features_for_vocal_model = features.drop(columns=['song', 't1', 't2', "spectral_centroid", "bandwidth"], errors='ignore')
    features_for_mridangam_model = features.drop(columns=['song', 't1', 't2',"zcr", "spectral_centroid", "bandwidth"], errors='ignore')
    result_violin = violin_model.predict(features_for_violin_model)
    result_voice = vocal_model.predict(features_for_vocal_model)
    result_mridangam = mridangam_model.predict(features_for_mridangam_model)
    
    final = convert(result_violin, result_voice, result_mridangam)
    
    if save_csv_path:
        final.to_csv(save_csv_path, index=False)
        print(f"Results saved to {save_csv_path}")
    
    return final

