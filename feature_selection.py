import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel



def select_features_from_csv(csv_file, target):
    """
    Selects the most important features from a CSV file using Random Forest.
    The CSV file should contain audio features and a target variable to predict.
    """
    print("Starting feature selection...")

    # Check if the CSV file exists
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print("Error: SampleData.csv not found.")
        return

    mfcc_df = pd.DataFrame()  # definie mfccs variable

    if 'mfccs' in df.columns:
        def parse_mfcc(x):
            try:
                arr = np.array(eval(x)) if isinstance(x, str) else np.array(x)
                if arr.size == 0:
                    return np.zeros(20)  # there are 20 different coefficients of mfcc
                if arr.ndim > 1:
                    arr = arr.flatten()
                return arr
            except:
                return np.zeros(20)
        
        df['mfccs'] = df['mfccs'].apply(parse_mfcc)
        max_mfccs = max(df['mfccs'].apply(len))
        mfcc_df = pd.DataFrame(df['mfccs'].to_list(), columns=[f'mfcc{i+1}' for i in range(max_mfccs)])
        df = pd.concat([df.drop(columns=['mfccs']), mfcc_df], axis=1)

    # define the features
    features_of_interest = ['rms', 'zcr', 'spectral_centroid', 'bandwidth']
    features_of_interest += [f'mfcc{i+1}' for i in range(20)]  # add from mfcc1 to mfcc20

    features_of_interest = [col for col in features_of_interest if col in df.columns]

    print("\n Selected Features:")
    print(features_of_interest)

    # apply a filter to the columns to only include the desired features
    X = df[features_of_interest].copy()
    X = X.select_dtypes(include=[np.number])
    y = df[target]

    # apply RandomForest to select the best features
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # print each importance (contribution of each)
    importances = rf.feature_importances_
    print("\nEach feature importance:")
    for feat, imp in zip(X.columns, importances):
        print(f"{feat}: {imp:.4f}")

    # restrictive adjusting
    selector = SelectFromModel(rf, threshold='median', prefit=True)
    selected_features = X.columns[selector.get_support()]

    print(f"\nSelection of most important features to predict '{target}':")
    print(selected_features.tolist())


