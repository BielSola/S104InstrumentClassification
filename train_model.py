from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import sample_data
import pandas as pd
import pickle

def train_model(list_of_tracks, what_to_predict='contains_violin'):
    print("0")
    metadata_df = sample_data.combine_features_and_metadata(list_of_tracks, chunk_size=0.25, sr=44100)
    print("1")
    X = metadata_df.drop(['song', 't1', 't2', 'contains_violin', 'contains_vocal', 'contains_mridangam'], axis=1)

    if what_to_predict == 'contains_violin':
        y = metadata_df['contains_violin']
    elif what_to_predict == 'contains_vocal':
        y = metadata_df['contains_vocal']
    elif what_to_predict == 'contains_mridangam':
        y = metadata_df['contains_mridangam']
        
    y = y.astype(int)  # Ensure y is of integer type for classification

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))
    


def train_model_csv(csv_file_path, what_to_predict='contains_violin'):
    """
    Train a model using data from a CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file containing training data
        what_to_predict (str): Which instrument to predict ('contains_violin', 'contains_vocal', 'contains_mridangam')
    """
    # Load data from CSV file
    try:
        metadata_df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}")
        return
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    #Prepare features and target variable    
    if what_to_predict == 'contains_violin':
        X = metadata_df.drop(['song', 't1', 't2',"zcr", "spectral_centroid", "bandwidth", 'contains_violin', 'contains_vocal', 'contains_mridangam'], axis=1)
        y = metadata_df['contains_violin']
    elif what_to_predict == 'contains_vocal':
        X = metadata_df.drop(['song', 't1', 't2', "spectral_centroid", "bandwidth", 'contains_violin', 'contains_vocal', 'contains_mridangam'], axis=1)
        y = metadata_df['contains_vocal']
    elif what_to_predict == 'contains_mridangam':
        X = metadata_df.drop(['song', 't1', 't2',"zcr", "spectral_centroid", "bandwidth", 'contains_violin', 'contains_vocal', 'contains_mridangam'], axis=1)
        y = metadata_df['contains_mridangam']
    else:
        print("Invalid prediction target. Choose 'contains_violin', 'contains_vocal', or 'contains_mridangam'")
        return
        
    y = y.astype(int)  # Ensure y is of integer type for classification

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the model
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))
    
    # Save model using pickle
    if what_to_predict == 'contains_violin':
        filename = 'violin_model.pkl'
    elif what_to_predict == 'contains_vocal':
        filename = 'vocal_model.pkl'
    elif what_to_predict == 'contains_mridangam':
        filename = 'mridangam_model.pkl'
    
    with open(filename, 'wb') as f:
        pickle.dump(clf, f)
    
    print(f"Model saved as {filename}")
    return clf
    


