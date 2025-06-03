from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import sample_data
import joblib


def train_model(list_of_tracks, what_to_predict='contains_violin'):
    metadata_df = sample_data.combine_features_and_metadata(list_of_tracks, chunk_size=0.25, sr=44100)
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
    
    if what_to_predict == 'contains_violin':
        return joblib.dump(clf, 'violin_model.pkl')
    elif what_to_predict == 'contains_vocal':
        return joblib.dump(clf, 'vocal_model.pkl')
    elif what_to_predict == 'contains_mridangam':
        return joblib.dump(clf, 'mridangam_model.pkl')



