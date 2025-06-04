import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Cargar el archivo CSV con features exportados
df = pd.read_csv('SampleData.csv')

# -----------------------------------------------------
# Asegúrate de expandir las MFCCs en columnas separadas
# -----------------------------------------------------
mfcc_df = pd.DataFrame()  # definimos la variable aunque no existan MFCCs todavía

if 'mfccs' in df.columns:
    df['mfccs'] = df['mfccs'].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)
    mfcc_df = pd.DataFrame(df['mfccs'].to_list(), columns=[f'mfcc_{i+1}' for i in range(df['mfccs'][0].shape[0])])
    df = pd.concat([df.drop(columns=['mfccs']), mfcc_df], axis=1)

# -----------------------------------------------------
# Definir las features de interés
# -----------------------------------------------------
features_of_interest = ['rms', 'zcr', 'spectral_centroid', 'bandwidth']

if not mfcc_df.empty:
    features_of_interest += [f'mfcc_{i+1}' for i in range(mfcc_df.shape[1])]
    
features_of_interest = [col for col in features_of_interest if col in df.columns]

# -----------------------------------------------------
# Elegir la variable objetivo que quieras predecir
# -----------------------------------------------------
target = 'contains_violin'  # o 'contains_vocal' o 'contains_mridangam'

# -----------------------------------------------------
# Filtrar las columnas para que solo incluyan las features deseadas
# -----------------------------------------------------
X = df[features_of_interest].copy()
X = X.select_dtypes(include=[np.number])
y = df[target]

# -----------------------------------------------------
# Entrenar el RandomForest y seleccionar las features más importantes
# -----------------------------------------------------
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X, y)

# Imprimir la importancia de cada feature
importances = rf.feature_importances_
print("\nImportancia de las features:")
for feat, imp in zip(X.columns, importances):
    print(f"{feat}: {imp:.4f}")

# Ajustar el umbral para ser más restrictivo
selector = SelectFromModel(rf, threshold='median', prefit=True)
selected_features = X.columns[selector.get_support()]

print(f"\nFeatures importantes para predecir '{target}':")
print(selected_features.tolist())

