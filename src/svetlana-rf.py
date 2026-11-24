import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score

df = pd.read_csv('../data/ai4i2020.csv')
features = ['Type', 'Air temperature [K]', 'Process temperature [K]', 
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
X = df[features]; y = df['Machine failure']
num_features = features[1:]; cat_features = ['Type']
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])
pipeline.fit(X_train, y_train)
y_val_pred = pipeline.predict(X_val)
print("=== СВЕТЛАНА: Random Forest (Validation) ===")
print(f"Recall: {recall_score(y_val, y_val_pred):.4f}")
print(f"F1-score: {f1_score(y_val, y_val_pred):.4f}")
y_test_pred = pipeline.predict(X_test)
print("\n=== СВЕТЛАНА: Random Forest (Test) ===")
print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_test_pred):.4f}")