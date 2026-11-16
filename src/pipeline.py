import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score

# 1. Загрузка
df = pd.read_csv('../data/ai4i2020.csv')

# Выбираем признаки
features = ['Type', 'Air temperature [K]', 'Process temperature [K]', 
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
target = 'Machine failure'

X = df[features]
y = df[target]

# 2. Предобработка
num_features = features[1:]
cat_features = ['Type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

# 3. Разбиение 60:20:20 (Train / Val / Test) СО СТРАТИФИКАЦИЕЙ
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

# 4. Обучение модели (Logistic Regression)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

# Обучаем на train
pipeline.fit(X_train, y_train)

# 5. Оценка на валидации
y_val_pred = pipeline.predict(X_val)

print("=== ОЦЕНКА НА ВАЛИДАЦИИ (Validation) ===")
print(f"Recall: {recall_score(y_val, y_val_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
print(f"F1-score: {f1_score(y_val, y_val_pred):.4f}")

# 6. Финальная оценка на тесте
y_test_pred = pipeline.predict(X_test)

print("\n=== ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТЕ (Test) ===")
print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_test_pred):.4f}")

print("\nПолный отчет по тестовой выборке:")
print(classification_report(y_test, y_test_pred))