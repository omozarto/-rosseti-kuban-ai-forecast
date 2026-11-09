import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score

# 1. Загрузка
df = pd.read_csv('../data/ai4i2020.csv')

# Выбираем признаки (исключаем ID и утечки типа TWF, HDF и т.д.)
features = ['Type', 'Air temperature [K]', 'Process temperature [K]', 
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
target = 'Machine failure'

X = df[features]
y = df[target]

# 2. Предобработка (кодируем категории и масштабируем числа)
numeric_features = ['Air temperature [K]', 'Process temperature [K]', 
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
categorical_features = ['Type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 3. Разбиение (стратифицируем, так как классы несбалансированы)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 4. ИСТИННАЯ БАЗОВАЯ ЛИНИЯ (БЕЗ ИИ)
# ==========================================
print("\n=== 1. БАЗОВАЯ ЛИНИЯ (БЕЗ ИИ - DummyClassifier) ===")
print("Стратегия: всегда предсказывать класс 0 (Норма)")

dummy_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DummyClassifier(strategy='most_frequent'))
])

dummy_pipeline.fit(X_train, y_train)
y_dummy_pred = dummy_pipeline.predict(X_test)

print(f"Recall: {recall_score(y_test, y_dummy_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_dummy_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_dummy_pred):.4f}")
print("(Модель не нашла ни одной поломки, так как всегда говорит 'Норма')")


# ==========================================
# 5. МОДЕЛЬ С ИИ (LOGISTIC REGRESSION)
# ==========================================
print("\n=== 2. МОДЕЛЬ С ИИ (Logistic Regression) ===")

ai_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

ai_pipeline.fit(X_train, y_train)
y_ai_pred = ai_pipeline.predict(X_test)

print(f"Recall: {recall_score(y_test, y_ai_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_ai_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_ai_pred):.4f}")

print("\nПолный отчет по модели с ИИ:")
print(classification_report(y_test, y_ai_pred))