# notebooks/eda_ai4i2020.py
# EDA для учебного стенда AI4I 2020
# Проект: Разработка ИИ-системы прогнозирования задержек и ошибок
# в данных на участках РЭС «Россети Кубань»

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Загрузка данных (файл ai4i2020.csv лежит в папке data/)
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ai4i2020.csv')
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("Проект: Россети Кубань — прогнозирование задержек и ошибок")
print("Датасет: AI4I 2020 Predictive Maintenance (учебный стенд)")
print("=" * 60)
print("Размер датасета:", df.shape)

# 2. Проверка качества данных
print("\n--- Пропуски ---")
print(df.isnull().sum())
print("Дубликатов:", df.duplicated().sum())
print("Уникальных UDI:", df['UDI'].nunique())

# 3. Описательные статистики
numeric_cols = ['Air temperature [K]', 'Process temperature [K]',
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
print("\n--- Описательные статистики ---")
print(df[numeric_cols].describe())

# 4. Целевая переменная
print("\n--- Machine failure ---")
print(df['Machine failure'].value_counts())
print("Доля отказов: {:.2%}".format(df['Machine failure'].mean()))

# 5. Режимы отказов
failure_modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
print("\n--- Режимы отказов ---")
for mode in failure_modes:
    c = df[mode].sum()
    print(f"{mode}: {c} ({c/len(df):.2%})")

# 6. Графики
OUT = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(OUT, exist_ok=True)

# Рис. 1. Частота отказов по типу продукта
plt.figure()
tf = df.groupby('Type')['Machine failure'].mean() * 100
ax = tf.plot(kind='bar', color=['orange', 'dodgerblue', 'purple'], edgecolor='black')
plt.title('Рис. 1. Частота отказов по типу продукта (Type)')
plt.ylabel('Доля отказов, %'); plt.xlabel('Type')
for i, v in enumerate(tf):
    ax.text(i, v + 0.1, f'{v:.2f}%', ha='center', fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig1_failure_by_type.png'), dpi=150)
plt.close() # Закрываем график, чтобы перейти к следующему

# Рис. 2. Число инцидентов по режимам отказа
plt.figure()
counts = [df[m].sum() for m in failure_modes]
plt.bar(failure_modes, counts, color=['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854'], edgecolor='black')
plt.title('Рис. 2. Число инцидентов по режимам отказа')
plt.ylabel('Количество')
for i, v in enumerate(counts):
    plt.text(i, v + 2, str(v), ha='center', fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig2_failure_modes.png'), dpi=150)
plt.close()

# Рис. 3. Boxplot Torque и Tool wear по классам
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(x='Machine failure', y='Torque [Nm]', data=df, ax=axes[0], palette=['green','red'])
axes[0].set_title('Torque [Nm] по классам'); axes[0].set_xticklabels(['Норма (0)','Отказ (1)'])
sns.boxplot(x='Machine failure', y='Tool wear [min]', data=df, ax=axes[1], palette=['green','red'])
axes[1].set_title('Tool wear [min] по классам'); axes[1].set_xticklabels(['Норма (0)','Отказ (1)'])
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig3_boxplots.png'), dpi=150)
plt.close()

# Рис. 4. Корреляционная матрица
plt.figure(figsize=(9, 7))
corr = df[numeric_cols + ['Machine failure']].corr()
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
plt.title('Рис. 4. Корреляционная матрица')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig4_correlation.png'), dpi=150)
plt.close()

# Рис. 5. Баланс классов
plt.figure()
cc = df['Machine failure'].value_counts()
ax = cc.plot(kind='bar', color=['#66c2a5','#fc8d62'], edgecolor='black')
plt.title('Рис. 5. Баланс классов (Machine failure)')
plt.ylabel('Количество наблюдений')
plt.xticks(ticks=[0, 1], labels=['Норма (0)','Отказ (1)'], rotation=0)
for i, v in enumerate(cc):
    plt.text(i, v + 100, f'{v}\n({v/len(df):.1%})', ha='center', fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig5_balance.png'), dpi=150)
plt.close()

# Рис. 6. Гистограммы
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col], bins=50, color='royalblue', edgecolor='black', alpha=0.7)
    axes[i].set_title(col)
axes[5].axis('off')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig6_histograms.png'), dpi=150)
plt.close()

print("\nEDA завершён. Все 6 графиков сохранены в папку data/.")