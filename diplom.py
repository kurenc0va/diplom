# -*- coding: utf-8 -*-
"""
Нейросетевой классификатор рамановских спектров сыворотки крови
Данные: Excel с двумя листами (health / heart disease).
Первый столбец – волновые числа (wavenumber),
остальные столбцы – интенсивности спектров.
"""

# ====================== ИМПОРТ БИБЛИОТЕК ======================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, callbacks

import warnings
warnings.filterwarnings('ignore')

# Для воспроизводимости результатов
np.random.seed(42)
tf.random.set_seed(42)

# ====================== 1. ЗАГРУЗКА ДАННЫХ ======================
file_path = r"C:\Users\Поля\OneDrive\Рабочий стол\диплом\Raman_krov_SSZ-zdorovye.xlsx"
print("Загрузка данных из файла:", file_path)

healthy_sheet = "health"
sick_sheet = "heart disease"

# Загрузка данных
df_healthy = pd.read_excel(file_path, sheet_name=healthy_sheet, engine='openpyxl')
df_sick = pd.read_excel(file_path, sheet_name=sick_sheet, engine='openpyxl')

# Переименование первого столбца
df_healthy.rename(columns={df_healthy.columns[0]: "wavenumber"}, inplace=True)
df_sick.rename(columns={df_sick.columns[0]: "wavenumber"}, inplace=True)

# Извлечение волновых чисел
wavenumbers = df_healthy["wavenumber"].values

# Удаление столбца волновых чисел
df_healthy = df_healthy.drop(columns=["wavenumber"])
df_sick = df_sick.drop(columns=["wavenumber"])

print("Здоровые (волновые числа × образцы):", df_healthy.shape)
print("Больные (волновые числа × образцы):", df_sick.shape)

# Транспонирование
df_healthy = df_healthy.T
df_sick = df_sick.T

# Установка волновых чисел в качестве названий столбцов
df_healthy.columns = wavenumbers
df_sick.columns = wavenumbers

print("После транспонирования:")
print("Здоровые (образцы × волновые числа):", df_healthy.shape)
print("Больные (образцы × волновые числа):", df_sick.shape)

# Преобразование в массивы
X_healthy = df_healthy.values.astype(np.float32)
X_sick = df_sick.values.astype(np.float32)

# Создание меток классов
y_healthy = np.zeros(X_healthy.shape[0], dtype=int)
y_sick = np.ones(X_sick.shape[0], dtype=int)

# Объединение данных
X_raw = np.vstack([X_healthy, X_sick])
y = np.hstack([y_healthy, y_sick])

# Перемешивание
indices = np.random.permutation(len(y))
X_raw = X_raw[indices]
y = y[indices]

print("\nОбщая форма X_raw:", X_raw.shape)
print("Распределение классов:", np.bincount(y))

# ====================== 2. ПРЕДОБРАБОТКА ======================
def savgol_smooth(X, window_length=11, polyorder=3):
    """Сглаживание фильтром Савицкого–Голая."""
    if window_length % 2 == 0:
        window_length += 1
    X_smooth = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_smooth[i, :] = signal.savgol_filter(
            X[i, :], window_length, polyorder, mode='nearest'
        )
    return X_smooth


def airPLS(x, lambda_=100, p=0.01, max_iter=10):
    """Вычитание фона методом airPLS."""
    m = len(x)
    w = np.ones(m)
    D = diags([1, -2, 1], [0, -1, -2], shape=(m, m - 2))

    for _ in range(max_iter):
        W = diags(w)
        Z = W + lambda_ * D.dot(D.T)
        z = spsolve(Z, w * x)
        d = x - z
        neg = d[d < 0]

        if len(neg) == 0:
            break

        w_new = p * (d > 0) + (1 - p) * (d < 0)

        if np.linalg.norm(w_new - w) < 1e-6:
            break

        w = w_new

    return z


def subtract_background(X, lambda_=100, p=0.01):
    """Удаление флуоресцентного фона."""
    X_corr = np.zeros_like(X)
    for i in range(X.shape[0]):
        bg = airPLS(X[i, :], lambda_, p)
        X_corr[i, :] = X[i, :] - bg
    return X_corr


def normalize_max(X):
    """Нормализация по максимуму."""
    max_vals = np.max(X, axis=1, keepdims=True)
    max_vals[max_vals == 0] = 1
    return X / max_vals


print("\nПрименение предобработки...")
X_smooth = savgol_smooth(X_raw)
X_bgcorr = subtract_background(X_smooth)
X_norm = normalize_max(X_bgcorr)

# Визуализация спектра
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(wavenumbers, X_raw[0], label='Исходный')
plt.plot(wavenumbers, X_smooth[0], '--', label='Сглаженный')
plt.title('Сглаживание спектра')
plt.xlabel('Wavenumber')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(wavenumbers, X_bgcorr[0], label='Без фона')
plt.plot(wavenumbers, X_norm[0], label='Нормированный')
plt.title('Предобработка спектра')
plt.xlabel('Wavenumber')
plt.legend()

plt.tight_layout()
plt.show()

# ====================== 3. ПОДГОТОВКА ДАННЫХ ======================
le = LabelEncoder()
y = le.fit_transform(y)
num_classes = len(le.classes_)

# Разделение данных
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_norm, y, test_size=0.2, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25,
    stratify=y_train_val, random_state=42
)

print("\nРазмеры выборок:")
print("Обучающая:", X_train.shape)
print("Валидационная:", X_val.shape)
print("Тестовая:", X_test.shape)

# Подготовка для CNN
X_train_cnn = X_train[..., np.newaxis]
X_val_cnn = X_val[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]

# ====================== 4. ПОСТРОЕНИЕ МОДЕЛИ ======================
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv1D(64, 5, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.25),

        layers.Conv1D(128, 5, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.25),

        layers.Conv1D(256, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.5),

        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation='softmax')
    ])
    return model


input_shape = (X_train_cnn.shape[1], 1)
model = create_cnn_model(input_shape, num_classes)
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Коллбэки
early_stop = callbacks.EarlyStopping(
    monitor='val_loss', patience=30, restore_best_weights=True, verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1
)

checkpoint = callbacks.ModelCheckpoint(
    'best_model.keras', monitor='val_accuracy',
    save_best_only=True, verbose=1
)

# ====================== 5. ОБУЧЕНИЕ ======================
print("\nНачало обучения...")
history = model.fit(
    X_train_cnn, y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# ====================== 6. ОЦЕНКА МОДЕЛИ ======================
best_model = keras.models.load_model('best_model.keras')

y_pred_proba = best_model.predict(X_test_cnn)
y_pred = np.argmax(y_pred_proba, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n=== Результаты на тестовой выборке ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# ROC-кривая
if num_classes == 2:
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    print(f"ROC-AUC: {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.legend()
    plt.show()

# Графики обучения
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['loss'], label='train_loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_title('Функция потерь')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='train_acc')
    ax2.plot(history.history['val_accuracy'], label='val_acc')
    ax2.set_title('Точность')
    ax2.legend()

    plt.show()


plot_training_history(history)

# ====================== 7. КРОСС-ВАЛИДАЦИЯ ======================
print("\n=== Кросс-валидация (Stratified 5-Fold) ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_norm, y)):
    print(f"\nFold {fold + 1}")

    X_tr, X_val_cv = X_norm[train_idx], X_norm[val_idx]
    y_tr, y_val_cv = y[train_idx], y[val_idx]

    X_tr_cnn = X_tr[..., np.newaxis]
    X_val_cnn_cv = X_val_cv[..., np.newaxis]

    model_cv = create_cnn_model(input_shape, num_classes)
    model_cv.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model_cv.fit(
        X_tr_cnn, y_tr,
        validation_data=(X_val_cnn_cv, y_val_cv),
        epochs=50,
        batch_size=32,
        verbose=0
    )

    _, acc = model_cv.evaluate(X_val_cnn_cv, y_val_cv, verbose=0)
    cv_scores.append(acc)
    print(f"Validation accuracy: {acc:.4f}")

print(f"\nСредняя точность: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

print("\nРабота завершена.")