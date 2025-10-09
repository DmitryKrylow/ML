import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff

# -----------------------------
# 1. Загрузка и объединение CSV
# -----------------------------
file_path = "C:\\Users\\dima2\\PycharmProjects\\lab4ML\\messidor_features.arff"

# Загружаем arff через scipy
data, meta = arff.loadarff(file_path)

# Преобразуем в DataFrame
df = pd.DataFrame(data)

# Преобразуем бинарный класс из bytes в int
df['Class'] = df['Class'].apply(lambda x: int(x.decode('utf-8')) if isinstance(x, bytes) else int(x))

# Признаки и целевая переменная
X = df.drop(columns=['Class']).values
y = df['Class'].values

print("Данные загружены. Размер:", X.shape)

# -----------------------------------
# 2. Разделение на train/val/test
# -----------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ------------------------------
# 3. Масштабирование признаков
# ------------------------------
scaler = StandardScaler()
scalerMinMax = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_minMax = scalerMinMax.fit_transform(X_train)
X_val_minMax = scalerMinMax.transform(X_val)
X_test_minMax = scalerMinMax.transform(X_test)

# ------------------------------
# 4. Обучение Perceptron
# ------------------------------
perceptron = Perceptron(max_iter=1000, penalty='l1', eta0=0.01, random_state=42)
perceptron.fit(X_train_scaled, y_train)
y_pred_perceptron = perceptron.predict(X_test_scaled)
acc_perceptron = accuracy_score(y_test, y_pred_perceptron)
print("Perceptron Accuracy on test set:", acc_perceptron)

# ------------------------------
# 5. Обучение MLPClassifier
# ------------------------------

activation = ['relu', 'tanh', 'logistic']
optimizers = ['sgd', 'adam', 'lbfgs']
for activator in activation:
    for optimizer in optimizers:
        mlp = MLPClassifier(
            hidden_layer_sizes=(100,50),
            activation=activator,
            solver=optimizer,
            alpha=0.005,
            learning_rate_init=0.01,
            max_iter=5000,
            random_state=42
        )
        mlp.fit(X_train_scaled, y_train)
        y_pred_mlp = mlp.predict(X_test_scaled)
        acc_mlp = accuracy_score(y_test, y_pred_mlp)
        print(f"MLPClassifier Accuracy on test set:{acc_mlp} optimizer {optimizer}, activator {activator}")

        mlp.fit(X_train_minMax, y_train)
        y_pred_mlp = mlp.predict(X_test_minMax)
        acc_mlp = accuracy_score(y_test, y_pred_mlp)
        print(f"MLPClassifier Accuracy on test set (MinMax):{acc_mlp} optimizer {optimizer}, activator {activator}")

# ------------------------------
# 6. Эксперименты MLP
# ------------------------------
learning_rates = [0.0001, 0.001, 0.01]
alphas = [0.0001, 0.001, 0.005]
results = []

for solver in optimizers:
    for lr in learning_rates:
        for alpha in alphas:
            mlp_exp = MLPClassifier(
                hidden_layer_sizes=(100,50),
                activation='relu',
                solver=solver,
                alpha=alpha,
                learning_rate_init=lr,
                max_iter=2000,
                random_state=42
            )
            mlp_exp.fit(X_train_scaled, y_train)
            val_acc = accuracy_score(y_val, mlp_exp.predict(X_val_scaled))
            results.append((solver, lr, alpha, val_acc))

results_df = pd.DataFrame(results, columns=['solver', 'learning_rate', 'alpha', 'val_accuracy'])
print("\nЭкспериментальные результаты для разных оптимизаторов:")
print(results_df)

# ------------------------------
# 7. Визуализация экспериментов
# ------------------------------
for solver in optimizers:
    subset = results_df[results_df['solver'] == solver]
    pivot_table = subset.pivot(index='alpha', columns='learning_rate', values='val_accuracy')
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="viridis")
    plt.title(f"Validation Accuracy ({solver.upper()})")
    plt.ylabel("Alpha (regularization)")
    plt.xlabel("Learning Rate")
    plt.show()

# ------------------------------
# 8. Подбор гиперпараметров MLP
# ------------------------------
from sklearn.model_selection import GridSearchCV

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100,50), (150,75)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'lbfgs', 'sgd'],
    'alpha': [0.0001, 0.001, 0.005],
    'learning_rate_init': [0.0001, 0.001, 0.01],
}

mlp = MLPClassifier(max_iter=10000, random_state=42)

grid_search = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,              # кросс-валидация на 3 фолда
    n_jobs=-1,         # использовать все ядра CPU
    verbose=2
)

print("\n🔍 Запуск подбора параметров (может занять несколько минут)...")
grid_search.fit(X_train_scaled, y_train)

# ------------------------------
# 9. Лучшие результаты
# ------------------------------
print("\nЛучшие параметры MLPClassifier:")
print(grid_search.best_params_)

print("\nЛучшая средняя точность (CV):", grid_search.best_score_)

# Проверка качества на тестовой выборке
best_mlp = grid_search.best_estimator_
test_acc = accuracy_score(y_test, best_mlp.predict(X_test_scaled))
print("\nТочность на тестовой выборке:", test_acc)
