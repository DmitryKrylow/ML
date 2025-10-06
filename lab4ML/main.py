import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Загрузка и объединение CSV
# -----------------------------
data_folder = "C:\\Users\\dima2\\PycharmProjects\\lab4ML" 
all_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

df_list = []
for file in all_files:
    df = pd.read_csv(os.path.join(data_folder, file))
    df_list.append(df)

data = pd.concat(df_list, ignore_index=True)
X = data.drop(columns=['Phase']).values
y = data['Phase'].values

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
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 4. Обучение Perceptron
# ------------------------------
perceptron = Perceptron(max_iter=1000, eta0=0.01, random_state=42)
perceptron.fit(X_train_scaled, y_train)
y_pred_perceptron = perceptron.predict(X_test_scaled)
acc_perceptron = accuracy_score(y_test, y_pred_perceptron)
print("Perceptron Accuracy on test set:", acc_perceptron)

# ------------------------------
# 5. Обучение MLPClassifier
# ------------------------------
mlp = MLPClassifier(
    hidden_layer_sizes=(50,),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=2000,
    random_state=42
)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)
acc_mlp = accuracy_score(y_test, y_pred_mlp)
print("MLPClassifier Accuracy on test set:", acc_mlp)

# ------------------------------
# 6. Эксперименты MLP
# ------------------------------
learning_rates = [0.0001, 0.001, 0.01]
alphas = [0.0001, 0.001, 0.01]
results = []

for lr in learning_rates:
    for alpha in alphas:
        mlp_exp = MLPClassifier(
            hidden_layer_sizes=(50,),
            activation='relu',
            solver='adam',
            alpha=alpha,
            learning_rate_init=lr,
            max_iter=2000,
            random_state=42
        )
        mlp_exp.fit(X_train_scaled, y_train)
        val_acc = accuracy_score(y_val, mlp_exp.predict(X_val_scaled))
        results.append((lr, alpha, val_acc))

results_df = pd.DataFrame(results, columns=['learning_rate','alpha','val_accuracy'])
print("\nЭкспериментальные результаты:")
print(results_df)

# ------------------------------
# 7. Визуализация экспериментов
# ------------------------------
pivot_table = results_df.pivot(index='alpha', columns='learning_rate', values='val_accuracy')
plt.figure(figsize=(8,6))
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="viridis")
plt.title("Validation Accuracy for different alpha and learning_rate")
plt.ylabel("Alpha (regularization)")
plt.xlabel("Learning Rate")

plt.show()
