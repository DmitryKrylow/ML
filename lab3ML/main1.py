import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

# === 1. Загрузка данных ===
data = pd.read_csv("CASP.csv")

# RMSD — целевая переменная
y = data.iloc[:, 0].values      # RMSD
X = data.iloc[:, 1:].values     # F1–F9

# === 2. Разделение выборки ===
def my_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    test_size = int(n_samples * test_size)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = my_train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# === 3. Линейная регрессия ===
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("R^2 на тестовой выборке (линейная):", r2_score(y_test, y_pred))
print("MSE линейная:", mean_squared_error(y_test, y_pred))

# === 4. Полиномиальная регрессия ===
degrees = [1,2,3,5]
train_scores = []
test_scores = []

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    train_scores.append(r2_score(y_train, y_train_pred))
    test_scores.append(r2_score(y_test, y_test_pred))

    print(f"MSE Полиномиальная (Train) degree {d}:", mean_squared_error(y_train, y_train_pred))
    print(f"MSE: Полиномиальная (Test) degree {d}:", mean_squared_error(y_test, y_test_pred))

plt.figure(figsize=(8, 5))
plt.plot(degrees, train_scores, marker='o', label="Train R²")
plt.plot(degrees, test_scores, marker='s', label="Test R²")
plt.xlabel("Степень полинома")
plt.ylabel("Точность (R²)")
plt.title("Полиномиальная регрессия: качество от степени")
plt.legend()
plt.grid(True)
plt.ylim(-1, 1)
plt.show()

# === 5. Регуляризация (Ridge и Lasso) ===
alphas = np.logspace(-4, 5, 20)
ridge_train_scores = []
ridge_test_scores = []
lasso_train_scores = []
lasso_test_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge_train = ridge.predict(X_train_scaled)
    y_pred_ridge_test = ridge.predict(X_test_scaled)
    ridge_train_scores.append(r2_score(y_train, y_pred_ridge_train))
    ridge_test_scores.append(r2_score(y_test, y_pred_ridge_test))

    print(f"MSE RIDGE (Train) alpha{alpha}:", mean_squared_error(y_train, y_pred_ridge_train))
    print(f"MSE: RIDGE (Test) alpha{alpha}:", mean_squared_error(y_test, y_pred_ridge_test))

    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)

    y_pred_lasso_train = lasso.predict(X_train_scaled)
    y_pred_lasso_test = lasso.predict(X_test_scaled)
    lasso_train_scores.append(r2_score(y_train, y_pred_lasso_train))
    lasso_test_scores.append(r2_score(y_test, y_pred_lasso_test))

    print(f"MSE LASSO (Train) alpha{alpha}:", mean_squared_error(y_train, y_pred_lasso_train))
    print(f"MSE: LASSO (Test) alpha{alpha}:", mean_squared_error(y_test, y_pred_lasso_test))

plt.figure(figsize=(8, 5))
plt.semilogx(alphas, ridge_train_scores, 'b--', label="Ridge Train")
plt.semilogx(alphas, ridge_test_scores, 'b', label="Ridge Test")
plt.semilogx(alphas, lasso_train_scores, 'r--', label="Lasso Train")
plt.semilogx(alphas, lasso_test_scores, 'r', label="Lasso Test")
plt.xlabel("Коэффициент регуляризации (α)")
plt.ylabel("Точность (R²)")
plt.title("Регуляризация: влияние α на качество модели")
plt.legend()
plt.grid(True)
plt.show()

