import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, Lasso

# Загружаем данные (замени путь на свой)
data = pd.read_csv("CASP.csv")

# Предположим, что последний столбец — целевая переменная
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


def my_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    test_size = int(n_samples * test_size)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = my_train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R^2 на тестовой выборке:", r2_score(y_test, y_pred))

#Полиномиальная регрессия
degrees = range(1, 5)
train_scores = []
test_scores = []

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    train_scores.append(r2_score(y_train, y_train_pred))
    test_scores.append(r2_score(y_test, y_test_pred))

plt.plot(degrees, train_scores, label="Train R^2")
plt.plot(degrees, test_scores, label="Test R^2")
plt.xlabel("Степень полинома")
plt.ylabel("Точность (R^2)")
plt.legend()
plt.show()

#Регуляризация
alphas = np.logspace(-4, 2, 20)  # значения от 1e-4 до 100
ridge_train_scores = []
ridge_test_scores = []

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    ridge_train_scores.append(r2_score(y_train, y_train_pred))
    ridge_test_scores.append(r2_score(y_test, y_test_pred))

plt.semilogx(alphas, ridge_train_scores, label="Train R^2")
plt.semilogx(alphas, ridge_test_scores, label="Test R^2")
plt.xlabel("Коэффициент регуляризации (alpha)")
plt.ylabel("Точность (R^2)")
plt.legend()
plt.show()