import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


m = 100
x_data = np.linspace(-3, 3, m)
y_data = 3 + np.sin(x_data) + np.random.uniform(-0.5, 0.5, m)

X = x_data.reshape(-1, 1)
y = y_data.reshape(-1, 1)


def plot_learning_curves(model, X, y, title):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_errors, val_errors = [], []
    for m in range(5, len(X_train) + 1):
        model.fit(X_train[:m], y_train[:m])

        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)

        train_errors.append(np.sqrt(mean_squared_error(y_train[:m], y_train_predict)))
        val_errors.append(np.sqrt(mean_squared_error(y_val, y_val_predict)))

    plt.figure(figsize=(9, 6))
    plt.plot(range(5, len(X_train) + 1), train_errors, "r-", linewidth=2, label="Навчальний набір (Train RMSE)")
    plt.plot(range(5, len(X_train) + 1), val_errors, "b-", linewidth=2, label="Перевірочний набір (Validation RMSE)")
    plt.xlabel("Розмір навчального набору")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 3)
    plt.show()

lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y, "Криві навчання — Лінійна Регресія (Ступінь 1)")

poly_reg_10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression())
])
plot_learning_curves(poly_reg_10, X, y, "Криві навчання — Поліноміальна Регресія (Ступінь 10)")

poly_reg_2 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression())
])
plot_learning_curves(poly_reg_2, X, y, "Криві навчання — Поліноміальна Регресія (Ступінь 2)")