import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

m = 100
x = np.linspace(-3, 3, m)
y = 3 + np.sin(x) + np.random.uniform(-0.5, 0.5, m)

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

lin_reg = LinearRegression()
lin_reg.fit(x, y)
y_lin_pred = lin_reg.predict(x)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(x)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_poly_pred = poly_reg.predict(X_poly)

mse_lin = mean_squared_error(y, y_lin_pred)
r2_lin = r2_score(y, y_lin_pred)
mse_poly = mean_squared_error(y, y_poly_pred)
r2_poly = r2_score(y, y_poly_pred)

print("=== Лінійна регресія ===")
print(f"Рівняння: y = {lin_reg.coef_[0][0]:.3f} * x + ({lin_reg.intercept_[0]:.3f})")
print(f"MSE = {mse_lin:.3f}, R² = {r2_lin:.3f}\n")

print("=== Поліноміальна регресія (ступінь 2) ===")
a1, a2 = poly_reg.coef_[0]
b = poly_reg.intercept_[0]
print(f"Рівняння: y = {a2:.3f} * x² + ({a1:.3f}) * x + ({b:.3f})")
print(f"MSE = {mse_poly:.3f}, R² = {r2_poly:.3f}")

x_new = np.linspace(-3, 3, 200).reshape(-1, 1)
y_lin_new = lin_reg.predict(x_new)
y_poly_new = poly_reg.predict(poly.transform(x_new))

plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', s=20, alpha=0.6, label='Згенеровані дані')
plt.plot(x_new, y_lin_new, 'r--', linewidth=3, label='Лінійна регресія')
plt.title(f"Лінійна Регресія (Ступінь 1)\n$R^2 = {r2_lin:.3f}$ (Варіант №9)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', s=20, alpha=0.6, label='Згенеровані дані')
plt.plot(x_new, y_poly_new, 'orange', linewidth=3, label='Поліноміальна регресія (ступінь 2)')
plt.title(f"Поліноміальна Регресія (Ступінь 2)\n$R^2 = {r2_poly:.3f}$ (Варіант №9)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()