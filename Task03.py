import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

input_file = 'data_multivar_regr.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

num_training = int(0.8 * len(X))
num_test = len(X) - num_training
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)
y_test_pred_linear = linear_regressor.predict(X_test)

print("Linear regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_linear), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred_linear), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred_linear), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred_linear), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred_linear), 2))

polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
X_test_transformed = polynomial.transform(X_test)
poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)
y_test_pred_poly = poly_linear_model.predict(X_test_transformed)

print("\nPolynomial regressor performance (degree 10):")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_poly), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred_poly), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred_poly), 2))

datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.transform(datapoint)

print("\nPredictions for new datapoint (7.75, 6.35, 5.56):")
print("Linear regression prediction:", linear_regressor.predict(datapoint))
print("Polynomial regression prediction:", poly_linear_model.predict(poly_datapoint))

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred_linear, color='blue', label='Linear Regression', alpha=0.7)
plt.scatter(y_test, y_test_pred_poly, color='red', label='Polynomial Regression (deg=10)', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2, label='Ideal Fit')

plt.title("Comparison: Linear vs Polynomial Regression", fontsize=14, fontweight='bold')
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
