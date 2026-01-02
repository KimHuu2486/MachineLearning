import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train: {x_train}, y_train: {y_train}")

print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples: m = {m}")

# plt.scatter(x_train, y_train, marker='x', c='r')

# plt.title("Housing Prices")
# plt.xlabel("Size of house (1000 sqft)")
# plt.ylabel("Price (1000s of dollars)")
# plt.show()

w = 200
b = 100
print(f"Initial weight (w): {w}, Initial bias (b): {b}")

def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b)

x_i = 1.2
cost_1200sqft = w * x_i + b
print(f"Predicted price for a 1200 sqft house: {cost_1200sqft} thousands dollars")

plt.plot(x_train, tmp_f_wb, c='b', label='Model Prediction')

plt.scatter(x_train, y_train, marker='x', c='r', label='Training Data')

plt.title("Housing Prices with Model Prediction")
plt.xlabel("Size of house (1000 sqft)")
plt.ylabel("Price (1000s of dollars)")
plt.legend()
plt.show()

