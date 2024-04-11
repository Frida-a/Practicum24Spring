import numpy as np
from sklearn.linear_model import LinearRegression


def collect_data(n, m):
    X = np.zeros((m, n + 1))  # +1 for log2(q)
    y = np.zeros(m)
    for i in range(m):
        for j in range(n):
            X[i, j] = float(input(f"Enter value for x[{j + 1}] for sample {i + 1}: "))
        X[i, n] = float(input(f"Enter processor count (q) for sample {i + 1}: "))
        y[i] = float(input(f"Enter execution time (T) for sample {i + 1}: "))
    return X, y


def log_transform(X, y):
    return np.log2(X), np.log2(y)


def fit_model(X_log, y_log):
    model = LinearRegression()
    model.fit(X_log, y_log)
    return model


def predict_single_sample(model, sample, n):
    if len(sample) != n + 1:
        raise ValueError("Sample must include input parameters and processor count.")
    log_sample = np.log2(sample).reshape(1, -1)
    log_predicted_time = model.predict(log_sample)
    predicted_time = 2 ** log_predicted_time[0]
    return predicted_time


n = int(input("Enter the number of input parameters (n): "))
m = int(input("Enter the number of data samples: "))

X, y = collect_data(n, m)
X_log, y_log = log_transform(X, y)
model = fit_model(X_log, y_log)

# predict one sample
sample = np.zeros(n + 1)
for i in range(n):
    sample[i] = float(input(f"Enter value for x[{i + 1}]: "))

sample[n] = float(input("Enter processor count (q): "))

# sample = [100, 150, 2]
predicted_time = predict_single_sample(model, sample, n)
print(f"Predicted execution time: {predicted_time}")

