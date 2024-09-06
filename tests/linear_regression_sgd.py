import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


class LinearRegressionSGD:
    def __init__(self, learning_rate=0.01, epochs=1000, batch_size=32):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for epoch in range(self.epochs):
            # Shuffle data at each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                # Get batch of data
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]   
                
                # Predictions
                y_pred = np.dot(X_batch, self.weights) + self.bias
                
                # Compute gradients
                dw = -(2 / len(y_batch)) * np.dot(X_batch.T, (y_batch - y_pred))
                db = -(2 / len(y_batch)) * np.sum(y_batch - y_pred)
                
                print("dw:")
                print(dw)
                print("db:", db)

                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db


            print(f"\nEPOCH: {epoch + 1}")
            print(self.weights)
            print(self.bias)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# Load the dataset
f = open("tests/winequality-white.csv")
f.close()
data = pd.read_csv("tests/winequality-white.csv", sep=';')

# Separate features and target
X = data.iloc[:, :-1].values  # Features (all columns except the last)
y = data.iloc[:, -1].values   # Target (quality score)

# Train the Linear Regression model
model = LinearRegressionSGD(learning_rate=1e-5, epochs=3, batch_size=X.shape[0])
model.fit(X, y)

# Predict on test set
y_pred = model.predict(X)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
print(f'Mean Squared Error on test set: {mse}')

# print(f"MSE for c++ weights:")
# print(mean_squared_error(X@(np.array([0.015, 0.00066, 0.00077, 0.016, 0.000112, 0.0819, 0.3239, 0.00231, 0.0074, 0.0011, 0.024]).T)+ 0.002, y))

