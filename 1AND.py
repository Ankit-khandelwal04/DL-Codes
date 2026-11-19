import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.where(x >= 1, 1, 0)

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

Y = np.array([0, 0, 0, 1])

weights = np.array([1.2, 0.6])
threshold = 1.0
bias = 0.0
learning_rate = 0.5
epochs = 10

for epoch in range(epochs):
    print(f"\n--- Epoch {epoch+1} ---")
    errors = 0
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        prediction = step_function(linear_output)
        
        error = Y[i] - prediction
        
        if error != 0:
            errors += 1
            weights += learning_rate * error * X[i]
            
        print(f"Input: {X[i]}, Target: {Y[i]}, Prediction: {prediction}, Weights: {weights}, Bias: {bias:.2f}")
    
    if errors == 0:
        print(f"\nTraining Converged Successfully at Epoch {epoch+1}!")
        break

print("\nFinal Weights:", weights)
print("Final Bias:", bias)

plt.figure(figsize=(6, 6))

# Scatter Plot for Data Points
for i in range(len(X)):
    if Y[i] == 0:
        plt.scatter(X[i, 0], X[i, 1], color='red', marker='o', s=100, 
                    label='Class 0' if i == 0 else "")
    else:
        plt.scatter(X[i, 0], X[i, 1], color='blue', marker='^', s=120, 
                    label='Class 1' if i == 3 else "")

# Plotting the Decision Boundary Line:
# For step_function with threshold 1.0: w1*x1 + w2*x2 + bias = 1.0
# => x2 = (1.0 - bias - w1*x1) / w2
x_values = np.linspace(-0.2, 1.2, 100)
if weights[1] != 0:
    y_values = (threshold - bias - weights[0] * x_values) / weights[1]
    plt.plot(x_values, y_values, 'k--', linewidth=2, label='Decision Boundary')

plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.xlabel("Input x1")
plt.ylabel("Input x2")
plt.title("Perceptron Decision Boundary for AND Gate")
plt.grid(True)
plt.legend()
plt.show()