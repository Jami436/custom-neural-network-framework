import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- IMPORTS ---
from my_framework import Sequential, Dense, ReLU, Sigmoid, BinaryCrossEntropy, Adam


# 1. Helper function to generate the synthetic "Moons" Data
def make_moons(n_samples=500, noise=0.1):
    np.random.seed(42)
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5
    
    X = np.vstack([np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y)]).T
    X += np.random.normal(0, noise, X.shape)
    
    Y = np.hstack([np.zeros(n_samples_out), np.ones(n_samples_in)]).reshape(-1, 1)
    return X, Y

# Generate the data points X and targets Y
X, Y = make_moons(n_samples=500, noise=0.1)

# Define deep neural architecture
model = Sequential([
    Dense(input_size=2, output_size=8),
    ReLU(),
    Dense(input_size=8, output_size=4),
    ReLU(),
    Dense(input_size=4, output_size=1),
    Sigmoid()
])

# Initialize adaptive optimizer and criterion loss
optimizer = Adam(learning_rate=0.01)
loss_fn = BinaryCrossEntropy

# Execute training pipeline
print("--- Starting Training on Non-Linear Moons Dataset ---")
model.fit(X, Y, epochs=500, loss_class=loss_fn, optimizer=optimizer, batch_size=32)

predictions = model.forward(X)
binary_predictions = (predictions > 0.5).astype(int)
accuracy = np.mean(binary_predictions == Y) * 100
print(f"\nTraining Complete! Final Accuracy: {accuracy:.2f}%")
