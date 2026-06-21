# Custom Neural Network Framework from Scratch

A modular, object-oriented Deep Learning framework built entirely from scratch using only Python and NumPy. This project bypasses high-level libraries like PyTorch or Keras to implement the underlying matrix calculus, structural nodes, and adaptive optimization algorithms from first principles.

## 🧠 Architecture Design

Every layer in this framework is modeled as an independent node within a directed computational graph. The architecture enforces a strict object-oriented contract allowing for arbitrary stacking of structural layers, non-linear activation functions, and pluggable optimizers.

- **Layer Contract (Base):** Enforces explicit abstract definitions for `.forward(input_data)` and `.backward(output_gradient, optimizer)`.
- **Structural Nodes:** Features a fully connected `Dense` layer implementing He/Xavier parameter initialization weights and systematic batch-dimension reduction for biases.
- **Activations:** Object-oriented tracking for non-linear activations including **ReLU** and a numerically stable **Sigmoid** activation function.
- **Optimization Algorithms:** Decoupled structural updates featuring standard **SGD** and a full state-tracking implementation of the **Adam (Adaptive Moment Estimation)** optimizer.

## 📈 Performance Verification

To verify the mathematical accuracy of the backpropagation chain, the model architecture was trained against a non-linear, synthetic interleaved "Moons" dataset (a problem impossible to solve using single-layer linear regression models).

### Training Evaluation Log
```text
--- Starting Training on Non-Linear Moons Dataset ---
Epoch 0/500   - Loss: 0.7981
Epoch 100/500 - Loss: 0.0063
Epoch 200/500 - Loss: 0.0017
Epoch 300/500 - Loss: 0.0007
Epoch 400/500 - Loss: 0.0003
Epoch 499/500 - Loss: 0.0002

Training Complete! Final Accuracy: 100.00%