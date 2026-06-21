import numpy as np

class Sequential:
    def __init__(self, layers=None):
        self.layers = [] if layers is None else layers

    def add(self, layer):
        """Dynamically add a new layer or activation to the execution graph."""
        self.layers.append(layer)

    def forward(self, input_data):
        """Pass inputs sequentially through every single layer node."""
        current_output = input_data
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output

    def backward(self, loss_gradient, optimizer):
        """Propagate gradients backward through the entire layer stack in reverse order."""
        current_gradient = loss_gradient
        for layer in reversed(self.layers):
            current_gradient = layer.backward(current_gradient, optimizer)
        return current_gradient

    def fit(self, X, Y, epochs, loss_class, optimizer, batch_size=None, verbose=True):
        """Automates the entire model training phase loop."""
        num_samples = len(X)
        
        for epoch in range(epochs):
            # Shuffle data at the start of each epoch for better stochastic variance
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            
            # Handle batching options
            if batch_size is None:
                batches_X = [X_shuffled]
                batches_Y = [Y_shuffled]
            else:
                batches_X = np.array_split(X_shuffled, int(np.ceil(num_samples / batch_size)))
                batches_Y = np.array_split(Y_shuffled, int(np.ceil(num_samples / batch_size)))
            
            epoch_loss = 0
            
            for batch_x, batch_y in zip(batches_X, batches_Y):
                # 1. Forward propagation
                predictions = self.forward(batch_x)
                
                # 2. Track total batch loss metrics
                epoch_loss += loss_class.forward(batch_y, predictions) * len(batch_x)
                
                # 3. Calculate initial loss derivative to spark backpropagation
                loss_grad = loss_class.backward(batch_y, predictions)
                
                # 4. Backward propagation step
                self.backward(loss_grad, optimizer)
            
            # Average out scalar loss over total samples
            epoch_loss /= num_samples
            
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}")