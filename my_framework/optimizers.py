import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layer, weights_gradient, biases_gradient):
        layer.weights -= self.learning_rate * weights_gradient
        layer.biases -= self.learning_rate * biases_gradient


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # State tracking for moments (maps unique layer IDs to their moment matrices)
        self.m_w, self.v_w = {}, {}
        self.m_b, self.v_b = {}, {}
        self.t = 0  # Time step counter

    def update(self, layer, weights_gradient, biases_gradient):
        layer_id = id(layer)
        
        # Initialize moment matrices if this is the first time seeing this layer
        if layer_id not in self.m_w:
            self.m_w[layer_id] = np.zeros_like(layer.weights)
            self.v_w[layer_id] = np.zeros_like(layer.weights)
            self.m_b[layer_id] = np.zeros_like(layer.biases)
            self.v_b[layer_id] = np.zeros_like(layer.biases)
            
        self.t += 1
        
        # --- Update Weights ---
        self.m_w[layer_id] = self.beta1 * self.m_w[layer_id] + (1 - self.beta1) * weights_gradient
        self.v_w[layer_id] = self.beta2 * self.v_w[layer_id] + (1 - self.beta2) * (weights_gradient ** 2)
        
        m_w_corrected = self.m_w[layer_id] / (1 - self.beta1 ** self.t)
        v_w_corrected = self.v_w[layer_id] / (1 - self.beta2 ** self.t)
        
        layer.weights -= (self.learning_rate / (np.sqrt(v_w_corrected) + self.epsilon)) * m_w_corrected

        # --- Update Biases ---
        self.m_b[layer_id] = self.beta1 * self.m_b[layer_id] + (1 - self.beta1) * biases_gradient
        self.v_b[layer_id] = self.beta2 * self.v_b[layer_id] + (1 - self.beta2) * (biases_gradient ** 2)
        
        m_b_corrected = self.m_b[layer_id] / (1 - self.beta1 ** self.t)
        v_b_corrected = self.v_b[layer_id] / (1 - self.beta2 ** self.t)
        
        layer.biases -= (self.learning_rate / (np.sqrt(v_b_corrected) + self.epsilon)) * m_b_corrected