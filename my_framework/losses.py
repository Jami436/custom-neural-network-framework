import numpy as np

class MSE:
    @staticmethod
    def forward(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    
    @staticmethod
    def backward(y_true, y_pred):
        n = len(y_true)
        return (2 / y_true.size) * (y_pred - y_true)
    
class BinaryCrossEntropy:
    @staticmethod
    def forward(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def backward(y_true, y_pred):
        n = len(y_true)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return(1 / n) * ((y_pred - y_true) / (y_pred * (1 - y_pred)))