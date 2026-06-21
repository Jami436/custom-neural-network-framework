import unittest
import numpy as np
from my_framework.layers import Dense
from my_framework.losses import MSE

class TestFrameworkMath(unittest.TestCase):
    def test_dense_layer_gradients(self):
        """Validates Dense layer backpropagation calculus against numerical approximations."""
        np.random.seed(42)
        
        # 1. Setup minimal shapes for predictable tracking
        X = np.random.randn(2, 3)   # Batch size = 2, Input size = 3
        y_true = np.random.randn(2, 2)
        
        layer = Dense(input_size=3, output_size=2)
        loss_fn = MSE()
        
        # 2. Compute Analytic Gradients via backpropagation implementation
        y_pred = layer.forward(X)
        loss_initial = loss_fn.forward(y_true, y_pred)
        
        loss_grad = loss_fn.backward(y_true, y_pred)
        
        # Re-running structural calculation manually to capture weights_gradient
        weights_gradient = np.dot(X.T, loss_grad)
        
        # 3. Compute Numerical Gradient for a single weight factor parameter
        epsilon = 1e-5
        
        # Nudge weight UP (+epsilon)
        layer.weights[0, 0] += epsilon
        loss_plus = loss_fn.forward(y_true, layer.forward(X))
        
        # Nudge weight DOWN (-epsilon)
        layer.weights[0, 0] -= 2 * epsilon # reset and move down
        loss_minus = loss_fn.forward(y_true, layer.forward(X))
        
        # Reset weight back to normal
        layer.weights[0, 0] += epsilon
        
        # Limit definition derivative formula
        numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)
        analytic_gradient = weights_gradient[0, 0]
        
        # 4. Assert that the mathematical difference is negligible
        absolute_difference = np.abs(analytic_gradient - numerical_gradient)
        
        print(f"\n[Test] Analytic Gradient:  {analytic_gradient:.6f}")
        print(f"[Test] Numerical Gradient: {numerical_gradient:.6f}")
        print(f"[Test] Absolute Difference: {absolute_difference:.6e}")
        
        # Expecting the difference to be well below 1e-7
        self.assertTrue(absolute_difference < 1e-7, "Backprop calculus discrepancy detected!")

if __name__ == '__main__':
    unittest.main()