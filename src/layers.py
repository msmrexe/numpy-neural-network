import numpy as np
from typing import Dict, Tuple, Optional

class Layer:
    """
    Abstract base class for all neural network layers.
    
    Layers must implement a `forward` and `backward` pass.
    Layers with learnable parameters should also populate `self.params` and `self.grads`.
    """
    def __init__(self):
        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        self.mode: str = 'train' # Default mode is 'train'
    
    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Computes the forward pass, returning the output."""
        raise NotImplementedError

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        """Computes the backward pass, returning the downstream gradient."""
        raise NotImplementedError
    
    def __call__(self, inp: np.ndarray) -> np.ndarray:
        """Convenience method to make layer objects callable."""
        return self.forward(inp)


class Linear(Layer):
    """
    A fully-connected (affine) layer: y = xW + b
    """
    def __init__(self, in_dim: int, out_dim: int, weight_scale: float = 1e-2):
        super().__init__()
        # Initialize parameters
        self.params['W'] = weight_scale * np.random.randn(in_dim, out_dim)
        self.params['b'] = np.zeros(out_dim)
        
        # Gradients will be initialized during the backward pass
        self.grads['W'] = np.zeros_like(self.params['W'])
        self.grads['b'] = np.zeros_like(self.params['b'])
        
        self._cache_input: Optional[np.ndarray] = None

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass for the linear layer.
        
        Args:
            inp (np.ndarray): Input data of shape (N, D_in).
            
        Returns:
            np.ndarray: Output of shape (N, D_out).
        """
        self._cache_input = inp
        out = inp.dot(self.params['W']) + self.params['b']
        return out

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass for the linear layer.
        
        Args:
            upstream_grad (np.ndarray): Gradient from the next layer, shape (N, D_out).
            
        Returns:
            np.ndarray: Downstream gradient (gradient w.r.t. input), shape (N, D_in).
        """
        if self._cache_input is None:
            raise RuntimeError("Must call forward() before backward().")
            
        # Reshape input if it was flattened
        N = self._cache_input.shape[0]
        inp_reshaped = self._cache_input.reshape(N, -1)
        
        # Compute gradients
        self.grads['W'] = inp_reshaped.T.dot(upstream_grad)
        self.grads['b'] = np.sum(upstream_grad, axis=0)
        
        downstream_grad = upstream_grad.dot(self.params['W'].T)
        
        # Reshape downstream gradient to match original input shape
        downstream_grad = downstream_grad.reshape(self._cache_input.shape)
        
        # Clear cache
        self._cache_input = None
        
        return downstream_grad


class ReLU(Layer):
    """
    Rectified Linear Unit (ReLU) activation function: f(x) = max(0, x)
    """
    def __init__(self):
        super().__init__()
        self._cache_input: Optional[np.ndarray] = None

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Computes the forward pass for ReLU."""
        self._cache_input = inp
        out = np.maximum(0, inp)
        return out

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        """Computes the backward pass for ReLU."""
        if self._cache_input is None:
            raise RuntimeError("Must call forward() before backward().")
            
        downstream_grad = upstream_grad * (self._cache_input > 0)
        self._cache_input = None
        return downstream_grad


class Sigmoid(Layer):
    """
    Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
    """
    def __init__(self):
        super().__init__()
        self._cache_output: Optional[np.ndarray] = None

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Computes the forward pass for Sigmoid."""
        out = 1.0 / (1.0 + np.exp(-inp))
        self._cache_output = out
        return out

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        """Computes the backward pass for Sigmoid."""
        if self._cache_output is None:
            raise RuntimeError("Must call forward() before backward().")
            
        sigmoid_deriv = self._cache_output * (1.0 - self._cache_output)
        downstream_grad = upstream_grad * sigmoid_deriv
        self._cache_output = None
        return downstream_grad


class BatchNorm(Layer):
    """
    Batch Normalization layer.
    
    Normalizes the activations of the previous layer at each batch,
    i.e., applies a transformation that maintains the mean activation 
    close to 0 and the activation standard deviation close to 1.
    """
    def __init__(self, dim: int, momentum: float = 0.9, eps: float = 1e-5):
        """
        Args:
            dim (int): The dimension of the input features (D).
            momentum (float): Momentum for updating running mean and variance.
            eps (float): Small constant for numerical stability.
        """
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        
        # Learnable parameters
        self.params['gamma'] = np.ones(dim)  # Scale
        self.params['beta'] = np.zeros(dim)   # Shift
        
        # Gradients
        self.grads['gamma'] = np.zeros_like(self.params['gamma'])
        self.grads['beta'] = np.zeros_like(self.params['beta'])
        
        # Running averages for test-time
        self.running_mean = np.zeros(dim)
        self.running_var = np.zeros(dim)
        
        self._cache: Optional[Tuple] = None

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass for Batch Normalization.
        Uses batch statistics in 'train' mode and running statistics in 'test' mode.
        """
        N, D = inp.shape
        
        if self.mode == 'train':
            # 1. Compute batch mean and variance
            sample_mean = np.mean(inp, axis=0)
            sample_var = np.var(inp, axis=0)
            
            # 2. Normalize
            x_centered = inp - sample_mean
            std_inv = 1.0 / np.sqrt(sample_var + self.eps)
            x_hat = x_centered * std_inv
            
            # 3. Scale and shift
            out = self.params['gamma'] * x_hat + self.params['beta']
            
            # 4. Update running averages
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * sample_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * sample_var
            
            # 5. Store cache for backward pass
            self._cache = (x_hat, x_centered, std_inv, sample_var)
        
        elif self.mode == 'test':
            # Use running averages for normalization
            x_hat = (inp - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.params['gamma'] * x_hat + self.params['beta']
            self._cache = None # No cache needed for test
        
        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be 'train' or 'test'.")
            
        return out

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass for Batch Normalization.
        """
        if self.mode == 'test':
            # In test mode, gradients just pass through the scaling
            # (This is a simplification; a full derivation is complex but
            # in practice, BN layers are frozen during inference)
             return upstream_grad * self.params['gamma'] / np.sqrt(self.running_var + self.eps)
            
        if self._cache is None:
            raise RuntimeError("Must call forward() in 'train' mode before backward().")
            
        N, D = upstream_grad.shape
        x_hat, x_centered, std_inv, sample_var = self._cache
        
        # 1. Gradients w.r.t. beta and gamma
        self.grads['beta'] = np.sum(upstream_grad, axis=0)
        self.grads['gamma'] = np.sum(upstream_grad * x_hat, axis=0)
        
        # 2. Gradient w.r.t. normalized input x_hat
        dx_hat = upstream_grad * self.params['gamma']
        
        # 3. Backprop through normalization
        dx_centered1 = dx_hat * std_inv
        dstd_inv = np.sum(dx_hat * x_centered, axis=0)
        
        dstd = (-1.0 / ((sample_var + self.eps))) * dstd_inv
        dvar = 0.5 * (1.0 / np.sqrt(sample_var + self.eps)) * dstd
        
        d_x_centered_sq = (1.0 / N) * np.ones((N, D)) * dvar
        dx_centered2 = 2.0 * x_centered * d_x_centered_sq
        
        # 4. Backprop through mean subtraction
        dx1 = dx_centered1 + dx_centered2
        dmu = -1.0 * np.sum(dx1, axis=0)
        
        dx2 = (1.0 / N) * np.ones((N, D)) * dmu
        
        # 5. Final downstream gradient w.r.t. input
        downstream_grad = dx1 + dx2
        
        self._cache = None # Clear cache
        return downstream_grad
