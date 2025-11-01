import numpy as np

class Loss:
    """
    Abstract base class for all loss functions.
    """
    def __init__(self):
        self.loss: float = 0.0
        self._cache: Dict[str, np.ndarray] = {}

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Computes the loss and returns it as a scalar."""
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        """Computes the gradient of the loss w.r.t. the prediction."""
        raise NotImplementedError
    
    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Convenience method to make loss objects callable."""
        return self.forward(prediction, target)


class MSELoss(Loss):
    """
    Mean Squared Error (MSE) Loss, typically for regression.
    Loss = (1/N) * sum((prediction - target)^2)
    """
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Computes the MSE loss.
        
        Args:
            prediction (np.ndarray): Predictions, shape (N, D_out).
            target (np.ndarray): Ground truth, shape (N, D_out).
        """
        if prediction.shape != target.shape:
            # Handle cases where regression output is (N, 1) but y is (N,)
            target = target.reshape(prediction.shape)
            
        N = prediction.shape[0]
        if N == 0:
            return 0.0
            
        diff = prediction - target
        self.loss = np.mean(np.sum(diff**2, axis=1))
        
        # Cache for backward pass
        self._cache['diff'] = diff
        self._cache['N'] = np.array(N)
        return self.loss

    def backward(self) -> np.ndarray:
        """
        Computes the gradient of MSE loss w.r.t. the prediction.
        dLoss/dPrediction = (2/N) * (prediction - target)
        """
        diff = self._cache.get('diff')
        N = self._cache.get('N')
        
        if diff is None or N is None:
            raise RuntimeError("Must call forward() before backward().")
        
        grad = (2.0 * diff) / N
        
        self._cache.clear()
        return grad


class SoftmaxCrossEntropyLoss(Loss):
    """
    Combined Softmax activation and Cross-Entropy loss.
    This is more numerically stable than a separate Softmax layer and
    Cross-Entropy loss.
    
    Expects raw logits (scores) as input.
    """
    def forward(self, logits: np.ndarray, target: np.ndarray) -> float:
        """
        Computes the Softmax Cross-Entropy loss.
        
        Args:
            logits (np.ndarray): Raw scores from the last layer, shape (N, C).
            target (np.ndarray): Ground truth labels, shape (N,) as integers.
        """
        N = logits.shape[0]
        if N == 0:
            return 0.0
            
        # 1. Shift logits for numerical stability (prevents overflow)
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        
        # 2. Compute softmax probabilities
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # 3. Compute negative log-likelihood loss
        # We need the log-probabilities of the correct classes
        correct_class_log_probs = -np.log(probs[np.arange(N), target] + 1e-9) # Add epsilon for stability
        
        self.loss = np.mean(correct_class_log_probs)
        
        # 4. Cache for backward pass
        self._cache['probs'] = probs
        self._cache['target'] = target
        self._cache['N'] = np.array(N)
        
        return self.loss

    def backward(self) -> np.ndarray:
        """
        Computes the gradient of the loss w.r.t. the input logits.
        The gradient is (Probs - Y_one_hot) / N.
        """
        probs = self._cache.get('probs')
        target = self._cache.get('target')
        N = self._cache.get('N')
        
        if probs is None or target is None or N is None:
            raise RuntimeError("Must call forward() before backward().")

        # The gradient is remarkably simple: (Probs - Y_one_hot)
        grad = probs.copy()
        grad[np.arange(N), target] -= 1
        
        # Average over the batch
        grad /= N
        
        self._cache.clear()
        return grad
