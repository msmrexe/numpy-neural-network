import numpy as np
from typing import Dict, Callable, Any

def sgd(w: np.ndarray, dw: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Performs simple Stochastic Gradient Descent (SGD) update.
    """
    config.setdefault('learning_rate', 1e-3)
    
    next_w = w - config['learning_rate'] * dw
    
    return next_w, config


def sgd_momentum(w: np.ndarray, dw: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Performs Stochastic Gradient Descent (SGD) with Momentum.
    
    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
    - velocity: A numpy array of the same shape as w and dw used to store
                a moving average of the gradients.
    """
    # Set default values
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))
    
    # Momentum update
    mu = config['momentum']
    lr = config['learning_rate']
    
    v = mu * v - lr * dw
    next_w = w + v
    
    # Store updated velocity in config
    config['velocity'] = v
    
    return next_w, config


# --- TODO (later): Add other update rules like RMSprop, Adam here ---

# Dictionary to map string names to update functions
OPTIMIZERS: Dict[str, Callable] = {
    'sgd': sgd,
    'sgd_momentum': sgd_momentum,
}
