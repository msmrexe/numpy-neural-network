import numpy as np
from random import randrange
from typing import Callable, Tuple, Any

def compute_numerical_gradient(func: Callable[[np.ndarray], float], 
                               x: np.ndarray, 
                               h: float = 1e-5) -> np.ndarray:
    """
    Computes the numerical gradient of a function `func` at point `x`.
    Uses the centered difference formula: (f(x+h) - f(x-h)) / (2*h)

    Args:
        func (Callable): A function that takes a single numpy array and returns a scalar.
        x (np.ndarray): The point (as a numpy array) at which to compute the gradient.
        h (float): The step size.

    Returns:
        np.ndarray: The numerical gradient, same shape as `x`.
    """
    grad = np.zeros_like(x)
    
    # Use np.nditer for efficient n-dimensional iteration
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        
        original_value = x[idx]
        
        # Evaluate f(x + h)
        x[idx] = original_value + h
        f_x_plus_h = func(x)
        
        # Evaluate f(x - h)
        x[idx] = original_value - h
        f_x_minus_h = func(x)
        
        # Restore original value
        x[idx] = original_value
        
        # Compute partial derivative
        grad[idx] = (f_x_plus_h - f_x_minus_h) / (2 * h)
        
        it.iternext()
        
    return grad

def compute_numerical_gradient_array(func: Callable[[np.ndarray], np.ndarray], 
                                     x: np.ndarray, 
                                     df: np.ndarray, 
                                     h: float = 1e-5) -> np.ndarray:
    """
    Computes numerical gradient for a function that takes a numpy array
    and returns a numpy array.

    Args:
        func (Callable): The function (e.g., forward pass)
        x (np.ndarray): The input array (e.g., weights)
        df (np.ndarray): The upstream derivative (gradient of loss w.r.t. function output)
        h (float): Step size.

    Returns:
        np.ndarray: The numerical gradient of the loss w.r.t. `x`.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        
        original_value = x[idx]
        
        # f(x + h)
        x[idx] = original_value + h
        pos = func(x)
        
        # f(x - h)
        x[idx] = original_value - h
        neg = func(x)
        
        # Restore
        x[idx] = original_value
        
        # Compute gradient
        grad[idx] = np.sum((pos - neg) * df) / (2 * h)
        
        it.iternext()
        
    return grad

def sparse_gradient_check(func: Callable[[np.ndarray], float], 
                          x: np.ndarray, 
                          analytic_grad: np.ndarray, 
                          num_checks: int = 10, 
                          h: float = 1e-5) -> None:
    """
    Performs a sparse gradient check by sampling random elements.
    Compares the numerical gradient to the analytic gradient.

    Args:
        func (Callable): Function that returns a scalar loss.
        x (np.ndarray): The parameter array to check.
        analytic_grad (np.ndarray): The computed analytic gradient.
        num_checks (int): Number of random elements to check.
        h (float): Step size for numerical gradient.
    """
    for _ in range(num_checks):
        # Select a random index to check
        idx = tuple([randrange(m) for m in x.shape])

        original_value = x[idx]
        
        # f(x + h)
        x[idx] = original_value + h
        f_x_plus_h = func(x)
        
        # f(x - h)
        x[idx] = original_value - h
        f_x_minus_h = func(x)
        
        # Restore
        x[idx] = original_value
        
        grad_numerical = (f_x_plus_h - f_x_minus_h) / (2 * h)
        grad_analytic = analytic_grad[idx]
        
        rel_error = relative_error(grad_numerical, grad_analytic)
        
        print(f"Index: {idx}")
        print(f"  Numerical: {grad_numerical:.6e}")
        print(f"  Analytic:  {grad_analytic:.6e}")
        print(f"  Relative Error: {rel_error:.6e}\n")

def relative_error(x: float, y: float) -> float:
    """
    Computes the relative error between two numbers.
    Uses a small epsilon to avoid division by zero.
    """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
