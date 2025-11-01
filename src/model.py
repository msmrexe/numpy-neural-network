import numpy as np
from typing import List, Dict
from src.layers import Layer
from src.losses import Loss

class Sequential:
    """
    A sequential model container.
    Layers are added in order, and data flows through them sequentially.
    """
    def __init__(self, layers: List[Layer], loss_fn: Loss, reg: float = 0.0):
        """
        Args:
            layers (List[Layer]): A list of layer instances.
            loss_fn (Loss): The loss function instance to use.
            reg (float): L2 regularization strength.
        """
        self.layers = layers
        self.loss_fn = loss_fn
        self.reg = reg
        
        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        
        self._collect_parameters()

    def _collect_parameters(self) -> None:
        """
        Iterates through layers and collects all learnable parameters
        and their corresponding gradient dicts into the model's
        `self.params` and `self.grads` dictionaries.
        
        Names are prefixed with `layer_{i}_` to avoid collisions.
        """
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params') and layer.params:
                for param_name, param_val in layer.params.items():
                    key = f"layer_{i}_{param_name}"
                    self.params[key] = param_val
                    # Point self.grads[key] to the layer's grad dict entry
                    self.grads[key] = layer.grads[param_name]
                    
                # Overwrite the layer's dicts to point to the central one
                # This ensures that when a layer updates its self.grads['W'],
                # it's *actually* updating self.grads['layer_i_W']
                layer.params = {k.split(f"layer_{i}_")[-1]: v for k, v in self.params.items() if k.startswith(f"layer_{i}_")}
                layer.grads = {k.split(f"layer_{i}_")[-1]: v for k, v in self.grads.items() if k.startswith(f"layer_{i}_")}

    def forward(self, inp: np.ndarray, mode: str = 'train') -> np.ndarray:
        """
        Performs a forward pass through all layers.
        
        Args:
            inp (np.ndarray): The input data.
            mode (str): 'train' or 'test'. Used to control behavior
                        of layers like BatchNorm.
        
        Returns:
            np.ndarray: The output of the final layer (raw scores/logits).
        """
        out = inp
        for layer in self.layers:
            layer.mode = mode
            out = layer.forward(out)
        return out

    def backward(self, upstream_grad: np.ndarray) -> None:
        """
        Performs a backward pass through all layers.
        Gradients are stored in `self.grads`.
        
        Args:
            upstream_grad (np.ndarray): The gradient from the loss function.
        """
        grad = upstream_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def compute_loss(self, X: np.ndarray, y: np.ndarray, mode: str = 'train') -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Computes the total loss (data + regularization) and gradients.
        
        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Ground truth labels.
            mode (str): 'train' or 'test'.
            
        Returns:
            Tuple[float, Dict]: A tuple containing:
                - loss (float): The total loss.
                - grads (Dict): The dictionary of gradients.
        """
        # 1. Forward pass
        scores = self.forward(X, mode)
        
        # 2. Compute data loss and initial gradient
        data_loss = self.loss_fn.forward(scores, y)
        dscores = self.loss_fn.backward()
        
        # 3. Backward pass (populates self.grads)
        self.backward(dscores)
        
        # 4. Add regularization loss and gradients
        reg_loss = 0.0
        for param_name, param_val in self.params.items():
            # Only regularize weights, not biases or BN params
            if param_name.endswith('_W'):
                reg_loss += 0.5 * self.reg * np.sum(param_val * param_val)
                
                # Add regularization gradient
                # self.grads[param_name] was populated by layer.backward()
                self.grads[param_name] += self.reg * param_val
                
        total_loss = data_loss + reg_loss
        
        return total_loss, self.grads
    
    def __call__(self, inp: np.ndarray, mode: str = 'train') -> np.ndarray:
        """Convenience method to make the model callable."""
        return self.forward(inp, mode)
