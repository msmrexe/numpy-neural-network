import numpy as np
import sys

# Add src to path to allow imports
sys.path.append(sys.path[0] + '/..')

from src.layers import Linear, ReLU, BatchNorm
from src.losses import MSELoss, SoftmaxCrossEntropyLoss
from src.utils.gradient_check import compute_numerical_gradient, relative_error

def check_linear_layer():
    """Checks the Linear layer's gradients."""
    print("Checking Linear Layer...")
    N, D_in, D_out = 3, 4, 5
    inp = np.random.randn(N, D_in)
    target = np.random.randn(N, D_out) # Using MSELoss
    
    layer = Linear(D_in, D_out)
    loss_fn = MSELoss()

    # Get analytic gradients
    out = layer.forward(inp)
    loss = loss_fn.forward(out, target)
    d_out = loss_fn.backward()
    d_inp_analytic = layer.backward(d_out)
    
    grad_W_analytic = layer.grads['W']
    grad_b_analytic = layer.grads['b']

    # --- Check d_inp ---
    f_inp = lambda x: loss_fn.forward(layer.forward(x), target)
    d_inp_numeric = compute_numerical_gradient(f_inp, inp)
    print(f"  d_inp rel error: {relative_error(d_inp_numeric, d_inp_analytic):.2e}")

    # --- Check dW ---
    f_W = lambda w: (layer.params.__setitem__('W', w), loss_fn.forward(layer.forward(inp), target))[1]
    grad_W_numeric = compute_numerical_gradient(f_W, layer.params['W'])
    print(f"  dW rel error:    {relative_error(grad_W_numeric, grad_W_analytic):.2e}")

    # --- Check db ---
    f_b = lambda b: (layer.params.__setitem__('b', b), loss_fn.forward(layer.forward(inp), target))[1]
    grad_b_numeric = compute_numerical_gradient(f_b, layer.params['b'])
    print(f"  db rel error:    {relative_error(grad_b_numeric, grad_b_analytic):.2e}")
    print("-"*20)

def check_relu_layer():
    """Checks the ReLU layer's gradient."""
    print("Checking ReLU Layer...")
    N, D = 5, 8
    inp = np.random.randn(N, D)
    target = np.random.randn(N, D) # Using MSELoss
    
    layer = ReLU()
    loss_fn = MSELoss()

    # Get analytic gradients
    out = layer.forward(inp)
    loss = loss_fn.forward(out, target)
    d_out = loss_fn.backward()
    d_inp_analytic = layer.backward(d_out)

    # --- Check d_inp ---
    f_inp = lambda x: loss_fn.forward(layer.forward(x), target)
    d_inp_numeric = compute_numerical_gradient(f_inp, inp)
    print(f"  d_inp rel error: {relative_error(d_inp_numeric, d_inp_analytic):.2e}")
    print("-"*20)

def check_batchnorm_layer():
    """Checks the BatchNorm layer's gradients."""
    print("Checking BatchNorm Layer...")
    N, D = 4, 5
    inp = np.random.randn(N, D)
    target = np.random.randn(N, D) # Using MSELoss

    layer = BatchNorm(D)
    layer.mode = 'train' # MUST be in train mode
    loss_fn = MSELoss()
    
    # Get analytic gradients
    out = layer.forward(inp)
    loss = loss_fn.forward(out, target)
    d_out = loss_fn.backward()
    d_inp_analytic = layer.backward(d_out)

    grad_gamma_analytic = layer.grads['gamma']
    grad_beta_analytic = layer.grads['beta']

    # --- Check d_inp ---
    f_inp = lambda x: loss_fn.forward(layer.forward(x), target)
    d_inp_numeric = compute_numerical_gradient(f_inp, inp)
    print(f"  d_inp rel error:   {relative_error(d_inp_numeric, d_inp_analytic):.2e}")

    # --- Check d_gamma ---
    f_gamma = lambda g: (layer.params.__setitem__('gamma', g), loss_fn.forward(layer.forward(inp), target))[1]
    grad_gamma_numeric = compute_numerical_gradient(f_gamma, layer.params['gamma'])
    print(f"  d_gamma rel error: {relative_error(grad_gamma_numeric, grad_gamma_analytic):.2e}")
    
    # --- Check d_beta ---
    f_beta = lambda b: (layer.params.__setitem__('beta', b), loss_fn.forward(layer.forward(inp), target))[1]
    grad_beta_numeric = compute_numerical_gradient(f_beta, layer.params['beta'])
    print(f"  d_beta rel error:  {relative_error(grad_beta_numeric, grad_beta_analytic):.2e}")
    print("-"*20)

def check_loss_functions():
    """Checks the Loss functions' gradients."""
    print("Checking Loss Functions...")
    N, C = 5, 3
    
    # --- SoftmaxCrossEntropyLoss ---
    logits = np.random.randn(N, C)
    target_labels = np.random.randint(C, size=N)
    
    loss_fn_softmax = SoftmaxCrossEntropyLoss()
    
    loss = loss_fn_softmax.forward(logits, target_labels)
    grad_analytic_softmax = loss_fn_softmax.backward()
    
    f_softmax = lambda x: loss_fn_softmax.forward(x, target_labels)
    grad_numeric_softmax = compute_numerical_gradient(f_softmax, logits)
    
    print(f"  SoftmaxCrossEntropyLoss rel error: {relative_error(grad_numeric_softmax, grad_analytic_softmax):.2e}")

    # --- MSELoss ---
    pred = np.random.randn(N, C)
    target_values = np.random.randn(N, C)
    
    loss_fn_mse = MSELoss()
    
    loss = loss_fn_mse.forward(pred, target_values)
    grad_analytic_mse = loss_fn_mse.backward()
    
    f_mse = lambda x: loss_fn_mse.forward(x, target_values)
    grad_numeric_mse = compute_numerical_gradient(f_mse, pred)
    
    print(f"  MSELoss rel error: {relative_error(grad_numeric_mse, grad_analytic_mse):.2e}")
    print("-"*20)


if __name__ == "__main__":
    print("===== Running Gradient Checks =====")
    
    # Set a fixed seed for reproducible checks
    np.random.seed(42)
    
    check_loss_functions()
    check_linear_layer()
    check_relu_layer()
    check_batchnorm_layer()
    
    print("===== Gradient Checks Complete =====")
