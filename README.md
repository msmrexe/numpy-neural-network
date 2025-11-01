# Neural Networks from Scratch (NumPy)

A modular deep learning library built from scratch using only NumPy. This project implements a sequential model API, a variety of layers (Linear, ReLU, BatchNorm), loss functions (MSE, SoftmaxCrossEntropy), and a robust training `Solver` to create and train multi-layer perceptrons for both classification and regression.

This project was developed for a Deep Learning course to demonstrate a foundational understanding of neural network mechanics, from forward propagation to backpropagation and optimization.

## Features

* **Object-Oriented Design:** A clean, "PyTorch-like" API with `Layer`, `Loss`, and `Sequential` base classes.
* **Modular Layers:** Easily stack layers, including `Linear`, `ReLU`, `Sigmoid`, and `BatchNorm`.
* **Robust Training:** A `Solver` class that handles all training, validation, and hyperparameter logic.
* **Optimizers:** Includes `sgd` and `sgd_momentum` update rules.
* **Versatile:** Capable of handling both `classification` (with `SoftmaxCrossEntropyLoss`) and `regression` (with `MSELoss`) tasks.
* **Utilities:** Comes with data loaders for MNIST, Fashion-MNIST, and California Housing, plus a numerical gradient checker for debugging.

## Core Concepts & Techniques

* **Backpropagation:** All layer gradients are analytically derived and implemented from scratch.
* **Batch Normalization:** Implemented as a layer with distinct `train` and `test` modes to stabilize training.
* **Numerical Stability:** Uses a combined `SoftmaxCrossEntropyLoss` to prevent overflow/underflow issues.
* **Modular Architecture:** The `Sequential` model is decoupled from the `Solver`, promoting clean code and reusability.
* **Logging & CLI:** All training scripts use `argparse` for hyperparameter tuning and `logging` to save results to files.

---

## How It Works

This library is composed of several core modules that work together to train a network.

### 1. Core Logic & Architecture

The project is built around two main components: the `Sequential` model and the `Solver`.

* **`src/model.py` (`Sequential`):** This class acts as a container. You initialize it with a list of `Layer` objects and a `Loss` object. It is responsible for:
    * Collecting all learnable parameters (weights, biases, gamma, beta) from its layers into a central `model.params` dictionary.
    * Performing a full forward pass by calling `layer.forward()` sequentially.
    * Performing a full backward pass by calling `layer.backward()` in reverse.
    * Computing the total loss (data loss + regularization).

* **`src/solver.py` (`Solver`):** This is the training engine. You give it the `model` and a `data` dictionary. It handles:
    * The main training loop (epochs, iterations).
    * Creating minibatches of data.
    * Calling `model.compute_loss()` to get the loss and gradients.
    * Calling the optimizer (e.g., `sgd_momentum`) to update every parameter in `model.params`.
    * Tracking loss history, validation metrics, and saving the best model.

### 2. Mathematical Foundations: Backpropagation

Our network is built on **backpropagation**, which is a practical application of the chain rule from calculus. To update a weight `W`, we must find how the final `Loss` $L$ changes with respect to `W` (i.e., $\frac{\partial L}{\partial W}$).

For a simple layer $y = f(x, W)$, the chain rule states:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W}$$

Here, $\frac{\partial L}{\partial y}$ is the "upstream gradient" (coming from the *next* layer) and $\frac{\partial y}{\partial W}$ is the "local gradient" (the derivative of the *current* layer). Each layer's `backward()` pass computes its local gradients, multiplies them by the upstream gradient, and passes the result $\frac{\partial L}{\partial x}$ *downstream* to the previous layer.

### 3. Core Implementations (The Math)

#### Linear Layer
* **Forward:** $y = xW + b$
* **Backward:** The layer receives the upstream gradient $\frac{\partial L}{\partial y}$ and computes three things:
    * $\frac{\partial L}{\partial W} = x^T \cdot \frac{\partial L}{\partial y}$ (Gradient for weights)
    * $\frac{\partial L}{\partial b} = \sum \frac{\partial L}{\partial y}$ (Gradient for biases)
    * $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot W^T$ (Downstream gradient to pass to the next layer)

#### ReLU Activation
* **Forward:** $f(x) = \max(0, x)$
* **Backward:** The local gradient is a simple gate: it is $1$ if $x > 0$ and $0$ otherwise. This means gradients only flow through neurons that were "active" during the forward pass.
    * $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot (x > 0)$

#### Batch Normalization
* **Forward (Train):** Normalizes activations within a batch $B$:
    1.  $\mu_B = \frac{1}{m} \sum_{i \in B} x_i$ (Find batch mean)
    2.  $\sigma^2_B = \frac{1}{m} \sum_{i \in B} (x_i - \mu_B)^2$ (Find batch variance)
    3.  $\hat{x_i} = \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}}$ (Normalize)
    4.  $y_i = \gamma \hat{x_i} + \beta$ (Scale and shift)
* **Backward:** This is the most complex backward pass, as the gradient $\frac{\partial L}{\partial y}$ must be propagated back through $\gamma$, $\beta$, and the normalization statistics ($\mu_B$, $\sigma^2_B$) to the input $x$.

#### Softmax Cross-Entropy Loss
For numerical stability, we combine the final activation and the loss function.
* **Forward:**
    1.  **Softmax:** $P_i = \frac{e^{z_i}}{\sum e^{z_j}}$ (Converts raw scores/logits $z$ to probabilities $P$).
    2.  **Cross-Entropy:** $L = - \frac{1}{N} \sum y_i \log(P_i)$ (Calculates loss, where $y_i$ is 1 for the true class).
* **Backward:** When combined, the derivative $\frac{\partial L}{\partial z}$ simplifies to a clean, stable expression that is perfect for starting backpropagation:
    * $\frac{\partial L}{\partial z} = \frac{1}{N} (P - Y_{onehot})$ (where $Y_{onehot}$ is the one-hot encoded target vector).

---

## Project Structure

```
numpy-neural-network/
├── .gitignore                 # Standard Python .gitignore
├── LICENSE                    # MIT License
├── README.md                  # This readme file
├── requirements.txt           # Project dependencies (numpy, sklearn)
├── notebook.ipynb             # Jupyter Notebook for demonstration
├── logs/                      # Directory for output log files
│   └── .gitkeep
├── src/                       # Main library source code
│   ├── __init__.py
│   ├── layers.py              # Layer implementations (Linear, ReLU, BN)
│   ├── losses.py              # Loss functions (MSE, SoftmaxCrossEntropy)
│   ├── model.py               # Sequential model class
│   ├── optimizer.py           # Update rules (SGD, Momentum)
│   ├── solver.py              # The Solver training class
│   └── utils/                 # Helper modules
│       ├── __init__.py
│       ├── data_utils.py      # Data loading (MNIST, etc.)
│       ├── gradient_check.py  # Numerical gradient checker
│       └── logger.py          # Logging setup
└── scripts/                   # Runnable training scripts
    ├── __init__.py
    ├── check_gradients.py     # Script to debug layer gradients
    ├── train_mnist.py         # Script to train on MNIST
    ├── train_fashion_mnist.py # Script to train on Fashion-MNIST
    └── train_regression.py    # Script to train on California Housing
```

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/msmrexe/numpy-neural-network.git
    cd numpy-neural-network
    ```

2.  **Set up the Environment:**
    (Recommended to use a virtual environment)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Run a Training Script:**
    The `scripts/` folder contains ready-to-run training scripts. You can use `argparse` to change hyperparameters.

    **Example: Train on MNIST**
    ```bash
    python scripts/train_mnist.py --epochs 10 --lr 0.01 --batch_size 128
    ```
    * Logs will be saved to `logs/train_mnist.log`.
    * Progress will be printed to the console.

    **Example: Train on California Housing (Regression)**
    ```bash
    python scripts/train_regression.py --epochs 30 --lr 0.005
    ```
    * Logs will be saved to `logs/train_regression.log`.

4.  **Run the Demonstration Notebook:**
    For a detailed breakdown and manual, step-by-step example of how to use the library, open the Jupyter Notebook:
    ```bash
    jupyter notebook notebook.ipynb
    ```

5.  **Check Layer Gradients (for Debugging):**
    You can verify that all `backward()` passes are implemented correctly by running the gradient checker.
    ```bash
    python scripts/check_gradients.py
    ```
    * You should see very small relative errors (e.g., `< 1e-7`) for all parameters.

---

## Author

Feel free to connect or reach out if you have any questions!

* **Maryam Rezaee**
* **GitHub:** [@msmrexe](https://github.com/msmrexe)
* **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
