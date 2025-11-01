import argparse
import logging
import sys
import numpy as np

# Add src to path to allow imports
sys.path.append(sys.path[0] + '/..')

from src.utils.logger import setup_logger
from src.utils.data_utils import load_california_housing
from src.model import Sequential
from src.layers import Linear, ReLU, BatchNorm
from src.losses import MSELoss
from src.solver import Solver

def main(args):
    """
    Main training and evaluation function for California Housing.
    """
    # 1. Setup Logger
    setup_logger(log_dir="logs", log_file="train_regression.log")
    logger = logging.getLogger(__name__)
    logger.info("Starting California Housing regression script...")
    logger.info(f"Arguments: {vars(args)}")

    # 2. Load Data
    X_train, y_train, X_val, y_val, X_test, y_test = load_california_housing()
    data = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val
    }
    
    in_dim = X_train.shape[1]
    out_dim = y_train.shape[1]
    logger.info(f"Data loaded: Input dim {in_dim}, Output dim {out_dim}")

    # 3. Define Model
    # Architecture: {Linear - BatchNorm - ReLU} x 2 - Linear(1)
    # No activation on the final layer for regression.
    model = Sequential(
        layers=[
            Linear(in_dim, args.hidden_dim),
            BatchNorm(args.hidden_dim),
            ReLU(),
            Linear(args.hidden_dim, args.hidden_dim // 2),
            BatchNorm(args.hidden_dim // 2),
            ReLU(),
            Linear(args.hidden_dim // 2, out_dim)
        ],
        loss_fn=MSELoss(),
        reg=args.reg
    )

    # 4. Configure Solver
    optim_config = {
        'learning_rate': args.lr,
        'momentum': args.momentum
    }
    
    solver = Solver(
        model, data,
        task_type='regression', # CRITICAL: Set task type
        update_rule=args.update_rule,
        optim_config=optim_config,
        lr_decay=args.lr_decay,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        print_every=100,
        verbose=True
    )

    # 5. Train Model
    logger.info("Starting model training...")
    solver.train()
    logger.info("Training complete.")

    # 6. Evaluate on Test Set
    test_rmse = solver.check_metric(X_test, y_test)
    logger.info(f"===== Final Test RMS Error: {test_rmse:.4f} =====")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a regression neural network on the California Housing dataset.")
    
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=5e-3, help='Initial learning rate.')
    parser.add_argument('--lr_decay', type=float, default=0.95, help='Learning rate decay factor per epoch.')
    parser.add_argument('--batch_size', type=int, default=64, help='Minibatch size.')
    parser.add_argument('--reg', type=float, default=1e-4, help='L2 regularization strength.')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Size of the first hidden layer.')
    parser.add_argument('--update_rule', type=str, default='sgd_momentum', choices=['sgd', 'sgd_momentum'], help='Optimizer update rule.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for sgd_momentum.')
    
    args = parser.parse_args()
    main(args)
