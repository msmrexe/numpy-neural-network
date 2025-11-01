import numpy as np
import pickle
import logging
from typing import Dict, Any, Optional
from src.model import Sequential
from src.optimizer import OPTIMIZERS

logger = logging.getLogger(__name__)

class Solver:
    """
    A Solver encapsulates all the logic necessary for training models.
    It performs stochastic gradient descent using different update rules.
    
    The solver accepts training and validation data and labels so it can
    periodically check accuracy on both training and validation
    data to watch out for overfitting.
    
    To train a model, you will first construct a Solver instance, passing the
    model, dataset, and various options to the constructor. You will then
    call the train() method.
    """
    def __init__(self, model: Sequential, data: Dict[str, np.ndarray], **kwargs):
        """
        Construct a new Solver instance.
        
        Required arguments:
        - model: A Sequential model object.
        - data: A dictionary of training and validation data containing:
          'X_train', 'y_train', 'X_val', 'y_val'
          
        Optional arguments:
        - task_type: 'classification' (default) or 'regression'.
        - update_rule: String name of an update rule in optimizer.py. Default is 'sgd'.
        - optim_config: A dictionary containing hyperparameters for the update rule.
        - lr_decay: A scalar for learning rate decay.
        - batch_size: Size of minibatches.
        - num_epochs: The number of epochs to run for.
        - print_every: Integer; training losses will be printed every N iterations.
        - verbose: Boolean; if false, print nothing.
        - checkpoint_name: If not None, save model checkpoints here every epoch.
        """
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        # Unpack keyword arguments
        self.task_type = kwargs.pop('task_type', 'classification')
        self.update_rule_name = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.num_train_samples = kwargs.pop('num_train_samples', 1000)
        self.num_val_samples = kwargs.pop('num_val_samples', None)
        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        if len(kwargs) > 0:
            raise ValueError(f"Unrecognized arguments: {list(kwargs.keys())}")

        if self.update_rule_name not in OPTIMIZERS:
            raise ValueError(f"Invalid update_rule '{self.update_rule_name}'")
        self.update_rule = OPTIMIZERS[self.update_rule_name]

        self._reset()

    def _reset(self):
        """Set up book-keeping variables for optimization."""
        self.epoch = 0
        self.best_val_metric = -np.inf if self.task_type == 'classification' else np.inf
        self.best_params = {}
        self.loss_history = []
        self.train_metric_history = []
        self.val_metric_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            self.optim_configs[p] = {k: v for k, v in self.optim_config.items()}

    def _step(self):
        """Make a single gradient update."""
        # 1. Get a minibatch
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]
        
        # 2. Compute loss and gradient
        loss, grads = self.model.compute_loss(X_batch, y_batch, mode='train')
        self.loss_history.append(loss)
        
        # 3. Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            
            next_w, next_config = self.update_rule(w, dw, config)
            
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def _save_checkpoint(self):
        """Saves the model state to a file."""
        if self.checkpoint_name is None:
            return
            
        checkpoint = {
            'model_params': self.model.params,
            'epoch': self.epoch,
            'best_val_metric': self.best_val_metric,
            'loss_history': self.loss_history,
            'train_metric_history': self.train_metric_history,
            'val_metric_history': self.val_metric_history,
        }
        filename = f"{self.checkpoint_name}_epoch_{self.epoch}.pkl"
        if self.verbose:
            logger.info(f'Saving checkpoint to "{filename}"')
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)

    def check_metric(self, X: np.ndarray, y: np.ndarray, num_samples: Optional[int] = None) -> float:
        """
        Check accuracy or RMS error of the model on the provided data.
        """
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N // self.batch_size
        if N % self.batch_size != 0:
            num_batches += 1
        
        scores_list = []
        for i in range(num_batches):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            scores_batch = self.model.forward(X[start:end], mode='test')
            scores_list.append(scores_batch)
            
        scores = np.vstack(scores_list)
        
        if self.task_type == 'classification':
            y_pred = np.argmax(scores, axis=1)
            metric = np.mean(y_pred == y)
        else: # regression
            y = y.reshape(scores.shape)
            metric = np.sqrt(np.mean(np.sum((scores - y)**2, axis=1))) # RMS Error
            
        return metric

    def train(self):
        """Run optimization to train the model."""
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        logger.info(f"Starting training for {self.num_epochs} epochs...")
        
        for t in range(num_iterations):
            self._step()

            if self.verbose and t % self.print_every == 0:
                logger.info(f"(Iteration {t+1} / {num_iterations}) loss: {self.loss_history[-1]:.4f}")

            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1

                # Decay learning rate
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

            # Check train and val metrics
            first_it = (t == 0)
            last_it = (t == num_iterations - 1)
            if first_it or last_it or epoch_end:
                train_metric = self.check_metric(self.X_train, self.y_train, num_samples=self.num_train_samples)
                val_metric = self.check_metric(self.X_val, self.y_val, num_samples=self.num_val_samples)
                
                self.train_metric_history.append(train_metric)
                self.val_metric_history.append(val_metric)

                if self.verbose:
                    metric_name = "Acc" if self.task_type == 'classification' else "RMSE"
                    logger.info(f"(Epoch {self.epoch} / {self.num_epochs}) Train {metric_name}: {train_metric:.4f}; Val {metric_name}: {val_metric:.4f}")

                # Keep track of the best model
                is_better = False
                if self.task_type == 'classification':
                    if val_metric > self.best_val_metric:
                        self.best_val_metric = val_metric
                        is_better = True
                else: # regression
                    if val_metric < self.best_val_metric:
                        self.best_val_metric = val_metric
                        is_better = True
                
                if is_better:
                    self.best_params = {k: v.copy() for k, v in self.model.params.items()}
                    if self.verbose:
                        logger.info(f"New best model found with val {metric_name}: {self.best_val_metric:.4f}")
                
                self._save_checkpoint()

        # At the end of training, swap the best params into the model
        self.model.params = self.best_params
        logger.info("Training finished. Best model parameters loaded.")
