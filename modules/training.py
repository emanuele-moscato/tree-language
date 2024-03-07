import numpy as np
import torch
from logger import get_logger


def training_step(training_data, model, loss_fn, optimizer, val_data=None):
    """
    Implements a training step for `model` using the algorithm implemented
    in `optimizer` and minimizing the loss `loss_fn`.
    """
    # Unpack the training data.
    x_train, y_train = training_data

    # Compute the loss function on the training data.
    y_pred = model(x_train)
    
    training_loss = loss_fn(y_pred, y_train)

    # Reset gradients and recompute it.
    optimizer.zero_grad()

    training_loss.backward()

    # Perform an optimization step.
    optimizer.step()

    if val_data is None:
        val_loss = None
    else:
        # Unpack the validation data.
        x_val, y_val = val_data

        # Compute validation loss.
        with torch.no_grad():
            y_pred_val = model(x_val)

            # val_loss = loss_fn(y_pred_val, y_val).numpy()
            val_loss = loss_fn(y_pred_val, y_val)
    
    # return training_loss.detach().numpy(), val_loss
    return training_loss.detach(), val_loss


class EarlyStopper:
    """
    Source: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        """
        Returns `True` if the validation loss exceeds its minimal value along
        the training history by more than `min_delta` for more than `patience`
        epochs. Else, returns `False`.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_nn_model(
        training_data,
        training_target,
        nn_model,
        loss_fn,
        optimizer,
        epochs,
        batch_size=None,
        early_stopper=None,
        val_data=None,
        val_target=None
    ):
    logger = get_logger(name='train_nn_model')

    epoch_counter = 0

    training_history = {
        'training_loss': [],
        'val_loss': []
    }

    training_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(training_data, training_target),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    n_batches_per_epoch = len(training_loader)
    
    logger.debug(f'Number of training steps per epoch: {n_batches_per_epoch}')

    # Training loop.
    for i in range(epochs):
        epoch_counter += 1
    
        training_loss_batches = []
    
        for batch in training_loader:
            training_batch, training_targets = batch
        
            training_loss_batch, _ = training_step(
                (training_batch, training_targets),
                nn_model,
                loss_fn,
                optimizer,
            )
    
            training_loss_batches.append(training_loss_batch)
    
        # Training loss for one epoch is computed as the average training
        # loss over the batches.
        training_loss = np.mean(training_loss_batches)
    
        training_history['training_loss'].append(float(training_loss))

        if val_data is not None:
            with torch.no_grad():
                val_loss = loss_fn(nn_model(val_data), val_target).numpy()
        else:
            val_loss = None

        training_history['val_loss'].append(
            float(val_loss) if val_loss is not None else None
        )
        
    
        if (i < 10) or (i % 50 == 0):
            logger.debug(
                f'Epoch: {epoch_counter}'
                f' | Training loss: {training_history["training_loss"][-1]}'
                f' | Validation loss: {training_history["val_loss"][-1]}'
            )

        if (val_data is not None) and (early_stopper is not None):
            if early_stopper.early_stop(training_history['val_loss'][-1]):
                logger.debug(
                    f'Early stopping epoch: {epoch_counter}'
                    f' | Training loss: {training_history["training_loss"][-1]}'
                    f' | Validation loss: {training_history["val_loss"][-1]}'
                )
                
                break
        elif (early_stopper is not None):
            if early_stopper.early_stop(training_history['training_loss'][-1]):
                logger.debug(
                    f'Early stopping epoch: {epoch_counter}'
                    f' | Training loss: {training_history["training_loss"][-1]}'
                )
                
                break

    logger.info(f'Last epoch: {epoch_counter}')

    return training_history