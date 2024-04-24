import numpy as np
import torch
import logging
from logger_tree_language import get_logger
from tqdm import trange
from model_evaluation_tree_language import compute_accuracy


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


def train_model(
        model,
        training_data,
        test_data,
        n_epochs,
        loss_fn=torch.nn.CrossEntropyLoss(),
        learning_rate=1e-3,
        batch_size=32,
        early_stopper=None
    ):
    """
    Trains a model.
    """
    logger = get_logger('train_model', level=logging.INFO)

    logger.info('Training model')

    epoch_counter = 0

    training_history = {
        'training_loss': [],
        'val_loss': [],
        'training_accuracy': [],
        'val_accuracy': []
    }

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=learning_rate
    )

    x_train, y_train = training_data
    x_test, y_test = test_data

    training_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # Training loop.
    with trange(n_epochs) as pbar:
        for i in pbar:
            epoch_counter += 1

            training_loss_batches = []
            training_accuracy_batches = []

            for batch in training_loader:
                training_batch, training_targets = batch
            
                training_loss_batch, _ = training_step(
                    (training_batch, training_targets),
                    model,
                    loss_fn,
                    optimizer,
                )

                training_loss_batches.append(training_loss_batch)

                # Compute the training accuracy over the batch and append it to
                # the corresponding list.
                training_accuracy_batch = compute_accuracy(model(training_batch), training_targets)
                training_accuracy_batches.append(training_accuracy_batch)

            # Training loss and accuracy for one epoch is computed as the average
            # training loss over the batches.
            training_loss = torch.tensor(training_loss_batches).mean()
            training_accuracy = torch.tensor(training_accuracy_batches).mean()

            training_history['training_loss'].append(training_loss)
            training_history['training_accuracy'].append(training_accuracy)

            if x_test is not None:
                with torch.no_grad():
                    val_loss = loss_fn(model(x_test), y_test)
                    val_accuracy = compute_accuracy(model(x_test), y_test)
            else:
                val_loss = None
                val_accuracy = None

            training_history['val_loss'].append(
                val_loss if val_loss is not None else None
            )

            training_history['val_accuracy'].append(
                val_accuracy if val_accuracy is not None else None
            )

            pbar.set_postfix(
                training_loss=training_history['training_loss'][-1],
                training_accuracy=training_history['training_accuracy'][-1],
                val_loss=training_history['val_loss'][-1],
                val_accuracy=training_history['val_accuracy'][-1]
            )
            # if (i < 50) or (i % 50 == 0):
            #     logger.debug(
            #         f'Epoch: {epoch_counter}'
            #         f' | Training loss: {training_history["training_loss"][-1]}'
            #         f' | Validation loss: {training_history["val_loss"][-1]}'
            #     )

            if (x_test is not None) and (early_stopper is not None):
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

    training_history['training_loss'] = torch.tensor(training_history['training_loss']).tolist()
    training_history['training_accuracy'] = torch.tensor(training_history['training_accuracy']).tolist()
    training_history['val_loss'] = torch.tensor(training_history['val_loss']).tolist()
    training_history['val_accuracy'] = torch.tensor(training_history['val_accuracy']).tolist()

    logger.info(f'Last epoch: {epoch_counter}')

    return model, training_history
