import sys
import logging
import torch

sys.path.append('../modules/')

from logger import get_logger
from tree_generation import (calcrho, generate_trees, compute_rho_entropy)
from models import FFNN
from training import training_step
from model_evaluation import compute_accuracy
from plotting import plot_training_history


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_dataset(
        n_samples_training,
        n_samples_test,
        k,
        matrix_type,
        **grammar_kwargs
    ):
    """
    Generates a grammar (transition tensors) and a dataset (set of leaves,
    for training and testing) given the grammar.
    """
    logger = get_logger('generate_dataset')

    logger.info('Generating training dataset')

    q = grammar_kwargs['q']

    rho = calcrho(matrix_type, **grammar_kwargs)
    rho, trees, roots_train, leaves_train = generate_trees(
        rho, n_samples_training, k, q
    )

    logger.info('Generating test dataset')

    _, _, roots_test, leaves_test = generate_trees(
        rho, n_samples_test, k, q
    )

    return rho, roots_train, leaves_train, roots_test, leaves_test


def data_preprocessing(roots_train, leaves_train, roots_test, leaves_test, q):
    """
    Preprocesses the training and test data (roots and leaves), which for now
    is just onw-hot encoding.
    """
    logger = get_logger('data_preprocessing')

    logger.info('Preprocessing data')

    x_train = torch.nn.functional.one_hot(torch.from_numpy(leaves_train), num_classes=q).to(dtype=torch.float32).to(device=device)
    y_train = torch.nn.functional.one_hot(torch.from_numpy(roots_train), num_classes=q).to(dtype=torch.float32).to(device=device)

    x_test = torch.nn.functional.one_hot(torch.from_numpy(leaves_test), num_classes=q).to(dtype=torch.float32).to(device=device)
    y_test = torch.nn.functional.one_hot(torch.from_numpy(roots_test), num_classes=q).to(dtype=torch.float32).to(device=device)

    return x_train, y_train, x_test, y_test


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
    for i in range(n_epochs):
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

        if (i < 50) or (i % 50 == 0):
            logger.debug(
                f'Epoch: {epoch_counter}'
                f' | Training loss: {training_history["training_loss"][-1]}'
                f' | Validation loss: {training_history["val_loss"][-1]}'
            )

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


def main():
    logger = get_logger('root_inference')

    # Generate data.
    n_samples_training = 8000
    n_samples_test = 2000
    k = 5
    q = 4
    matrix_type = 'mixed_index_sets'
    grammar_kwargs = dict(
        q=q,
        sigma=0.1,
        epsilon=0.0
    )

    rho, roots_train, leaves_train, roots_test, leaves_test = (
        generate_dataset(
            n_samples_training=n_samples_training,
            n_samples_test=n_samples_test,
            k=k,
            matrix_type=matrix_type,
            **grammar_kwargs
        )
    )
    rho_entropy = compute_rho_entropy(rho, q)

    logger.info(f'Entropy of the transition matrices: {rho_entropy}')

    # Data preprocessing.
    x_train, y_train, x_test, y_test = data_preprocessing(
        roots_train, leaves_train, roots_test, leaves_test, q
    )

    # Model definition.
    logger.info('Instantiating model')

    dims = [leaves_train.shape[-1], 64, q]

    model = FFNN(
        dims=dims,
        activation='relu',
        output_activation='softmax',
        batch_normalization=False,
        concatenate_last_dim=True
    ).to(device=device)

    logger.info(f'N params (model): {sum(p.numel() for p in model.parameters())}')

    model, training_history = train_model(
        model,
        training_data=(x_train, y_train),
        test_data=(x_test, y_test),
        n_epochs=150,
        loss_fn=torch.nn.CrossEntropyLoss(),
        learning_rate=1e-3,
        batch_size=32,
        early_stopper=None
    )

    baseline_accuracy = 1. / q

    plot_training_history(
        training_history,
        baseline_accuracy=baseline_accuracy,
        savefig_dir='./',
        exp_id='test'
    )

    logger.info(f'Final test accuracy: {training_history["val_accuracy"][-1]} (baseline: {baseline_accuracy})')


if __name__ == '__main__':
    main()
