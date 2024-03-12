import sys
import numpy as np
import torch

sys.path.append('../modules/')

from logger import get_logger
from tree_generation import (
    generate_dataset_lognormal, generate_dataset_simplified,
    compute_rho_entropy)
from models import FFNN
from training import training_step
from model_evaluation import compute_accuracy
from plotting import plot_training_history


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    logger = get_logger('root_inference')

    # Generate data.
    logger.info('Generating dataset')

    n_samples = 10000
    k = 5
    q = 3
    sigma = 5.
    eps = 0.02

    # rho, trees, roots, leaves = generate_dataset_lognormal(n_samples, k, q, sigma)
    rho, trees, roots, leaves = generate_dataset_simplified(n_samples, k, q, eps=eps)

    rho_entropy = compute_rho_entropy(rho, q)

    # Train-test split.
    logger.info('Splitting training and test data')

    test_frac = .2

    test_indices = np.random.choice(range(leaves.shape[0]), int(leaves.shape[0] * test_frac), replace=False)
    train_indices = np.array(list(set(range(leaves.shape[0])) - set(test_indices)))

    x_train = torch.nn.functional.one_hot(torch.from_numpy(leaves[train_indices, :]), num_classes=q).to(dtype=torch.float32).to(device=device)
    y_train = torch.nn.functional.one_hot(torch.from_numpy(roots[train_indices]), num_classes=q).to(dtype=torch.float32).to(device=device)
    x_test = torch.nn.functional.one_hot(torch.from_numpy(leaves[test_indices, :]), num_classes=q).to(dtype=torch.float32).to(device=device)
    y_test = torch.nn.functional.one_hot(torch.from_numpy(roots[test_indices]), num_classes=q).to(dtype=torch.float32).to(device=device)

    # Model definition.
    logger.info('Instantiating model')

    dims = [leaves.shape[-1], 64, q]

    model = FFNN(
        dims=dims,
        activation='relu',
        output_activation='softmax',
        batch_normalization=False,
        concatenate_last_dim=True
    ).to(device=device)

    print(f'N params: {sum(p.numel() for p in model.parameters())}')

    loss_fn = torch.nn.CrossEntropyLoss()

    epoch_counter = 0

    training_history = {
        'training_loss': [],
        'val_loss': [],
        'training_accuracy': [],
        'val_accuracy': []
    }

    # Training.
    logger.info('Training')

    learning_rate = 1e-3
    batch_size = 32

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=learning_rate
    )

    # early_stopper = EarlyStopper(
    #     patience=5,
    #     min_delta=0.
    # )
    early_stopper = None

    training_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    n_epochs = 150

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

    baseline_accuracy = 1. / q

    plot_training_history(training_history, baseline_accuracy=baseline_accuracy)

    logger.info(f'Final test accuracy: {training_history["val_accuracy"][-1]} (baseline: {baseline_accuracy})')


if __name__ == '__main__':
    main()
