import logging
from tqdm import trange
import torch
from logger_tree_language import get_logger


def mask_sequences(sequences, mask_rate, reshaped_mask_idx, device):
    """
    Performs random masking of the elements of the input sequence,
    substituting them with the token `mask_idx` with probability `mask_rate`.
    """
    # Generate a boolean mask of shape `sequences.shape` with elements
    # having a `mask_rate` probability of being True (corresponding to
    # the elements to mask).
    mask = (torch.rand(size=sequences.shape) < mask_rate).to(device=device)

    # Mask the sequences: replace the elements corresponding to the True
    # entries of the mask with the `mask_idx` index.
    masked_sequences = torch.where(
        mask,
        reshaped_mask_idx,
        sequences
    )
    
    return masked_sequences, mask


def training_step_mlm(batch, masked_batch, mask, model, loss_fn, optimizer):
    """
    Performs a training step for masked language modeling. The predictions are
    computed for the batch (both masked and non-masked tokens), as well as the
    loss (a tensor of values), then we average over the loss values
    corresponding to the masked tokens only.
    """
    # Compute predictions on the masked batch of sequences.
    pred = model(masked_batch)
    
    # Compute the average loss between the predicted masked tokens and the
    # actual ones in the masked positions.
    training_loss = loss_fn(
        torch.permute(
            pred,
            (0, 2, 1)),
        batch
    )[mask].mean()

    # Reset gradients and recompute it.
    optimizer.zero_grad()

    training_loss.backward()

    # Perform an optimization step.
    optimizer.step()
    
    # return training_loss.detach().numpy(), val_loss
    return training_loss.detach()


def compute_masked_accuracy(pred_logits, actual_sequence, mask):
    """
    Compute the reconstruction accuracy on the masked tokens.
    """
    return (
        torch.argmax(pred_logits, axis=-1) == actual_sequence
    )[mask].to(dtype=torch.float32).mean()


def train_model_mlm(
        sequences,
        model,
        n_epochs,
        learning_rate,
        batch_size,
        mask_rate,
        mask_idx,
        device,
        optimizer=None,
        training_history=None
    ):
    """
    Trains a model for mask language modeling. Pass an optimizer to resume
    training with that optimizer, otherwise a new one is initialized.
    """
    logger = get_logger('train_model_mlm', level=logging.INFO)

    logger.info('Training model')

    if training_history is None:
        epoch_counter = 0

        training_history = {
            'training_loss': [],
            'training_accuracy': [],
        }
    else:
        # Resume training from the last epoch, as inferred by the length of
        # the provided training history.
        epoch_counter = len(
            training_history[list(training_history.keys())[0]]
        )

        logger.info(f'Resuming training from epoch {epoch_counter}')

    if optimizer is None:
        logger.info('Instantiating a new optimizer')

        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=learning_rate
        )

    loss_fn = loss_fn = torch.nn.CrossEntropyLoss(
        reduction='none'
    )

    training_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(sequences),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # Broadcast `mask_idx` to the same shape of a batch of sequences and send
    # it to the selected device.
    reshaped_mask_idx = (
        mask_idx.repeat(sequences[:batch_size, ...].shape).to(device=device)
    )

    # Training loop.
    with trange(n_epochs) as pbar:
        for _ in pbar:
            epoch_counter += 1

            training_loss_batches = []
            training_accuracy_batches = []

            for batch in training_loader:
                # In this case, batch is a 1-element list (the batch of
                # sequences).
                batch = batch[0]

                # Perform random masking on the batch.
                masked_batch, mask = mask_sequences(
                    batch,
                    mask_rate,
                    reshaped_mask_idx,
                    device
                )

                training_loss_batch = training_step_mlm(
                    batch,
                    masked_batch,
                    mask,
                    model,
                    loss_fn,
                    optimizer
                )

                training_loss_batches.append(training_loss_batch)

                # Compute the training accuracy over the batch and append it
                # to the corresponding list.
                training_accuracy_batch = compute_masked_accuracy(
                    model(masked_batch),
                    batch,
                    mask
                )
                training_accuracy_batches.append(training_accuracy_batch)

            # Training loss and accuracy for one epoch is computed as the average
            # training loss over the batches.
            training_loss = torch.tensor(training_loss_batches).mean()
            training_accuracy = torch.tensor(training_accuracy_batches).mean()
            
            training_history['training_loss'].append(training_loss)
            training_history['training_accuracy'].append(training_accuracy)

            pbar.set_postfix(
                training_loss=training_history['training_loss'][-1],
                training_accuracy=training_history['training_accuracy'][-1],
            )

    training_history['training_loss'] = torch.tensor(training_history['training_loss']).tolist()
    training_history['training_accuracy'] = torch.tensor(training_history['training_accuracy']).tolist()

    logger.info(f'Last epoch: {epoch_counter}')

    return model, optimizer, training_history