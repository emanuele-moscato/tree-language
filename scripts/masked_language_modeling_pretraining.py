import os
import sys
import json
import numpy as np
import torch

sys.path.append('../modules/')

from logger_tree_language import get_logger
from models import TransformerClassifier
from pytorch_utilities import create_linear_lr_schedulers, lr_linear_update
from masked_language_modeling import train_model_mlm
from plotting import plot_training_history


DATA_PATH = (
    '../data/mlm_data/slrm_data/labeled_data_fixed_4_8_1.0_0.00000.npy'
)
SEED = 0
EXP_ID = 'mlm_pretraining_1'
LOG_FILE_PATH = f'../logs/{EXP_ID}.txt'
MODEL_DIR = f'../models/{EXP_ID}/'
BATCH_SIZE = 32  # 32 is the standard (e.g. in Keras)
N_EPOCHS = 6000
WARMUP_UPDATES_FRAC = 0.15
MASK_RATE = 0.1
CHECKPOINTING_PERIOD_EPOCHS = int(N_EPOCHS) / 20


def main():
    logger = get_logger(
        'masked_language_modeling',
        log_file_path=LOG_FILE_PATH
    )

    logger.info(
        'Masked language modeling training'
        f' | Experiment ID: {EXP_ID}'
        f' | Log file: {LOG_FILE_PATH}'
        f' | Model dir: {MODEL_DIR}'
        f' | Batch size: {BATCH_SIZE}'
        f' | N epochs: {N_EPOCHS}'
        f' | Fraction of warmup updates: {WARMUP_UPDATES_FRAC}'
        f' | Token masking rate: {MASK_RATE}'
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read data.
    logger.info(f'Reading data from: {DATA_PATH}')

    q, k, sigma, epsilon, _, leaves_seeds, _ = np.load(
        DATA_PATH, allow_pickle=True
    )

    shuffled_indices = np.random.choice(
        range(leaves_seeds.shape[1]), leaves_seeds.shape[1], replace=False
    )

    leaves = leaves_seeds[..., SEED].T
    leaves = leaves[shuffled_indices, :]

    # Generate a vocabulary (in this case, "text" is already tokenized).
    vocab = torch.arange(q).to(dtype=torch.int64)
    mask_idx = vocab.max() + 1

    # Enalarge the vocabulary with the special tokens.
    vocab = torch.hstack(
        [vocab, torch.Tensor(mask_idx).to(dtype=torch.int64)]
    )

    # Data preprocessing.
    leaves = torch.from_numpy(leaves).to(device=device).to(dtype=torch.int64)

    # Define model.
    logger.info('Instantiating model')

    seq_len = int(2 ** k)
    embedding_size = 128
    vocab_size = vocab.shape[0]

    # Instantiate model.
    model_params = dict(
        seq_len=seq_len,
        embedding_size=embedding_size,
        n_tranformer_layers=2,  # Good: 4 (also acceptable: 2).
        n_heads=1,  # Good: 1.
        vocab_size=vocab_size,
        encoder_dim_feedforward=2 * embedding_size,
        positional_encoding=True,
        n_special_tokens=1,  # We assume the special tokens correspond to the last `n_special_tokens` indices.
        embedding_agg=None,
        decoder_hidden_sizes=[64],  # Good: [64]
        decoder_activation='relu',  # Good: 'relu'
        decoder_output_activation='identity'
    )

    model = TransformerClassifier(
        **model_params
    ).to(device=device)

    # Instantiate optimizer.
    optimizer_params = dict(
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-08,  # 1e-6 for BERT, 1e-8 for default.
        weight_decay=0  # 0.01 for BERT, 0 for default.
    )

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        **optimizer_params
    )

    # Create a directory in which to save the train model (and checkpoints) and the
    # optimizer's final state.
    if MODEL_DIR is not None:
        # Create model directory if it doesn't exist.
        if not os.path.exists(MODEL_DIR):
            logger.info(f'Creating model directory: {MODEL_DIR}')

            os.makedirs(MODEL_DIR)

        # Save model's and optimizer's (hyper)parameters.
        model_params_path = os.path.join(MODEL_DIR, 'model_params.json')

        with open(model_params_path, 'w') as f:
            json.dump(model_params, f)
        
        optimizer_params_path = os.path.join(MODEL_DIR, 'optimizer_params.json')

        with open(optimizer_params_path, 'w') as f:
            json.dump(optimizer_params, f)

    # Set up learning rate scheduling.
    n_updates_total = int(leaves.shape[0] / BATCH_SIZE) * N_EPOCHS
    n_updates_warmup = int(n_updates_total * WARMUP_UPDATES_FRAC)
    n_updates_decay = n_updates_total - n_updates_warmup

    logger.info(
        f'Setting up learning rate scheduling with {n_updates_warmup} warmup '
        f'update steps and {n_updates_decay} decay update steps'
    )

    lr_scheduler_warmup, lr_scheduler_decay = create_linear_lr_schedulers(
        optimizer,
        n_updates_warmup,
        n_updates_decay
    )
    lr_schedule_fn = lambda update_counter: lr_linear_update(
        update_counter,
        lr_scheduler_warmup,
        lr_scheduler_decay,
        n_updates_warmup
    )

    # Model training.
    model, optimizer, training_history = train_model_mlm(
        sequences=leaves,
        model=model,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        mask_rate=MASK_RATE,
        mask_idx=mask_idx,
        device=device,
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn,
        checkpointing_period_epochs=CHECKPOINTING_PERIOD_EPOCHS,
        model_dir=MODEL_DIR,
        checkpoint_id=EXP_ID,
        tensorboard_log_dir=f'../tensorboard_logs/{EXP_ID}/'
    )

    plot_training_history(
        training_history,
        savefig_dir='../data/mlm_data/',
        exp_id=EXP_ID
    )

if __name__ == '__main__':
    main()