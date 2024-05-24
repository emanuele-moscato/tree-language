# WARNING
# This script is still incomplete, plese DON'T USE IT  as it is!

import os
import sys
import torch

sys.path.append('../modules/')

from logger_tree_language import get_logger
from utilities import read_data
from pytorch_utilities import (load_checkpoint,
    replace_decoder_with_classification_head, freeze_encoder_weights)
from model_evaluation import compute_accuracy
from training import train_model


PRETRAINING_DATA_PATH = '../../data/mlm_data/slrm_data/labeled_data_fixed_4_8_1.0_0.00000.npy'
DATA_PATH = '../../data/mlm_data/slrm_data/labeled_data_fixed_validation_4_8_1.0_0.00000.npy'
MODEL_DIR = '../../models/mlm_pretraining_2/'
EXP_ID = 'root_inference_fine_tuning_1'
LOG_FILE_PATH = f'../logs/{EXP_ID}.txt'
SEED = 0
DEVICE_INDEX = 1
N_VAL_SAMPLES = 2000
DECODER_HIDDEN_DIM = [64]
BATCH_SIZE = 32
N_EPOCHS = 200
LEARNING_RATE = 1e-4
BATCH_SIZE = 32


def main():
    logger = get_logger(
        'root_inference_fine_tuning',
        log_file_path=LOG_FILE_PATH
    )

    device = torch.device(
        f"cuda:{DEVICE_INDEX}" if torch.cuda.is_available() else "cpu"
    )

    logger.info(
        'Root inference fine-tuning'
        f' | Experiment ID: {EXP_ID}'
        f' | Log file: {LOG_FILE_PATH}'
        f' | Device index: {DEVICE_INDEX}'
        f' | Model dir: {MODEL_DIR}'
        f' | N epochs: {N_EPOCHS}'
        f' | Batch size: {BATCH_SIZE}'
        f' | Learning rate: {LEARNING_RATE}'
        # f' | Fraction of warmup updates: {WARMUP_UPDATES_FRAC}'
        f' | N validation samples: {N_VAL_SAMPLES}'
    )

    # Read data.
    logger.info(f'Reading training data from: {DATA_PATH}')

    q, k, sigma, epsilon, roots, leaves, _ = read_data(
        DATA_PATH, seed=SEED
    )

    # Train-test split (the data has already been shuffled inside
    # `read_data`).
    leaves_train = leaves[:-N_VAL_SAMPLES, :]
    roots_train = roots[:-N_VAL_SAMPLES]

    leaves_test = leaves[-N_VAL_SAMPLES:, :]
    roots_test = roots[-N_VAL_SAMPLES:]

    # Data preprocessing.
    # Note: for consistency with previous root inference attempts, we keep the
    #       targets as one-hot-encoded vectors.
    leaves_train = torch.from_numpy(leaves_train).to(device=device).to(dtype=torch.int64)
    leaves_test = torch.from_numpy(leaves_test).to(device=device).to(dtype=torch.int64)

    roots_train = torch.nn.functional.one_hot(
        torch.from_numpy(roots_train).to(dtype=torch.int64), num_classes=q
    ).to(dtype=torch.float32).to(device=device)
    roots_test = torch.nn.functional.one_hot(
        torch.from_numpy(roots_test).to(dtype=torch.int64), num_classes=q
    ).to(dtype=torch.float32).to(device=device)

    logger.info(
        f'N training samples: {leaves_train.shape[0]}'
        f' | N test samples: {leaves_test.shape[0]}'
    )

    # Load model.
    logger.info(f'Loading model from: {MODEL_DIR}')

    checkpoint_epochs = sorted([
        int(f.split('_')[-1].split('.')[0])
        for f in os.listdir(MODEL_DIR)
        if '.pt' in f
    ])

    selected_checkpoint_epoch = checkpoint_epochs[-1]

    checkpoint_id = [f for f in os.listdir(MODEL_DIR) if f'{selected_checkpoint_epoch}.pt' in f][0]

    logger.info(f'Selected checkpoint: {checkpoint_id}')

    pretrained_model, _, _ = load_checkpoint(
        MODEL_DIR,
        checkpoint_id,
        device=device
    )

    # Replace the model's decoder head with a newly-initialized one and freeze
    # the encoder's weights.
    logger.info(
        'Replacing the decoder head with a FFNN with hidden layers: '
        f'{DECODER_HIDDEN_DIM}'
    )

    classification_model = replace_decoder_with_classification_head(
        pretrained_model,
        n_classes=q,
        device=device,
        embedding_agg='flatten',
        head_hidden_dim=[64],
        head_activation='relu',
        head_output_activation='identity'
    )

    freeze_encoder_weights(classification_model, trainable_modules=['decoder'])

    # Compute initial accuracy (before training).
    initial_accuracy = compute_accuracy(
        classification_model(leaves_test).detach(),
        roots_test
    ).cpu()

    logger.info(f'Initial validation accuracy: {initial_accuracy}')

    loss_fn = torch.nn.CrossEntropyLoss()

    _, fine_tuning_training_history = train_model(
        model=classification_model,
        training_data=(leaves_train, roots_train),
        test_data=(leaves_test, roots_test),
        n_epochs=N_EPOCHS,
        loss_fn=loss_fn,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        early_stopper=None
    )

if __name__ == '__main__':
    main()