import os
import sys
import json
import torch

sys.path.append('../modules/')

from logger_tree_language import get_logger
from utilities import read_data
from models import (TransformerClassifier,
    replace_classification_head_with_decoder, freeze_encoder_weights)
from model_evaluation_tree_language import compute_accuracy
from masked_language_modeling import train_model_mlm
from plotting import plot_training_history


CLASSIFICATION_MODEL_PATH = '../models/inverse_pretraining/classification_model_epoch_200.pt'
DATA_PATH = '../data/inverse_pretraining/labeled_data_fixed_4_4_1.0_0.00000.npy'
EXP_ID = 'inverse_pretraining_3'
LOG_FILE_PATH = f'../logs/{EXP_ID}.txt'
MODEL_DIR = f'../models/inverse_pretraining/{EXP_ID}/'
DEVICE_INDEX = 1
SEED = 0
N_TRAINING_SAMPLES = 2 ** 17
N_VAL_SAMPLES = 10000
FREEZE_ENCODER_WEIGHTS = True
BATCH_SIZE = 32  # 32 is the standard (e.g. in Keras)
MASK_RATE = 0.1
SINGLE_MASK = True
N_EPOCHS = 500
CHECKPOINTING_PERIOD_EPOCHS = int(N_EPOCHS) / 1


def main():
    logger = get_logger(
        EXP_ID,
        log_file_path=LOG_FILE_PATH
    )

    device = torch.device(
        f"cuda:{DEVICE_INDEX}" if torch.cuda.is_available() else "cpu"
    )

    logger.info(
        'Inverse pretraining: MLM training starting from a trained '
        'classification model'
        f' | Experiment ID: {EXP_ID}'
        f' | Log file: {LOG_FILE_PATH}'
        f' | Device index: {DEVICE_INDEX}'
        f' | Model dir: {MODEL_DIR}'
        f' | Batch size: {BATCH_SIZE}'
        f' | N epochs: {N_EPOCHS}'
        # f' | Fraction of warmup updates: {WARMUP_UPDATES_FRAC}'
        f' | Token masking rate: {MASK_RATE}'
        f' | Token masking rate: {SINGLE_MASK}'
        f' | N validation samples: {N_VAL_SAMPLES}'
    )

    # Read data.
    logger.info(f'Reading training data from: {DATA_PATH}')

    q, k, sigma, epsilon, roots, leaves, rho = read_data(DATA_PATH, SEED)

    classification_training_roots = roots[:N_TRAINING_SAMPLES]
    classification_training_leaves = torch.from_numpy(leaves[:N_TRAINING_SAMPLES, :]).to(device=device).to(dtype=torch.int64)

    val_roots = torch.nn.functional.one_hot(torch.from_numpy(roots[-N_VAL_SAMPLES:]).to(device=device).to(dtype=torch.int64)).to(dtype=torch.float32)
    val_leaves = torch.from_numpy(leaves[-N_VAL_SAMPLES:]).to(device=device).to(dtype=torch.int64)

    # Generate a vocabulary (in this case, "text" is already tokenized).
    # Original vocabulary used by the classification model.
    classification_vocab = torch.arange(q).to(dtype=torch.int64)
    mask_idx = classification_vocab.max() + 1

    # Extended vocabulary used for MLM (additional token = mask).
    vocab = torch.hstack(
        [classification_vocab, torch.Tensor(mask_idx).to(dtype=torch.int64)]
    )

    # Data preprocessing.
    # classification_training_roots = roots[:2**17]
    classification_training_leaves = (
        torch.from_numpy(leaves[:2**17, :])
        .to(device=device)
        .to(dtype=torch.int64)
    )

    val_roots = torch.nn.functional.one_hot(
        torch.from_numpy(roots[-N_VAL_SAMPLES:])
        .to(device=device)
        .to(dtype=torch.int64)
    ).to(dtype=torch.float32)
    val_leaves = (
        torch.from_numpy(leaves[-N_VAL_SAMPLES:])
        .to(device=device)
        .to(dtype=torch.int64)
    )

    # Load the trained classification model.
    logger.info('Loading trained classification model')

    classification_model = TransformerClassifier(
        seq_len=leaves.shape[-1],
        embedding_size=128,
        n_tranformer_layers=4,
        n_heads=1,
        vocab_size=q,
        encoder_dim_feedforward=2048,
        positional_encoding=True,
        n_special_tokens=0,
        embedding_agg='flatten',
        decoder_hidden_sizes=[64],
        decoder_activation='relu',
        decoder_output_activation='identity',
    )

    # Note: comment the following two lines to avoid loading a pre-trained
    #       classification model, if needed.
    torch_checkpoint = torch.load(CLASSIFICATION_MODEL_PATH)
    classification_model.load_state_dict(torch_checkpoint['model_state_dict'])

    classification_model = classification_model.to(device=device)

    # Check the validation accuracy on the classification task for the trained
    # model.
    logger.info(
        'Classification accuracy (validation set): '
        f'{compute_accuracy(classification_model(val_leaves).detach(), val_roots)}'
    )

    # Replace the loaded model's classification head with a decoder for MLM.
    logger.info("Replacing the model's classification head ")

    mlm_model = replace_classification_head_with_decoder(
        original_model=classification_model,
        n_classes=q,
        device=device,
        decoder_hidden_dim=[64]
    )

    if FREEZE_ENCODER_WEIGHTS:
        logger.info("Freezing the encoder's weights")

        freeze_encoder_weights(mlm_model, trainable_modules=['decoder'])

    # Get the parameters for the MLM model.
    mlm_model_params = dict(
        seq_len=mlm_model.seq_len,
        embedding_size=mlm_model.embedding_size,
        n_tranformer_layers=mlm_model.n_tranformer_layers,
        n_heads=mlm_model.n_heads,
        vocab_size=int(mlm_model.vocab_size),
        encoder_dim_feedforward=mlm_model.encoder_dim_feedforward,
        positional_encoding=mlm_model.positional_encoding,
        n_special_tokens=1,
        embedding_agg=mlm_model.embedding_agg,
        decoder_hidden_sizes=mlm_model.decoder_hidden_sizes,
        decoder_activation=mlm_model.decoder_activation,
        decoder_output_activation=mlm_model.decoder_output_activation
    )

    # Instantiate optimizer.
    optimizer_params = dict(
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-08,  # 1e-6 for BERT.
        weight_decay=0  # 0.01 for BERT.
    )

    optimizer = torch.optim.Adam(
        params=mlm_model.parameters(),
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
            json.dump(mlm_model_params, f)
        
        optimizer_params_path = os.path.join(MODEL_DIR, 'optimizer_params.json')

        with open(optimizer_params_path, 'w') as f:
            json.dump(optimizer_params, f)

    # Model training.
    _, _, training_history = train_model_mlm(
        sequences=classification_training_leaves,
        model=mlm_model,
        n_epochs=N_EPOCHS,
        batch_size=32,
        mask_rate=0.1,
        mask_idx=mask_idx,
        device=device,
        optimizer=optimizer,
        lr_schedule_fn=None,
        training_history=None,
        checkpointing_period_epochs=CHECKPOINTING_PERIOD_EPOCHS,
        model_dir=MODEL_DIR,
        checkpoint_id=EXP_ID,
        tensorboard_log_dir=f'../tensorboard_logs/{EXP_ID}/',
        val_sequences=val_leaves,
        single_mask=True,
        test_data_factorized=None
    )

    plot_training_history(
        training_history,
        savefig_dir='../data/inverse_pretraining_data/',
        exp_id=EXP_ID
    )


if __name__ == '__main__':
    main()
