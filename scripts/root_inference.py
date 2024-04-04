import os
import sys
import logging
import numpy as np
import torch

sys.path.append('../modules/')

from logger import get_logger
from tree_generation import (generate_dataset, compute_rho_entropy)
from data_preprocessing import preprocess_data
from models import FFNN
from training import train_model
from model_evaluation import load_experiment_catalog, save_experiment_info
from plotting import plot_training_history


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATA_TARGET_DIR = '../data/nn_nb_bp_comparison'
PARAMS_SOURCE_PATH = '../data/params_to_test_sigma_eps/NB_bestperfs.npy'
EXPERIMENT_CATALOG_PATH = os.path.join(
    DATA_TARGET_DIR, 'experiment_catalog.csv')


def main():
    logger = get_logger(
        'root_inference',
        level=logging.INFO,
        log_file_path='../logs/nn_nb_bp_comparison_best_logs.txt'
    )

    # Read parameters from file.
    seeds, epsilons, sigmas, nb_accuracies, bp_accuracies = np.load(
        PARAMS_SOURCE_PATH,
        allow_pickle=True
    )

    # i: index on epsilons.
    for i in range(seeds.shape[0]):
        # j: index on sigmas.
        for j in range(seeds.shape[1]):
            seed = seeds[i, j]
            epsilon = epsilons[i]
            sigma = sigmas[j]
            nb_accuracy = nb_accuracies[i, j]
            bp_accuracy = bp_accuracies[i, j]

            experiment_id = f'nb_best_eps{i}_sigma{j}'

            logger.info(
                f'Experiment {i * seeds.shape[1] +j+1} of '
                f'{seeds.shape[0] * seeds.shape[1]} '
                f'| (i, j)=({i}, {j}) | epsilon={epsilon} | sigma={sigma}'
            )

            # Generate data.
            n_samples_training = 5000
            n_samples_test = 2000
            k = 4
            q = 4
            matrix_type = 'mixed_index_sets'
            grammar_kwargs = dict(
                q=q,
                sigma=sigma,
                epsilon=epsilon
            )

            np.random.seed(seed)

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
            x_train, y_train, x_test, y_test = preprocess_data(
                roots_train, leaves_train, roots_test, leaves_test, q, device
            )

            # Model definition.
            logger.info('Instantiating model')

            model_params = dict(
                dims=[leaves_train.shape[-1], 64, q]
            )

            model = FFNN(
                dims=model_params['dims'],
                activation='relu',
                output_activation='softmax',
                batch_normalization=False,
                concatenate_last_dim=True
            ).to(device=device)

            n_params_model = sum(p.numel() for p in model.parameters())

            logger.info(f'N params (model): {n_params_model}')

            training_params = dict(
                n_epochs=150,
                learning_rate=1e-3,
                batch_size=32,
            )

            model, training_history = train_model(
                model,
                training_data=(x_train, y_train),
                test_data=(x_test, y_test),
                n_epochs=training_params['n_epochs'],
                loss_fn=torch.nn.CrossEntropyLoss(),
                learning_rate=training_params['learning_rate'],
                batch_size=training_params['batch_size'],
                early_stopper=None
            )

            baseline_accuracy = 1. / q

            logger.info(
                f'Final test accuracy: {training_history["val_accuracy"][-1]} '
                f'(baseline: {baseline_accuracy})'
            )

            experiment_params = dict(
                experiment_id=experiment_id,
                q=q,
                k=k,
                matrix_type=matrix_type,
                matrix_entropy=rho_entropy,
                np_seed=seed,
                naive_bayes_accuracy=nb_accuracy,
                bp_accuracy=bp_accuracy,
                sigma=grammar_kwargs['sigma'],
                epsilon=grammar_kwargs['epsilon'],
                n_samples_training=n_samples_training,
                n_samples_test=n_samples_test,
                **model_params,
                n_params_model=n_params_model,
                **training_params,
                final_training_accuracy=training_history["training_accuracy"][-1],
                final_test_accuracy=training_history["val_accuracy"][-1],
                baseline_accuracy=baseline_accuracy
            )

            save_experiment_info(EXPERIMENT_CATALOG_PATH, **experiment_params)

            plots_dir_exp = os.path.join(
                DATA_TARGET_DIR, f'plots_{experiment_id}/'
            )

            if not os.path.exists(plots_dir_exp):
                os.makedirs(plots_dir_exp)

            plot_training_history(
                training_history,
                baseline_accuracy=baseline_accuracy,
                savefig_dir=plots_dir_exp,
                exp_id=experiment_id
            )


if __name__ == '__main__':
    main()
