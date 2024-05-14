import numpy as np


def read_data(file_path, seed):
    q, k, sigma, epsilon, roots_seeds, leaves_seeds, rho_seeds = np.load(file_path, allow_pickle=True)

    # The last index corresponds to the seed that generated the
    # data/transition tensors: select one.
    seed = 0
    
    shuffled_indices = np.random.choice(range(leaves_seeds.shape[1]), leaves_seeds.shape[1], replace=False)
    
    roots = roots_seeds[:, seed]
    roots = roots[shuffled_indices]
    
    leaves = leaves_seeds[..., seed].T
    leaves = leaves[shuffled_indices, :]
    rho = rho_seeds[..., seed]

    return q, k, sigma, epsilon, roots, leaves, rho
