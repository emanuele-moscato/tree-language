import numpy as np 
import torch
from torch import nn
import sys
import os

sys.path.append('./Emanuele_git/tree-language/modules')

from training import train_model
from models import TransformerClassifier
from models_custom import TransformerClassifierNoFeedforward
from logger_tree_language import get_logger

print('Running sigma = 0, l = 4 with feedforward up to largest (2**20) data LINEAR READOUT.',flush=True)

print('Imports done.',flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Running on device:', device,flush=True)

#seeds = [0,1,15,31]
#seeds = [31]
seeds = [0]
#P = 2**np.arange(18,20) # Number of training samples that will be used
P = 2**np.arange(9,21) # Number of training samples that will be used
#P = 2**np.arange(20,21)
N_test = int(1e4) # Number of test samples
sigma = 1.0
epsilon = 0.0
q = 4
l = 4
factorized_layers = np.flip(np.arange(1,l+1))
#factorized_layers = np.arange(1,l+1)

loss_fn = nn.CrossEntropyLoss()
num_epochs = 20
embedding_size = 128
n_layers = [4]
n_head = 1

[q,l,sigma,epsilon,x0s,xis,M_s] = np.load('./sim_data/labeled_data_fixed_{}_{}_{}_{:.5f}.npy'.format(q,l,sigma,epsilon),allow_pickle=True)

for factorized_layer in factorized_layers:
    #[_,_,_,_,x0s_factorized,xis_factorized,_,_] = np.load('./sim_data/labeled_data_fixed_factorized_{}_{}_{}_{:.5f}_{}.npy'.format(q,l,sigma,epsilon,factorized_layer),allow_pickle=True)
    [_,_,_,_,x0s_factorized,xis_factorized,_,_] = np.load('./sim_data/labeled_data_fixed_factorizedNew_LARGE_{}_{}_{}_{:.5f}_{}.npy'.format(q,l,sigma,epsilon,factorized_layer),allow_pickle=True)
    for n_layer in n_layers:
        for i in range(len(seeds)):
            """ x0_test = x0s[:,seeds[i]]
            xi_test = xis[:,:,seeds[i]]
            x0 = x0s_factorized[:,seeds[i]]
            xi = xis_factorized[:,:,seeds[i]] """
            x0_test = x0s[:,seeds[i]]
            xi_test = xis[:,:,seeds[i]]
            x0 = x0s_factorized[:,i]
            xi = xis_factorized[:,:,i]
            y_test = nn.functional.one_hot(torch.from_numpy(x0_test[-N_test:]).to(dtype=torch.int64), num_classes=q).to(dtype=torch.float32).to(device=device)
            x_test = torch.from_numpy(xi_test[:,-N_test:].T).to(device=device).int()
            for p in P:
                torch.cuda.empty_cache()
                x_train = torch.from_numpy(xi[:,:p].T).to(device=device).int()
                y_train = nn.functional.one_hot(torch.from_numpy(x0[:p]).to(dtype=torch.int64), num_classes=q).to(dtype=torch.float32).to(device=device)
                model = TransformerClassifier(
                    seq_len=int(2**l),
                    embedding_size=embedding_size,
                    n_tranformer_layers=n_layer,
                    n_heads=n_head,
                    vocab_size=q,
                    embedding_agg='flatten',
                    positional_encoding=True,
                    decoder_hidden_sizes=[],
                ).to(device=device)
                _, training_history = train_model(
                    model=model,
                    training_data=(x_train, y_train),
                    test_data=(x_test, y_test),
                    n_epochs=num_epochs,
                    loss_fn=loss_fn,
                    learning_rate=1e-4,
                    batch_size=32,
                    early_stopper=None
                )
                # Save the training history and settings
                np.save('./sim_data/Transformer_wPE_flat_factorizedNew_LinearReadout_{}_{}_{:.2f}_{:.5f}_{}_{}_{}_{}.npy'.format(q,l,sigma,epsilon,seeds[i],p,n_layer,factorized_layer),np.array([q,l,sigma,epsilon,seeds[i],p,n_layer,training_history,embedding_size],dtype=object))
                # Save the actual model
                model_dir = './models/model_factorizedNew_LinearReadout_{}_{}_{:.2f}_{:.5f}_{}_{}_{}_{}'.format(q,l,sigma,epsilon,seeds[i],p,n_layer,factorized_layer)
                if model_dir is not None:
                    # Create model directory if it doesn't exist.
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                checkpoint_id = 'model'
                checkpoint_path = os.path.join(model_dir,checkpoint_id + f'_epoch_{num_epochs}.pt')
                torch.save(
                        {
                            'model_state_dict': model.state_dict(),
                            'training_history': training_history
                        },
                        checkpoint_path
                    )