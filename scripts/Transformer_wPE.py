import numpy as np 
import torch
from torch import nn
import sys
import os

sys.path.append('../modules/')

from training import train_model
from models import TransformerClassifier
from models_custom import TransformerClassifierNoFeedforward,TransformerClassifierNoResiduals
from logger_tree_language import get_logger

print('Running sigma = 1 with l = 4, 4 then 1 layers of attention with the new and fixed factorized validation LINEAR READOUT.',flush=True)

print('Imports done.',flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Running on device:', device,flush=True)

#seeds = [0,1,15,31]
seeds = [0]
P = 2**np.arange(9,18) # Number of training samples that will be used
N_test = int(1e4) # Number of test samples
sigma = 1.0
epsilon = 0.0
q = 4
l = 4
loss_fn = nn.CrossEntropyLoss()
num_epochs = 200
embedding_size = 128
#n_layers = [1,2,4]
n_layers = [2,3]
n_head = 1

[q,l,sigma,epsilon,x0s,xis,M_s] = np.load('./sim_data/labeled_data_fixed_{}_{}_{}_{:.5f}.npy'.format(q,l,sigma,epsilon),allow_pickle=True)

factorized_layers = np.flip(np.arange(1,l+1))

for n_layer in n_layers:
    for seed in seeds:
        x0 = x0s[:,seed]
        xi = xis[:,:,seed]
        # Need to re-shuffle the data
        indices = np.arange(x0s.shape[0])
        np.random.shuffle(indices)
        x0 = x0s[indices,seed]
        xi = xis[:,indices,seed]
        y_test = nn.functional.one_hot(torch.from_numpy(x0[-N_test:]).to(dtype=torch.int64), num_classes=q).to(dtype=torch.float32).to(device=device)
        x_test = torch.from_numpy(xi[:,-N_test:].T).to(device=device).int()
        test_data_factorized = []
        for i in range(len(factorized_layers)):
            [_,_,_,_,x0s_factorized,xis_factorized,_,_] = np.load('./sim_data/labeled_data_fixed_factorizedNew_{}_{}_{}_{:.5f}_{}.npy'.format(q,l,sigma,epsilon,factorized_layers[i]),allow_pickle=True)
            x0_factorized = x0s_factorized[:,seed]
            xi_factorized = xis_factorized[:,:,seed]
            y_test_factorized = nn.functional.one_hot(torch.from_numpy(x0_factorized[-N_test:]).to(dtype=torch.int64), num_classes=q).to(dtype=torch.float32).to(device=device)
            x_test_factorized = torch.from_numpy(xi_factorized[:,-N_test:].T).to(device=device).int()
            test_data_factorized.append((x_test_factorized,y_test_factorized))
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
                early_stopper=None,
                test_data_factorized=test_data_factorized
            )
            # Save the training history and settings
            np.save('./sim_data/Transformer_wPE_flat_wfactorizedvalNew_LinearReadout_{}_{}_{:.2f}_{:.5f}_{}_{}_{}.npy'.format(q,l,sigma,epsilon,seed,p,n_layer),np.array([q,l,sigma,epsilon,seed,p,n_layer,training_history,embedding_size],dtype=object))
            # Save the actual model
            model_dir = './models/model_wfactorizedvalNew_LinearReadout_{}_{}_{:.2f}_{:.5f}_{}_{}_{}'.format(q,l,sigma,epsilon,seed,p,n_layer)
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