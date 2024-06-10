import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class PottsModel(nn.Module):
    def __init__(self, seq_len, q):
        super(PottsModel, self).__init__()
        self.J = nn.Parameter(torch.randn(seq_len, seq_len, q, q),requires_grad=True)
        self.h = nn.Parameter(torch.randn(seq_len, q),requires_grad=True)
        self.q = q

    def forward(self, x): # x needs to be 1H encoded i.e. of dim (batch_size,seq_len,q)
        h_term = self.h[None,...]

        # Compute the contribution from J, removing the self-interactions
        J_term = torch.einsum('kjb,rjab->kra', x, self.J) - torch.einsum('krb,rrab->kra', x, self.J)

        # Compute the total energy
        energy = h_term + J_term
        prob = torch.exp(torch.einsum('kra,kra -> kr',x,energy))/torch.exp(energy).sum(dim=-1)
        
        # Compute the pseudo-likelihood
        npll = -torch.log(prob).sum(dim=-1)
        return npll
    
    def infer_symbols(self, x):
        h_term = self.h[None,...]

        # Compute the contribution from J, removing the self-interactions
        J_term = torch.einsum('kjb,rjab->kra', x, self.J) - torch.einsum('krb,rrab->kra', x, self.J)

        energy = h_term + J_term

        # Compute the total energy
        probs = F.softmax(energy,dim=-1)

        # Infer the symbols
        x_inferred = torch.argmax(probs,dim=-1)

        return (x_inferred == torch.argmax(x,dim=-1)).to(dtype=torch.float32).mean()

def Potts_training_step(model,data,optimizer):
    loss = model(data).mean()
    optimizer.zero_grad()
    loss.backward()
    # Perform an optimization step.
    optimizer.step()
    return loss.detach()

def train_Potts(model,data,optimizer,n_epochs,batch_size=32):
    training_history = {
        'training_loss': [],
        'learning_rate': []
    }
    training_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    update_counter = 0
    for epoch in range(n_epochs):
        losses = []
        for batch in training_loader:
            update_counter += 1
            loss = Potts_training_step(model,batch[0],optimizer)
            losses.append(loss.item())
        training_history['training_loss'].append(np.mean(losses))
        training_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        print('Epoch:',epoch,'Loss:',np.mean(losses),flush=True)
    return model,training_history