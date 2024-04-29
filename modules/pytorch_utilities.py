import os
import json
import torch
from models import TransformerClassifier


def optimizer_to(optim, device):
    """
    Sends a PyTorch optimizer's parameters to the selected device,
    as it doesn't have a native `to` method. Useful when loading
    an optimizer's state from a state dict (`load_state_dict`
    method) to resume training on a GPU).
    """
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def load_checkpoint(model_dir, checkpoint_id, device):
    """
    Loads a saved checkpoint (trained model, optimizer and training history)
    and sends the model's and optimizer's parameters to the specified device.
    """
    # Read saved model's (hyper) parameters.
    model_params_path = os.path.join(model_dir, 'model_params.json')
    
    with open(model_params_path, 'r') as f:
        model_params_loaded = json.load(f)
    
    optimizer_params_path = os.path.join(model_dir, 'optimizer_params.json')
    
    with open(optimizer_params_path, 'r') as f:
        optimizer_params_loaded = json.load(f)

    # Instantiate new model.
    model_loaded = TransformerClassifier(
        **model_params_loaded
    )
    
    # Instantiate new optimizer.
    optimizer_loaded = torch.optim.Adam(
        params=model_loaded.parameters(),
        **optimizer_params_loaded
    )
    
    # Load checkpoint.
    checkpoint_path = os.path.join(model_dir, checkpoint_id)
    
    checkpoint = torch.load(checkpoint_path)
    
    # Load data from the checkpoint into the model/optimizer/other variables.
    model_loaded.load_state_dict(checkpoint['model_state_dict'])
    optimizer_loaded.load_state_dict(checkpoint['optimizer_state_dict'])
    training_history_loaded = checkpoint['training_history']
    
    # Send loaded model and optimizer to the chosen device.
    model_loaded = model_loaded.to(device=device)
    optimizer_to(optimizer_loaded, device)

    return model_loaded, optimizer_loaded, training_history_loaded
