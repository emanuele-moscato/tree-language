import torch


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
