import torch
from logger import get_logger


def preprocess_data(roots_train, leaves_train, roots_test, leaves_test, q, device):
    """
    Preprocesses the training and test data (roots and leaves), which for now
    is just onw-hot encoding.
    """
    logger = get_logger('data_preprocessing')

    logger.info('Preprocessing data')

    x_train = torch.nn.functional.one_hot(torch.from_numpy(leaves_train), num_classes=q).to(dtype=torch.float32).to(device=device)
    y_train = torch.nn.functional.one_hot(torch.from_numpy(roots_train), num_classes=q).to(dtype=torch.float32).to(device=device)

    x_test = torch.nn.functional.one_hot(torch.from_numpy(leaves_test), num_classes=q).to(dtype=torch.float32).to(device=device)
    y_test = torch.nn.functional.one_hot(torch.from_numpy(roots_test), num_classes=q).to(dtype=torch.float32).to(device=device)

    return x_train, y_train, x_test, y_test