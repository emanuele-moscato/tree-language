import torch.nn as nn


class FFNN(nn.Module):
    """
    Subclass of the `Module` object representing an arbitrary feed-forward NN
    with the specified number of hidden layers (and their dimension) and
    activation function.

    TO DO:
      * Restructure how dimensions are passed to the constructor?
      * Add optional dropout regularization.
    """
    def __init__(
            self,
            dims,
            activation='relu',
            output_activation='identity',
            batch_normalization=False
        ):
        """
        Class constructor. `dims` is a list of int representing the
        dimension of each layer (the first being the input dimension, while
        the last is the output one).
        """
        super().__init__()

        self.batch_normalization = batch_normalization

        # In order for PyTorch to be able to detect submodules, they either
        # need to be attributes of the `Module` subclass or they need to be
        # put in ad-hoc containers provided by PyTorch itself such as
        # `ModuleList` objects.
        # Note: [n layers] = [n dims] - 1 (we include input and output dims).
        if batch_normalization:
            self.batch_norm_layers = nn.ModuleList([
                nn.BatchNorm1d(num_features=dims[i])
                for i in range(len(dims) - 1)
            ])

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=dims[i], out_features=dims[i+1])
            for i in range(len(dims) - 1)
        ])

        # Note: [total n activations] = [n layers] (here we split them into
        #       "intermediate" activations and the output one, after the final
        #       layer.
        n_intermediate_activations = len(dims) - 2

        if activation.lower() == 'relu':
            self.activations = nn.ModuleList([
                nn.ReLU()
                for _ in range(n_intermediate_activations)
            ])
        elif activation.lower() == 'tanh':
            self.activations = nn.ModuleList([
                nn.Tanh()
                for _ in range(n_intermediate_activations)
            ])
        elif activation.lower() == 'identity':
            self.activations = nn.ModuleList([
                nn.Identity()
                for _ in range(n_intermediate_activations)
            ])
        else:
            raise NotImplementedError(
                f'Support for activations {activation} not implemented'
            )

        if output_activation == 'identity':
            self.activations.append(nn.Identity())
        elif output_activation == 'sigmoid':
            self.activations.append(nn.Sigmoid())
        elif output_activation == 'softmax':
            self.activations.append(nn.Softmax(dim=-1))
        else:
            raise Exception(
                f'Selected output activation {output_activation} not '
                'recognized'
            )

    def forward(self, x):
        """
        Forward pass of the model.
        """
        out = x

        for i, (layer, activation) in enumerate(
                zip(self.linear_layers, self.activations)
            ):
            if self.batch_normalization:
                out = self.batch_norm_layers[i](out)

            out = activation(layer(out))

        return out