import numpy as np
import torch
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
            batch_normalization=False,
            concatenate_last_dim=False
        ):
        """
        Class constructor. `dims` is a list of int representing the
        dimension of each layer (the first being the input dimension, while
        the last is the output one).
        """
        super().__init__()

        self.batch_normalization = batch_normalization
        self.concatenate_last_dim = concatenate_last_dim

        if concatenate_last_dim:
            dims[0] = dims[0] * dims[-1]

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

        if self.concatenate_last_dim:
            out = out.reshape(out.shape[0], -1)

        for i, (layer, activation) in enumerate(
                zip(self.linear_layers, self.activations)
            ):
            if self.batch_normalization:
                out = self.batch_norm_layers[i](out)

            out = activation(layer(out))

        return out
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MeanAggLayer(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
            
        self.avg_pool = nn.AvgPool1d(
            kernel_size=seq_len,
        )

    def forward(self, x):
        return self.avg_pool(torch.permute(x, dims=(0, 2, 1))).squeeze()


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        seq_len,
        embedding_size,
        n_tranformer_layers,
        n_heads,
        n_classes,
        embedding_agg='mean'
    ):
        super().__init__()

        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.n_tranformer_layers = n_tranformer_layers
        self.n_heads = n_heads
        self.n_classes = n_classes
        self.embedding_agg = embedding_agg

        # Embedding.
        self.input_embedding = nn.Embedding(n_classes, embedding_size)
        self.positional_embedding = PositionalEncoding(
            d_model=embedding_size,
            dropout=0.1,
            max_len=5000
        )

        # Single encoder layer.
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=n_heads,
            dim_feedforward=2048,
            batch_first=True
        )

        # Stack of encoder layers.
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=n_tranformer_layers
        )

        if self.embedding_agg == 'mean':
            self.embedding_agg_layer = MeanAggLayer(
                seq_len=self.seq_len
            )

            # Final FFNN.
            self.final_layer = nn.Linear(
                embedding_size,
                n_classes
            )
        elif self.embedding_agg == 'flatten':
            self.embedding_agg_layer = nn.Flatten(start_dim=-2, end_dim=-1)

            self.final_layer = nn.Linear(
                seq_len * embedding_size,
                n_classes
            )
        else:
            raise NotImplementedError(
                f'Embedding aggregation {embedding_agg} not implemented'
            )

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.positional_embedding(x)

        x = self.transformer_encoder(x)

        x = self.embedding_agg_layer(x)

        x = self.final_layer(x)

        return x