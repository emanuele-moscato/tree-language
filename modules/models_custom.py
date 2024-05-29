from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from logger_tree_language import get_logger


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
            dropout_p=None,
            concatenate_last_dim=False
        ):
        """
        Class constructor. `dims` is a list of int representing the
        dimension of each layer (the first being the input dimension, while
        the last is the output one).
        """
        super().__init__()

        self.batch_normalization = batch_normalization
        self.dropout_p = dropout_p
        self.concatenate_last_dim = concatenate_last_dim

        if concatenate_last_dim:
            dims[0] = dims[0] * dims[-1]

        # In order for PyTorch to be able to detect submodules, they either
        # need to be attributes of the `Module` subclass or they need to be
        # put in ad-hoc containers provided by PyTorch itself such as
        # `ModuleList` objects.
        # Notes:
        #  * [n layers] = [n dims] - 1 (we include input and output dims).
        #  * [n batch normalization layers] = [n layers] = [n dims] - 1
        #    (one batch normalization layer before each linear layer).
        if batch_normalization:
            self.batch_norm_layers = nn.ModuleList([
                nn.BatchNorm1d(num_features=dims[i])
                for i in range(len(dims) - 1)
            ])

        # If specified, we place a dropout layer after the activation function
        # of each HIDDEN linear layer.
        # Note: [n dropout layers] = [n hidden linear layers] = [n dims] - 2
        if dropout_p is not None:
            self.dropout_layers = nn.ModuleList([
                nn.Dropout(p=dropout_p)
                for i in range(len(dims) - 2)
            ])

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=dims[i], out_features=dims[i+1])
            for i in range(len(dims) - 1)
        ])

        # Note: [total n activations] = [n layers] (here we split them into
        #       "intermediate" activations and the output one, after the final
        #       layer).
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

            # If using dropout layers, apply dropout after the activation
            # function of each hidden linear layer (not after the last
            # - output - one).
            if (self.dropout_p) and (i != len(self.linear_layers) - 1):
                out = self.dropout_layers[i](out)

        return out
    

class PositionalEncoding(nn.Module):
    """
    PyTorch module implementing sinusoidal positional encoding.
    """
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
    """
    PyTorch module implementing averaging over the second dimension of a
    tensor. Example: for tensors of shape `(batch_shape, seq_len, model_dim)`
    (as in transformers), this takes the average over the `seq_len` dimension
    (i.e. over the tokens in the sequences) in order to obtain tensor of
    shape `(batch_shape, model_dim)`, i.e. a "sentence embedding" vector of
    dimension `(model_dim,)` for each sample in the batch.
    """
    def __init__(self, seq_len):
        super().__init__()
            
        self.avg_pool = nn.AvgPool1d(
            kernel_size=seq_len,
        )

    def forward(self, x):
        return self.avg_pool(torch.permute(x, dims=(0, 2, 1))).squeeze()


class TransformerClassifier(nn.Module):
    """
    A transformer-based classifier (encoder only), using semanting embedding
    and sinusoidal positional embeddings and with a final feed-forward NN.
    """
    def __init__(
        self,
        seq_len,
        embedding_size,
        n_tranformer_layers,
        n_heads,
        vocab_size,
        encoder_dim_feedforward=2048,
        positional_encoding=True,
        n_special_tokens=0,  # We assume the special tokens correspond to the last `n_special_tokens` indices.
        embedding_agg='mean',
        decoder_hidden_sizes=[],
        decoder_activation='relu',
        decoder_output_activation='identity'
    ):
        super().__init__()

        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.positional_encoding = positional_encoding
        self.n_tranformer_layers = n_tranformer_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.encoder_dim_feedforward = encoder_dim_feedforward
        self.embedding_agg = embedding_agg
        self.decoder_hidden_sizes = decoder_hidden_sizes
        self.decoder_activation = decoder_activation
        self.decoder_output_activation = decoder_output_activation


        # Embedding.
        self.input_embedding = nn.Embedding(vocab_size, embedding_size)
        if self.positional_encoding:
            self.positional_embedding = PositionalEncoding(
                d_model=embedding_size,
                dropout=0.1,
                max_len=5000
            )
        else:
            self.positional_embedding = nn.Identity()

        # Single encoder layer.
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=n_heads,
            dim_feedforward=encoder_dim_feedforward,
            batch_first=True
        )

        # Stack of encoder layers.
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=n_tranformer_layers
        )

        # Aggregation/stacking of token representations (the input dimension
        # of the final layer is adapted accordingly).
        if self.embedding_agg == 'mean':
            self.embedding_agg_layer = MeanAggLayer(
                seq_len=self.seq_len
            )

            decoder_input_dim = embedding_size
        elif self.embedding_agg == 'flatten':
            self.embedding_agg_layer = nn.Flatten(start_dim=-2, end_dim=-1)

            decoder_input_dim = seq_len * embedding_size
        elif (
            (self.embedding_agg == 'flatten')
            or (self.embedding_agg is None)
        ):
            # Note: in this case THE OUTPUT SHAPE FOR THE WHOLE MODEL is
            #       different (used e.g. for (masked) language modeling).
            #       Example: if the input shape is 
            #                   (batch_size, seq_len, hidden_dim)
            #                then the output shape is
            #                   (batch_size, seq_len, vocab_size),
            #                where in this case we need to exclude the special
            #                tokens.
            self.embedding_agg_layer = nn.Identity()

            decoder_input_dim = embedding_size
        else:
            raise NotImplementedError(
                f'Embedding aggregation {embedding_agg} not implemented'
            )
        
        # Decoder (FFNN).
        decoder_dims = (
            [decoder_input_dim]
            + decoder_hidden_sizes
            + [vocab_size - n_special_tokens]
        )

        self.decoder = FFNN(
            dims=decoder_dims,
            activation=decoder_activation,
            output_activation=decoder_output_activation,
            batch_normalization=False,
            concatenate_last_dim=False
        )

    def forward(self, x, src_key_padding_mask=None):
        """
        Model's forward pass.

        Note: `src_key_padding_mask` should have shape `(batch_size, seq_len)`
              and be a boolean mask with value `True` correspnding to the
              positions to mask along each sequence in the batch.

              See: https://stackoverflow.com/questions/62170439/difference-between-src-mask-and-src-key-padding-mask
        """
        x = self.input_embedding(x)
        x = self.positional_embedding(x)

        x = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )

        x = self.embedding_agg_layer(x)

        x = self.decoder(x)

        return x
    

def replace_decoder_with_classification_head(
        original_model,
        n_classes,
        device,
        embedding_agg='flatten',
        head_hidden_dim=[64],
        head_activation='relu',
        head_output_activation='identity',
        head_batch_normalization=False,
        head_dropout_p=None
    ):
    """
    Replaces the decoder of a `TransformerClassifier` model pretrained
    with masked language modeling with a new classification head. The
    aggregation operation of the tokens' latent representation is modified
    accordingly (by flattening or averaging, according to the specified
    option).
    """
    model = deepcopy(original_model)
    
    # Replace the aggregation operation from `None` (no aggregation, as
    # needed for MLM) to `flatten`.
    if embedding_agg == 'flatten':
        model.embedding_agg = 'flatten'
        model.embedding_agg_layer = torch.nn.Flatten(start_dim=-2, end_dim=-1)
    
        decoder_input_dim = model.seq_len * model.embedding_size
    elif embedding_agg == 'mean':
        model.embedding_agg_layer = MeanAggLayer(
                seq_len=model.seq_len
            )

        decoder_input_dim = model.embedding_size
    else:
        raise NotImplementedError(
            f'Embeddings aggregation {embedding_agg} not available'
        )
    
    # Replace the decoder with a FFNN of appropriate size.
    model.decoder = FFNN(
        dims=(
            [decoder_input_dim]
            + head_hidden_dim
            + [n_classes]
        ),
        activation=head_activation,
        output_activation=head_output_activation,
        batch_normalization=head_batch_normalization,
        dropout_p=head_dropout_p,
        concatenate_last_dim=False
    ).to(device=device)

    return model


def freeze_encoder_weights(model, trainable_modules=['decoder']):
    """
    Freezes the weights in all the submodules of `model` whose name
    don't appear in the `trainable_modules` list. The submodules for which the
    weigths are frozen are also set to inference mode (dropout layers are
    deactivated so the submodule's output is deterministic).
    """
    logger = get_logger('freeze_encoder_weights')

    for submodule in model.named_children():
        if submodule[0] not in trainable_modules:
            for p in submodule[1].parameters():
                p.requires_grad = False

            # Set all the PyTorch modules that are not meant to remain
            # trainable to inference mode.
            submodule[1].train(mode=False)

        logger.info(
            f'Module: {submodule[0]}'
            f'| N parameters: {sum([p.numel() for p in submodule[1].parameters()])}'
            f' | Parameters trainable: {all([p.requires_grad for p in submodule[1].parameters()])}'
            f' | Training mode: {submodule[1].training}'
        )

class CustomTransformerEncoderLayer(nn.Module):
    """
    Custom transformer encoder layer with multi-head self-attention but without the feedforward layer.
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self attention
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        return src

class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        return output
    
class TransformerClassifierNoFeedforward(nn.Module):
    """
    Custom version of the TransformerClassifier that is free of feedforward layers in the encoder, i.e. purely attention-based.
    """
    def __init__(
        self,
        seq_len,
        embedding_size,
        n_tranformer_layers,
        n_heads,
        vocab_size,
        positional_encoding=True,
        n_special_tokens=0,  # We assume the special tokens correspond to the last `n_special_tokens` indices.
        embedding_agg='mean',
        decoder_hidden_sizes=[],
        decoder_activation='relu',
        decoder_output_activation='identity'
    ):
        super().__init__()

        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.positional_encoding = positional_encoding
        self.n_tranformer_layers = n_tranformer_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.embedding_agg = embedding_agg
        self.decoder_hidden_sizes = decoder_hidden_sizes
        self.decoder_activation = decoder_activation
        self.decoder_output_activation = decoder_output_activation


        # Embedding.
        self.input_embedding = nn.Embedding(vocab_size, embedding_size)
        if self.positional_encoding:
            self.positional_embedding = PositionalEncoding(
                d_model=embedding_size,
                dropout=0.1,
                max_len=5000
            )
        else:
            self.positional_embedding = nn.Identity()

        # Single encoder layer.
        self.encoder_layer = CustomTransformerEncoderLayer(
            d_model=embedding_size,
            nhead=n_heads,
            dropout=0.1
        )

        # Stack of encoder layers.
        self.transformer_encoder = CustomTransformerEncoder(
            self.encoder_layer,
            num_layers=n_tranformer_layers
        )

        # Aggregation/stacking of token representations (the input dimension
        # of the final layer is adapted accordingly).
        if self.embedding_agg == 'mean':
            self.embedding_agg_layer = MeanAggLayer(
                seq_len=self.seq_len
            )

            decoder_input_dim = embedding_size
        elif self.embedding_agg == 'flatten':
            self.embedding_agg_layer = nn.Flatten(start_dim=-2, end_dim=-1)

            decoder_input_dim = seq_len * embedding_size
        elif (
            (self.embedding_agg == 'flatten')
            or (self.embedding_agg is None)
        ):
            # Note: in this case THE OUTPUT SHAPE FOR THE WHOLE MODEL is
            #       different (used e.g. for (masked) language modeling).
            #       Example: if the input shape is 
            #                   (batch_size, seq_len, hidden_dim)
            #                then the output shape is
            #                   (batch_size, seq_len, vocab_size),
            #                where in this case we need to exclude the special
            #                tokens.
            self.embedding_agg_layer = nn.Identity()

            decoder_input_dim = embedding_size
        else:
            raise NotImplementedError(
                f'Embedding aggregation {embedding_agg} not implemented'
            )
        
        # Decoder (FFNN).
        decoder_dims = (
            [decoder_input_dim]
            + decoder_hidden_sizes
            + [vocab_size - n_special_tokens]
        )

        self.decoder = FFNN(
            dims=decoder_dims,
            activation=decoder_activation,
            output_activation=decoder_output_activation,
            batch_normalization=False,
            concatenate_last_dim=False
        )

    def forward(self, x, src_key_padding_mask=None):
        x = self.input_embedding(x)
        x = self.positional_embedding(x)

        x = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )

        x = self.embedding_agg_layer(x)

        x = self.decoder(x)

        return x