from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from logger_tree_language import get_logger
from typing import Optional, Union, Callable
from models import PositionalEncoding,MeanAggLayer,FFNN


class CustomTransformerEncoderLayer(nn.Module):
    """
    Custom transformer encoder layer with multi-head self-attention but
    without the feedforward layer.

    Note: by default, this custom layer applies layer normalization BOTH
          BEFORE AND AFTER the self-attention block. The resulting
          architecture is
            [layer norm] --> [self-attention] --> [layer norm]
    """
    def __init__(self, d_model, nhead, dropout=0.1, device=None, dtype=None, bias: bool = True, batch_first: bool = True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,bias=bias, batch_first=batch_first,**factory_kwargs)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def _sa_block(self, x: Tensor,attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,attn_mask=attn_mask,key_padding_mask=key_padding_mask,need_weights=False, is_causal=is_causal)[0]
        return self.dropout(x)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self attention
        src = src + self._sa_block(self.norm1(src), src_mask, src_key_padding_mask)
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


class CustomTransformerEncoderLayerNoResidualConn(nn.Module):
    """
    Custom transformer encoder layer with multi-head self-attention with the
    feedforward layer but WITHOUT RESIDUAL CONNECTIONS.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = nn.functional.relu,
        layer_norm_eps: float = 1e-5, batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True, device=None, dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        # Multi-head attention layer.
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs
        )

        # Feedforward part of the model.
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        # Layer normalizations.
        # Option for using layer normalization before or after the
        # self-attention and the feed-forward block.
        self.norm_first = norm_first
        # Layer normalization before or after the self-attention block.
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        # Layer normalization before or after the feed-forward block.
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)

        # Dropout layers.
        # Dropout used after the self-attention layer.
        self.dropout1 = nn.Dropout(dropout)
        # Dropour used after the feed-forward layer.
        self.dropout2 = nn.Dropout(dropout)

        # Activation function to use in the feed-forward block.
        self.activation = activation

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False
    ) -> Tensor:
        """
        Wrapper around the call to the self-attention block (a closure on the
        additional options, actually).
        """
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal
        )[0]

        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        """
        Forward pass for the feed-forward block. Resulting architecture:
            [linear] --> [activation] --> [dropout] --> [linear] --> [dorpout]
        """
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))

        return self.dropout2(x)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False
    ) -> Tensor:
        """
        Forward pass of the model. According to the choice of positioning of
        layer normalization, the architecture is:
          * If layer normalization is applied BEFORE the blocks,
                [layer norm] -> [self-attention] -> [layer norm] -> [feed-forward]
          * If layer normalization is applied AFTER the blocks,
                [self-attention] -> [layer norm] -> [feed-forward] -> [layer norm]
        """
        # WARNING: as we never use masking in our experiments, the following
        #          transformations are not really relevant. Please have a look
        #          at them if masking is needed.
        src_key_padding_mask = nn.functional._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=nn.functional._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = nn.functional._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src

        if self.norm_first:
            x = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = self._ff_block(self.norm2(x))
        else:
            x = self.norm1(self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(self._ff_block(x))

        return x


class TransformerClassifierNoFeedforward(nn.Module):
    """
    Custom version of the TransformerClassifier that is free of residual
    connections in the encoder blocks.
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

        # Single encoder layer, custom version with no residual connections.
        self.encoder_layer = CustomTransformerEncoderLayerNoResidualConn(
            d_model=embedding_size,
            nhead=n_heads,
            dropout=0.1,
            dim_feedforward=2048
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
