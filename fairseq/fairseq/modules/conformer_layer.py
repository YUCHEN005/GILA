# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import nn
from typing import Optional, Tuple
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
    ESPNETMultiHeadedAttention,
    RelPositionMultiHeadedAttention,
    RotaryPositionMultiHeadedAttention,
)
from fairseq.utils import get_activation_fn
import math


class ConvolutionModule(torch.nn.Module):
    """Convolution block used in the conformer block"""

    def __init__(
        self,
        embed_dim,
        channels,
        depthwise_kernel_size,
        dropout,
        activation_fn="swish",
        bias=False,
        export=False,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            channels: Number of channels in depthwise conv layers
            depthwise_kernel_size: Depthwise conv layer kernel size
            dropout: dropout value
            activation_fn: Activation function to use after depthwise convolution kernel
            bias: If bias should be added to conv layers
            export: If layernorm should be exported to jit
        """
        super(ConvolutionModule, self).__init__()
        assert (
            depthwise_kernel_size - 1
        ) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        self.layer_norm = LayerNorm(embed_dim, export=export)
        self.pointwise_conv1 = torch.nn.Conv1d(
            embed_dim,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.glu = torch.nn.GLU(dim=1)
        self.depthwise_conv = torch.nn.Conv1d(
            channels,
            channels,
            depthwise_kernel_size,
            stride=1,
            padding=(depthwise_kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.batch_norm = torch.nn.BatchNorm1d(channels)
        self.activation = get_activation_fn(activation_fn)(channels)
        self.pointwise_conv2 = torch.nn.Conv1d(
            channels,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input of shape B X T X C
        Returns:
          Tensor of shape B X T X C
        """
        x = self.layer_norm(x)
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = self.glu(x)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class FeedForwardModule(torch.nn.Module):
    """Positionwise feed forward layer used in conformer"""

    def __init__(
        self,
        input_feat,
        hidden_units,
        dropout1,
        dropout2,
        activation_fn="swish",
        bias=True,
    ):
        """
        Args:
            input_feat: Input feature dimension
            hidden_units: Hidden unit dimension
            dropout1: dropout value for layer1
            dropout2: dropout value for layer2
            activation_fn: Name of activation function
            bias: If linear layers should have bias
        """

        super(FeedForwardModule, self).__init__()
        self.layer_norm = LayerNorm(input_feat)
        self.w_1 = torch.nn.Linear(input_feat, hidden_units, bias=bias)
        self.w_2 = torch.nn.Linear(hidden_units, input_feat, bias=bias)
        self.dropout1 = torch.nn.Dropout(dropout1)
        self.dropout2 = torch.nn.Dropout(dropout2)
        self.activation = get_activation_fn(activation_fn)(hidden_units)

    def forward(self, x):
        """
        Args:
            x: Input Tensor of shape  T X B X C
        Returns:
            Tensor of shape T X B X C
        """
        x = self.layer_norm(x)
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.w_2(x)
        return self.dropout2(x)



class ConformerEncoderLayer(torch.nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100. We currently don't support relative positional encoding in MHA"""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        dropout,
        use_fp16,
        depthwise_conv_kernel_size=31,
        activation_fn="swish",
        attn_type=None,
        pos_enc_type="abs",
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            ffn_embed_dim: FFN layer dimension
            attention_heads: Number of attention heads in MHA
            dropout: dropout value
            depthwise_conv_kernel_size: Size of kernel in depthwise conv layer in convolution module
            activation_fn: Activation function name to use in convulation block and feed forward block
            attn_type: MHA implementation from ESPNET vs fairseq
            pos_enc_type: Positional encoding type - abs, rope, rel_pos
        """
        self.pos_enc_type = pos_enc_type
        super(ConformerEncoderLayer, self).__init__()

        self.ffn1 = FeedForwardModule(
            embed_dim,
            ffn_embed_dim,
            dropout,
            dropout,
        )

        self.self_attn_layer_norm = LayerNorm(embed_dim, export=False)
        self.self_attn_dropout = torch.nn.Dropout(dropout)
        if attn_type == "espnet":
            if self.pos_enc_type == "rel_pos":
                self.self_attn = RelPositionMultiHeadedAttention(
                    embed_dim,
                    attention_heads,
                    dropout=dropout,
                )
            elif self.pos_enc_type == "rope":
                self.self_attn = RotaryPositionMultiHeadedAttention(
                    embed_dim, attention_heads, dropout=dropout, precision=use_fp16
                )
            elif self.pos_enc_type == "abs":
                self.self_attn = ESPNETMultiHeadedAttention(
                    embed_dim,
                    attention_heads,
                    dropout=dropout,
                )
            else:
                raise Exception(f"Unsupported attention type {self.pos_enc_type}")
        else:
            # Default to fairseq MHA
            self.self_attn = MultiheadAttention(
                embed_dim,
                attention_heads,
                dropout=dropout,
            )

        self.conv_module = ConvolutionModule(
            embed_dim=embed_dim,
            channels=128,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            activation_fn=activation_fn,
        )

        self.ffn2 = FeedForwardModule(
            embed_dim,
            ffn_embed_dim,
            dropout,
            dropout,
            activation_fn=activation_fn,
        )
        self.final_layer_norm = LayerNorm(embed_dim, export=False)

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[torch.Tensor],
        position_emb: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: Tensor of shape T X B X C
            encoder_padding_mask: Optional mask tensor
            positions:
        Returns:
            Tensor of shape T X B X C
        """
        residual = x
        x = self.ffn1(x)
        x = x * 0.5 + residual
        residual = x
        x = self.self_attn_layer_norm(x)
        if self.pos_enc_type == "rel_pos":
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                pos_emb=position_emb,
                need_weights=False,
            )
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
            )
        x = self.self_attn_dropout(x)
        x = x + residual

        residual = x
        # TBC to BTC
        x = x.transpose(0, 1)
        x = self.conv_module(x)
        # BTC to TBC
        x = x.transpose(0, 1)
        x = residual + x

        residual = x
        x = self.ffn2(x)

        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x, attn

class ConformerWav2Vec2EncoderLayer(ConformerEncoderLayer):
    """Encoder layer for Wav2vec2 encoder"""

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
        position_emb=None,
    ):
        return super().forward(x, self_attn_padding_mask, position_emb)



class GIConformerEncoderLayer(torch.nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100. We currently don't support relative positional encoding in MHA"""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        dropout,
        use_fp16,
        depthwise_conv_kernel_size=31,
        activation_fn="swish",
        attn_type=None,
        pos_enc_type="abs",
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            ffn_embed_dim: FFN layer dimension
            attention_heads: Number of attention heads in MHA
            dropout: dropout value
            depthwise_conv_kernel_size: Size of kernel in depthwise conv layer in convolution module
            activation_fn: Activation function name to use in convulation block and feed forward block
            attn_type: MHA implementation from ESPNET vs fairseq
            pos_enc_type: Positional encoding type - abs, rope, rel_pos
        """
        self.pos_enc_type = pos_enc_type
        super(GIConformerEncoderLayer, self).__init__()

        self.ffn1_audio = FeedForwardModule(
            embed_dim,
            ffn_embed_dim,
            dropout,
            dropout,
        )
        self.ffn1_video = FeedForwardModule(
            embed_dim,
            ffn_embed_dim,
            dropout,
            dropout,
        )

        self.self_attn_layer_norm_audio = LayerNorm(embed_dim, export=False)
        self.self_attn_layer_norm_video = LayerNorm(embed_dim, export=False)
        self.cross_attn_layer_norm_audio_video = LayerNorm(embed_dim, export=False)
        self.cross_attn_layer_norm_video_audio = LayerNorm(embed_dim, export=False)

        self.self_attn_dropout_audio = torch.nn.Dropout(dropout)
        self.self_attn_dropout_video = torch.nn.Dropout(dropout)
        self.cross_attn_dropout_audio_video = torch.nn.Dropout(dropout)
        self.cross_attn_dropout_video_audio = torch.nn.Dropout(dropout)

        if attn_type == "espnet":
            if self.pos_enc_type == "rel_pos":
                self.self_attn_audio = RelPositionMultiHeadedAttention(embed_dim, attention_heads, dropout=dropout,)
                self.self_attn_video = RelPositionMultiHeadedAttention(embed_dim, attention_heads, dropout=dropout,)
                self.cross_attn_audio_video = RelPositionMultiHeadedAttention(embed_dim, attention_heads, dropout=dropout,)
                self.cross_attn_video_audio = RelPositionMultiHeadedAttention(embed_dim, attention_heads, dropout=dropout,)
            
            elif self.pos_enc_type == "rope":
                self.self_attn_audio = RotaryPositionMultiHeadedAttention(embed_dim, attention_heads, dropout=dropout, precision=use_fp16)
                self.self_attn_video = RotaryPositionMultiHeadedAttention(embed_dim, attention_heads, dropout=dropout, precision=use_fp16)
                self.cross_attn_audio_video = RotaryPositionMultiHeadedAttention(embed_dim, attention_heads, dropout=dropout, precision=use_fp16)
                self.cross_attn_video_audio = RotaryPositionMultiHeadedAttention(embed_dim, attention_heads, dropout=dropout, precision=use_fp16)
            
            elif self.pos_enc_type == "abs":
                self.self_attn_audio = ESPNETMultiHeadedAttention(embed_dim, attention_heads, dropout=dropout,)
                self.self_attn_video = ESPNETMultiHeadedAttention(embed_dim, attention_heads, dropout=dropout,)
                self.cross_attn_audio_video = ESPNETMultiHeadedAttention(embed_dim, attention_heads, dropout=dropout,)
                self.cross_attn_video_audio = ESPNETMultiHeadedAttention(embed_dim, attention_heads, dropout=dropout,)
            
            else:
                raise Exception(f"Unsupported attention type {self.pos_enc_type}")

        else:
            # Default to fairseq MHA
            self.self_attn_audio = MultiheadAttention(embed_dim, attention_heads, dropout=dropout, self_attention=True,)
            self.self_attn_video = MultiheadAttention(embed_dim, attention_heads, dropout=dropout, self_attention=True,)
            self.cross_attn_audio_video = MultiheadAttention(embed_dim, attention_heads, dropout=dropout, encoder_decoder_attention=True,)
            self.cross_attn_video_audio = MultiheadAttention(embed_dim, attention_heads, dropout=dropout, encoder_decoder_attention=True,)

        self.conv_module_audio = ConvolutionModule(
            embed_dim=embed_dim,
            channels=128,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            activation_fn=activation_fn,
        )
        self.conv_module_video = ConvolutionModule(
            embed_dim=embed_dim,
            channels=128,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            activation_fn=activation_fn,
        )

        self.ffn2_audio = FeedForwardModule(
            embed_dim,
            ffn_embed_dim,
            dropout,
            dropout,
            activation_fn=activation_fn,
        )
        self.ffn2_video = FeedForwardModule(
            embed_dim,
            ffn_embed_dim,
            dropout,
            dropout,
            activation_fn=activation_fn,
        )

        self.final_layer_norm_audio = LayerNorm(embed_dim, export=False)
        self.final_layer_norm_video = LayerNorm(embed_dim, export=False)

        self.co_attention = CoAttention(embed_dim)

    def forward(
        self,
        x_a, x_v, x,
        encoder_padding_mask: Optional[torch.Tensor],
        position_emb: Optional[Tuple[torch.Tensor]],
    ):
        """
        Args:
            x: Tensor of shape T X B X C
            encoder_padding_mask: Optional mask tensor
            positions:
        Returns:
            Tensor of shape T X B X C
        """
        pe_a, pe_v, pe = position_emb
        feat_a, feat_v = None, None

        # audio ffn1
        residual_a = x_a
        x_a = self.ffn1_audio(x_a)
        x_a = x_a * 0.5 + residual_a
        residual_a = x_a

        # video ffn1
        residual_v = x_v
        x_v = self.ffn1_video(x_v)
        x_v = x_v * 0.5 + residual_v
        residual_v = x_v

        if self.pos_enc_type == "rel_pos":
            # audio self-attention
            x_a = self.self_attn_layer_norm_audio(x_a)
            x_a, attn_a = self.self_attn_audio(
                query=x_a,
                key=x_a,
                value=x_a,
                key_padding_mask=encoder_padding_mask,
                pos_emb=pe_a,
                need_weights=False,
            )
            x_a = self.self_attn_dropout_audio(x_a)
            x_a = x_a + residual_a
            residual_a = x_a

            # video self-attention
            x_v = self.self_attn_layer_norm_video(x_v)
            x_v, attn_v = self.self_attn_video(
                query=x_v,
                key=x_v,
                value=x_v,
                key_padding_mask=encoder_padding_mask,
                pos_emb=pe_v,
                need_weights=False,
            )
            x_v = self.self_attn_dropout_video(x_v)
            x_v = x_v + residual_v
            residual_v = x_v

            # layer norm before cross-attention layers
            input_a = self.cross_attn_layer_norm_audio_video(x_a)
            input_v = self.cross_attn_layer_norm_video_audio(x_v)

            # T x B x C -> B x T x C
            feat_a = input_a.transpose(0, 1).contiguous()
            feat_v = input_v.transpose(0, 1).contiguous()

            # audio-video cross-attention
            x_a, attn_av = self.cross_attn_audio_video(
                query=input_a,
                key=input_v,
                value=input_v,
                key_padding_mask=encoder_padding_mask,
                pos_emb=pe_v,
                need_weights=False,
            )
            x_a = self.cross_attn_dropout_audio_video(x_a)
            x_a = x_a + residual_a
            residual_a = x_a

            # video-audio cross-attention
            x_v, attn_va = self.cross_attn_video_audio(
                query=input_v,
                key=input_a,
                value=input_a,
                key_padding_mask=encoder_padding_mask,
                pos_emb=pe_a,
                need_weights=False,
            )
            x_v = self.cross_attn_dropout_video_audio(x_v)
            x_v = x_v + residual_v
            residual_v = x_v
        
        else:
            # audio self-attention
            x_a = self.self_attn_layer_norm_audio(x_a)
            x_a, attn_a = self.self_attn_audio(
                query=x_a,
                key=x_a,
                value=x_a,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
            )
            x_a = self.self_attn_dropout_audio(x_a)
            x_a = x_a + residual_a
            residual_a = x_a

            # video self-attention
            x_v = self.self_attn_layer_norm_video(x_v)
            x_v, attn_v = self.self_attn_video(
                query=x_v,
                key=x_v,
                value=x_v,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
            )
            x_v = self.self_attn_dropout_video(x_v)
            x_v = x_v + residual_v
            residual_v = x_v

            # layer norm before cross-attention layers
            input_a = self.cross_attn_layer_norm_audio_video(x_a)
            input_v = self.cross_attn_layer_norm_video_audio(x_v)

            # T x B x C -> B x T x C
            feat_a = input_a.transpose(0, 1).contiguous()
            feat_v = input_v.transpose(0, 1).contiguous()

            # audio-video cross-attention
            x_a, attn_av = self.cross_attn_audio_video(
                query=input_a,
                key=input_v,
                value=input_v,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
            )
            x_a = self.cross_attn_dropout_audio_video(x_a)
            x_a = x_a + residual_a
            residual_a = x_a

            # video-audio cross-attention
            x_v, attn_va = self.cross_attn_video_audio(
                query=input_v,
                key=input_a,
                value=input_a,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
            )
            x_v = self.cross_attn_dropout_video_audio(x_v)
            x_v = x_v + residual_v
            residual_v = x_v
        
        # audio conv
        x_a = x_a.transpose(0, 1)
        x_a = self.conv_module_audio(x_a)
        x_a = x_a.transpose(0, 1)
        x_a = residual_a + x_a
        residual_a = x_a
        # audio ffn2
        x_a = self.ffn2_audio(x_a)
        x_a = x_a * 0.5 + residual_a

        # video conv
        x_v = x_v.transpose(0, 1)
        x_v = self.conv_module_video(x_v)
        x_v = x_v.transpose(0, 1)
        x_v = residual_v + x_v
        residual_v = x_v
        # video ffn2
        x_v = self.ffn2_video(x_v)
        x_v = x_v * 0.5 + residual_v

        # final layernorm
        x_a = self.final_layer_norm_audio(x_a)
        x_v = self.final_layer_norm_video(x_v)

        # av fusion
        x = self.co_attention(x_a, x_v, x)

        return x_a, x_v, x, attn_av, (feat_a, feat_v)

class GIConformerWav2Vec2EncoderLayer(GIConformerEncoderLayer):
    """Encoder layer for Wav2vec2 encoder"""

    def forward(
        self,
        x_a: torch.Tensor,
        x_v: torch.Tensor,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
        position_emb=(None, None, None),
    ):
        return super().forward(x_a, x_v, x, self_attn_padding_mask, position_emb)



class CoAttention(nn.Module):
    def __init__(
            self,
            embedding_dim: float = 768,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.layer_norm = LayerNorm(embedding_dim)

        self.conv_audio = nn.Sequential(
            nn.Conv1d(embedding_dim, embedding_dim, 1, 1, 0),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU(),
        )
        self.conv_video = nn.Sequential(
            nn.Conv1d(embedding_dim, embedding_dim, 1, 1, 0),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU(),
        )

    def forward(
            self,
            x_a: torch.Tensor,
            x_v: torch.Tensor,
            x: torch.Tensor,
    ):
        # (T, B, C) -> (B, T, C)
        x_a = x_a.transpose(0, 1).contiguous()
        x_v = x_v.transpose(0, 1).contiguous()
        x = x.transpose(0, 1).contiguous()

        # audio residual
        q, k, v = x, x_a.transpose(1, 2).contiguous(), x_a
        attn_map = torch.softmax(torch.matmul(q, k) / math.sqrt(self.embedding_dim), dim=-1)
        residual_audio = self.conv_audio(torch.matmul(attn_map, v).transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

        # video residual
        q, k, v = x, x_v.transpose(1, 2).contiguous(), x_v
        attn_map = torch.softmax(torch.matmul(q, k) / math.sqrt(self.embedding_dim), dim=-1)
        residual_video = self.conv_video(torch.matmul(attn_map, v).transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

        # output, (B, T, C) -> (T, B, C)
        x = x + residual_audio + residual_video
        x = x.transpose(0, 1).contiguous()

        return self.layer_norm(x)


