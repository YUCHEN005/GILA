# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os, sys
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import (
    ConvFeatureExtractionModel,
    TransformerEncoder,
    GITransformerEncoder,
    ConformerEncoder,
    GIConformerEncoder,
)
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,
    MultiheadAttention,
    SamePad,
    TransposeLast,
)
from copy import deepcopy

from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import buffered_arange, index_put, is_xla_tensor

DBG = True if len(sys.argv) == 1 else False

if DBG:
    from hubert_pretraining import (
        AVHubertPretrainingConfig,
        AVHubertPretrainingTask,
    )
    from resnet import ResEncoder

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    from utils import compute_mask_indices
    from decoder import TransformerDecoder

else:
    from .hubert_pretraining import (
        AVHubertPretrainingConfig,
        AVHubertPretrainingTask,
    )
    from .resnet import ResEncoder
    from .utils import compute_mask_indices
    from .decoder import TransformerDecoder

from omegaconf import II

logger = logging.getLogger(__name__)

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(
    ["static", "uniform", "normal", "poisson"]
)
LAYER_TYPE_CHOICES = ChoiceEnum(["transformer", "conformer"])


@dataclass
class AVHubertConfig(FairseqDataclass):
    label_rate: int = II("task.label_rate")
    input_modality: str = II("task.input_modality")
    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
                    "norm with d groups in the first conv block, whereas layer_norm "
                    "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    gi_model_layers: int = field(
        default=0, metadata={"help": "num cross-modal encoder layers in the transformer"}
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )
    layer_type: LAYER_TYPE_CHOICES = field(
        default="transformer", metadata={"help": "layer type in encoder"}
    )

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={
            "help": "dropout to apply to the features (after feat extr)"
        },
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many "
                    "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=False,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
                    "layers in form of a python list that contains "
                    "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    quantize_targets_audio: bool = field(
        default=False, metadata={"help": "use quantized targets for audio"}
    )
    quantize_targets_video: bool = field(
        default=False, metadata={"help": "use quantized targets for video"}
    )

    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    quantizer_depth: int = field(
        default=1,
        metadata={"help": "number of quantizer layers"},
    )
    quantizer_factor: int = field(
        default=3,
        metadata={
            "help": "dimensionality increase for inner quantizer layers (if depth > 1)"
        },
    )
    latent_vars: int = field(
        default=320,
        metadata={"help": "number of latent variables V in each group of the codebook"},
    )
    latent_groups: int = field(
        default=2,
        metadata={"help": "number of groups G of latent variables in the codebook"},
    )
    latent_dim: int = field(
        default=0,
        metadata={
            "help": "if > 0, uses this dimensionality for latent variables. "
            "otherwise uses final_dim / latent_groups"
        },
    )

    # masking
    mask_length_audio: int = field(default=10, metadata={"help": "mask length"})
    mask_prob_audio: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_length_image: int = field(default=10, metadata={"help": "mask length"})
    mask_prob_image: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
                    "(used for more complex distributions), "
                    "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={
            "help": "min space between spans (if no overlap is enabled)"
        },
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
                    "(used for more complex distributions), "
                    "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={
            "help": "min space between spans (if no overlap is enabled)"
        },
    )

    # negative sample selection
    negatives_from_everywhere: bool = field(
        default=False,
        metadata={"help": "sample negatives from everywhere, not just masked states"},
    )
    num_negatives: int = field(
        default=100,
        metadata={"help": "number of negative examples from the same sample"},
    )
    cross_sample_negatives: int = field(
        default=0, metadata={"help": "number of negative examples from the any sample"}
    )
    codebook_negatives: int = field(
        default=0, metadata={"help": "number of negative examples codebook"}
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={
            "help": "number of filters for convolutional positional embeddings"
        },
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={
            "help": "number of groups for convolutional positional embedding"
        },
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "legacy (to be removed)"},
    )
    max_positions: int = field(default=100000, metadata={"help": "Max positions"})

    # Conformer
    depthwise_conv_kernel_size: int = field(
        default=31,
        metadata={
            "help": "depthwise-conv-kernel-size for convolution in conformer layer"
        },
    )
    attn_type: str = field(
        default="",
        metadata={"help": "if espnet use ESPNET MHA"},
    )
    pos_enc_type: str = field(
        default="abs",
        metadata={"help": "Positional encoding type to use in conformer"},
    )
    fp16: bool = field(default=False, metadata={"help": "If fp16 is being used"})

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )
    resnet_relu_type: str = field(default='prelu', metadata={"help": 'relu type for resnet'})
    resnet_weights: Optional[str] = field(default=None, metadata={"help": 'resnet weights'})
    sim_type: str = field(default='cosine', metadata={"help": 'similarity type'})

    sub_encoder_layers: int = field(default=0, metadata={'help': 'number of transformer layers for single modality'})
    audio_feat_dim: int = field(default=-1, metadata={'help': 'audio feature dimension'})
    modality_dropout: float = field(default=0, metadata={'help': 'drop one modality'})
    audio_dropout: float = field(default=0, metadata={'help': 'drop audio feature'})
    modality_fuse: str = field(default='concat', metadata={'help': 'fusing two modalities: add,concat'})
    selection_type: str = field(default='same_other_seq', metadata={
        'help': 'type of selectig images, same_other_seq: replace masked span with span from another sequence, same_seq: repace masked span with span of the same sequence'})
    masking_type: str = field(default='input', metadata={'help': 'input or feature masking'})

    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(
        default=6, metadata={"help": "num of decoder layers"}
    )
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layernorm before each decoder block"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings "
                    "(outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.1,
        metadata={
            "help": "dropout probability for attention weights "
                    "inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
                    "inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"},
    )
    no_scale_embedding: bool = field(default=True, metadata={'help': 'scale embedding'})


class SubModel(nn.Module):
    def __init__(self, resnet=None, input_dim=None, cfg=None):
        super().__init__()
        self.resnet = resnet
        self.proj = nn.Linear(input_dim, cfg.encoder_embed_dim)
        self.encoder = TransformerEncoder(cfg) if cfg.encoder_layers > 0 else None

    def forward(self, x):
        if self.resnet is not None:
            x = self.resnet(x)
        x = self.proj(x.transpose(1, 2))
        if self.encoder is not None:
            x = self.encoder(x)[0].transpose(1, 2)
        else:
            x = x.transpose(1, 2)
        return x


@register_model("av_hubert", dataclass=AVHubertConfig)
class AVHubertModel(BaseFairseqModel):
    def __init__(
            self,
            cfg: AVHubertConfig,
            task_cfg: AVHubertPretrainingConfig,
            dictionaries: List[Dictionary],
            **kwargs
    ) -> None:
        super().__init__()
        logger.info(f"HubertModel Config: {cfg}")

        feature_ds_rate = 1
        self.feat2tar_ratio = cfg.label_rate * feature_ds_rate / task_cfg.sample_rate
        sub_cfg = deepcopy(cfg)
        sub_cfg.encoder_layers = sub_cfg.sub_encoder_layers
        resnet = ResEncoder(relu_type=cfg.resnet_relu_type, weights=cfg.resnet_weights)
        self.feature_extractor_audio = SubModel(resnet=None, input_dim=cfg.audio_feat_dim, cfg=sub_cfg)
        self.feature_extractor_video = SubModel(resnet=resnet, input_dim=resnet.backend_out, cfg=sub_cfg)
        self.modality_dropout, self.audio_dropout = cfg.modality_dropout, cfg.audio_dropout
        self.modality_fuse = cfg.modality_fuse
        self.encoder_embed_dim = cfg.encoder_embed_dim
        if self.modality_fuse == 'concat':
            self.embed = cfg.encoder_embed_dim * 2
        elif self.modality_fuse == 'add':
            self.embed = cfg.encoder_embed_dim
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob_image, self.mask_prob_audio = cfg.mask_prob_image, cfg.mask_prob_audio
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length_image, self.mask_length_audio = cfg.mask_length_image, cfg.mask_length_audio
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_input_audio = nn.Dropout(cfg.dropout_input)
        self.dropout_input_video = nn.Dropout(cfg.dropout_input)

        self.dropout_features = nn.Dropout(cfg.dropout_features)
        self.dropout_features_audio = nn.Dropout(cfg.dropout_features)
        self.dropout_features_video = nn.Dropout(cfg.dropout_features)

        self.dropout_av = nn.Dropout(cfg.dropout_input)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask
        self.sim_type = cfg.sim_type
        self.selection_type = cfg.selection_type
        self.masking_type = cfg.masking_type

        final_dim = (
            cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        )

        if cfg.quantize_targets_audio:
            vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else final_dim
            self.quantizer_a0 = GumbelVectorQuantizer(
                dim=cfg.encoder_embed_dim,
                num_vars=cfg.latent_vars,
                temp=cfg.latent_temp,
                groups=cfg.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
                weight_proj_depth=cfg.quantizer_depth,
                weight_proj_factor=cfg.quantizer_factor,
            )
            self.project_q_a0 = nn.Linear(vq_dim, final_dim)
            self.quantizer_a3 = GumbelVectorQuantizer(
                dim=cfg.encoder_embed_dim,
                num_vars=cfg.latent_vars,
                temp=cfg.latent_temp,
                groups=cfg.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
                weight_proj_depth=cfg.quantizer_depth,
                weight_proj_factor=cfg.quantizer_factor,
            )
            self.project_q_a3 = nn.Linear(vq_dim, final_dim)
        else:
            self.quantizer_a0 = None
            self.project_q_a0 = nn.Linear(cfg.encoder_embed_dim, final_dim)
            self.quantizer_a3 = None
            self.project_q_a3 = nn.Linear(cfg.encoder_embed_dim, final_dim)

        if cfg.quantize_targets_video:
            vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else final_dim
            self.quantizer_v0 = GumbelVectorQuantizer(
                dim=cfg.encoder_embed_dim,
                num_vars=cfg.latent_vars,
                temp=cfg.latent_temp,
                groups=cfg.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
                weight_proj_depth=cfg.quantizer_depth,
                weight_proj_factor=cfg.quantizer_factor,
            )
            self.project_q_v0 = nn.Linear(vq_dim, final_dim)
            self.quantizer_v3 = GumbelVectorQuantizer(
                dim=cfg.encoder_embed_dim,
                num_vars=cfg.latent_vars,
                temp=cfg.latent_temp,
                groups=cfg.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
                weight_proj_depth=cfg.quantizer_depth,
                weight_proj_factor=cfg.quantizer_factor,
            )
            self.project_q_v3 = nn.Linear(vq_dim, final_dim)
        else:
            self.quantizer_v0 = None
            self.project_q_v0 = nn.Linear(cfg.encoder_embed_dim, final_dim)
            self.quantizer_v3 = None
            self.project_q_v3 = nn.Linear(cfg.encoder_embed_dim, final_dim)
        
        self.negatives_from_everywhere = cfg.negatives_from_everywhere
        self.n_negatives = cfg.num_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.audio_feat_dim).uniform_() if self.masking_type == 'input' else torch.FloatTensor(
                cfg.encoder_embed_dim).uniform_()
        )

        self.gi_model = GITransformerEncoder(cfg)
        self.encoder = TransformerEncoder(cfg)
        self.proj_av = nn.Linear(cfg.encoder_embed_dim * 3, cfg.encoder_embed_dim)
        self.layer_norm_av = LayerNorm(cfg.encoder_embed_dim * 3)

        self.layer_norm = LayerNorm(self.embed)
        self.layer_norm_audio = LayerNorm(cfg.encoder_embed_dim)
        self.layer_norm_video = LayerNorm(cfg.encoder_embed_dim)


        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )
            self.target_glu_a0 = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )
            self.target_glu_a3 = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )
            self.target_glu_v0 = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )
            self.target_glu_v3 = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )
        else:
            self.target_glu = None
            self.target_glu_a0 = None
            self.target_glu_a3 = None
            self.target_glu_v0 = None
            self.target_glu_v3 = None

        self.untie_final_proj = cfg.untie_final_proj
        if self.untie_final_proj:
            self.final_proj = nn.Linear(
                cfg.encoder_embed_dim, final_dim * len(dictionaries)
            )
            self.final_proj_a0 = nn.Linear(
                cfg.encoder_embed_dim, final_dim * len(dictionaries)
            )
            self.final_proj_a3 = nn.Linear(
                cfg.encoder_embed_dim, final_dim * len(dictionaries)
            )
            self.final_proj_v0 = nn.Linear(
                cfg.encoder_embed_dim, final_dim * len(dictionaries)
            )
            self.final_proj_v3 = nn.Linear(
                cfg.encoder_embed_dim, final_dim * len(dictionaries)
            )
        else:
            self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)
            self.final_proj_a0 = nn.Linear(cfg.encoder_embed_dim, final_dim)
            self.final_proj_a3 = nn.Linear(cfg.encoder_embed_dim, final_dim)
            self.final_proj_v0 = nn.Linear(cfg.encoder_embed_dim, final_dim)
            self.final_proj_v3 = nn.Linear(cfg.encoder_embed_dim, final_dim)

        # modules below are not needed during fine-tuning
        if any([d is None for d in dictionaries]):
            logger.info(
                "cannot find dictionary. assume will be used for fine-tuning"
            )
        else:
            self.num_classes = [len(d) for d in dictionaries]
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(sum(self.num_classes), final_dim)
            )
            nn.init.uniform_(self.label_embs_concat)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: AVHubertConfig, task: AVHubertPretrainingTask):
        """Build a new model instance."""

        kwargs = {}
        model = AVHubertModel(cfg, task.cfg, task.dictionaries, **kwargs)
        return model

    def apply_input_mask(self, x, padding_mask, target_list):
        B, C, T = x.shape[:3]
        is_audio = True if len(x.shape) == 3 else False
        if is_audio:
            mask_prob, mask_length = self.mask_prob_audio, self.mask_length_audio
        else:
            mask_prob, mask_length = self.mask_prob_image, self.mask_length_image
        if mask_prob > 0:

            mask_indices, starts, ends, batch_indexes = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_prob,
                mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices_np = mask_indices
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = x.transpose(1, 2).contiguous()  # [B, T, C, H, W]
            if B == 1:
                x[mask_indices] = 0
            elif is_audio:
                x[mask_indices] = self.mask_emb
            elif self.selection_type == 'same_other_seq':
                perm = (torch.arange(B) + torch.randint(low=1, high=B, size=(1,))) % B
                x_perm = x[perm]
                x[mask_indices] = x_perm[mask_indices]
            elif self.selection_type == 'same_seq':
                batch_indexes_, other_indexes = [], []
                for batch_index, start, end in zip(batch_indexes, starts, ends):
                    length = end - start
                    other_start = np.setdiff1d(np.arange(T), np.arange(max(0, start - length), end))
                    if len(other_start) > 0:
                        other_start = np.random.choice(other_start, size=1)
                    else:
                        other_start = 0
                    other_end = other_start + length
                    other_indexes.append(np.arange(other_start, other_end).clip(max=T - 1))
                    batch_indexes_.append(np.zeros([length], dtype=np.int64) + batch_index)
                batch_indexes, other_indexes = np.concatenate(batch_indexes_), np.concatenate(other_indexes)
                x[mask_indices] = x[batch_indexes, other_indexes]

            x = x.transpose(1, 2).contiguous()
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            logger.info(f"No mask channel prob for input masking")
        return x, mask_indices

    def apply_feature_mask(self, x, padding_mask, target_list):
        B, T, C = x.shape
        assert self.mask_prob_audio == self.mask_prob_image and self.mask_length_audio == self.mask_length_image, f"masking prob/length for image/audio be same for feature masking"
        mask_prob, mask_length = self.mask_prob_audio, self.mask_length_image
        if mask_prob > 0:
            mask_indices, _, _, _ = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_prob,
                mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices, _, _, _ = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                    .to(x.device)
                    .unsqueeze(1)
                    .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def forward_features(self, source: torch.Tensor, modality: str) -> torch.Tensor:
        extractor = eval(f"self.feature_extractor_{modality}")
        if self.feature_grad_mult > 0:
            features = extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = extractor(source)
        return features

    def forward_targets(
            self, features: torch.Tensor, mask_indices: torch.Tensor, target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
            if mask_indices is not None:
                mask_indices = mask_indices[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, mask_indices, target_list

    def forward_padding_mask(
            self, features: torch.Tensor, padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def compute_logits(self, feats, emb_mat):
        # feats: [B, T, F], emb_mat: [V, F]
        if self.sim_type == 'dot':
            logits = torch.matmul(feats, emb_mat.transpose(0, 1))
        elif self.sim_type == 'cosine':
            batch_size, timesteps, emb_dim = feats.size()
            feats_ = feats.view(-1, emb_dim)
            nom = (feats_.unsqueeze(dim=1) * emb_mat.unsqueeze(dim=0)).sum(dim=-1)  # [B*T, V]
            denom = (feats_ ** 2).sum(dim=-1).sqrt().unsqueeze(dim=1) * (emb_mat ** 2).sum(dim=-1).sqrt().unsqueeze(
                dim=0)  # [B*T, V]
            logits = (nom / denom.clamp(min=1e-6)).view(batch_size, timesteps, -1)
        else:
            raise NotImplementedError
        logits = logits / self.logit_temp
        return logits

    def forward(
            self,
            source: torch.Tensor,
            target_list: Optional[List[torch.Tensor]] = None,
            padding_mask: Optional[torch.Tensor] = None,
            mask: bool = True,
            features_only: bool = False,
            output_layer: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        src_audio, src_video = source['audio'], source['video']
        if mask and self.masking_type == 'input':
            src_video, mask_indices_video = self.apply_input_mask(src_video, padding_mask, target_list)
            src_audio, mask_indices_audio = self.apply_input_mask(src_audio, padding_mask, target_list)
            mask_indices = torch.logical_or(mask_indices_audio, mask_indices_video)
        else:
            src_audio, src_video, mask_indices = src_audio, src_video, None

        features_audio = self.forward_features(src_audio, modality='audio')  # features: [B, F, T]
        features_video = self.forward_features(src_video, modality='video')
        modality_drop_prob, audio_drop_prob = np.random.random(), np.random.random()
        if self.training:
            if modality_drop_prob < self.modality_dropout:
                if audio_drop_prob < self.audio_dropout:
                    features_audio = 0 * features_audio
                else:
                    features_video = 0 * features_video
        if self.modality_fuse == 'concat':
            features = torch.cat([features_audio, features_video], dim=1)
        elif self.modality_fuse == 'add':
            features = features_audio + features_video
        if target_list is not None:
            features, mask_indices, target_list = self.forward_targets(features, mask_indices, target_list)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        if self.masking_type == 'feature' and mask:
            x, mask_indices = self.apply_feature_mask(features, padding_mask, target_list)
        else:
            x = features

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1
        )

        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features}

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)
        proj_x = self.final_proj(x)
        if self.untie_final_proj:
            proj_x_list = proj_x.chunk(len(self.num_classes), dim=-1)
        else:
            proj_x_list = [proj_x for _ in self.num_classes]
        logit_list = [self.compute_logits(proj, emb).view(-1, num_class) for proj, emb, num_class in
                      zip(proj_x_list, label_embs_list, self.num_classes)]  # [[B*T, V]]
        mask, unmask = torch.logical_and(mask_indices, ~padding_mask).view(-1), torch.logical_and(~mask_indices,
                                                                                                  ~padding_mask).view(
            -1)  # [B*T]
        logit_m_list, logit_u_list = [logit[mask] for logit in logit_list], [logit[unmask] for logit in logit_list]
        target_m_list, target_u_list = [target.view(-1)[mask].long() for target in target_list], [
            target.view(-1)[unmask].long() for target in target_list]
        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "target_m_list": target_m_list,
            "target_u_list": target_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }
        return result

    def extract_features(
            self,
            source: torch.Tensor,
            padding_mask: Optional[torch.Tensor] = None,
            mask: bool = False,
            ret_conv: bool = False,
            output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        return feature, res["padding_mask"]
    
    def generate_feature_mask(self, x, padding_mask, feature_type):
        B, T, C = x.shape
        if feature_type == "audio":
            mask_prob, mask_length = self.mask_prob_audio, self.mask_length_audio
        elif feature_type == "video":
            mask_prob, mask_length = self.mask_prob_image, self.mask_length_image
        
        if mask_prob > 0:
            mask_indices, _, _, _ = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_prob,
                mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            # x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        # if self.mask_channel_prob > 0:
        #     mask_channel_indices, _, _, _ = compute_mask_indices(
        #         (B, C),
        #         None,
        #         self.mask_channel_prob,
        #         self.mask_channel_length,
        #         self.mask_channel_selection,
        #         self.mask_channel_other,
        #         no_overlap=self.no_mask_channel_overlap,
        #         min_space=self.mask_channel_min_space,
        #     )
        #     mask_channel_indices = (
        #         torch.from_numpy(mask_channel_indices)
        #             .to(x.device)
        #             .unsqueeze(1)
        #             .expand(-1, T, -1)
        #     )
        #     x[mask_channel_indices] = 0

        return mask_indices
    
    def sample_negatives(self, y, num, padding_count=None):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        # FIXME: what happens if padding_count is specified?
        cross_high = tsz * bsz
        high = tsz - (padding_count or 0)
        with torch.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"

            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs
    
    def compute_preds(self, x, y, negatives):
        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)

        logits = logits / self.logit_temp

        if is_xla_tensor(logits) or neg_is_pos.any():
            fillval = -float(2 ** 30)
            if not hasattr(self, "_inftensor"):
                self._inftensor = (
                    torch.tensor(fillval).to(x.device)
                    if is_xla_tensor(logits)
                    else float("-inf")
                )
            logits[1:] = index_put(logits[1:], neg_is_pos, self._inftensor)

        return logits

    def extract_finetune(self, source, padding_mask=None, mask=False, ret_conv=False, output_layer=None, padding_count=None):
        src_audio, src_video = source['audio'], source['video']
        # print(f"src_audio.shape: {src_audio.shape}, src_video.shape: {src_video.shape}")
        if mask and self.masking_type == 'input':
            src_video, mask_indices_video = self.apply_input_mask(src_video, padding_mask, target_list=None)
            src_audio, mask_indices_audio = self.apply_input_mask(src_audio, padding_mask, target_list=None)
            mask_indices = torch.logical_or(mask_indices_audio,
                                            mask_indices_video)  # mask_indices not used in fine-tuning
        else:
            src_audio, src_video, mask_indices = src_audio, src_video, None

        if src_audio is not None and src_video is None:
            features_audio = self.forward_features(src_audio, modality='audio')
            features_video = features_audio.new_zeros(features_audio.size(0), self.encoder_embed_dim,
                                                      features_audio.size(-1))
        elif src_audio is None and src_video is not None:
            features_video = self.forward_features(src_video, modality='video')
            features_audio = features_video.new_zeros(features_video.size(0), self.encoder_embed_dim,
                                                      features_video.size(-1))
        elif src_audio is not None and src_video is not None:
            features_video = self.forward_features(src_video, modality='video')  # features: [B, C, T]
            features_audio = self.forward_features(src_audio, modality='audio')  # features: [B, C, T]

        if self.modality_fuse == 'concat':
            features = torch.cat([features_audio, features_video], dim=1)
        elif self.modality_fuse == 'add':
            features = features_audio + features_video

        features_pen_audio = features_audio.float().pow(2).mean()
        features_audio = features_audio.transpose(1, 2)  # [B, T, C]
        features_audio = self.layer_norm_audio(features_audio)  # [B, T, C]
        unmasked_features_audio = features_audio.clone()

        features_pen_video = features_video.float().pow(2).mean()
        features_video = features_video.transpose(1, 2)  # [B, T, C]
        features_video = self.layer_norm_video(features_video)  # [B, T, C]
        unmasked_features_video = features_video.clone()

        features_pen = features.float().pow(2).mean()
        features = features.transpose(1, 2)  # [B, T, 2C]
        features = self.layer_norm(features)  # [B, T, 2C]
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)  # [B, T, C]

        features_audio = self.dropout_input_audio(features_audio)
        features_video = self.dropout_input_video(features_video)
        features = self.dropout_input(features)

        unmasked_features_audio = self.dropout_features_audio(unmasked_features_audio)
        unmasked_features_video = self.dropout_features_video(unmasked_features_video)
        unmasked_features = self.dropout_features(unmasked_features)

        mask_indices_audio = self.generate_feature_mask(features_audio, padding_mask, feature_type="audio")
        mask_indices_video = self.generate_feature_mask(features_video, padding_mask, feature_type="video")

        if not is_xla_tensor(features_audio) and mask_indices_audio is not None:
            # tpu-comment: reducing the size in a dynamic way causes
            # too many recompilations on xla.
            y_a = unmasked_features_audio[mask_indices_audio].view(
                unmasked_features_audio.size(0), -1, unmasked_features_audio.size(-1)
            )
        else:
            y_a = unmasked_features_audio
        
        if not is_xla_tensor(features_video) and mask_indices_video is not None:
            # tpu-comment: reducing the size in a dynamic way causes
            # too many recompilations on xla.
            y_v = unmasked_features_video[mask_indices_video].view(
                unmasked_features_video.size(0), -1, unmasked_features_video.size(-1)
            )
        else:
            y_v = unmasked_features_video

        x_a, x_v, x = features_audio, features_video, features

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool

        ### Global Interaction Model ###
        x_a, x_v, x, feats = self.gi_model(
            x_a, x_v, x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1
        )

        ### Local Alignment Method ###
        ## layer-wise contrastive learning
        lw_align_logits = self.get_lw_align_logits(feats)

        ## cross-layer contrastive learning
        # a3 -> v0
        num_vars_v0, code_ppl_v0, prob_ppl_v0, curr_temp_v0 = None, None, None, None
        if self.quantizer_v0:
            q_v0 = self.quantizer_v0(y_v, produce_targets=False)
            y_v0 = q_v0["x"]
            num_vars_v0 = q_v0["num_vars"]
            code_ppl_v0 = q_v0["code_perplexity"]
            prob_ppl_v0 = q_v0["prob_perplexity"]
            curr_temp_v0 = q_v0["temp"]

            y_v0 = self.project_q_v0(y_v0)

            if self.negatives_from_everywhere:
                neg_cands_v0 = self.quantizer_v0(unmasked_features_video, produce_targets=False)["x"]
                negs_v0, _ = self.sample_negatives(
                    neg_cands_v0,
                    y_v0.size(1),
                    padding_count=padding_count,
                )
                negs_v0 = self.project_q_v0(negs_v0)
            else:
                negs_v0, _ = self.sample_negatives(
                    y_v0,
                    y_v0.size(1),
                    padding_count=padding_count,
                )

            if self.codebook_negatives > 0:
                cb_negs_v0 = self.quantizer_v0.sample_from_codebook(
                    y_v0.size(0) * y_v0.size(1), self.codebook_negatives
                )
                cb_negs_v0 = cb_negs_v0.view(
                    self.codebook_negatives, y_v0.size(0), y_v0.size(1), -1
                )  # order doesnt matter
                cb_negs_v0 = self.project_q_v0(cb_negs_v0)
                negs_v0 = torch.cat([negs_v0, cb_negs_v0], dim=0)
        else:
            y_v0 = self.project_q_v0(y_v)

            if self.negatives_from_everywhere:
                negs_v0, _ = self.sample_negatives(
                    unmasked_features_video,
                    y_v0.size(1),
                    padding_count=padding_count,
                )
                negs_v0 = self.project_q_v0(negs_v0)
            else:
                negs_v0, _ = self.sample_negatives(
                    y_v0,
                    y_v0.size(1),
                    padding_count=padding_count,
                )

        if self.target_glu_v0:
            y_v0 = self.target_glu_v0(y_v0)
            negs_v0 = self.target_glu_v0(negs_v0)

        x_a3 = x_a.clone()
        if not is_xla_tensor(x_a):
            # tpu-comment: reducing the size in a dynamic way causes
            # too many recompilations on xla.
            x_a3 = x_a3[mask_indices_video].view(x_a3.size(0), -1, x_a3.size(-1))

        x_a3 = self.final_proj_a3(x_a3)
        x_a3 = self.compute_preds(x_a3, y_v0, negs_v0)

        logits_a3_v0 = {
            "x": x_a3,
            "padding_mask": padding_mask,
            "features_pen": x_a.float().pow(2).mean(), # |a3|
        }

        if prob_ppl_v0 is not None:
            logits_a3_v0["prob_perplexity"] = prob_ppl_v0
            logits_a3_v0["code_perplexity"] = code_ppl_v0
            logits_a3_v0["num_vars"] = num_vars_v0
            logits_a3_v0["temp"] = curr_temp_v0



        # v0 -> a3
        num_vars_a3, code_ppl_a3, prob_ppl_a3, curr_temp_a3 = None, None, None, None
        y_a3 = x_a.clone()
        if not is_xla_tensor(x_a):
            # tpu-comment: reducing the size in a dynamic way causes
            # too many recompilations on xla.
            y_a3 = y_a3[mask_indices_video].view(y_a3.size(0), -1, y_a3.size(-1))
        if self.quantizer_a3:
            q_a3 = self.quantizer_a3(y_a3, produce_targets=False)
            y_a3 = q_a3["x"]
            num_vars_a3 = q_a3["num_vars"]
            code_ppl_a3 = q_a3["code_perplexity"]
            prob_ppl_a3 = q_a3["prob_perplexity"]
            curr_temp_a3 = q_a3["temp"]

            y_a3 = self.project_q_a3(y_a3)

            if self.negatives_from_everywhere:
                neg_cands_a3 = self.quantizer_a3(x_a.clone(), produce_targets=False)["x"]
                negs_a3, _ = self.sample_negatives(
                    neg_cands_a3,
                    y_a3.size(1),
                    padding_count=padding_count,
                )
                negs_a3 = self.project_q_a3(negs_a3)
            else:
                negs_a3, _ = self.sample_negatives(
                    y_a3,
                    y_a3.size(1),
                    padding_count=padding_count,
                )

            if self.codebook_negatives > 0:
                cb_negs_a3 = self.quantizer_a3.sample_from_codebook(
                    y_a3.size(0) * y_a3.size(1), self.codebook_negatives
                )
                cb_negs_a3 = cb_negs_a3.view(
                    self.codebook_negatives, y_a3.size(0), y_a3.size(1), -1
                )  # order doesnt matter
                cb_negs_a3 = self.project_q_a3(cb_negs_a3)
                negs_a3 = torch.cat([negs_a3, cb_negs_a3], dim=0)
        else:
            y_a3 = self.project_q_a3(y_a3)

            if self.negatives_from_everywhere:
                negs_a3, _ = self.sample_negatives(
                    x_a.clone(),
                    y_a3.size(1),
                    padding_count=padding_count,
                )
                negs_a3 = self.project_q_a3(negs_a3)
            else:
                negs_a3, _ = self.sample_negatives(
                    y_a3,
                    y_a3.size(1),
                    padding_count=padding_count,
                )

        if self.target_glu_a3:
            y_a3 = self.target_glu_a3(y_a3)
            negs_a3 = self.target_glu_a3(negs_a3)

        x_v0 = self.final_proj_v0(y_v.clone())
        x_v0 = self.compute_preds(x_v0, y_a3, negs_a3)

        logits_v0_a3 = {
            "x": x_v0,
            "padding_mask": padding_mask,
            "features_pen": features_pen_video, # |v0|
        }

        if prob_ppl_a3 is not None:
            logits_v0_a3["prob_perplexity"] = prob_ppl_a3
            logits_v0_a3["code_perplexity"] = code_ppl_a3
            logits_v0_a3["num_vars"] = num_vars_a3
            logits_v0_a3["temp"] = curr_temp_a3



        # v3 -> a0
        num_vars_a0, code_ppl_a0, prob_ppl_a0, curr_temp_a0 = None, None, None, None
        if self.quantizer_a0:
            q_a0 = self.quantizer_a0(y_a, produce_targets=False)
            y_a0 = q_a0["x"]
            num_vars_a0 = q_a0["num_vars"]
            code_ppl_a0 = q_a0["code_perplexity"]
            prob_ppl_a0 = q_a0["prob_perplexity"]
            curr_temp_a0 = q_a0["temp"]

            y_a0 = self.project_q_a0(y_a0)

            if self.negatives_from_everywhere:
                neg_cands_a0 = self.quantizer_a0(unmasked_features_audio, produce_targets=False)["x"]
                negs_a0, _ = self.sample_negatives(
                    neg_cands_a0,
                    y_a0.size(1),
                    padding_count=padding_count,
                )
                negs_a0 = self.project_q_a0(negs_a0)
            else:
                negs_a0, _ = self.sample_negatives(
                    y_a0,
                    y_a0.size(1),
                    padding_count=padding_count,
                )

            if self.codebook_negatives > 0:
                cb_negs_a0 = self.quantizer_a0.sample_from_codebook(
                    y_a0.size(0) * y_a0.size(1), self.codebook_negatives
                )
                cb_negs_a0 = cb_negs_a0.view(
                    self.codebook_negatives, y_a0.size(0), y_a0.size(1), -1
                )  # order doesnt matter
                cb_negs_a0 = self.project_q_a0(cb_negs_a0)
                negs_a0 = torch.cat([negs_a0, cb_negs_a0], dim=0)
        else:
            y_a0 = self.project_q_a0(y_a)

            if self.negatives_from_everywhere:
                negs_a0, _ = self.sample_negatives(
                    unmasked_features_audio,
                    y_a0.size(1),
                    padding_count=padding_count,
                )
                negs_a0 = self.project_q_a0(negs_a0)
            else:
                negs_a0, _ = self.sample_negatives(
                    y_a0,
                    y_a0.size(1),
                    padding_count=padding_count,
                )

        if self.target_glu_a0:
            y_a0 = self.target_glu_a0(y_a0)
            negs_a0 = self.target_glu_a0(negs_a0)

        x_v3 = x_v.clone()
        if not is_xla_tensor(x_v):
            # tpu-comment: reducing the size in a dynamic way causes
            # too many recompilations on xla.
            x_v3 = x_v3[mask_indices_audio].view(x_v3.size(0), -1, x_v3.size(-1))

        x_v3 = self.final_proj_v3(x_v3)
        x_v3 = self.compute_preds(x_v3, y_a0, negs_a0)

        logits_v3_a0 = {
            "x": x_v3,
            "padding_mask": padding_mask,
            "features_pen": x_v.float().pow(2).mean(), # |v3|
        }

        if prob_ppl_a0 is not None:
            logits_v3_a0["prob_perplexity"] = prob_ppl_a0
            logits_v3_a0["code_perplexity"] = code_ppl_a0
            logits_v3_a0["num_vars"] = num_vars_a0
            logits_v3_a0["temp"] = curr_temp_a0



        # a0 -> v3
        num_vars_v3, code_ppl_v3, prob_ppl_v3, curr_temp_v3 = None, None, None, None
        y_v3 = x_v.clone()
        if not is_xla_tensor(x_v):
            # tpu-comment: reducing the size in a dynamic way causes
            # too many recompilations on xla.
            y_v3 = y_v3[mask_indices_audio].view(y_v3.size(0), -1, y_v3.size(-1))
        if self.quantizer_v3:
            q_v3 = self.quantizer_v3(y_v3, produce_targets=False)
            y_v3 = q_v3["x"]
            num_vars_v3 = q_v3["num_vars"]
            code_ppl_v3 = q_v3["code_perplexity"]
            prob_ppl_v3 = q_v3["prob_perplexity"]
            curr_temp_v3 = q_v3["temp"]

            y_v3 = self.project_q_v3(y_v3)

            if self.negatives_from_everywhere:
                neg_cands_v3 = self.quantizer_v3(x_v.clone(), produce_targets=False)["x"]
                negs_v3, _ = self.sample_negatives(
                    neg_cands_v3,
                    y_v3.size(1),
                    padding_count=padding_count,
                )
                negs_v3 = self.project_q_v3(negs_v3)
            else:
                negs_v3, _ = self.sample_negatives(
                    y_v3,
                    y_v3.size(1),
                    padding_count=padding_count,
                )

            if self.codebook_negatives > 0:
                cb_negs_v3 = self.quantizer_v3.sample_from_codebook(
                    y_v3.size(0) * y_v3.size(1), self.codebook_negatives
                )
                cb_negs_v3 = cb_negs_v3.view(
                    self.codebook_negatives, y_v3.size(0), y_v3.size(1), -1
                )  # order doesnt matter
                cb_negs_v3 = self.project_q_v3(cb_negs_v3)
                negs_v3 = torch.cat([negs_v3, cb_negs_v3], dim=0)
        else:
            y_v3 = self.project_q_v3(y_v3)

            if self.negatives_from_everywhere:
                negs_v3, _ = self.sample_negatives(
                    x_v.clone(),
                    y_v3.size(1),
                    padding_count=padding_count,
                )
                negs_v3 = self.project_q_v3(negs_v3)
            else:
                negs_v3, _ = self.sample_negatives(
                    y_v3,
                    y_v3.size(1),
                    padding_count=padding_count,
                )

        if self.target_glu_v3:
            y_v3 = self.target_glu_v3(y_v3)
            negs_v3 = self.target_glu_v3(negs_v3)

        x_a0 = self.final_proj_a0(y_a.clone())
        x_a0 = self.compute_preds(x_a0, y_v3, negs_v3)

        logits_a0_v3 = {
            "x": x_a0,
            "padding_mask": padding_mask,
            "features_pen": features_pen_audio, # |a0|
        }

        if prob_ppl_v3 is not None:
            logits_a0_v3["prob_perplexity"] = prob_ppl_v3
            logits_a0_v3["code_perplexity"] = code_ppl_v3
            logits_a0_v3["num_vars"] = num_vars_v3
            logits_a0_v3["temp"] = curr_temp_v3



        # Fusion
        x = torch.cat([x_a, x_v, x], dim=-1)  # [B, T, 3C]
        x = self.layer_norm_av(x)  # [B, T, 3C]
        x = self.proj_av(x)  # [B, T, C]
        x = self.dropout_av(x)  # [B, T, C]

        # print(f"x.shape[:2]: {x.shape[:2]}")

        # Encoder
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1
        )

        return x, padding_mask, lw_align_logits, logits_a3_v0, logits_v0_a3, logits_v3_a0, logits_a0_v3
    
    def get_lw_align_logits(self, feats):
        lw_align_logits = []
        for feat in feats:
            # B x T x C
            feat_a, feat_v = feat
            assert feat_a.shape == feat_v.shape
            # B x T x C -> B x C x T
            feat_v = feat_v.transpose(1, 2).contiguous()
            # B x T x T
            matrix = torch.matmul(feat_a, feat_v)
            lw_align_logits.append(matrix)
        
        return lw_align_logits
    
    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None

    def get_extra_losses(self, net_output):
        pen = []

        if "prob_perplexity" in net_output:
            pen.append(
                (net_output["num_vars"] - net_output["prob_perplexity"])
                / net_output["num_vars"]
            )

        if "features_pen" in net_output:
            pen.append(net_output["features_pen"])

        return pen

    def get_logits(self, net_output):
        logits = net_output["x"]
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

    def get_targets(self, sample, net_output, expand_steps=True):
        x = net_output["x"]
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    # def get_extra_losses(self, net_output):
    #     extra_losses = []
    #     names = []
    #     if "features_pen" in net_output:
    #         extra_losses.append(net_output["features_pen"])
    #         names.append("features_pen")

    #     return extra_losses, names

    # def get_logits(self, net_output, is_masked=True):
    #     raise NotImplementedError

    # def get_targets(self, net_output, is_masked=True):
    #     raise NotImplementedError

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(
            x.float(), targets.float(), dim=-1
        ).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits
