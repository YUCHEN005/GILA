# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.utils import is_xla_tensor
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    weight_lw: float = field(
        default=0.0,
        metadata={"help": "weight of alignment loss"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    infonce: bool = field(
        default=False,
        metadata={
            "help": "if set, uses cross entropy instead of binary cross entropy (i.e. InfoNCE loss)"
        },
    )
    weights_cl: Optional[List[float]] = field(
        default=None,
        metadata={"help": "weights for audio loss and video loss"},
    )
    weights_cl_a3: Optional[List[float]] = field(
        default=None,
        metadata={"help": "weights for additional audio loss terms (not first one)"},
    )
    weights_cl_v0: Optional[List[float]] = field(
        default=None,
        metadata={"help": "weights for additional video loss terms (not first one)"},
    )
    weights_cl_v3: Optional[List[float]] = field(
        default=None,
        metadata={"help": "weights for additional audio loss terms (not first one)"},
    )
    weights_cl_a0: Optional[List[float]] = field(
        default=None,
        metadata={"help": "weights for additional video loss terms (not first one)"},
    )
    log_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "output keys to log"},
    )


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        weight_lw=0,
        infonce=False,
        weights_cl=None,
        weights_cl_a3=None,
        weights_cl_v0=None,
        weights_cl_v3=None,
        weights_cl_a0=None,
        log_keys=None,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.weight_lw = weight_lw
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.infonce = infonce
        self.weights_cl = weights_cl
        self.weights_cl_a3 = weights_cl_a3
        self.weights_cl_v0 = weights_cl_v0
        self.weights_cl_v3 = weights_cl_v3
        self.weights_cl_a0 = weights_cl_a0
        self.log_keys = [] if log_keys is None else log_keys

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output, lw_align_logits, aux_output = model(**sample["net_input"])
        logits_a3_v0, logits_v0_a3, logits_v3_a0, logits_a0_v3 = aux_output["logits_a3_v0"], aux_output["logits_v0_a3"], aux_output["logits_v3_a0"], aux_output["logits_a0_v3"]
        encoder = model.encoder.w2v_model


        ### ASR loss
        loss_asr, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        ### Local alignment loss
        ## Layer-wise alignment loss
        loss_lw = torch.tensor(0, dtype=torch.float16).to(loss_asr.device)
        for matrix in lw_align_logits:
            B, T, _ = matrix.shape      # (B, T, T)
            tgt = torch.arange(T).unsqueeze(0).repeat(B, 1).to(matrix.device)       # (B, T)
            loss1 = self.ce_loss(matrix, tgt)
            mask1 = torch.isfinite(loss1)
            loss1 = loss1.masked_select(mask1).mean()
            # print(f"loss1: {loss1}, all finite? {mask1.all()}, num_inf: {B*T - mask1.sum()}")

            loss2 = self.ce_loss(matrix.transpose(1, 2), tgt)
            mask2 = torch.isfinite(loss2)
            loss2 = loss2.masked_select(mask2).mean()
            # print(f"loss2: {loss2}, all finite? {mask2.all()}, num_inf: {B*T - mask2.sum()}")

            loss_lw = loss_lw + (loss1 + loss2) / 2

        # loss_lw = loss_lw / len(lw_align_logits) if lw_align_logits else loss_lw


        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss_asr.data,
            "loss_asr": loss_asr.data,
            "loss_lw": loss_lw.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)


        ## Cross-layer alignment loss
        # a3 -> v0
        logits_a3 = encoder.get_logits(logits_a3_v0).float()
        target_a3 = encoder.get_targets(sample, logits_a3_v0)
        losses_a3 = []
        # XXX: handle weights on xla.
        weights_a3 = None
        if hasattr(encoder, "get_target_weights") and not self.infonce:
            weights_a3 = encoder.get_target_weights(target_a3, logits_a3_v0)
            if torch.is_tensor(weights_a3):
                weights_a3 = weights_a3.float()
        
        reduction = "none" if not reduce else "sum"
        if self.infonce:
            loss_a3 = F.cross_entropy(logits_a3, target_a3, reduction=reduction)
        else:
            loss_a3 = F.binary_cross_entropy_with_logits(
                logits_a3, target_a3.float(), weights_a3, reduction=reduction
            )

        if 'sample_size' in sample:
            sample_size_a3 = sample['sample_size']
        elif 'mask_indices' in sample['net_input']:
            sample_size_a3 = sample['net_input']['mask_indices'].sum()
        else:
            sample_size_a3 = target_a3.numel() if self.infonce else target_a3.long().sum().item()
        losses_a3.append(loss_a3.detach().clone())

        if self.weights_cl_a3 is not None:
            assert hasattr(encoder, "get_extra_losses")
            extra_losses_a3 = encoder.get_extra_losses(logits_a3_v0)
            if torch.is_tensor(extra_losses_a3):
                extra_losses_a3 = [extra_losses_a3]
            if len(self.weights_cl_a3) == 1 and len(extra_losses_a3) != 1:
                self.weights_cl_a3 = [self.weights_cl_a3[0]] * len(extra_losses_a3)
            assert len(extra_losses_a3) == len(
                self.weights_cl_a3
            ), f"{len(extra_losses_a3)}, {len(self.weights_cl_a3)}"
            for p, coef in zip(extra_losses_a3, self.weights_cl_a3):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size_a3
                    loss_a3 += p
                    losses_a3.append(p)
        
        logging_output["loss_a3"] = loss_a3.item() if reduce else loss_a3.detach()
        if len(losses_a3) > 1:
            for i, l in enumerate(losses_a3):
                logging_output[f"loss_a3_{i}"] = l.item()
        
        for lk in self.log_keys:
            # Only store "logits" and "target" for computing MAP and MAUC
            # during validation
            if lk == "logits":
                if not self.training:
                    logging_output["logits_a3"] = logits_a3.cpu().numpy()
            elif lk == "target":
                if not self.training:
                    # If the targets have been mixed with the predictions of
                    # teacher models, find the original targets
                    if hasattr(encoder, "get_original_targets"):
                        original_target_a3 = encoder.get_original_targets(sample, logits_a3_v0)
                    else:
                        original_target_a3 = target_a3
                    logging_output["target_a3"] = original_target_a3.cpu().numpy()
            elif lk in logits_a3_v0:
                value = logits_a3_v0[lk]
                if not is_xla_tensor(value):
                    value = float(value)
                logging_output[f"{lk}_a3"] = value
        
        if self.infonce:
            with torch.no_grad():
                if logits_a3.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert logits_a3.dim() > 1, logits_a3.shape
                    max = logits_a3.argmax(-1) == 0
                    min = logits_a3.argmin(-1) == 0
                    both = max & min
                    corr = max.long().sum().item() - both.long().sum().item()
                    count = float(max.numel())
                logging_output["correct_a3"] = corr
                logging_output["count_a3"] = count
        


        # v0 -> a3
        logits_v0 = encoder.get_logits(logits_v0_a3).float()
        target_v0 = encoder.get_targets(sample, logits_v0_a3)
        losses_v0 = []
        # XXX: handle weights on xla.
        weights_v0 = None
        if hasattr(encoder, "get_target_weights") and not self.infonce:
            weights_v0 = encoder.get_target_weights(target_v0, logits_v0_a3)
            if torch.is_tensor(weights_v0):
                weights_v0 = weights_v0.float()
        
        reduction = "none" if not reduce else "sum"
        if self.infonce:
            loss_v0 = F.cross_entropy(logits_v0, target_v0, reduction=reduction)
        else:
            loss_v0 = F.binary_cross_entropy_with_logits(
                logits_v0, target_v0.float(), weights_v0, reduction=reduction
            )

        if 'sample_size' in sample:
            sample_size_v0 = sample['sample_size']
        elif 'mask_indices' in sample['net_input']:
            sample_size_v0 = sample['net_input']['mask_indices'].sum()
        else:
            sample_size_v0 = target_v0.numel() if self.infonce else target_v0.long().sum().item()
        losses_v0.append(loss_v0.detach().clone())

        if self.weights_cl_v0 is not None:
            assert hasattr(encoder, "get_extra_losses")
            extra_losses_v0 = encoder.get_extra_losses(logits_v0_a3)
            if torch.is_tensor(extra_losses_v0):
                extra_losses_v0 = [extra_losses_v0]
            if len(self.weights_cl_v0) == 1 and len(extra_losses_v0) != 1:
                self.weights_cl_v0 = [self.weights_cl_v0[0]] * len(extra_losses_v0)
            assert len(extra_losses_v0) == len(
                self.weights_cl_v0
            ), f"{len(extra_losses_v0)}, {len(self.weights_cl_v0)}"
            for p, coef in zip(extra_losses_v0, self.weights_cl_v0):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size_v0
                    loss_v0 += p
                    losses_v0.append(p)
        
        logging_output["loss_v0"] = loss_v0.item() if reduce else loss_v0.detach()
        if len(losses_v0) > 1:
            for i, l in enumerate(losses_v0):
                logging_output[f"loss_v0_{i}"] = l.item()
        
        for lk in self.log_keys:
            # Only store "logits" and "target" for computing MAP and MAUC
            # during validation
            if lk == "logits":
                if not self.training:
                    logging_output["logits_v0"] = logits_v0.cpu().numpy()
            elif lk == "target":
                if not self.training:
                    # If the targets have been mixed with the predictions of
                    # teacher models, find the original targets
                    if hasattr(encoder, "get_original_targets"):
                        original_target_v0 = encoder.get_original_targets(sample, logits_v0_a3)
                    else:
                        original_target_v0 = target_v0
                    logging_output["target_v0"] = original_target_v0.cpu().numpy()
            elif lk in logits_v0_a3:
                value = logits_v0_a3[lk]
                if not is_xla_tensor(value):
                    value = float(value)
                logging_output[f"{lk}_v0"] = value
        
        if self.infonce:
            with torch.no_grad():
                if logits_v0.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert logits_v0.dim() > 1, logits_v0.shape
                    max = logits_v0.argmax(-1) == 0
                    min = logits_v0.argmin(-1) == 0
                    both = max & min
                    corr = max.long().sum().item() - both.long().sum().item()
                    count = float(max.numel())
                logging_output["correct_v0"] = corr
                logging_output["count_v0"] = count
        

        # v3 -> a0
        logits_v3 = encoder.get_logits(logits_v3_a0).float()
        target_v3 = encoder.get_targets(sample, logits_v3_a0)
        losses_v3 = []
        # XXX: handle weights on xla.
        weights_v3 = None
        if hasattr(encoder, "get_target_weights") and not self.infonce:
            weights_v3 = encoder.get_target_weights(target_v3, logits_v3_a0)
            if torch.is_tensor(weights_v3):
                weights_v3 = weights_v3.float()
        
        reduction = "none" if not reduce else "sum"
        if self.infonce:
            loss_v3 = F.cross_entropy(logits_v3, target_v3, reduction=reduction)
        else:
            loss_v3 = F.binary_cross_entropy_with_logits(
                logits_v3, target_v3.float(), weights_v3, reduction=reduction
            )

        if 'sample_size' in sample:
            sample_size_v3 = sample['sample_size']
        elif 'mask_indices' in sample['net_input']:
            sample_size_v3 = sample['net_input']['mask_indices'].sum()
        else:
            sample_size_v3 = target_v3.numel() if self.infonce else target_v3.long().sum().item()
        losses_v3.append(loss_v3.detach().clone())

        if self.weights_cl_v3 is not None:
            assert hasattr(encoder, "get_extra_losses")
            extra_losses_v3 = encoder.get_extra_losses(logits_v3_a0)
            if torch.is_tensor(extra_losses_v3):
                extra_losses_v3 = [extra_losses_v3]
            if len(self.weights_cl_v3) == 1 and len(extra_losses_v3) != 1:
                self.weights_cl_v3 = [self.weights_cl_v3[0]] * len(extra_losses_v3)
            assert len(extra_losses_v3) == len(
                self.weights_cl_v3
            ), f"{len(extra_losses_v3)}, {len(self.weights_cl_v3)}"
            for p, coef in zip(extra_losses_v3, self.weights_cl_v3):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size_v3
                    loss_v3 += p
                    losses_v3.append(p)
        
        logging_output["loss_v3"] = loss_v3.item() if reduce else loss_v3.detach()
        if len(losses_v3) > 1:
            for i, l in enumerate(losses_v3):
                logging_output[f"loss_v3_{i}"] = l.item()
        
        for lk in self.log_keys:
            # Only store "logits" and "target" for computing MAP and MAUC
            # during validation
            if lk == "logits":
                if not self.training:
                    logging_output["logits_v3"] = logits_v3.cpu().numpy()
            elif lk == "target":
                if not self.training:
                    # If the targets have been mixed with the predictions of
                    # teacher models, find the original targets
                    if hasattr(encoder, "get_original_targets"):
                        original_target_v3 = encoder.get_original_targets(sample, logits_v3_a0)
                    else:
                        original_target_v3 = target_v3
                    logging_output["target_v3"] = original_target_v3.cpu().numpy()
            elif lk in logits_v3_a0:
                value = logits_v3_a0[lk]
                if not is_xla_tensor(value):
                    value = float(value)
                logging_output[f"{lk}_v3"] = value
        
        if self.infonce:
            with torch.no_grad():
                if logits_v3.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert logits_v3.dim() > 1, logits_v3.shape
                    max = logits_v3.argmax(-1) == 0
                    min = logits_v3.argmin(-1) == 0
                    both = max & min
                    corr = max.long().sum().item() - both.long().sum().item()
                    count = float(max.numel())
                logging_output["correct_v3"] = corr
                logging_output["count_v3"] = count
        

        # a0 -> v3
        logits_a0 = encoder.get_logits(logits_a0_v3).float()
        target_a0 = encoder.get_targets(sample, logits_a0_v3)
        losses_a0 = []
        # XXX: handle weights on xla.
        weights_a0 = None
        if hasattr(encoder, "get_target_weights") and not self.infonce:
            weights_a0 = encoder.get_target_weights(target_a0, logits_a0_v3)
            if torch.is_tensor(weights_a0):
                weights_a0 = weights_a0.float()
        
        reduction = "none" if not reduce else "sum"
        if self.infonce:
            loss_a0 = F.cross_entropy(logits_a0, target_a0, reduction=reduction)
        else:
            loss_a0 = F.binary_cross_entropy_with_logits(
                logits_a0, target_a0.float(), weights_a0, reduction=reduction
            )

        if 'sample_size' in sample:
            sample_size_a0 = sample['sample_size']
        elif 'mask_indices' in sample['net_input']:
            sample_size_a0 = sample['net_input']['mask_indices'].sum()
        else:
            sample_size_a0 = target_a0.numel() if self.infonce else target_a0.long().sum().item()
        losses_a0.append(loss_a0.detach().clone())

        if self.weights_cl_a0 is not None:
            assert hasattr(encoder, "get_extra_losses")
            extra_losses_a0 = encoder.get_extra_losses(logits_a0_v3)
            if torch.is_tensor(extra_losses_a0):
                extra_losses_a0 = [extra_losses_a0]
            if len(self.weights_cl_a0) == 1 and len(extra_losses_a0) != 1:
                self.weights_cl_a0 = [self.weights_cl_a0[0]] * len(extra_losses_a0)
            assert len(extra_losses_a0) == len(
                self.weights_cl_a0
            ), f"{len(extra_losses_a0)}, {len(self.weights_cl_a0)}"
            for p, coef in zip(extra_losses_a0, self.weights_cl_a0):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size_a0
                    loss_a0 += p
                    losses_a0.append(p)
        
        logging_output["loss_a0"] = loss_a0.item() if reduce else loss_a0.detach()
        if len(losses_a0) > 1:
            for i, l in enumerate(losses_a0):
                logging_output[f"loss_a0_{i}"] = l.item()
        
        for lk in self.log_keys:
            # Only store "logits" and "target" for computing MAP and MAUC
            # during validation
            if lk == "logits":
                if not self.training:
                    logging_output["logits_a0"] = logits_a0.cpu().numpy()
            elif lk == "target":
                if not self.training:
                    # If the targets have been mixed with the predictions of
                    # teacher models, find the original targets
                    if hasattr(encoder, "get_original_targets"):
                        original_target_a0 = encoder.get_original_targets(sample, logits_a0_v3)
                    else:
                        original_target_a0 = target_a0
                    logging_output["target_a0"] = original_target_a0.cpu().numpy()
            elif lk in logits_a0_v3:
                value = logits_a0_v3[lk]
                if not is_xla_tensor(value):
                    value = float(value)
                logging_output[f"{lk}_a0"] = value
        
        if self.infonce:
            with torch.no_grad():
                if logits_a0.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert logits_a0.dim() > 1, logits_a0.shape
                    max = logits_a0.argmax(-1) == 0
                    min = logits_a0.argmin(-1) == 0
                    both = max & min
                    corr = max.long().sum().item() - both.long().sum().item()
                    count = float(max.numel())
                logging_output["correct_a0"] = corr
                logging_output["count_a0"] = count
        

        loss_a3_v0 = (loss_a3 + loss_v0) / 2
        loss_v3_a0 = (loss_v3 + loss_a0) / 2


        ### Final loss
        loss = loss_asr + self.weight_lw * loss_lw + self.weights_cl[0] * loss_a3_v0 + self.weights_cl[1] * loss_v3_a0
        logging_output["loss"] = loss.data

        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)

        loss_asr_sum = sum(log.get("loss_asr", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        loss_lw_sum = sum(log.get("loss_lw", 0) for log in logging_outputs)


        loss_a3_sum = sum(log.get("loss_a3", 0) for log in logging_outputs)
        loss_a3_0_sum = sum(log.get("loss_a3_0", 0) for log in logging_outputs)
        correct_a3_sum = sum(log.get("correct_a3", 0) for log in logging_outputs)
        count_a3_sum = sum(log.get("count_a3", 0) for log in logging_outputs)

        loss_v0_sum = sum(log.get("loss_v0", 0) for log in logging_outputs)
        loss_v0_0_sum = sum(log.get("loss_v0_0", 0) for log in logging_outputs)
        correct_v0_sum = sum(log.get("correct_v0", 0) for log in logging_outputs)
        count_v0_sum = sum(log.get("count_v0", 0) for log in logging_outputs)


        loss_v3_sum = sum(log.get("loss_v3", 0) for log in logging_outputs)
        loss_v3_0_sum = sum(log.get("loss_v3_0", 0) for log in logging_outputs)
        correct_v3_sum = sum(log.get("correct_v3", 0) for log in logging_outputs)
        count_v3_sum = sum(log.get("count_v3", 0) for log in logging_outputs)

        loss_a0_sum = sum(log.get("loss_a0", 0) for log in logging_outputs)
        loss_a0_0_sum = sum(log.get("loss_a0_0", 0) for log in logging_outputs)
        correct_a0_sum = sum(log.get("correct_a0", 0) for log in logging_outputs)
        count_a0_sum = sum(log.get("count_a0", 0) for log in logging_outputs)
        

        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        metrics.log_scalar(
            "loss_asr", loss_asr_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "loss_lw", loss_lw_sum / sample_size / math.log(2), sample_size, round=3
        )


        metrics.log_scalar(
            "loss_a3", loss_a3_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_a3_0", loss_a3_0_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "correct_a3", correct_a3_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "count_a3", count_a3_sum / sample_size / math.log(2), sample_size, round=3
        )
        

        metrics.log_scalar(
            "loss_v0", loss_v0_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_v0_0", loss_v0_0_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "correct_v0", correct_v0_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "count_v0", count_v0_sum / sample_size / math.log(2), sample_size, round=3
        )


        metrics.log_scalar(
            "loss_v3", loss_v3_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_v3_0", loss_v3_0_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "correct_v3", correct_v3_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "count_v3", count_v3_sum / sample_size / math.log(2), sample_size, round=3
        )
        

        metrics.log_scalar(
            "loss_a0", loss_a0_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_a0_0", loss_a0_0_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "correct_a0", correct_a0_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "count_a0", count_a0_sum / sample_size / math.log(2), sample_size, round=3
        )


        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
