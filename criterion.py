import torch
import torch.nn.functional as F
from torch import nn
import logging
from utils import span_cxw_to_xx, generalized_temporal_iou
from isp_loss import EnhancedISPLoss

logger = logging.getLogger(__name__)

def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "none"):
    """
    Focal Loss implementation for binary classification
    Args:
        inputs: Logits (before sigmoid)
        targets: Binary targets (0 or 1)
        alpha: Weight for positive class (0 < alpha < 1)
        gamma: Focusing parameter
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss

class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, losses, eos_coef, span_loss_type="l1", temperature=0.07):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.span_loss_type = span_loss_type
        self.eos_coef = eos_coef
        self.temperature = temperature
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        
        # ========== [æ–°å¢ž] åˆå§‹åŒ– ISP Loss ==========
        self.isp_loss_module = EnhancedISPLoss(
            temperature=temperature,
            local_weight=0.3
        )

    def _check_and_clamp(self, idx, target_shape, name="Unknown"):
        b_idx, s_idx = idx
        B, L = target_shape[:2]
        if b_idx.numel() > 0 and b_idx.max().item() >= B:
             raise ValueError(f"Batch index error in {name}")
        if s_idx.numel() > 0:
            s_idx = torch.clamp(s_idx, max=L-1, min=0)
        return b_idx, s_idx

    def loss_labels(self, outputs, targets, indices, num_spans, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        idx = self._check_and_clamp(idx, src_logits.shape, name="loss_labels")
        b_idx, s_idx = idx
        
        target_classes_o = torch.cat([torch.zeros_like(t["labels"][J]) for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 1, dtype=torch.int64, device=src_logits.device)
        target_classes[b_idx, s_idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_labels': loss_ce}

    def loss_spans(self, outputs, targets, indices, num_spans):
        assert 'pred_spans' in outputs
        idx = self._get_src_permutation_idx(indices)
        idx = self._check_and_clamp(idx, outputs['pred_spans'].shape, name="loss_spans")
        b_idx, s_idx = idx
        src_spans = outputs['pred_spans'][b_idx, s_idx]
        
        target_spans_list = []
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            cur_spans = targets[batch_idx]['spans'] 
            if len(cur_spans) == 0: continue
            tgt_idx = torch.clamp(tgt_idx, max=len(cur_spans) - 1)
            target_spans_list.append(cur_spans[tgt_idx])
            
        if len(target_spans_list) == 0:
            return {'loss_span': src_spans.sum() * 0, 'loss_giou': src_spans.sum() * 0}

        target_spans_cw = torch.cat(target_spans_list, dim=0).clamp(min=0.0, max=1.0)
        target_spans_xx = span_cxw_to_xx(target_spans_cw)
        
        loss_span = F.l1_loss(src_spans, target_spans_xx, reduction='none')
        loss_giou = 1 - torch.diag(generalized_temporal_iou(src_spans, target_spans_xx))
        return {'loss_span': loss_span.sum() / num_spans, 'loss_giou': loss_giou.sum() / num_spans}

    def loss_saliency(self, outputs, targets, indices, num_spans):
        if 'saliency_scores' not in outputs: return {}
        saliency_scores = outputs['saliency_scores']
        video_mask = outputs['video_mask']
        num_valid_frames = video_mask.sum().clamp(min=1.0)
        
        gt_saliency = torch.zeros_like(saliency_scores)
        L = saliency_scores.shape[1]
        grid = torch.arange(L, device=saliency_scores.device).float().unsqueeze(0)
        
        # 生成高斯分布的 GT
        for i, t in enumerate(targets):
            spans = t['spans']
            if len(spans) == 0: continue
            centers = spans[:, 0] * L
            widths = spans[:, 1] * L
            # 注意：这里稍微调小了 sigma 的分母，让 GT 更尖锐一点，更容易学习
            for c, w in zip(centers, widths):
                sigma = (w / 3.0).clamp(min=1.0) 
                gaussian = torch.exp(- (grid - c)**2 / (2 * sigma**2))
                gt_saliency[i] = torch.max(gt_saliency[i], gaussian.squeeze(0))
        
        # 制作成 0/1 标签用于 Focal Loss (或者保留软标签)
        # Focal Loss 通常配合 0/1 标签，但软标签在 DETR 类方法中也常用
        # 这里我们保留高斯软标签，但截断一下以减少噪声
        gt_saliency = gt_saliency.clamp(0, 1)

        if video_mask is not None:
            saliency_scores = saliency_scores * video_mask.float()
            gt_saliency = gt_saliency * video_mask.float()
            
            # 如果是 Padding 区域，强制 Mask 掉 Loss
            # (Focal Loss function 没有 mask 参数，所以我们手动置零)
            
        # 使用 Focal Loss
        # alpha=0.25 (降低背景权重), gamma=2.0 (关注难样本)
        loss = sigmoid_focal_loss(
            saliency_scores, 
            gt_saliency, 
            alpha=0.25, 
            gamma=2.0, 
            reduction='none'
        )
        
        if video_mask is not None:
            loss = loss * video_mask.float()
            
        return {'loss_saliency': loss.sum() / num_valid_frames}

    def loss_quality(self, outputs, targets, indices, num_spans):
        assert 'pred_quality' in outputs
        src_quality = outputs['pred_quality']
        idx = self._get_src_permutation_idx(indices)
        
        # [ä¿®æ”¹å‰] é”™è¯¯ä»£ç ï¼šä¼ é€’äº†æ•´æ•° shape[1]
        # b_idx, s_idx = self._check_and_clamp(idx, src_quality.shape[1]) 
        
        # [ä¿®æ”¹åŽ] æ­£ç¡®ä»£ç ï¼šä¼ é€’å®Œæ•´çš„ shape å…ƒç»„ï¼Œå¹¶åŠ ä¸Š name å‚æ•°æ–¹ä¾¿è°ƒè¯•
        b_idx, s_idx = self._check_and_clamp(idx, src_quality.shape, name="loss_quality")
        
        pred_quality = src_quality[b_idx, s_idx].squeeze(-1)
        
        src_spans = outputs['pred_spans'][b_idx, s_idx]
        target_spans_cw = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0).clamp(max=1.0)
        target_spans_xx = span_cxw_to_xx(target_spans_cw)
        
        inter_min = torch.max(src_spans[:, 0], target_spans_xx[:, 0])
        inter_max = torch.min(src_spans[:, 1], target_spans_xx[:, 1])
        inter_len = (inter_max - inter_min).clamp(min=0)
        union_len = (src_spans[:, 1] - src_spans[:, 0]) + (target_spans_xx[:, 1] - target_spans_xx[:, 0]) - inter_len
        gt_iou = inter_len / (union_len + 1e-6)
        
        loss_tensor = F.l1_loss(pred_quality.sigmoid(), gt_iou.detach(), reduction='none')
        return {'loss_quality': loss_tensor.mean()}

    def loss_recfw(self, outputs, targets, indices, num_spans):
        if 'recfw_words_logit' not in outputs or outputs['recfw_words_logit'] is None: return {}
        logits = outputs['recfw_words_logit']
        mask_indices = outputs['masked_indices']
        if mask_indices is None or mask_indices.sum() == 0: return {'loss_recfw': torch.tensor(0.0, device=logits.device)}
        
        vocab_limit = logits.shape[-1] - 1
        gt_words = torch.stack([torch.clamp(t['words_id'], max=vocab_limit) for t in targets])
        loss = F.cross_entropy(logits[mask_indices], gt_words[mask_indices])
        return {'loss_recfw': loss}

    def loss_contrastive(self, outputs, targets, indices, num_spans):
        if 'proj_txt_emb' not in outputs: return {}
        txt_emb = outputs['proj_txt_emb']
        vid_emb = outputs['proj_vid_emb']
        bs = txt_emb.shape[0]
        # ç®€å•å–ç¬¬ä¸€ä¸ªåŒ¹é…
        pos_vid_feats = []
        valid_mask = []
        for i, (src_idx, _) in enumerate(indices):
            if len(src_idx) > 0:
                pos_vid_feats.append(vid_emb[i, min(src_idx[0], vid_emb.shape[1]-1)])
                valid_mask.append(True)
            else:
                pos_vid_feats.append(torch.zeros_like(txt_emb[i]))
                valid_mask.append(False)
        
        if not any(valid_mask): return {'loss_contrastive': torch.tensor(0.0, device=txt_emb.device)}
        
        pos_vid_feats = torch.stack(pos_vid_feats)
        valid_mask = torch.tensor(valid_mask, device=txt_emb.device)
        logits = torch.matmul(txt_emb, pos_vid_feats.t()) / self.temperature
        return {'loss_contrastive': F.cross_entropy(logits[valid_mask], torch.arange(bs, device=txt_emb.device)[valid_mask])}

    # ========== [å…³é”®ä¿®å¤] ISP Loss å®žçŽ° ==========
    def loss_isp(self, outputs, targets, indices, num_spans):
        if 'isp_video_feat' not in outputs or 'isp_text_feat' not in outputs:
            return {}
        
        video_feat = outputs['isp_video_feat'] # [B, L, D]
        text_feat = outputs['isp_text_feat']   # [B, L, D]
        video_mask = outputs.get('video_mask', None)
        # [Fix] ç›´æŽ¥ä»Ž Model è¾“å‡ºèŽ·å–çœŸå®žçš„ words_mask
        text_mask = outputs.get('words_mask', None)

        # å³ä½¿ Mask ä¸º Noneï¼ŒISP Loss å†…éƒ¨ä¹Ÿä¼šå¤„ç†ï¼ˆå‡è®¾å…¨ Trueï¼‰
        # ä½†æœ‰äº† words_maskï¼Œæ•ˆæžœä¼šæ›´å¥½
        isp_loss, _ = self.isp_loss_module(
            video_feat, text_feat, video_mask, text_mask
        )
        return {'loss_isp': isp_loss}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_spans, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'spans': self.loss_spans,
            'quality': self.loss_quality,
            'recfw': self.loss_recfw,
            'contrastive': self.loss_contrastive,
            'saliency': self.loss_saliency,
            'isp': self.loss_isp,  # æ³¨å†Œ ISP
        }
        if loss not in loss_map: return {}
        return loss_map[loss](outputs, targets, indices, num_spans, **kwargs)

    def forward(self, outputs, targets):
        clean_targets = []
        for t in targets:
            new_t = {k: v for k, v in t.items() if k not in ['labels', 'spans', 'words_id']}
            if 'labels' in t: new_t['labels'] = torch.zeros_like(t['labels'])
            if 'spans' in t: new_t['spans'] = t['spans'].clone().clamp(0, 1)
            if 'words_id' in t: new_t['words_id'] = t['words_id']
            clean_targets.append(new_t)
        targets = clean_targets

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_spans = max(sum(len(t["labels"]) for t in targets), 1)

        losses = {}
        # [Fix] ç¡®ä¿ isp åœ¨ loss åˆ—è¡¨ä¸­
        current_losses = self.losses + ['quality', 'recfw', 'contrastive', 'saliency', 'isp']
        for loss in current_losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_spans))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss in ['recfw', 'contrastive', 'saliency', 'isp']: continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_spans)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses