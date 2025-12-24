import torch
import torch.nn.functional as F
from torch import nn
import math
import logging
from utils import span_cxw_to_xx, generalized_temporal_iou

logger = logging.getLogger(__name__)

# 修改 criterion.py 开头的这个函数
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # [修改] 
    # 原来: return loss.mean(1).sum() / num_boxes
    # 改为: return loss.sum() / num_boxes
    return loss.sum() / num_boxes

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

    def _check_and_clamp(self, idx, target_shape, name="Unknown"):
        """
        [侦探函数] 
        在访问 GPU 张量前，先在 CPU/GPU 上检查索引是否合法。
        如果不合法，打印详细错误并抛出 ValueError (比 CUDA assert 好调试得多)。
        """
        b_idx, s_idx = idx
        B, L = target_shape[:2]
        
        # 检查 Batch Index
        if b_idx.numel() > 0:
            max_b = b_idx.max().item()
            if max_b >= B:
                err_msg = f"❌ CRITICAL ERROR in {name}: Batch index {max_b} >= Batch size {B}!"
                print(err_msg)
                raise ValueError(err_msg)
        
        # 检查 Query/Sequence Index
        if s_idx.numel() > 0:
            max_s = s_idx.max().item()
            if max_s >= L:
                err_msg = f"❌ CRITICAL ERROR in {name}: Query index {max_s} >= Max Length {L}!"
                print(err_msg)
                # 紧急修复：强制 Clamp，防止 CUDA 崩溃，让你能看到上面的报错
                s_idx = torch.clamp(s_idx, max=L-1)
                
            # 双重保险：防止负数索引 (虽然 matcher 不会返回负数)
            if s_idx.min().item() < 0:
                print(f"❌ CRITICAL ERROR in {name}: Found negative index {s_idx.min().item()}!")
                s_idx = torch.clamp(s_idx, min=0)

        return b_idx, s_idx

    def loss_labels(self, outputs, targets, indices, num_spans, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # [B, Nq, C]
        idx = self._get_src_permutation_idx(indices)
        
        # [侦探检查]
        idx = self._check_and_clamp(idx, src_logits.shape, name="loss_labels")
        b_idx, s_idx = idx

        target_classes_o = torch.cat([torch.zeros_like(t["labels"][J]) for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 1, dtype=torch.int64, device=src_logits.device)
        
        # 安全赋值
        target_classes[b_idx, s_idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_labels': loss_ce}

    def loss_spans(self, outputs, targets, indices, num_spans):
        assert 'pred_spans' in outputs
        idx = self._get_src_permutation_idx(indices)
        
        # [侦探检查] 也就是你之前报错的地方
        idx = self._check_and_clamp(idx, outputs['pred_spans'].shape, name="loss_spans")
        b_idx, s_idx = idx
        
        src_spans = outputs['pred_spans'][b_idx, s_idx]
        
        target_spans_list = []
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            cur_spans = targets[batch_idx]['spans'] 
            num_gt = len(cur_spans)
            if num_gt == 0: continue

            if tgt_idx.max() >= num_gt:
                # print(f"⚠️ Warning: Fix GT index {tgt_idx.max()} >= {num_gt}")
                tgt_idx = torch.clamp(tgt_idx, max=num_gt - 1)
            
            selected_spans = cur_spans[tgt_idx]
            target_spans_list.append(selected_spans)
            
        if len(target_spans_list) == 0:
            return {'loss_span': src_spans.sum() * 0, 'loss_giou': src_spans.sum() * 0}

        target_spans_cw = torch.cat(target_spans_list, dim=0)
        target_spans_cw = torch.clamp(target_spans_cw, min=0.0, max=1.0)
        target_spans_xx = span_cxw_to_xx(target_spans_cw)
        
        loss_span = F.l1_loss(src_spans, target_spans_xx, reduction='none')
        loss_giou = 1 - torch.diag(generalized_temporal_iou(src_spans, target_spans_xx))
        
        return {
            'loss_span': loss_span.sum() / num_spans,
            'loss_giou': loss_giou.sum() / num_spans
        }

    def loss_saliency(self, outputs, targets, indices, num_spans):
        if 'saliency_scores' not in outputs: return {}
        saliency_scores = outputs['saliency_scores']
        
        gt_saliency = torch.zeros_like(saliency_scores)
        L = saliency_scores.shape[1]
        
        for i, t in enumerate(targets):
            spans = t['spans']
            if len(spans) == 0: continue
            
            # 这里的 spans 已经被 forward 清洗过，应该是 <= 1.0 的
            # 转换为特征图坐标
            starts = (spans[:, 0] - spans[:, 1] / 2) * L
            ends = (spans[:, 0] + spans[:, 1] / 2) * L
            
            for s, e in zip(starts, ends):
                # [侦探检查] Saliency 索引
                s_idx = int(math.floor(s.item()))
                e_idx = int(math.ceil(e.item()))
                
                # 显式截断
                s_idx = max(0, min(L, s_idx))
                e_idx = max(0, min(L, e_idx))
                
                if e_idx > s_idx:
                    gt_saliency[i, s_idx:e_idx] = 1.0
        
        loss = sigmoid_focal_loss(saliency_scores, gt_saliency, num_spans)
        return {'loss_saliency': loss}

    def loss_quality(self, outputs, targets, indices, num_spans):
        assert 'pred_quality' in outputs
        src_quality = outputs['pred_quality']
        idx = self._get_src_permutation_idx(indices)
        
        # [侦探检查]
        idx = self._check_and_clamp(idx, src_quality.shape, name="loss_quality")
        b_idx, s_idx = idx

        matched_quality = src_quality[b_idx, s_idx].squeeze(-1)
        src_spans = outputs['pred_spans'][b_idx, s_idx]
        
        target_spans_cw = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_spans_cw = torch.clamp(target_spans_cw, max=1.0)
        target_spans_xx = span_cxw_to_xx(target_spans_cw) 
        
        inter_min = torch.max(src_spans[:, 0], target_spans_xx[:, 0])
        inter_max = torch.min(src_spans[:, 1], target_spans_xx[:, 1])
        inter_len = (inter_max - inter_min).clamp(min=0)
        union_len = (src_spans[:, 1] - src_spans[:, 0]) + \
                    (target_spans_xx[:, 1] - target_spans_xx[:, 0]) - inter_len
        gt_iou = inter_len / (union_len + 1e-6)

        loss_quality = F.l1_loss(matched_quality.sigmoid(), gt_iou.detach(), reduction='sum')
        return {'loss_quality': loss_quality / num_spans}

    def loss_recfw(self, outputs, targets, indices, num_spans):
        if 'recfw_words_logit' not in outputs or outputs['recfw_words_logit'] is None:
            return {}
        logits = outputs['recfw_words_logit']
        mask_indices = outputs['masked_indices']
        if mask_indices is None or mask_indices.sum() == 0:
            return {'loss_recfw': torch.tensor(0.0, device=logits.device)}
        
        vocab_limit = logits.shape[-1] - 1
        gt_words = torch.stack([torch.clamp(t['words_id'], max=vocab_limit) for t in targets])
        
        masked_logits = logits[mask_indices]
        masked_labels = gt_words[mask_indices]
        loss = F.cross_entropy(masked_logits, masked_labels)
        return {'loss_recfw': loss}

    def loss_contrastive(self, outputs, targets, indices, num_spans):
        if 'proj_txt_emb' not in outputs or 'proj_vid_emb' not in outputs: return {}
        txt_emb = outputs['proj_txt_emb']
        vid_emb = outputs['proj_vid_emb']
        bs = txt_emb.shape[0]
        pos_vid_feats = []
        valid_mask = [] 
        for i, (src_idx, _) in enumerate(indices):
            if len(src_idx) > 0:
                idx = src_idx[0]
                idx = min(idx, vid_emb.shape[1] - 1)
                pos_vid_feats.append(vid_emb[i, idx])
                valid_mask.append(True)
            else:
                pos_vid_feats.append(torch.zeros_like(txt_emb[i]))
                valid_mask.append(False)
        if not any(valid_mask):
            return {'loss_contrastive': torch.tensor(0.0, device=txt_emb.device)}
        pos_vid_feats = torch.stack(pos_vid_feats)
        valid_mask = torch.tensor(valid_mask, device=txt_emb.device)
        logits = torch.matmul(txt_emb, pos_vid_feats.t()) / self.temperature
        labels = torch.arange(bs, device=txt_emb.device)
        loss = F.cross_entropy(logits[valid_mask], labels[valid_mask])
        return {'loss_contrastive': loss}

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
        }
        if loss not in loss_map: return {}
        return loss_map[loss](outputs, targets, indices, num_spans, **kwargs)

    def forward(self, outputs, targets):
        # [终极防火墙] Targets 数据清洗
        clean_targets = []
        for i, t in enumerate(targets):
            new_t = {}
            for k, v in t.items():
                if k not in ['labels', 'spans', 'words_id']:
                    new_t[k] = v
            
            # 1. Labels 强制为 0
            if 'labels' in t:
                new_t['labels'] = torch.zeros_like(t['labels'])
            
            # 2. Spans 强制归一化 (0~1)
            if 'spans' in t:
                spans = t['spans'].clone()
                if spans.numel() > 0:
                    spans = torch.clamp(spans, min=0.0, max=1.0)
                new_t['spans'] = spans

            # 3. Words_id 强制限制范围
            if 'words_id' in t:
                 # 保守一点，设为 49000
                 new_t['words_id'] = torch.clamp(t['words_id'], min=0, max=49000) 
                 
            clean_targets.append(new_t)
            
        targets = clean_targets

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        
        num_spans = sum(len(t["labels"]) for t in targets)
        num_spans = torch.as_tensor([num_spans], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_spans = torch.clamp(num_spans, min=1).item()

        losses = {}
        # 确保 quality 和 recfw 总是被计算
        current_losses = self.losses + ['quality', 'recfw', 'contrastive', 'saliency']
        for loss in current_losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_spans))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss in ['recfw', 'contrastive', 'saliency']: continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_spans)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses