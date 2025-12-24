import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import copy
import math

# ===============================================================
# 1. 基础网络组件
# ===============================================================

def _get_clones(module, N):
    """克隆 N 个相同的 Layer"""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """[修复] 补回缺失的激活函数获取方法"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class LinearLayer(nn.Module):
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super().__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        )

    def forward(self, x):
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

def inverse_sigmoid(x, eps=1e-3):
    """Sigmoid 的反函数，用于 DETR 类模型初始化 Anchor"""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

# ===============================================================
# 2. Span / Box 工具 (增强数值稳定性，防止 NaN)
# ===============================================================

def span_cxw_to_xx(cxw_spans):
    """
    (Center, Width) -> (Start, End)
    增加 clamp 确保数值在 [0, 1] 之间，且防止 width 为 0
    """
    center = cxw_spans[..., 0]
    width = cxw_spans[..., 1].clamp(min=1e-6) # [保护] 防止宽度为0
    
    x1 = center - 0.5 * width
    x2 = center + 0.5 * width
    return torch.stack([x1, x2], dim=-1).clamp(min=0.0, max=1.0)

def temporal_iou(spans1, spans2):
    """
    计算 IOU，增加数值保护
    spans1, spans2: [..., 2] (start, end)
    """
    # 确保数值合法
    spans1 = spans1.clamp(min=0, max=1)
    spans2 = spans2.clamp(min=0, max=1)
    
    areas1 = (spans1[:, 1] - spans1[:, 0]).clamp(min=1e-6)
    areas2 = (spans2[:, 1] - spans2[:, 0]).clamp(min=1e-6)

    left = torch.max(spans1[:, None, 0], spans2[:, 0])
    right = torch.min(spans1[:, None, 1], spans2[:, 1])

    inter = (right - left).clamp(min=0)
    union = areas1[:, None] + areas2 - inter

    # [保护] 分母加 epsilon
    iou = inter / (union + 1e-6)
    return iou, union

def generalized_temporal_iou(spans1, spans2):
    """
    计算 GIoU，防止 loss=nan
    """
    iou, union = temporal_iou(spans1, spans2)

    left = torch.min(spans1[:, None, 0], spans2[:, 0])
    right = torch.max(spans1[:, None, 1], spans2[:, 1])
    
    enclosing_area = (right - left).clamp(min=1e-6) # [保护] 防止除以0

    giou = iou - (enclosing_area - union) / enclosing_area
    return giou

# ===============================================================
# 3. 评估指标计算工具
# ===============================================================

def compute_temporal_iou(pred_spans, gt_spans):
    """Engine 用于评估的 IoU 计算"""
    if not isinstance(pred_spans, torch.Tensor): pred_spans = torch.tensor(pred_spans)
    if not isinstance(gt_spans, torch.Tensor): gt_spans = torch.tensor(gt_spans)
    
    if pred_spans.ndim == 1: pred_spans = pred_spans.unsqueeze(0)
    if gt_spans.ndim == 1: gt_spans = gt_spans.unsqueeze(0)

    tmin = torch.max(pred_spans[:, 0].unsqueeze(1), gt_spans[:, 0].unsqueeze(0))
    tmax = torch.min(pred_spans[:, 1].unsqueeze(1), gt_spans[:, 1].unsqueeze(0))
    
    intersection = (tmax - tmin).clamp(min=0)
    
    pred_len = (pred_spans[:, 1] - pred_spans[:, 0]).clamp(min=1e-8)
    gt_len = (gt_spans[:, 1] - gt_spans[:, 0]).clamp(min=1e-8)
    
    union = pred_len.unsqueeze(1) + gt_len.unsqueeze(0) - intersection
    return intersection / (union + 1e-8)

def calculate_stats(results, iou_thresholds):
    """同时计算 Recall@1, Recall@5, mIoU"""
    r1_results = {th: 0.0 for th in iou_thresholds}
    r5_results = {th: 0.0 for th in iou_thresholds}
    miou_sum = 0.0
    n_samples = len(results)
    
    if n_samples == 0:
        return {}

    for item in results:
        pred_scores = item['pred_scores']
        pred_spans = item['pred_spans']
        gt_spans = item['gt_spans']
        
        # 转换为 torch tensor
        if not isinstance(pred_scores, torch.Tensor): pred_scores = torch.tensor(pred_scores)
        if not isinstance(pred_spans, torch.Tensor): pred_spans = torch.tensor(pred_spans)
        if not isinstance(gt_spans, torch.Tensor): gt_spans = torch.tensor(gt_spans) # 确保 GT 也是 tensor

        # 排序
        sorted_idx = torch.argsort(pred_scores, descending=True)
        pred_spans_sorted = pred_spans[sorted_idx]
        
        # Top-1
        top1_span = pred_spans_sorted[0].unsqueeze(0)
        iou_top1 = compute_temporal_iou(top1_span, gt_spans) 
        max_iou_r1 = iou_top1.max().item() if iou_top1.numel() > 0 else 0
        
        # Top-5
        k = min(5, len(pred_spans_sorted))
        top5_spans = pred_spans_sorted[:k]
        iou_top5 = compute_temporal_iou(top5_spans, gt_spans)
        max_iou_r5 = iou_top5.max().item() if iou_top5.numel() > 0 else 0
        
        miou_sum += max_iou_r1
        
        for th in iou_thresholds:
            if max_iou_r1 >= th:
                r1_results[th] += 1.0
            if max_iou_r5 >= th:
                r5_results[th] += 1.0
                
    metrics = {}
    for th in iou_thresholds:
        metrics[f"R1@{th}"] = (r1_results[th] / n_samples) * 100
        metrics[f"R5@{th}"] = (r5_results[th] / n_samples) * 100
        
    metrics["mIoU"] = (miou_sum / n_samples) * 100
    return metrics

def calculate_mAP(results, iou_thresholds):
    """计算 mAP"""
    ap_results = {th: [] for th in iou_thresholds}
    
    for item in results:
        pred_spans = item['pred_spans']
        pred_scores = item['pred_scores']
        gt_spans = item['gt_spans']
        
        # 转 numpy
        if isinstance(pred_spans, torch.Tensor): pred_spans = pred_spans.cpu().numpy()
        if isinstance(pred_scores, torch.Tensor): pred_scores = pred_scores.cpu().numpy()
        if isinstance(gt_spans, torch.Tensor): gt_spans = gt_spans.cpu().numpy()
        
        sorted_idx = np.argsort(pred_scores)[::-1]
        pred_spans_sorted = pred_spans[sorted_idx]
        
        num_gt = len(gt_spans)
        if num_gt == 0:
            for th in iou_thresholds: ap_results[th].append(0.0)
            continue

        iou_mat = compute_temporal_iou(pred_spans_sorted, gt_spans).numpy()
        
        for th in iou_thresholds:
            gt_matched = np.zeros(num_gt, dtype=bool)
            tp = np.zeros(len(pred_spans_sorted))
            fp = np.zeros(len(pred_spans_sorted))
            
            for i in range(len(pred_spans_sorted)):
                if iou_mat.shape[1] > 0:
                    max_iou_idx = np.argmax(iou_mat[i])
                    max_iou = iou_mat[i][max_iou_idx]
                else:
                    max_iou = 0
                
                if max_iou >= th:
                    if not gt_matched[max_iou_idx]:
                        tp[i] = 1.0
                        gt_matched[max_iou_idx] = True
                    else:
                        fp[i] = 1.0
                else:
                    fp[i] = 1.0
            
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
            ap = np.sum(precision * tp) / num_gt
            ap_results[th].append(ap)
            
    mAP_scores = {f"mAP@{th}": np.mean(vals) * 100 for th, vals in ap_results.items()}
    avg_map = np.mean([mAP_scores[f"mAP@{th}"] for th in iou_thresholds if th >= 0.5])
    mAP_scores["mAP@Avg"] = avg_map
    
    return mAP_scores