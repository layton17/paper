import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from utils import generalized_temporal_iou, span_cxw_to_xx

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_span=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # --- [Add this Block] Debugging Data Safety ---
        num_classes = outputs["pred_logits"].shape[-1]
        all_labels = torch.cat([v["labels"] for v in targets])
        if len(all_labels) > 0:
            max_label = all_labels.max().item()
            if max_label >= num_classes:
                raise ValueError(f"Data Error: Found label {max_label} but model only outputs {num_classes} classes (indices 0-{num_classes-1}).")

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_span = outputs["pred_spans"].flatten(0, 1)

        tgt_ids = torch.cat([torch.zeros_like(v["labels"]) for v in targets])
        tgt_span = torch.cat([v["spans"] for v in targets])

        if len(tgt_ids) == 0:
            return [(torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)) for _ in range(bs)]

        # 1. Classification Cost
        cost_class = -out_prob[:, tgt_ids]

        # 2. Span L1 Cost
        tgt_span_xx = span_cxw_to_xx(tgt_span)
        cost_span = torch.cdist(out_span, tgt_span_xx, p=1)

        # 3. GIoU Cost
        cost_giou = -generalized_temporal_iou(out_span, tgt_span_xx)

        # Total Cost
        C = self.cost_span * cost_span + self.cost_giou * cost_giou + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        # ===============================================================
        # [新增] 强制清洗 NaN/Inf，防止匈牙利算法崩溃或返回错误索引
        # ===============================================================
        if torch.isnan(C).any() or torch.isinf(C).any():
            # print("Warning: Matcher Cost Matrix contains NaN/Inf! Cleaning up...")
            # 将 NaN 替换为 0，正无穷替换为大数，负无穷替换为小数
            C = torch.nan_to_num(C, nan=100.0, posinf=1000.0, neginf=-1000.0)
        # ===============================================================

        sizes = [len(v["spans"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]