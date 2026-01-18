import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
from utils import span_cxw_to_xx, calculate_stats, calculate_mAP
# [新增] 引入 torchvision 的 NMS，这是提升 R1 的关键工具
from torchvision.ops import nms 

logger = logging.getLogger(__name__)

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    results = []
    
    logger.info("Evaluating...")
    for batch in tqdm(data_loader, desc="Inference"):
        video_feat = batch['video_feat'].to(device)
        video_mask = batch['video_mask'].to(device)
        words_id = batch['words_id'].to(device)
        words_mask = batch['words_mask'].to(device)
        
        outputs = model(video_feat, video_mask, words_id, words_mask, is_training=False)
        
        pred_logits = outputs['pred_logits']
        pred_spans = outputs['pred_spans']   # [B, N, 2] (Start, End)
        pred_quality = outputs['pred_quality']
        
        # 1. 计算融合分数 (Quality-Aware Ranking)
        prob = F.softmax(pred_logits, -1)
        scores = prob[..., 0] # 前景概率
        quality_scores = pred_quality.sigmoid().squeeze(-1) # 预测的 IoU
        
        
        #alpha = 0.7
        #combined_scores = (scores ** alpha) * (quality_scores ** (1-alpha))
        combined_scores = scores * quality_scores
        
        # 2. 坐标处理
        # pred_spans 已经是 start/end 格式，直接限制在 [0, 1] 范围内
        pred_spans_xx = pred_spans.clamp(min=0.0, max=1.0)
        
        targets = batch['targets']
        for i, target in enumerate(targets):
            duration = target['duration']
            
            # 获取当前样本的预测结果
            cur_spans = pred_spans_xx[i]    # [N, 2]
            cur_scores = combined_scores[i] # [N]
            
            # 3. [关键改进] 执行 NMS (Non-Maximum Suppression)
            # 目的：去除那些堆积在一起的重复框，保证 Top-1 是唯一的最佳框
            
            # 构造 Boxes [N, 4] -> (x1, y1, x2, y2)
            # 因为是 1D 时序检测，我们将 y 轴设为 0-1 的伪坐标，只在 x 轴做 NMS
            boxes = torch.zeros((cur_spans.shape[0], 4), device=device)
            boxes[:, 0] = cur_spans[:, 0] # x1 (start)
            boxes[:, 2] = cur_spans[:, 1] # x2 (end)
            boxes[:, 1] = 0.0             # y1
            boxes[:, 3] = 1.0             # y2
            
            # 执行 NMS, IoU 阈值
            keep_indices = nms(boxes, cur_scores, iou_threshold=0.45)
            
            # 根据 NMS 结果筛选预测
            final_spans = cur_spans[keep_indices]
            final_scores = cur_scores[keep_indices]
            
            # 转换 GT (Ground Truth)
            gt_spans_tensor = target['spans'].to(device) if isinstance(target['spans'], torch.Tensor) else torch.tensor(target['spans'], device=device)
            gt_spans_xx = span_cxw_to_xx(gt_spans_tensor)
            
            # 保存结果 (还原到真实时间尺度 * duration)
            results.append({
                "video_id": target['video_id'],
                "pred_spans": final_spans.cpu().numpy() * duration,
                "pred_scores": final_scores.cpu().numpy(),
                "gt_spans": gt_spans_xx.cpu().numpy() * duration
            })
            
    if not results:
        return {}

    recall_thds = [0.5, 0.7]
    map_thds_full = [round(x * 0.05, 2) for x in range(10, 20)]
    
    metrics = {}
    metrics.update(calculate_stats(results, recall_thds)) 
    metrics.update(calculate_mAP(results, map_thds_full)) 
    
    logger.info("------------------------------------------------")
    logger.info(f"R1@0.5: {metrics.get('R1@0.5', 0):.2f} | R1@0.7: {metrics.get('R1@0.7', 0):.2f}")
    logger.info(f"R5@0.5: {metrics.get('R5@0.5', 0):.2f} | R5@0.7: {metrics.get('R5@0.7', 0):.2f}")
    logger.info(f"mAP@0.5: {metrics.get('mAP@0.5', 0):.2f} | mAP@Avg: {metrics.get('mAP@Avg', 0):.2f}")
    logger.info(f"mIoU: {metrics.get('mIoU', 0):.2f}")
    logger.info("------------------------------------------------")
    
    return metrics