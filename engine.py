import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
from utils import span_cxw_to_xx, calculate_stats, calculate_mAP

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
        pred_spans = outputs['pred_spans']   # [B, N, 2] (Start, End) - BAM 原生输出
        pred_quality = outputs['pred_quality']
        
        prob = F.softmax(pred_logits, -1)
        scores = prob[..., 0]
        quality_scores = pred_quality.sigmoid().squeeze(-1)
        combined_scores = scores * (quality_scores ** 1.2)
        #combined_scores = scores
        
        # [修正] pred_spans 已经是 start/end，不需要转换！
        # 如果模型输出在 [0,1]，直接用
        pred_spans_xx = pred_spans.clamp(min=0.0, max=1.0)
        
        targets = batch['targets']
        for i, target in enumerate(targets):
            duration = target['duration']
            
            # [修正] GT 是 cxw，需要转为 xx
            gt_spans_tensor = target['spans'].to(device) if isinstance(target['spans'], torch.Tensor) else torch.tensor(target['spans'], device=device)
            gt_spans_xx = span_cxw_to_xx(gt_spans_tensor)
            
            results.append({
                "video_id": target['video_id'],
                "pred_spans": pred_spans_xx[i].cpu().numpy() * duration,
                "pred_scores": combined_scores[i].cpu().numpy(),
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