"""
Initial Semantic Projection (ISP) Loss
来自论文: Watching What You Want (W2W)

核心思想:
1. CLIP预训练的视频和文本特征在语义空间中并不完美对齐
2. 通过显式的对比学习Loss，强制文本和视频特征在初始阶段就对齐
3. 这种对齐是moment retrieval任务成功的基础
"""

import torch
from torch import nn
import torch.nn.functional as F

class ISPLoss(nn.Module):
    def __init__(self, temperature=0.07, use_hard_negatives=True):
        super().__init__()
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        
    def forward(self, video_feat, text_feat, video_mask=None, text_mask=None, temperature=None):
        # [动态温度] 优先使用传入的 temperature
        current_temp = temperature if temperature is not None else self.temperature
        
        # 格式修正逻辑
        if video_mask is not None and video_feat.dim() == 3:
            mask_B, mask_L = video_mask.shape
            if video_feat.shape[0] == mask_L and video_feat.shape[1] == mask_B:
                video_feat = video_feat.permute(1, 0, 2)
            elif video_feat.shape[0] == mask_B and video_feat.shape[1] == mask_L:
                pass
            else:
                if video_feat.shape[0] < video_feat.shape[1]:
                    video_feat = video_feat.permute(1, 0, 2)
        else:
            if video_feat.dim() == 3 and video_feat.shape[0] < video_feat.shape[1]:
                video_feat = video_feat.permute(1, 0, 2)
        
        if text_mask is not None and text_feat.dim() == 3:
            mask_B, mask_L = text_mask.shape
            if text_feat.shape[0] == mask_L and text_feat.shape[1] == mask_B:
                text_feat = text_feat.permute(1, 0, 2)
            elif text_feat.shape[0] == mask_B and text_feat.shape[1] == mask_L:
                pass
            else:
                if text_feat.shape[0] < text_feat.shape[1]:
                    text_feat = text_feat.permute(1, 0, 2)
        else:
            if text_feat.dim() == 3 and text_feat.shape[0] < text_feat.shape[1]:
                text_feat = text_feat.permute(1, 0, 2)
        
        B = video_feat.shape[0]
        
        # 全局池化
        if video_mask is not None:
            video_mask_expanded = video_mask.unsqueeze(-1).float()
            video_global = (video_feat * video_mask_expanded).sum(dim=1) / (video_mask_expanded.sum(dim=1) + 1e-6)
        else:
            video_global = video_feat.mean(dim=1)
        
        if text_mask is not None:
            text_mask_expanded = text_mask.unsqueeze(-1).float()
            text_global = (text_feat * text_mask_expanded).sum(dim=1) / (text_mask_expanded.sum(dim=1) + 1e-6)
        else:
            text_global = text_feat.mean(dim=1)
        
        video_global = F.normalize(video_global, p=2, dim=-1)
        text_global = F.normalize(text_global, p=2, dim=-1)
        
        # 使用当前温度计算相似度
        sim_matrix = torch.matmul(video_global, text_global.t()) / current_temp
        
        labels = torch.arange(B, device=sim_matrix.device)
        loss_v2t = F.cross_entropy(sim_matrix, labels)
        loss_t2v = F.cross_entropy(sim_matrix.t(), labels)
        
        loss = (loss_v2t + loss_t2v) / 2.0
        
        hard_neg_loss = 0.0
        if self.use_hard_negatives and B > 1:
            neg_mask = 1 - torch.eye(B, device=sim_matrix.device)
            hard_neg_sim = (sim_matrix * neg_mask).max(dim=1)[0]
            margin = 0.2
            hard_neg_loss = F.relu(hard_neg_sim - margin).mean()
            loss = loss + 0.1 * hard_neg_loss
        
        with torch.no_grad():
            pos_sim = torch.diagonal(sim_matrix).mean() * current_temp 
            if B > 1:
                neg_mask = 1 - torch.eye(B, device=sim_matrix.device)
                neg_sim = ((sim_matrix * neg_mask).sum() / (B * (B - 1))) * current_temp
            else:
                neg_sim = torch.tensor(0.0, device=sim_matrix.device)
            preds = sim_matrix.argmax(dim=1)
            accuracy = (preds == labels).float().mean()
        
        info = {
            'loss_v2t': loss_v2t.item(),
            'loss_t2v': loss_t2v.item(),
            'hard_neg_loss': hard_neg_loss.item() if isinstance(hard_neg_loss, torch.Tensor) else hard_neg_loss,
            'pos_similarity': pos_sim.item(),
            'neg_similarity': neg_sim.item(),
            'accuracy': accuracy.item()
        }
        
        return loss, info


class EnhancedISPLoss(nn.Module):
    def __init__(self, temperature=0.07, local_weight=0.3):
        super().__init__()
        self.temperature = temperature
        self.local_weight = local_weight
        self.global_isp = ISPLoss(temperature=temperature, use_hard_negatives=True)
        
    def local_alignment_loss(self, video_feat, text_feat, video_mask=None, text_mask=None):
        B, L_v, D = video_feat.shape
        L_t = text_feat.shape[1]
        video_feat_norm = F.normalize(video_feat, p=2, dim=-1)
        text_feat_norm = F.normalize(text_feat, p=2, dim=-1)
        sim = torch.bmm(text_feat_norm, video_feat_norm.transpose(1, 2))
        max_sim_per_token, _ = sim.max(dim=2)
        if text_mask is not None:
            max_sim_per_token = max_sim_per_token * text_mask.float()
            num_valid = text_mask.sum(dim=1, keepdim=True).float() + 1e-6
        else:
            num_valid = L_t
        local_loss = -(max_sim_per_token.sum(dim=1) / num_valid).mean()
        return local_loss
    
    def forward(self, video_feat, text_feat, video_mask=None, text_mask=None, temperature=None):
        # 修正维度
        if video_mask is not None and video_feat.dim() == 3:
            mask_B, mask_L = video_mask.shape
            if video_feat.shape[0] == mask_L and video_feat.shape[1] == mask_B:
                video_feat = video_feat.permute(1, 0, 2)
            elif video_feat.shape[0] == mask_B and video_feat.shape[1] == mask_L:
                pass
            else:
                if video_feat.shape[0] < video_feat.shape[1]:
                    video_feat = video_feat.permute(1, 0, 2)
        else:
            if video_feat.dim() == 3 and video_feat.shape[0] < video_feat.shape[1]:
                video_feat = video_feat.permute(1, 0, 2)
        
        if text_mask is not None and text_feat.dim() == 3:
            mask_B, mask_L = text_mask.shape
            if text_feat.shape[0] == mask_L and text_feat.shape[1] == mask_B:
                text_feat = text_feat.permute(1, 0, 2)
            elif text_feat.shape[0] == mask_B and text_feat.shape[1] == mask_L:
                pass
            else:
                if text_feat.shape[0] < text_feat.shape[1]:
                    text_feat = text_feat.permute(1, 0, 2)
        else:
            if text_feat.dim() == 3 and text_feat.shape[0] < text_feat.shape[1]:
                text_feat = text_feat.permute(1, 0, 2)
        
        # 传递动态温度
        global_loss, global_info = self.global_isp(
            video_feat, text_feat, video_mask, text_mask, temperature=temperature
        )
        
        local_loss = self.local_alignment_loss(
            video_feat, text_feat, video_mask, text_mask
        )
        
        total_loss = global_loss + self.local_weight * local_loss
        
        info = global_info.copy()
        info['local_loss'] = local_loss.item()
        info['global_loss'] = global_loss.item()
        
        return total_loss, info

# ================== 测试代码 ==================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing ISP Loss")
    print("=" * 60)
    
    B, L_v, L_t, D = 8, 75, 32, 256
    
    # 创建模拟数据
    video_feat = torch.randn(L_v, B, D)  # [L_v, B, D]
    text_feat = torch.randn(L_t, B, D)   # [L_t, B, D]
    
    video_mask = torch.ones(B, L_v, dtype=torch.bool)
    video_mask[:, 60:] = False  # 模拟padding
    
    text_mask = torch.ones(B, L_t, dtype=torch.bool)
    text_mask[:, 25:] = False  # 模拟padding
    
    # 测试基础ISP Loss
    print("\n1. Testing Basic ISP Loss")
    print("-" * 60)
    isp_loss = ISPLoss(temperature=0.07, use_hard_negatives=True)
    loss, info = isp_loss(video_feat, text_feat, video_mask, text_mask)
    
    print(f"Loss: {loss.item():.4f}")
    print("Loss components:")
    for k, v in info.items():
        print(f"  {k}: {v:.4f}")
    
    # 测试增强版ISP Loss
    print("\n2. Testing Enhanced ISP Loss")
    print("-" * 60)
    enhanced_isp = EnhancedISPLoss(temperature=0.07, local_weight=0.3)
    total_loss, enhanced_info = enhanced_isp(video_feat, text_feat, video_mask, text_mask)
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print("Loss components:")
    for k, v in enhanced_info.items():
        print(f"  {k}: {v:.4f}")
    
    # 测试梯度反传
    print("\n3. Testing Gradient Backpropagation")
    print("-" * 60)
    video_feat_grad = video_feat.clone().requires_grad_(True)
    text_feat_grad = text_feat.clone().requires_grad_(True)
    
    loss, _ = isp_loss(video_feat_grad, text_feat_grad, video_mask, text_mask)
    loss.backward()
    
    print(f"Video feature gradient norm: {video_feat_grad.grad.norm().item():.4f}")
    print(f"Text feature gradient norm: {text_feat_grad.grad.norm().item():.4f}")
    
    print("\n✅ All tests passed!")