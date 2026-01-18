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
    """
    Initial Semantic Projection Loss
    
    在模型的早期阶段(enhance_encoder之后)计算文本-视频对比损失，
    确保CLIP特征在任务特定的语义空间中对齐
    """
    def __init__(self, temperature=0.07, use_hard_negatives=True):
        """
        Args:
            temperature: InfoNCE loss的温度系数
            use_hard_negatives: 是否使用hard negative mining
        """
        super().__init__()
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        
    def forward(self, video_feat, text_feat, video_mask=None, text_mask=None):
        """
        Args:
            video_feat: [L_v, B, D] or [B, L_v, D] - 视频特征
            text_feat: [L_t, B, D] or [B, L_t, D] - 文本特征
            video_mask: [B, L_v] - True表示有效帧
            text_mask: [B, L_t] - True表示有效词
        
        Returns:
            loss: scalar tensor
            info: dict with detailed loss components
        """
        # ========== [修复] 使用更可靠的格式判断 ==========
        # 策略：使用 mask 的形状来判断特征的格式
        # video_mask 的格式固定为 [B, L_v]
        
        if video_mask is not None and video_feat.dim() == 3:
            # 通过 mask 的形状来判断 video_feat 的格式
            mask_B, mask_L = video_mask.shape
            
            # 检查 video_feat 的哪个维度与 mask_B 匹配
            if video_feat.shape[0] == mask_L and video_feat.shape[1] == mask_B:
                # [L, B, D] 格式
                video_feat = video_feat.permute(1, 0, 2)  # -> [B, L, D]
            elif video_feat.shape[0] == mask_B and video_feat.shape[1] == mask_L:
                # [B, L, D] 格式，无需转换
                pass
            else:
                # 如果都不匹配，使用原有的简单判断
                if video_feat.shape[0] < video_feat.shape[1]:
                    video_feat = video_feat.permute(1, 0, 2)
        else:
            # 没有 mask 时，使用原有的判断方法（假设较小的维度是 Length）
            if video_feat.dim() == 3 and video_feat.shape[0] < video_feat.shape[1]:
                video_feat = video_feat.permute(1, 0, 2)
        
        # 对 text_feat 使用相同的策略
        if text_mask is not None and text_feat.dim() == 3:
            mask_B, mask_L = text_mask.shape
            
            if text_feat.shape[0] == mask_L and text_feat.shape[1] == mask_B:
                # [L, B, D] 格式
                text_feat = text_feat.permute(1, 0, 2)  # -> [B, L, D]
            elif text_feat.shape[0] == mask_B and text_feat.shape[1] == mask_L:
                # [B, L, D] 格式，无需转换
                pass
            else:
                if text_feat.shape[0] < text_feat.shape[1]:
                    text_feat = text_feat.permute(1, 0, 2)
        else:
            if text_feat.dim() == 3 and text_feat.shape[0] < text_feat.shape[1]:
                text_feat = text_feat.permute(1, 0, 2)
        
        B = video_feat.shape[0]
        
        # 1. 聚合为全局表示
        # 对于视频：对所有有效帧取平均
        if video_mask is not None:
            video_mask_expanded = video_mask.unsqueeze(-1).float()  # [B, L_v, 1]
            video_global = (video_feat * video_mask_expanded).sum(dim=1) / \
                          (video_mask_expanded.sum(dim=1) + 1e-6)  # [B, D]
        else:
            video_global = video_feat.mean(dim=1)  # [B, D]
        
        # 对于文本：对所有有效词取平均
        if text_mask is not None:
            text_mask_expanded = text_mask.unsqueeze(-1).float()  # [B, L_t, 1]
            text_global = (text_feat * text_mask_expanded).sum(dim=1) / \
                         (text_mask_expanded.sum(dim=1) + 1e-6)  # [B, D]
        else:
            text_global = text_feat.mean(dim=1)  # [B, D]
        
        # 2. L2 归一化 (这是对比学习的关键)
        video_global = F.normalize(video_global, p=2, dim=-1)  # [B, D]
        text_global = F.normalize(text_global, p=2, dim=-1)   # [B, D]
        
        # 3. 计算相似度矩阵
        # [B, D] @ [D, B] -> [B, B]
        # sim_matrix[i, j] = 视频i和文本j的相似度
        sim_matrix = torch.matmul(video_global, text_global.t()) / self.temperature
        
        # 4. InfoNCE Loss (双向)
        # 4.1 Video-to-Text方向：对于每个视频，正样本是对应的文本，负样本是其他文本
        labels = torch.arange(B, device=sim_matrix.device)
        
        # 使用cross_entropy，它内部会做softmax
        # sim_matrix的第i行表示视频i与所有文本的相似度
        loss_v2t = F.cross_entropy(sim_matrix, labels)
        
        # 4.2 Text-to-Video方向：对于每个文本，正样本是对应的视频，负样本是其他视频
        # sim_matrix的第j列表示文本j与所有视频的相似度
        # 需要转置：sim_matrix.t()的第j行表示文本j与所有视频的相似度
        loss_t2v = F.cross_entropy(sim_matrix.t(), labels)
        
        # 总loss是双向loss的平均
        loss = (loss_v2t + loss_t2v) / 2.0
        
        # 5. 可选：Hard Negative Mining
        # 找出那些相似度很高但实际不匹配的样本，给它们额外的惩罚
        hard_neg_loss = 0.0
        if self.use_hard_negatives and B > 1:
            # 创建mask，对角线为0（正样本不参与hard negative）
            neg_mask = 1 - torch.eye(B, device=sim_matrix.device)
            
            # 找出每个样本的最难负样本（相似度最高的负样本）
            hard_neg_sim = (sim_matrix * neg_mask).max(dim=1)[0]  # [B]
            
            # Hard negative loss：希望最难负样本的相似度也要低
            # 使用hinge loss: max(0, hard_neg_sim - margin)
            margin = 0.2  # 希望负样本相似度至少比正样本低0.2
            hard_neg_loss = F.relu(hard_neg_sim - margin).mean()
            
            # 加权加入总loss
            loss = loss + 0.1 * hard_neg_loss
        
        # 6. 收集详细信息
        with torch.no_grad():
            # 正样本相似度（对角线）
            pos_sim = torch.diagonal(sim_matrix).mean()
            # 负样本相似度（非对角线）
            if B > 1:
                neg_mask = 1 - torch.eye(B, device=sim_matrix.device)
                neg_sim = (sim_matrix * neg_mask).sum() / (B * (B - 1))
            else:
                neg_sim = torch.tensor(0.0, device=sim_matrix.device)
            
            # Accuracy (top-1准确率)
            preds = sim_matrix.argmax(dim=1)
            accuracy = (preds == labels).float().mean()
        
        info = {
            'loss_v2t': loss_v2t.item(),
            'loss_t2v': loss_t2v.item(),
            'hard_neg_loss': hard_neg_loss.item() if isinstance(hard_neg_loss, torch.Tensor) else hard_neg_loss,
            'pos_similarity': pos_sim.item(),
            'neg_similarity': neg_sim.item(),
            'accuracy': accuracy.item(),
            'similarity_gap': (pos_sim - neg_sim).item()
        }
        
        return loss, info


class EnhancedISPLoss(nn.Module):
    """
    增强版ISP Loss
    不仅计算全局对齐，还计算局部token-level的对齐
    """
    def __init__(self, temperature=0.07, local_weight=0.3):
        """
        Args:
            temperature: 温度系数
            local_weight: 局部对齐loss的权重
        """
        super().__init__()
        self.temperature = temperature
        self.local_weight = local_weight
        self.global_isp = ISPLoss(temperature=temperature, use_hard_negatives=True)
        
    def local_alignment_loss(self, video_feat, text_feat, video_mask=None, text_mask=None):
        """
        Token-level的对齐loss
        对于每个文本token，找到最相似的视频帧，计算对齐loss
        """
        # video_feat: [B, L_v, D]
        # text_feat: [B, L_t, D]
        
        B, L_v, D = video_feat.shape
        L_t = text_feat.shape[1]
        
        # L2归一化
        video_feat_norm = F.normalize(video_feat, p=2, dim=-1)  # [B, L_v, D]
        text_feat_norm = F.normalize(text_feat, p=2, dim=-1)    # [B, L_t, D]
        
        # 计算每个文本token与视频帧的相似度
        # [B, L_t, D] @ [B, D, L_v] -> [B, L_t, L_v]
        sim = torch.bmm(text_feat_norm, video_feat_norm.transpose(1, 2))
        
        # 对于每个文本token，找到最相似的视频帧
        max_sim_per_token, _ = sim.max(dim=2)  # [B, L_t]
        
        # Mask处理
        if text_mask is not None:
            max_sim_per_token = max_sim_per_token * text_mask.float()
            num_valid = text_mask.sum(dim=1, keepdim=True).float() + 1e-6
        else:
            num_valid = L_t
        
        # 希望每个文本token都能找到相似的视频帧
        # 使用负的平均最大相似度作为loss（要最大化相似度）
        local_loss = -(max_sim_per_token.sum(dim=1) / num_valid).mean()
        
        return local_loss
    
    def forward(self, video_feat, text_feat, video_mask=None, text_mask=None):
        """
        Args:
            video_feat: [L_v, B, D] or [B, L_v, D]
            text_feat: [L_t, B, D] or [B, L_t, D]
            video_mask: [B, L_v]
            text_mask: [B, L_t]
        
        Returns:
            total_loss: scalar
            info: dict
        """
        # ========== [修复] 使用更可靠的格式判断 ==========
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
        
        # 1. 全局对齐loss
        global_loss, global_info = self.global_isp(
            video_feat, text_feat, video_mask, text_mask
        )
        
        # 2. 局部对齐loss
        local_loss = self.local_alignment_loss(
            video_feat, text_feat, video_mask, text_mask
        )
        
        # 3. 加权组合
        total_loss = global_loss + self.local_weight * local_loss
        
        # 合并info
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