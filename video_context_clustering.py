"""
Video Context Clustering Module
来自论文: Keyword-guided Video Moment Retrieval (CVPR 2024)

核心思想:
1. 将相似的video clips聚类到一起
2. 通过聚类中心增强clip之间的区分度
3. 解决相邻clips高度相似的问题
"""

import torch
from torch import nn
import torch.nn.functional as F

class VideoContextClustering(nn.Module):
    def __init__(self, hidden_dim, num_clusters=8, nhead=8, dropout=0.1):
        """
        Args:
            hidden_dim: 特征维度
            num_clusters: 聚类中心数量
            nhead: multi-head attention的头数
            dropout: dropout比例
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        
        # 1. 聚类中心 (可学习参数)
        self.cluster_centers = nn.Parameter(
            torch.randn(num_clusters, hidden_dim) * 0.02
        )
        
        # 2. 软分配网络 (计算每个clip属于每个聚类的概率)
        self.assignment_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_clusters)
        )
        
        # 3. 聚类上下文融合
        self.cluster_attn = nn.MultiheadAttention(
            hidden_dim, nhead, dropout=dropout, batch_first=False
        )
        
        self.cluster_norm = nn.LayerNorm(hidden_dim)
        
        # 4. 门控融合 (决定使用多少聚类信息)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, video_feat, video_mask=None):
        """
        Args:
            video_feat: [L, B, D] - 视频特征
            video_mask: [B, L] - padding mask (True为有效帧)
        
        Returns:
            clustered_feat: [L, B, D] - 聚类增强后的特征
            cluster_info: dict - 聚类信息（用于可视化和分析）
        """
        L, B, D = video_feat.shape
        
        # 1. 计算软分配 (每个clip属于每个聚类的概率)
        # [L, B, D] -> [B, L, D]
        feat_bld = video_feat.permute(1, 0, 2)
        
        # [B, L, num_clusters]
        assignment_logits = self.assignment_net(feat_bld)
        
        # Mask处理 (padding位置的分配概率设为0)
        if video_mask is not None:
            # video_mask: [B, L], True表示有效
            mask_expanded = video_mask.unsqueeze(-1)  # [B, L, 1]
            assignment_logits = assignment_logits.masked_fill(~mask_expanded, -1e9)
        
        # Softmax得到软分配概率
        assignment_probs = F.softmax(assignment_logits, dim=-1)  # [B, L, num_clusters]
        
        # 2. 更新聚类中心 (基于当前batch的clips)
        # 使用软分配加权平均
        # feat_bld: [B, L, D], assignment_probs: [B, L, K]
        # 目标: batch_centers: [B, K, D]
        
        # [B, K, L] @ [B, L, D] -> [B, K, D]
        batch_centers = torch.bmm(
            assignment_probs.transpose(1, 2),  # [B, K, L]
            feat_bld  # [B, L, D]
        )
        
        # 归一化 (每个聚类中心是其成员的加权平均)
        weights_sum = assignment_probs.sum(dim=1, keepdim=True).transpose(1, 2)  # [B, K, 1]
        batch_centers = batch_centers / (weights_sum + 1e-6)
        
        # 3. 融合可学习聚类中心和batch聚类中心
        # 可学习中心提供稳定性，batch中心提供适应性
        # 使用0.5:0.5的混合
        global_centers = self.cluster_centers.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]
        fused_centers = 0.5 * global_centers + 0.5 * batch_centers
        
        # 4. 使用聚类中心增强video特征
        # video_feat: [L, B, D] as query
        # fused_centers: [B, K, D] -> [K, B, D] as key/value
        centers_kbd = fused_centers.permute(1, 0, 2)  # [K, B, D]
        
        # Cross-attention: video clips attend to cluster centers
        cluster_out, cluster_attn_weights = self.cluster_attn(
            query=video_feat,       # [L, B, D]
            key=centers_kbd,        # [K, B, D]
            value=centers_kbd,      # [K, B, D]
            need_weights=True
        )
        
        # 5. 残差连接 + LayerNorm
        video_feat_clustered = self.cluster_norm(
            video_feat + self.dropout(cluster_out)
        )
        
        # 6. 门控融合 (自适应决定使用多少聚类信息)
        concat_feat = torch.cat([video_feat, video_feat_clustered], dim=-1)  # [L, B, 2D]
        gate_weight = self.gate(concat_feat)  # [L, B, D]
        
        # 门控融合: 原始特征 * (1-gate) + 聚类特征 * gate
        output_feat = (1 - gate_weight) * video_feat + gate_weight * video_feat_clustered
        
        # 收集聚类信息用于分析
        cluster_info = {
            'assignment_probs': assignment_probs.detach(),  # [B, L, K]
            'cluster_centers': fused_centers.detach(),      # [B, K, D]
            'gate_weights': gate_weight.mean(dim=(0, 1)).detach(),  # [D] -> scalar
            'attn_weights': cluster_attn_weights.detach() if cluster_attn_weights is not None else None
        }
        
        return output_feat, cluster_info


class KeywordWeightDetection(nn.Module):
    """
    关键词权重检测模块 (Keyword-DETR的另一个核心)
    为查询中的每个词分配重要性权重
    """
    def __init__(self, hidden_dim, nhead=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Self-attention提取词之间的依赖关系
        self.word_self_attn = nn.MultiheadAttention(
            hidden_dim, nhead, dropout=dropout, batch_first=False
        )
        self.word_norm = nn.LayerNorm(hidden_dim)
        
        # 2. 重要性预测器
        self.importance_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出[0,1]的重要性分数
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text_feat, text_mask=None):
        """
        Args:
            text_feat: [L_txt, B, D] - 文本特征
            text_mask: [B, L_txt] - True表示有效词
        
        Returns:
            weighted_text: [L_txt, B, D] - 加权后的文本特征
            word_weights: [B, L_txt] - 每个词的重要性权重
        """
        L_txt, B, D = text_feat.shape
        
        # 1. Self-attention增强词特征
        key_padding_mask = ~text_mask if text_mask is not None else None
        
        attn_out, _ = self.word_self_attn(
            query=text_feat,
            key=text_feat,
            value=text_feat,
            key_padding_mask=key_padding_mask
        )
        
        enhanced_text = self.word_norm(text_feat + self.dropout(attn_out))
        
        # 2. 预测每个词的重要性
        # [L_txt, B, D] -> [B, L_txt, D] -> [B, L_txt, 1] -> [B, L_txt]
        word_weights = self.importance_predictor(
            enhanced_text.permute(1, 0, 2)
        ).squeeze(-1)
        
        # Mask处理
        if text_mask is not None:
            word_weights = word_weights * text_mask.float()
        
        # 3. 加权文本特征
        # [B, L_txt] -> [L_txt, B, 1]
        weights_expanded = word_weights.transpose(0, 1).unsqueeze(-1)
        weighted_text = enhanced_text * weights_expanded
        
        return weighted_text, word_weights


# ================== 测试代码 ==================
if __name__ == "__main__":
    # 测试Video Context Clustering
    print("=" * 60)
    print("Testing Video Context Clustering")
    print("=" * 60)
    
    B, L, D = 4, 75, 256
    video_feat = torch.randn(L, B, D)
    video_mask = torch.ones(B, L, dtype=torch.bool)
    video_mask[:, 60:] = False  # 模拟padding
    
    vcc = VideoContextClustering(hidden_dim=D, num_clusters=8)
    clustered_feat, cluster_info = vcc(video_feat, video_mask)
    
    print(f"Input shape: {video_feat.shape}")
    print(f"Output shape: {clustered_feat.shape}")
    print(f"Assignment probs shape: {cluster_info['assignment_probs'].shape}")
    print(f"Cluster centers shape: {cluster_info['cluster_centers'].shape}")
    print(f"Average gate weight: {cluster_info['gate_weights'].item():.4f}")
    
    # 测试Keyword Weight Detection
    print("\n" + "=" * 60)
    print("Testing Keyword Weight Detection")
    print("=" * 60)
    
    L_txt = 32
    text_feat = torch.randn(L_txt, B, D)
    text_mask = torch.ones(B, L_txt, dtype=torch.bool)
    text_mask[:, 20:] = False  # 模拟padding
    
    kwd = KeywordWeightDetection(hidden_dim=D)
    weighted_text, word_weights = kwd(text_feat, text_mask)
    
    print(f"Input text shape: {text_feat.shape}")
    print(f"Output text shape: {weighted_text.shape}")
    print(f"Word weights shape: {word_weights.shape}")
    print(f"Sample word weights (batch 0): {word_weights[0, :10].tolist()}")
    
    print("\n✅ All tests passed!")
