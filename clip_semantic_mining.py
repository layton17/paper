"""
Clip Semantic Mining Module
来自论文: Multi-level Event-centric Multi-modal Representation Learning (MESM)

核心思想:
1. 除了word-level的跨模态交互，还需要sentence-level和clip-level的交互
2. 通过多层次的语义挖掘，捕获更丰富的跨模态对应关系
3. 使用渐进式的特征增强策略
"""

import torch
from torch import nn
import torch.nn.functional as F

class ClipSemanticMining(nn.Module):
    """
    Clip级别的语义挖掘
    在已有的word-level交互基础上，增加clip-level的全局语义对齐
    """
    def __init__(self, hidden_dim, nhead=8, num_layers=2, dropout=0.1):
        """
        Args:
            hidden_dim: 特征维度
            nhead: multi-head attention头数
            num_layers: 挖掘层数
            dropout: dropout比例
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 1. Sentence-level表示提取
        # 从word-level特征聚合为sentence-level
        self.sentence_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 注意力权重计算
        self.word_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 2. Clip-level表示提取
        # 从frame-level特征聚合为clip-level
        self.clip_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 时序注意力权重
        self.frame_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 3. Clip-Sentence Cross-Attention
        self.clip_sent_attn = nn.MultiheadAttention(
            hidden_dim, nhead, dropout=dropout, batch_first=False
        )
        self.clip_sent_norm = nn.LayerNorm(hidden_dim)
        
        # 4. 多层渐进式挖掘
        self.mining_layers = nn.ModuleList([
            ClipMiningLayer(hidden_dim, nhead, dropout)
            for _ in range(num_layers)
        ])
        
        # 5. 最终融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def aggregate_to_sentence(self, word_feat, word_mask=None):
        """
        从word-level聚合为sentence-level表示
        
        Args:
            word_feat: [L_word, B, D] - 词级特征
            word_mask: [B, L_word] - True表示有效词
        
        Returns:
            sent_feat: [B, D] - 句子级特征
        """
        L_word, B, D = word_feat.shape
        
        # 计算注意力权重
        word_feat_bld = word_feat.permute(1, 0, 2)  # [B, L_word, D]
        attn_scores = self.word_attention(word_feat_bld)  # [B, L_word, 1]
        
        # Mask处理
        if word_mask is not None:
            attn_scores = attn_scores.masked_fill(~word_mask.unsqueeze(-1), -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, L_word, 1]
        
        # 加权聚合
        sent_feat = (word_feat_bld * attn_weights).sum(dim=1)  # [B, D]
        sent_feat = self.sentence_aggregator(sent_feat)  # [B, D]
        
        return sent_feat
    
    def aggregate_to_clip(self, frame_feat, frame_mask=None, num_clips=None):
        """
        从frame-level聚合为clip-level表示
        
        Args:
            frame_feat: [L_frame, B, D] - 帧级特征
            frame_mask: [B, L_frame] - True表示有效帧
            num_clips: int or None - 聚合成多少个clips
                      如果为None，则聚合为单个clip
        
        Returns:
            clip_feat: [K, B, D] if num_clips else [B, D]
        """
        L_frame, B, D = frame_feat.shape
        frame_feat_bld = frame_feat.permute(1, 0, 2)  # [B, L_frame, D]
        
        if num_clips is None or num_clips == 1:
            # 聚合为单个clip
            attn_scores = self.frame_attention(frame_feat_bld)  # [B, L_frame, 1]
            
            if frame_mask is not None:
                attn_scores = attn_scores.masked_fill(~frame_mask.unsqueeze(-1), -1e9)
            
            attn_weights = F.softmax(attn_scores, dim=1)  # [B, L_frame, 1]
            clip_feat = (frame_feat_bld * attn_weights).sum(dim=1)  # [B, D]
            clip_feat = self.clip_aggregator(clip_feat)
            
            return clip_feat
        else:
            # 聚合为多个clips
            # 使用均匀分割 + 注意力聚合
            clip_size = L_frame // num_clips
            if clip_size == 0:
                clip_size = 1
                num_clips = L_frame
            
            clip_feats = []
            for i in range(num_clips):
                start_idx = i * clip_size
                end_idx = min((i + 1) * clip_size, L_frame)
                
                # 提取当前clip的帧
                clip_frames = frame_feat_bld[:, start_idx:end_idx, :]  # [B, clip_len, D]
                
                # 注意力聚合
                clip_attn_scores = self.frame_attention(clip_frames)  # [B, clip_len, 1]
                clip_attn_weights = F.softmax(clip_attn_scores, dim=1)
                clip_agg = (clip_frames * clip_attn_weights).sum(dim=1)  # [B, D]
                clip_agg = self.clip_aggregator(clip_agg)
                
                clip_feats.append(clip_agg)
            
            # [K, B, D]
            clip_feats = torch.stack(clip_feats, dim=0)
            return clip_feats
    
    def forward(self, video_feat, text_feat, video_mask=None, text_mask=None, num_clips=4):
        """
        Args:
            video_feat: [L_v, B, D] - 视频帧特征
            text_feat: [L_t, B, D] - 文本词特征
            video_mask: [B, L_v]
            text_mask: [B, L_t]
            num_clips: 将视频分成多少个clips进行挖掘
        
        Returns:
            enhanced_video: [L_v, B, D] - 增强后的视频特征
            mining_info: dict - 挖掘过程的信息
        """
        L_v, B, D = video_feat.shape
        L_t = text_feat.shape[0]
        
        # 1. 聚合为sentence和clip表示
        sent_feat = self.aggregate_to_sentence(text_feat, text_mask)  # [B, D]
        clip_feats = self.aggregate_to_clip(video_feat, video_mask, num_clips)  # [K, B, D]
        
        # 2. Clip-Sentence交互
        # sent_feat: [B, D] -> [1, B, D]
        sent_feat_expanded = sent_feat.unsqueeze(0)  # [1, B, D]
        
        # Cross-attention: clips attend to sentence
        clip_enhanced, _ = self.clip_sent_attn(
            query=clip_feats,           # [K, B, D]
            key=sent_feat_expanded,     # [1, B, D]
            value=sent_feat_expanded    # [1, B, D]
        )
        
        clip_enhanced = self.clip_sent_norm(
            clip_feats + self.dropout(clip_enhanced)
        )
        
        # 3. 多层渐进式挖掘
        current_video = video_feat
        for layer in self.mining_layers:
            current_video = layer(
                current_video,
                clip_enhanced,
                sent_feat_expanded,
                video_mask,
                text_mask
            )
        
        # 4. 融合原始特征和挖掘特征
        concat_feat = torch.cat([video_feat, current_video], dim=-1)  # [L_v, B, 2D]
        enhanced_video = self.fusion(concat_feat)  # [L_v, B, D]
        
        # 收集信息
        mining_info = {
            'sentence_feat': sent_feat.detach(),
            'clip_feats': clip_feats.detach(),
            'num_clips': num_clips
        }
        
        return enhanced_video, mining_info


class ClipMiningLayer(nn.Module):
    """
    单层的Clip挖掘模块
    从clip-level知识向frame-level传播
    """
    def __init__(self, hidden_dim, nhead=8, dropout=0.1):
        super().__init__()
        
        # 1. Frame-to-Clip attention
        self.frame_to_clip_attn = nn.MultiheadAttention(
            hidden_dim, nhead, dropout=dropout, batch_first=False
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # 2. Frame-to-Sentence attention
        self.frame_to_sent_attn = nn.MultiheadAttention(
            hidden_dim, nhead, dropout=dropout, batch_first=False
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 3. FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, frame_feat, clip_feat, sent_feat, frame_mask=None, sent_mask=None):
        """
        Args:
            frame_feat: [L_v, B, D]
            clip_feat: [K, B, D]
            sent_feat: [1, B, D]
            frame_mask: [B, L_v]
            sent_mask: [B, L_t] (not used here, for interface consistency)
        
        Returns:
            enhanced_frame: [L_v, B, D]
        """
        # 1. Frames attend to clips
        clip_out, _ = self.frame_to_clip_attn(
            query=frame_feat,
            key=clip_feat,
            value=clip_feat
        )
        frame_feat = self.norm1(frame_feat + self.dropout(clip_out))
        
        # 2. Frames attend to sentence
        sent_out, _ = self.frame_to_sent_attn(
            query=frame_feat,
            key=sent_feat,
            value=sent_feat
        )
        frame_feat = self.norm2(frame_feat + self.dropout(sent_out))
        
        # 3. FFN
        ffn_out = self.ffn(frame_feat)
        frame_feat = self.norm3(frame_feat + self.dropout(ffn_out))
        
        return frame_feat


# ================== 测试代码 ==================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Clip Semantic Mining")
    print("=" * 60)
    
    B, L_v, L_t, D = 4, 75, 32, 256
    num_clips = 4
    
    # 创建模拟数据
    video_feat = torch.randn(L_v, B, D)
    text_feat = torch.randn(L_t, B, D)
    
    video_mask = torch.ones(B, L_v, dtype=torch.bool)
    video_mask[:, 60:] = False
    
    text_mask = torch.ones(B, L_t, dtype=torch.bool)
    text_mask[:, 25:] = False
    
    # 创建模块
    csm = ClipSemanticMining(
        hidden_dim=D,
        nhead=8,
        num_layers=2,
        dropout=0.1
    )
    
    # 前向传播
    enhanced_video, mining_info = csm(
        video_feat, text_feat,
        video_mask, text_mask,
        num_clips=num_clips
    )
    
    print(f"\nInput video shape: {video_feat.shape}")
    print(f"Input text shape: {text_feat.shape}")
    print(f"Output enhanced video shape: {enhanced_video.shape}")
    print(f"\nMining Info:")
    print(f"  Sentence feat shape: {mining_info['sentence_feat'].shape}")
    print(f"  Clip feats shape: {mining_info['clip_feats'].shape}")
    print(f"  Number of clips: {mining_info['num_clips']}")
    
    # 测试梯度
    print("\nTesting gradient backpropagation...")
    video_grad = video_feat.clone().requires_grad_(True)
    text_grad = text_feat.clone().requires_grad_(True)
    
    enhanced, _ = csm(video_grad, text_grad, video_mask, text_mask, num_clips)
    loss = enhanced.sum()
    loss.backward()
    
    print(f"Video gradient norm: {video_grad.grad.norm().item():.4f}")
    print(f"Text gradient norm: {text_grad.grad.norm().item():.4f}")
    
    print("\n✅ All tests passed!")
