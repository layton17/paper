import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
from utils import _get_activation_fn, _get_clones, MLP, inverse_sigmoid

# ----------------- Position Encoding -----------------

class PositionEmbeddingSine(nn.Module):
    """
    修正后的位置编码，适配 Mask 定义: True=有效, False=Padding
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
        
        # [Fix] True=1, 累加得到位置索引
        y_embed = mask.cumsum(1, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)

        pos_x = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        return pos_x

def build_position_encoding(args):
    N_steps = args.hidden_dim
    position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    return position_embedding, None 

def gen_sineembed_for_position(pos_tensor, num_pos_feats=256, only_center=False):
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / num_pos_feats)
    
    center_embed = pos_tensor[:, :, 0] * scale
    pos_x = center_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)

    if only_center:
        return pos_x

    span_embed = pos_tensor[:, :, 1] * scale
    pos_w = span_embed[:, :, None] / dim_t
    pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

    pos = torch.cat((pos_x, pos_w), dim=2)
    return pos

# ----------------- Decoder Layers -----------------

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=True):
        super().__init__()
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model * 2, d_model)
        
        self.cross_attn = nn.MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)
        self.ca_out_proj = nn.Linear(d_model * 2, d_model)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.nhead = nhead
        self.normalize_before = normalize_before 

    def forward(self, tgt, memory, memory_key_padding_mask=None, pos=None, query_pos=None, query_sine_embed=None, **kwargs):
        if self.normalize_before:
            tgt_norm = self.norm2(tgt)
        else:
            tgt_norm = tgt

        q_content = self.ca_qcontent_proj(tgt_norm)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)
        k_pos = self.ca_kpos_proj(pos)
        
        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape
        
        q_pos = self.ca_qpos_proj(query_pos) if query_pos is not None else 0
        q = q_content + q_pos
        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        
        q_total = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        
        k = k_content.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k_total = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q_total, key=k_total, value=v, key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = self.ca_out_proj(tgt2)
        
        tgt = tgt + self.dropout2(tgt2)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        if self.normalize_before:
            tgt_norm = self.norm3(tgt)
        else:
            tgt_norm = tgt
            
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt_norm))))
        tgt = tgt + self.dropout3(tgt2)
        
        if not self.normalize_before:
            tgt = self.norm3(tgt)
            
        return tgt, None

# ----------------- Boundary Decoder (核心修改部分) -----------------

def bilinear_sampling(value, sampling_locations):
    """
    value: [L_src, B, N_head, D_head]
    sampling_locations: [B, N_query, N_head, N_point, 1]
    """
    L_src, B, n_head, d_head = value.shape
    
    # 1. Prepare Value for grid_sample: [B*H, D_h, 1, L]
    value_trans = value.permute(1, 2, 3, 0).flatten(0, 1) # [B*H, D_h, L]
    value_trans = value_trans.unsqueeze(2) # [B*H, D_h, 1, L]
    
    # 2. Prepare Grid: [B*H, 1, Nq*Np, 2]
    _, Nq, _, Np, _ = sampling_locations.shape
    # [B, Nq, H, P, 1] -> [B, H, Nq, P, 1] -> [B*H, Nq*P]
    locs = sampling_locations.permute(0, 2, 1, 3, 4).flatten(0, 1).flatten(1, 2).squeeze(-1) 
    grid = torch.stack([locs, torch.zeros_like(locs)], dim=-1).unsqueeze(1) 
    
    # 3. Grid Sample
    # out: [B*H, D_h, 1, Nq*Np]
    out = F.grid_sample(value_trans, grid, align_corners=False, padding_mode='zeros')
    
    # 4. Reshape back
    # [B*H, D_h, 1, Nq*Np] -> [B, H, D_h, Nq, Np]
    out = out.squeeze(2).view(B, n_head, d_head, Nq, Np)
    
    # 5. Permute to [B, Nq, H, Np, D_h] (Match attn_weights shape)
    out = out.permute(0, 3, 1, 4, 2) 
    
    return out

class BoundaryDeformation(nn.Module):
    def __init__(self, d_model, nhead, num_points=4, sampling_ratio=2):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_points = num_points # 每个边界附近的采样点数
        
        # 1. 生成采样偏移量 (Learnable Offsets)
        self.sampling_offsets = nn.Linear(d_model, nhead * num_points * 2) # *2 for start/end
        self.sampling_offsets.weight.data.zero_()
        self.sampling_offsets.bias.data.zero_()
        
        # 2. 注意力权重 (Attention Weights)
        self.attention_weights = nn.Linear(d_model, nhead * num_points * 2)
        self.attention_weights.weight.data.zero_()
        self.attention_weights.bias.data.zero_()
        
        # 3. 投影层
        self.value_proj = nn.Linear(d_model * 2, d_model) # 处理 concat 的 features
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, pro_features, features, boundary_points, window_size):
        """
        pro_features: [Nq, B, D]
        """
        Nq, B, D = pro_features.shape
        L, _, _ = features.shape
        
        # 1. 投影 Value
        value = self.value_proj(features) 
        value = value.view(L, B, self.nhead, self.d_model // self.nhead)
        
        # 2. 计算偏移和权重
        query = pro_features.permute(1, 0, 2) # [B, Nq, D]
        offsets = self.sampling_offsets(query).view(B, Nq, self.nhead, self.num_points * 2, 1)
        attn_weights = self.attention_weights(query).view(B, Nq, self.nhead, self.num_points * 2)
        attn_weights = F.softmax(attn_weights, dim=-1).unsqueeze(-1) # [B, Nq, H, P, 1]
        
        # 3. 计算采样位置 (Sampling Locations)
        center = boundary_points[..., 0]
        width = boundary_points[..., 1]
        start = center - 0.5 * width
        end = center + 0.5 * width
        
        base_locs = torch.stack([start, end], dim=-1).unsqueeze(2).unsqueeze(-1)
        base_locs = base_locs.repeat(1, 1, self.nhead, self.num_points, 1) # [B, Nq, H, 2P, 1]
        
        sampling_locs = base_locs + offsets * width.view(B, Nq, 1, 1, 1) * 0.5
        sampling_grid = sampling_locs.clamp(0.0, 1.0) * 2.0 - 1.0
        
        # 4. 采样
        # sampled_value: [B, Nq, H, P, D_h] (Fix: Now matches attn_weights)
        sampled_value = bilinear_sampling(value, sampling_grid) 
        
        # 5. 聚合 (Weighted Sum)
        # [B, Nq, H, P, D_h] * [B, Nq, H, P, 1] -> sum over P -> [B, Nq, H, D_h]
        out = (sampled_value * attn_weights).sum(-2) 
        
        # 6. Flatten & Project
        out = out.flatten(2) # [B, Nq, D]
        out = out.permute(1, 0, 2) # [Nq, B, D] (Transformer expects Seq-First)
        
        return self.output_proj(out)

class BoundaryDecoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu"):
        super().__init__()
        self.boundary_deformation = BoundaryDeformation(d_model, nhead)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model) # Pre-Norm for FFN
        self.norm2 = nn.LayerNorm(d_model) # Post-Norm for Attn
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, features, proposal_points, pro_features, window_size):
        # features: [L, B, 2D]
        # pro_features: [Nq, B, D]
        
        # Pre-Norm (Optional, keeping consistent with standard Transfomer)
        tgt_norm = self.norm2(pro_features)
        
        # Deformation Attention
        tgt2 = self.boundary_deformation(tgt_norm, features, proposal_points, window_size)
        tgt = pro_features + self.dropout2(tgt2)
        
        # FFN
        tgt_norm = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt_norm))))
        tgt = tgt + self.dropout3(tgt2)
        
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, boundary_decoder_layer, num_layers, d_model, nhead, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.boundary_layers = _get_clones(boundary_decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.d_model = d_model
        
        self.ref_point_head = MLP(d_model*2, d_model*2, d_model, 2)
        self.bbox_embed = None # 外部赋值
        self.boundary_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1), nn.ReLU(),
            nn.Conv1d(d_model, d_model, 3, padding=1)
        )

    def forward(self, tgt, memory, memory_key_padding_mask=None, pos=None, refpoints_unsigmoid=None):
        # ... (boundary_mem 计算保持不变) ...
        mem_perm = memory.permute(1, 2, 0)
        boundary_mem = self.boundary_conv(mem_perm).permute(2, 0, 1) 
        mem_boundary_feat = torch.cat([memory, boundary_mem], dim=-1)

        output = tgt
        intermediate = []
        
        # reference_points: [B, Nq, 2] (Start, End) from sigmoid
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points_hist = [reference_points]
        
        for layer_id, (layer, boundary_layer) in enumerate(zip(self.layers, self.boundary_layers)):
            # 1. 准备 Reference Points (Center, Width) 格式
            centers = (reference_points[..., 0] + reference_points[..., 1]) / 2
            widths = (reference_points[..., 1] - reference_points[..., 0]).clamp(min=1e-6)
            ref_cw = torch.stack([centers, widths], dim=-1) # [B, Nq, 2]
            
            # 2. 生成 Query Position Embedding
            # gen_sineembed_for_position 输出通常是 [B, Nq, D]
            query_sine_embed = gen_sineembed_for_position(
                ref_cw, 
                num_pos_feats=self.d_model, 
                only_center=False
            )
            query_pos = self.ref_point_head(query_sine_embed) # [B, Nq, D]
            
            # [核心修复] !!! 必须 permute 为 [Nq, B, D] 以匹配 Transformer 输入 (tgt) !!!
            query_pos = query_pos.permute(1, 0, 2)
            query_sine_embed = query_sine_embed.permute(1, 0, 2)
            
            # 3. Stream 1: Content Update (Global Context)
            # 现在 input (output) 是 [Nq, B, D], query_pos 也是 [Nq, B, D]，可以相加了
            output, _ = layer(output, memory, memory_key_padding_mask, pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed)
            
            # 4. Stream 2: Boundary Update (Local Boundary Refinement)
            # Boundary Layer 需要 Batch-First [B, Nq, 2]，ref_cw 已经是这个格式了
            window_size = ref_cw[..., 1].contiguous()

            output = boundary_layer(
                features=mem_boundary_feat,      
                proposal_points=ref_cw,          # 传入 [B, Nq, 2]
                pro_features=output,             # 这里 output 还是 [Nq, B, D]
                window_size=window_size
            )

            if self.bbox_embed is not None:
                # output 当前形状是 [Nq, B, D] (Sequence First)
                # reference_points 形状是 [B, Nq, 2] (Batch First)
                # 必须先 permute output，使其变成 [B, Nq, D]
                tmp = self.bbox_embed[layer_id](output.permute(1, 0, 2))
                
                # 现在 tmp 是 [B, Nq, 2]，可以和 reference_points 相加了
                tmp[..., :2] += inverse_sigmoid(reference_points)
                reference_points = tmp[..., :2].sigmoid()
                ref_points_hist.append(reference_points)

            if self.return_intermediate:
                # 中间结果通常保持 Sequence First 以供可能的后续处理，或者也可以转置
                # 这里保持原样即可，DETR 默认 intermediate 是 [L, Nq, B, D] 或 [L, B, Nq, D]
                # 但为了简便，存 raw output 即可
                intermediate.append(output)

        return torch.stack(intermediate), torch.stack(ref_points_hist), boundary_mem