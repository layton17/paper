import torch
from torch import nn, Tensor
from typing import Optional
from utils import _get_activation_fn, _get_clones

class T2V_TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=True):
        super().__init__()
        # 改为 Cross-Attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, query, key, value=None, key_padding_mask=None, 
                pos_q=None, pos_k=None, **kwargs):
        if value is None:
            value = key
            
        # --- [修改开始] Pre-Norm 逻辑 ---
        
        # 1. Self/Cross Attention Block
        if self.normalize_before:
            query_norm = self.norm1(query) # Pre-Norm
        else:
            query_norm = query

        # 注意：使用归一化后的 query_norm 加上位置编码
        q = self.with_pos_embed(query_norm, pos_q)
        k = self.with_pos_embed(key, pos_k)
        
        src2 = self.cross_attn(query=q, key=k, value=value, 
                               key_padding_mask=key_padding_mask)[0]
        
        # Residual
        src = query + self.dropout1(src2)
        
        if not self.normalize_before:
            src = self.norm1(src) # Post-Norm

        # 2. Feed Forward Block
        if self.normalize_before:
            src_norm = self.norm2(src) # Pre-Norm
        else:
            src_norm = src
            
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(src2)
        
        if not self.normalize_before:
            src = self.norm2(src) # Post-Norm
            
        return src
        

class T2V_TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, key, value=None, key_padding_mask=None, 
                pos_q=None, pos_k=None, **kwargs):
        output = query
        for layer in self.layers:
            output = layer(query=output, key=key, value=value, 
                           key_padding_mask=key_padding_mask,
                           pos_q=pos_q, pos_k=pos_k, **kwargs)
        if self.norm is not None:
            output = self.norm(output)
        return output