import torch
from torch import nn
import torch.nn.functional as F
import math
from utils import LinearLayer, MLP, _get_clones
from utils import span_cxw_to_xx 
from mesm_layers import T2V_TransformerEncoderLayer, T2V_TransformerEncoder
from bam_layers import (
    TransformerDecoder, 
    TransformerDecoderLayer, 
    BoundaryDecoderLayer, 
    build_position_encoding
)

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

# MultiContextPerception 保持不变
class MultiContextPerception(nn.Module):
    def __init__(self, hidden_dim, nhead=8, dropout=0.1, dataset_name='charades'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        
        self.wts_lin = nn.Linear(hidden_dim, 1)
        self.ec_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.ec_norm = nn.LayerNorm(hidden_dim)
        
        if 'charades' in dataset_name.lower() or 'tvsum' in dataset_name.lower():
            self.stride_groups = [[1, 2], [4, 8], [16, 24], [32, 48]]
        else:
            self.stride_groups = [[1], [2], [4], [8]]
            
        self.cc_attns = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
            for _ in range(len(self.stride_groups))
        ])
        
        self.cc_norm = nn.LayerNorm(hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x, txt_feat, video_mask=None, txt_mask=None):
        attn_weights = self.wts_lin(txt_feat) 
        if txt_mask is not None:
            attn_weights = attn_weights.masked_fill((~txt_mask).transpose(0,1).unsqueeze(-1), -1e9)
        attn_weights = torch.softmax(attn_weights, dim=0)
        sent_feat = (attn_weights * txt_feat).sum(dim=0) 
        
        scores = torch.sum(x * sent_feat.unsqueeze(0), dim=-1) 
        if video_mask is not None:
            scores = scores.masked_fill((~video_mask).transpose(0, 1), -1e9)
        
        L, B, D = x.shape
        k = max(int(L * 0.5), 1)
        _, topk_indices = torch.topk(scores, k, dim=0)
        
        ec_key_mask = torch.ones((B, L), device=x.device, dtype=torch.bool)
        ec_key_mask.scatter_(1, topk_indices.transpose(0, 1), False) 
        if video_mask is not None:
            ec_key_mask = ec_key_mask | (~video_mask)
            
        ec_out, _ = self.ec_attn(x, x, x, key_padding_mask=ec_key_mask)
        x_ec = self.ec_norm(x + self.dropout(ec_out))

        x_perm = x.permute(1, 2, 0)
        def get_pooled_keys_and_mask(strides, mask_in=None):
            keys_list = []
            masks_list = []
            for s in strides:
                curr_x = x_perm
                curr_m = mask_in.float().unsqueeze(1) if mask_in is not None else None
                if curr_x.shape[-1] % s != 0:
                    pad_len = s - (curr_x.shape[-1] % s)
                    curr_x = F.pad(curr_x, (0, pad_len))
                    if curr_m is not None:
                        curr_m = F.pad(curr_m, (0, pad_len))
                pooled = F.max_pool1d(curr_x, kernel_size=s, stride=s)
                keys_list.append(pooled.permute(2, 0, 1))
                if curr_m is not None:
                    pooled_m = F.max_pool1d(curr_m, kernel_size=s, stride=s)
                    masks_list.append(pooled_m.squeeze(1))
            keys_out = torch.cat(keys_list, dim=0)
            mask_out = None
            if masks_list:
                mask_out = torch.cat(masks_list, dim=1)
                mask_out = ~(mask_out.bool())
            return keys_out, mask_out

        total_cc_out = 0
        for i, strides in enumerate(self.stride_groups):
            keys, mask = get_pooled_keys_and_mask(strides, video_mask)
            cc_out, _ = self.cc_attns[i](x, keys, keys, key_padding_mask=mask)
            total_cc_out = total_cc_out + cc_out
        
        x_cc = self.cc_norm(x + self.dropout(total_cc_out))
        out = x_ec + x_cc
        out2 = self.linear2(self.dropout(self.activation(self.linear1(out))))
        return self.fusion_norm(out + self.dropout(out2))


class MESM_W2W_BAM(nn.Module):
    def __init__(self, args, text_encoder=None):
        super().__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        
        self.text_encoder = text_encoder
        self.input_txt_proj = nn.Sequential(
            nn.LayerNorm(args.t_feat_dim),
            nn.Linear(args.t_feat_dim, args.hidden_dim)
        )
        self.input_vid_proj = nn.Sequential(
            nn.LayerNorm(args.v_feat_dim),
            nn.Linear(args.v_feat_dim, args.hidden_dim)
        )
        
        self.vid_pos_embed, _ = build_position_encoding(args)
        self.txt_pos_embed = nn.Embedding(args.max_q_l, args.hidden_dim)

        enhance_layer = T2V_TransformerEncoderLayer(args.hidden_dim, args.nheads)
        self.enhance_encoder = T2V_TransformerEncoder(enhance_layer, num_layers=2)
        
        align_layer = T2V_TransformerEncoderLayer(args.hidden_dim, args.nheads)
        self.t2v_encoder = T2V_TransformerEncoder(align_layer, num_layers=3)

        self.rec_fw = getattr(args, 'rec_fw', True)
        self.vocab_size = getattr(args, 'vocab_size', 49408)
        if self.rec_fw:
            self.masked_token = nn.Parameter(torch.randn(args.hidden_dim), requires_grad=True)
            self.output_txt_proj = nn.Sequential(
                nn.LayerNorm(args.hidden_dim),
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, self.vocab_size)
            )

        dataset_name = getattr(args, 'dataset_name', 'charades')
        self.w2w_context = MultiContextPerception(args.hidden_dim, args.nheads, dataset_name=dataset_name)
        # self.w2w_gate = nn.Parameter(torch.tensor(0.1))  <-- 删除或注释掉这一行

        # [新增] 非线性融合层
        # 将 Raw 和 W2W 特征拼接后，通过 MLP 进行深度融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(args.hidden_dim * 2, args.hidden_dim),
            nn.LayerNorm(args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_dim, args.hidden_dim)
        )
        # 保留一个 Gate 来控制融合特征对原始特征的残差贡献
        self.fusion_gate = nn.Parameter(torch.tensor(0.0))

        bam_layer = TransformerDecoderLayer(args.hidden_dim, args.nheads)
        boundary_layer = BoundaryDecoderLayer(args.hidden_dim, nhead=args.nheads)
        
        self.transformer_decoder = TransformerDecoder(
            bam_layer, 
            boundary_layer, 
            args.dec_layers, 
            args.hidden_dim, 
            args.nheads, 
            return_intermediate=True
        )
        self.query_embed = nn.Embedding(args.num_queries, 3)      
        self.class_embed = nn.Linear(args.hidden_dim, 2)
        self.span_embed = MLP(args.hidden_dim, args.hidden_dim, 2, 3)
        self.quality_proj = MLP(args.hidden_dim, args.hidden_dim * 2, 1, 3)
        self.saliency_proj = nn.Linear(args.hidden_dim, 1)

        self.vid_con_proj = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim)
        )
        self.txt_con_proj = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim)
        )

        self.class_embed = _get_clones(self.class_embed, args.dec_layers)
        self.span_embed = _get_clones(self.span_embed, args.dec_layers)
        self.quality_proj = _get_clones(self.quality_proj, args.dec_layers)
        
        self.transformer_decoder.bbox_embed = self.span_embed
        
        nn.init.constant_(self.saliency_proj.bias, -2.0)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_embed_layer in self.class_embed:
            nn.init.constant_(class_embed_layer.bias, 0)
            class_embed_layer.bias.data[0] = bias_value
            class_embed_layer.bias.data[1] = 0.0
        
        for span_embed_layer in self.span_embed:
            nn.init.constant_(span_embed_layer.layers[-1].weight.data, 0)
            nn.init.constant_(span_embed_layer.layers[-1].bias.data, 0)
            
        for q_proj in self.quality_proj:
            nn.init.constant_(q_proj.layers[-1].weight.data, 0)
            nn.init.constant_(q_proj.layers[-1].bias.data, 0)

    def _mask_words(self, src_txt, src_txt_mask, masked_token):
        src_txt_t = src_txt.permute(1, 0, 2).clone() 
        masked_token_vec = masked_token.view(1, 1, -1).to(src_txt.device)
        if src_txt_mask is not None:
             words_length = src_txt_mask.sum(dim=1).long()
        else:
             words_length = torch.full((src_txt.shape[1],), src_txt.shape[0], device=src_txt.device)

        mask_selection = torch.zeros_like(src_txt_mask, dtype=torch.bool)
        for i, l in enumerate(words_length):
            l = int(l)
            if l <= 1: continue
            num_masked = max(int(l * 0.3), 1)
            choices = torch.randperm(l, device=src_txt.device)[:num_masked]
            mask_selection[i, choices] = True
            
        mask_broadcast = mask_selection.unsqueeze(-1)
        masked_src_txt = torch.where(mask_broadcast, masked_token_vec, src_txt_t)
        return masked_src_txt.permute(1, 0, 2), mask_selection

    def forward(self, video_feat, video_mask, words_id, words_mask, is_training=False):
        # ------------------------------------------------------------------
        # 0. [BAM-DETR 核心] 强制约束 Anchor 参数 (In-place Constraint)
        # ------------------------------------------------------------------
        # 这段代码必须在最前面执行，它直接修改参数的数值，保证 Anchor 永远合法
        
        # 约束 1: 左距离 (dim 1) <= Anchor中心 (dim 0)
        # 逻辑: logit(d_s) <= logit(p)  =>  d_s <= p  =>  Start = p - d_s >= 0
        self.query_embed.weight.data[..., 1] = torch.minimum(
            self.query_embed.weight[..., 0], 
            self.query_embed.weight[..., 1]
        )
        
        # 约束 2: 右距离 (dim 2) <= 1 - Anchor中心
        # 逻辑: logit(d_e) <= logit(1-p) => d_e <= 1-p => End = p + d_e <= 1
        # inverse_sigmoid 是为了把概率空间转换回 logit 空间进行比较
        p_prob = self.query_embed.weight[..., 0].sigmoid()
        # 注意：为了数值稳定，使用 clamp
        p_prob = p_prob.clamp(min=1e-4, max=1-1e-4) 
        limit_right = inverse_sigmoid(1.0 - p_prob)
        
        self.query_embed.weight.data[..., 2] = torch.minimum(
            limit_right, 
            self.query_embed.weight[..., 2]
        )

        # ------------------------------------------------------------------
        # 1. 文本和视频特征编码 (Encoder) - 保持不变
        # ------------------------------------------------------------------
        if self.text_encoder:
            txt_out = self.text_encoder(words_id)
            if isinstance(txt_out, dict):
                words_feat = txt_out['last_hidden_state']
            else:
                words_feat = txt_out 
        else:
            words_feat = words_id
            
        src_txt = self.input_txt_proj(words_feat).permute(1, 0, 2)
        src_vid = self.input_vid_proj(video_feat).permute(1, 0, 2)
        
        pos_v = self.vid_pos_embed(src_vid.permute(1, 0, 2), video_mask).permute(1, 0, 2)
        pos_t = self.txt_pos_embed.weight[:src_txt.shape[0]].unsqueeze(1).repeat(1, src_txt.shape[1], 1)

        enhanced_vid = self.enhance_encoder(
            query=src_vid, key=src_txt, 
            key_padding_mask=~words_mask, 
            pos_q=pos_v, pos_k=pos_t
        )
        
        recfw_words_logit = None
        masked_indices = None
        if self.rec_fw and is_training:
            masked_src_txt, mask_selection = self._mask_words(src_txt, words_mask, self.masked_token)
            masked_indices = mask_selection
            rec_out_text_len = self.enhance_encoder(
                query=masked_src_txt, key=src_vid,
                key_padding_mask=~video_mask, 
                pos_q=pos_t, pos_k=pos_v
            )
            recfw_words_logit = self.output_txt_proj(rec_out_text_len).permute(1, 0, 2)

        f_aligned = self.t2v_encoder(
            query=enhanced_vid, key=src_txt, 
            key_padding_mask=~words_mask, 
            pos_q=pos_v, pos_k=pos_t
        )

        f_raw = f_aligned
        f_w2w = self.w2w_context(
            x=f_aligned, txt_feat=src_txt, video_mask=video_mask, txt_mask=words_mask
        )
        
        # [修改] 使用融合层
        # f_context = f_raw + self.w2w_gate * f_w2w <-- 替换这一行
        
        f_concat = torch.cat([f_raw, f_w2w], dim=-1) # [B, L, 2D]
        f_fused = self.fusion_layer(f_concat)        # [B, L, D]
        
        # 采用残差结构：原始特征 + 门控融合特征
        # 这样既保留了 f_raw 的定位敏感性，又注入了 f_w2w 的上下文信息
        f_context = f_raw + torch.sigmoid(self.fusion_gate) * f_fused
        
        # ------------------------------------------------------------------
        # 2. Anchor 生成 (使用清洗过的参数)
        # ------------------------------------------------------------------
        
        # [A] Saliency Scores
        memory = f_context.permute(1, 0, 2) # [B, L, D]
        saliency_scores = self.saliency_proj(memory).squeeze(-1) # [B, L]
        if torch.isnan(saliency_scores).any() or torch.isinf(saliency_scores).any():
            saliency_scores = torch.nan_to_num(saliency_scores, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # [B] 生成 Static Anchors (非对称)
        raw_anchor = self.query_embed.weight.unsqueeze(0).repeat(memory.shape[0], 1, 1)
        anchor_prob = raw_anchor.sigmoid()
        
        p = anchor_prob[..., 0]   # Center
        d_s = anchor_prob[..., 1] # Left Dist
        d_e = anchor_prob[..., 2] # Right Dist
        
        # 计算 Start / End
        # 因为前面已经做了硬约束，这里理论上不需要再 clamp，但保留以防万一
        start = (p - d_s).clamp(min=1e-4, max=1-1e-4)
        end   = (p + d_e).clamp(min=1e-4, max=1-1e-4)
        
        # 转回 Logit 域传给 Decoder
        query_pos = torch.stack([
            inverse_sigmoid(start), 
            inverse_sigmoid(end)
        ], dim=-1) # [B, Nq, 2]
        
        # [C] Content Query 全 0
        tgt = torch.zeros((self.args.num_queries, memory.shape[0], self.hidden_dim), 
                          device=memory.device, dtype=memory.dtype)

        # ------------------------------------------------------------------
        # 3. Transformer Decoder
        # ------------------------------------------------------------------
        hs, refs, boundary_mem = self.transformer_decoder(
            tgt, f_context, memory_key_padding_mask=~video_mask, 
            pos=pos_v, refpoints_unsigmoid=query_pos
        )

        outputs_class = torch.stack([self.class_embed[i](hs[i]) for i in range(len(hs))])
        outputs_quality = torch.stack([self.quality_proj[i](hs[i]) for i in range(len(hs))])
        outputs_coord = refs 

        # ------------------------------------------------------------------
        # 4. 对比学习特征准备
        # ------------------------------------------------------------------
        txt_feat_perm = src_txt.permute(1, 0, 2)
        txt_mask_float = words_mask.float().unsqueeze(-1)
        sent_feat = (txt_feat_perm * txt_mask_float).sum(dim=1) / (txt_mask_float.sum(dim=1) + 1e-6)
        proj_txt_emb = self.txt_con_proj(sent_feat) 
        
        vid_query_feat = hs[-1].permute(1, 0, 2)
        proj_vid_emb = self.vid_con_proj(vid_query_feat) 

        proj_txt_emb = F.normalize(proj_txt_emb, p=2, dim=-1)
        proj_vid_emb = F.normalize(proj_vid_emb, p=2, dim=-1)

        # ------------------------------------------------------------------
        # 5. 输出组装
        # ------------------------------------------------------------------
        out = {
            'pred_logits': outputs_class[-1].permute(1, 0, 2), 
            'pred_spans': outputs_coord[-1], 
            'pred_quality': outputs_quality[-1].permute(1, 0, 2),
            'saliency_scores': saliency_scores,
            'video_mask': video_mask,
            'recfw_words_logit': recfw_words_logit,
            'masked_indices': masked_indices,
            'proj_txt_emb': proj_txt_emb,
            'proj_vid_emb': proj_vid_emb
        }
        
        if self.args.aux_loss:
             out['aux_outputs'] = [
                {
                    'pred_logits': a.permute(1, 0, 2), 
                    'pred_spans': b,
                    'pred_quality': c.permute(1, 0, 2)
                }
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_quality[:-1])
            ]
            
        return out
def build_model(args):
    if not hasattr(args, 'vocab_size'): args.vocab_size = 49408
    if not hasattr(args, 'rec_fw'): args.rec_fw = True
    return MESM_W2W_BAM(args)