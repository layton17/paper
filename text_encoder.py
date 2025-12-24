import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# =================================================================
# 1. 基础组件
# =================================================================

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

# =================================================================
# 2. CLIP Text Encoder
# =================================================================

class CLIPTextEncoder(nn.Module):
    def __init__(self, embed_dim=512, context_length=77, vocab_size=49408, transformer_width=512, transformer_heads=8, transformer_layers=12):
        super().__init__()
        self.context_length = context_length
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads, attn_mask=self.build_attention_mask())
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.token_embedding.weight.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        eos_x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        return dict(last_hidden_state=x, pooler_output=eos_x)

# =================================================================
# 3. GloVe Text Encoder (带错误处理的健壮版本)
# =================================================================

class GloVe(object):
    def __init__(self, glove_path, dim=300):
        self.glove_path = glove_path
        self.dim = dim
        self.glove = self._load()
        # 初始化特殊 token
        if "<PAD>" not in self.glove:
            self.glove["<PAD>"] = torch.zeros(self.dim)
        if "<UNK>" not in self.glove:
            self.glove["<UNK>"] = torch.randn(self.dim)

    def get(self, word):
        return self.glove.get(word, self.glove["<UNK>"])

    def _load(self):
        """ 健壮的 GloVe 加载函数，自动跳过坏行 """
        glove = dict()
        logger.info(f"Loading GloVe embeddings from {self.glove_path} ...")
        
        try:
            # errors='ignore' 防止编码错误
            with open(self.glove_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in tqdm(f, desc="Parsing GloVe"):
                    split_line = line.strip().split()
                    
                    # [修复 1] 长度检查：如果切分后长度不足 dim + 1，说明该行截断了，跳过
                    if len(split_line) < self.dim + 1:
                        continue
                        
                    try:
                        # [修复 2] 尝试转换向量，如果包含非数字字符则捕获异常
                        # 取最后 dim 个元素作为向量
                        vals = [float(val) for val in split_line[-self.dim:]]
                        embedding = torch.tensor(vals, dtype=torch.float32)
                        
                        # 剩下的部分是单词 (处理可能包含空格的短语)
                        word = " ".join(split_line[:-self.dim])
                        glove[word] = embedding
                    except ValueError:
                        # 再次捕获诸如 'tendineae' 出现在向量区的情况
                        continue
                        
        except FileNotFoundError:
            logger.error(f"GloVe file not found at {self.glove_path}")
            # 返回空字典，后续会全部初始化为 UNK
            return dict()
            
        logger.info(f"Successfully loaded {len(glove)} words.")
        return glove

class GloveTextEncoder(nn.Module):
    def __init__(self, vocab_list, glove_path):
        super(GloveTextEncoder, self).__init__()
        # 假设维度为 300，如果你的文件是其他维度，请修改这里
        self.glove_loader = GloVe(glove_path, dim=300) 
        dim = self.glove_loader.dim
        
        self.emb = nn.Embedding(num_embeddings=len(vocab_list), embedding_dim=dim)
        
        # 填充权重矩阵
        weight_matrix = torch.zeros((len(vocab_list), dim))
        for i, word in enumerate(vocab_list):
            weight_matrix[i] = self.glove_loader.get(word)
            
        self.emb.weight.data.copy_(weight_matrix)
        # 冻结 GloVe 参数
        for param in self.emb.parameters():
            param.requires_grad = False

    def forward(self, word_ids):
        # word_ids: [Batch, Len]
        return self.emb(word_ids)