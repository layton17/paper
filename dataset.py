import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import logging

# 尝试导入 CLIP
try:
    import clip
except ImportError:
    clip = None

logger = logging.getLogger(__name__)

class VideoDataset(Dataset):
    def __init__(self, args, is_training=True):
        self.args = args
        self.is_training = is_training
        self.feature_dir = args.feature_dir
        self.clip_visual_path = getattr(args, 'clip_visual_path', None)
        self.max_v_l = args.max_v_l
        self.max_q_l = args.max_q_l
        
        # 1. 加载标注文件
        anno_path = args.annotation_path
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"Annotation file not found at {anno_path}")
            
        logger.info(f"Loading annotations from {anno_path}...")
        
        if anno_path.endswith('.txt'):
            self.annotations = self._load_txt(anno_path)
        else:
            try:
                with open(anno_path, 'r') as f:
                    self.annotations = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Failed to load JSON from {anno_path}.")
                raise

        # 2. 文本处理准备
        self.text_encoder_type = args.text_encoder_type
        self.vocab = None
        
        if self.text_encoder_type == 'glove':
            self.word2idx = self._build_vocab()
            self.vocab = list(self.word2idx.keys())
            logger.info(f"Vocab size: {len(self.vocab)}")
        elif self.text_encoder_type == 'clip':
            if clip is None:
                logger.warning("Warning: 'clip' module not found. Please install openai-clip.")
        else:
            logger.info(f"Loading cached text features from {args.cached_text_path}...")
            self.cached_text_feats = torch.load(args.cached_text_path)

    def _load_txt(self, path):
        data = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    meta, sentence = line.split('##')
                    parts = meta.split()
                    vid_id = parts[0]
                    start = float(parts[1])
                    end = float(parts[2])
                    data.append({
                        "video_id": vid_id,
                        "timestamps": [[start, end]],
                        "sentence": sentence.strip()
                    })
                except Exception as e:
                    logger.warning(f"Skipping line: {line}. Error: {e}")
        logger.info(f"Loaded {len(data)} samples from txt.")
        return data

    def _build_vocab(self):
        word_counts = {}
        for sample in self.annotations:
            sentence = sample['sentence'].lower().replace('.', '').replace(',', '')
            for w in sentence.split():
                word_counts[w] = word_counts.get(w, 0) + 1
        word2idx = {"<PAD>": 0, "<UNK>": 1}
        for w, count in word_counts.items():
            if count >= 1: 
                word2idx[w] = len(word2idx)
        return word2idx

    def _read_file(self, vid_id, dir_path):
        if not dir_path or not os.path.exists(dir_path): return None
        p_npz = os.path.join(dir_path, f"{vid_id}.npz")
        p_npy = os.path.join(dir_path, f"{vid_id}.npy")
        
        feat = None
        if os.path.exists(p_npz):
            try:
                data = np.load(p_npz)
                if isinstance(data, np.lib.npyio.NpzFile):
                    for key in ['features', 'arr_0', 'embedding']:
                        if key in data:
                            feat = data[key]; break
                    if feat is None and len(data.files) > 0: feat = data[data.files[0]]
                else: feat = data
            except Exception as e: logger.error(f"Error loading {p_npz}: {e}")
        elif os.path.exists(p_npy):
            feat = np.load(p_npy)
            
        if feat is None: return None
        feat = torch.from_numpy(feat).float()
        
        if feat.dim() == 2:
            if feat.shape[0] in [512, 2304, 2048] and feat.shape[1] not in [512, 2304, 2048]:
                 feat = feat.transpose(0, 1)
        if feat.dim() == 1: feat = feat.unsqueeze(0)
        return feat

    def _load_video_feature(self, vid_id):
        feat_main = self._read_file(vid_id, self.feature_dir)
        if feat_main is None:
            raise FileNotFoundError(f"Main feature not found for {vid_id} in {self.feature_dir}")

        if self.clip_visual_path:
            feat_aux = self._read_file(vid_id, self.clip_visual_path)
            if feat_aux is None:
                raise FileNotFoundError(f"Aux feature not found for {vid_id} in {self.clip_visual_path}")
            min_len = min(feat_main.shape[0], feat_aux.shape[0])
            feat_main = feat_main[:min_len]
            feat_aux = feat_aux[:min_len]
            feat = torch.cat([feat_main, feat_aux], dim=1)
        else:
            feat = feat_main
            
        raw_seq_len = feat.shape[0]
        target_len = self.max_v_l
        
        if raw_seq_len >= target_len:
            idxs = torch.linspace(0, raw_seq_len - 1, target_len).long()
            feat = feat[idxs]
            mask = torch.ones(target_len, dtype=torch.bool)
        else:
            pad_len = target_len - raw_seq_len
            padding = torch.zeros(pad_len, feat.shape[1])
            feat = torch.cat([feat, padding], dim=0)
            mask = torch.cat([torch.ones(raw_seq_len, dtype=torch.bool), 
                              torch.zeros(pad_len, dtype=torch.bool)], dim=0)
        return feat, mask, raw_seq_len

    def _process_text(self, sentence):
        if self.text_encoder_type == 'clip':
            tokens = clip.tokenize(sentence, truncate=True).squeeze(0)
            if self.max_q_l < 77: tokens = tokens[:self.max_q_l]
            mask = (tokens != 0)
            return tokens, mask
        elif self.text_encoder_type == 'glove':
            words = sentence.lower().replace('.', '').replace(',', '').split()
            words = words[:self.max_q_l]
            ids = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words]
            pad_len = self.max_q_l - len(ids)
            mask = [True] * len(ids) + [False] * pad_len
            ids = ids + [self.word2idx["<PAD>"]] * pad_len
            return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    def __len__(self): return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations[idx]
        vid_id = sample['video_id']
        sentence = sample['sentence']
        
        video_feat, video_mask, raw_seq_len = self._load_video_feature(vid_id)
        time_unit = self.args.feature_stride / self.args.feature_fps
        duration = sample.get('duration', raw_seq_len * time_unit)
        
        # [修改] 始终生成 txt_ids (整数标签) 用于 Loss 计算
        if self.args.text_encoder_type == 'precomputed':
            # 1. 获取预计算特征 (用于模型输入)
            words_feat = self.cached_text_feats[sentence].float()
            
            # 2. 实时生成 Token IDs (用于 GT 标签)
            tokens = clip.tokenize(sentence, truncate=True).squeeze(0)
            words_mask = (tokens != 0)
            
            # 对齐长度
            if self.max_q_l < 77:
                tokens = tokens[:self.max_q_l]
                words_mask = words_mask[:self.max_q_l]
                if words_feat.shape[0] > self.max_q_l:
                    words_feat = words_feat[:self.max_q_l]
            
            # words_id 传特征给模型，txt_ids 传 ID 给 Loss
            words_id = words_feat 
            txt_ids = tokens.long()
        else:
            words_id, words_mask = self._process_text(sentence)
            txt_ids = words_id # 此时 words_id 本身就是 ID
        
        gt_spans = []
        if 'timestamps' in sample:
            for st, ed in sample['timestamps']:
                st = max(0, min(st, duration))
                ed = max(0, min(ed, duration))
                if ed <= st: continue 
                center = (st + ed) / 2.0 / duration
                width = (ed - st) / duration
                gt_spans.append([center, width])
        
        if len(gt_spans) == 0: gt_spans.append([0.5, 0.1]) 
            
        target = {
            "spans": torch.tensor(gt_spans, dtype=torch.float32),
            "labels": torch.zeros(len(gt_spans), dtype=torch.long),
            "video_id": vid_id,
            "duration": duration,
            "raw_txt": sentence,
            "words_id": txt_ids
        }

        return {
            "video_feat": video_feat,
            "video_mask": video_mask,
            "words_id": words_id,   # 输入 (Precomputed Features 或 IDs)
            "txt_ids": txt_ids,     # 标签 (始终是 IDs)
            "words_mask": words_mask,
            "targets": target 
        }

def collate_fn(batch):
    batch_data = {
        "video_feat": torch.stack([b['video_feat'] for b in batch]),
        "video_mask": torch.stack([b['video_mask'] for b in batch]),
        "words_id": torch.stack([b['words_id'] for b in batch]),
        "txt_ids": torch.stack([b['txt_ids'] for b in batch]), # Stack txt_ids
        "words_mask": torch.stack([b['words_mask'] for b in batch]),
        "targets": [b['targets'] for b in batch] 
    }
    return batch_data