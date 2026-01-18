"""
main_optimized.py - 优化版本

主要改进:
1. bbox_embed 使用更高的学习率 (5倍)
2. 使用 CosineAnnealingWarmRestarts 学习率调度
3. 打印当前学习率便于监控
"""

import torch
import os
import time
import random
import numpy as np
import logging
import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

# 引入自定义模块
from config import get_args_parser
from dataset import VideoDataset, collate_fn
from model import build_model
from text_encoder import CLIPTextEncoder, GloveTextEncoder
from matcher import HungarianMatcher
from criterion import SetCriterion
from engine import evaluate

# 配置基础 Logger (控制台输出)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch):
    model.train()
    criterion.train()
    
    total_loss = 0
    
    # [统计变量初始化] - 确保初始化所有 8 个 Loss
    total_span_loss = 0
    total_giou_loss = 0
    total_label_loss = 0
    total_quality_loss = 0
    total_cont_loss = 0
    total_saliency_loss = 0
    total_recfw_loss = 0   
    total_isp_loss = 0
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch} Train")
    
    for i, batch in pbar:
        video_feat = batch['video_feat'].to(device)
        video_mask = batch['video_mask'].to(device)
        words_id = batch['words_id'].to(device)
        words_mask = batch['words_mask'].to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in batch['targets']]

        # Forward
        outputs = model(video_feat, video_mask, words_id, words_mask, is_training=True)
        
        # Loss Calculation
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        # 计算加权总 Loss
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Backward
        optimizer.zero_grad()
        losses.backward()
        if hasattr(model.args, 'clip_max_norm') and model.args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), model.args.clip_max_norm)
        optimizer.step()
        
        total_loss += losses.item()
        
        # [提取所有 Loss]
        l_span = loss_dict.get('loss_span', torch.tensor(0.0)).item()
        l_giou = loss_dict.get('loss_giou', torch.tensor(0.0)).item()
        l_label = loss_dict.get('loss_labels', torch.tensor(0.0)).item()
        l_qual = loss_dict.get('loss_quality', torch.tensor(0.0)).item()
        l_cont = loss_dict.get('loss_contrastive', torch.tensor(0.0)).item()
        l_sal = loss_dict.get('loss_saliency', torch.tensor(0.0)).item()
        l_rec = loss_dict.get('loss_recfw', torch.tensor(0.0)).item()    
        l_isp = loss_dict.get('loss_isp', torch.tensor(0.0)).item()
        
        # [累加统计]
        total_span_loss += l_span
        total_giou_loss += l_giou
        total_label_loss += l_label
        total_quality_loss += l_qual
        total_cont_loss += l_cont
        total_saliency_loss += l_sal
        total_recfw_loss += l_rec    
        total_isp_loss += l_isp
        
        pbar.set_postfix({
            'Loss': f"{losses.item():.2f}",     
            'Span': f"{l_span:.2f}",
            'IoU': f"{l_giou:.2f}",
            'Cls': f"{l_label:.2f}",
            'ISP': f"{l_isp:.3f}"
        })
    
    # [计算 Epoch 平均值]
    num_batches = len(data_loader)
    avg_loss = total_loss / num_batches
    
    avg_span = total_span_loss / num_batches
    avg_giou = total_giou_loss / num_batches
    avg_label = total_label_loss / num_batches
    avg_qual = total_quality_loss / num_batches
    avg_cont = total_cont_loss / num_batches
    avg_sal = total_saliency_loss / num_batches
    avg_rec = total_recfw_loss / num_batches   
    avg_isp = total_isp_loss / num_batches
    
    logger.info(
        f"Epoch [{epoch}] Avg Loss: {avg_loss:.4f}\n"
        f"  - Label: {avg_label:.4f} | Quality: {avg_qual:.4f}\n"
        f"  - Span:  {avg_span:.4f}  | GIoU:    {avg_giou:.4f}\n"
        f"  - Sal:   {avg_sal:.4f}   | Cont:    {avg_cont:.4f}\n"
        f"  - Rec:   {avg_rec:.4f}   | ISP:     {avg_isp:.4f}"
    )
    
    return avg_loss

def main(args):
    device = torch.device(args.device)
    set_seed(args.seed)
    logger.info(f"Checking Modules Status:")
    logger.info(f"  - VCC: {args.use_vcc}")
    logger.info(f"  - KWD: {args.use_kwd}")
    logger.info(f"  - CSM: {args.use_csm}")
    
    # 1. 创建 Checkpoint 保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 日志设置
    exp_name = os.path.basename(os.path.normpath(args.save_dir))
    log_dir = os.path.join("logs", exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_{timestamp}.log")
    
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    root_logger = logging.getLogger('')
    for h in root_logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            root_logger.removeHandler(h)
    root_logger.addHandler(file_handler)
    
    logger.info(f"Log file created at: {log_path}")
    
    # ----------------------
    logger.info("=" * 60)
    logger.info("Training Configuration:")
    for key, value in sorted(vars(args).items()):
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)
    # -----------------------
    logger.info(f"initializing Dataset: {args.dataset_name}")
    
    # -----------------------------------------------------------
    # 2. 加载数据集
    # -----------------------------------------------------------
    dataset_train = VideoDataset(args, is_training=True)
    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    logger.info(f"Train dataset size: {len(dataset_train)}")

    test_anno_path = args.annotation_path.replace("train.txt", "test.txt")
    dataloader_val = None
    if os.path.exists(test_anno_path):
        logger.info(f"Loading Validation Dataset from: {test_anno_path}")
        args_val = type(args)(**vars(args)) 
        args_val.annotation_path = test_anno_path
        dataset_val = VideoDataset(args_val, is_training=False)
        
        if args.text_encoder_type == 'glove':
            dataset_val.word2idx = dataset_train.word2idx
            dataset_val.vocab = dataset_train.vocab
            
        dataloader_val = DataLoader(
            dataset_val, batch_size=args.batch_size, shuffle=False, 
            collate_fn=collate_fn, num_workers=4, pin_memory=True
        )

    # -----------------------------------------------------------
    # 3. 初始化 Text Encoder
    # -----------------------------------------------------------
    text_encoder = None
    if args.text_encoder_type == 'clip':
        logger.info("Building CLIP Text Encoder...")
        text_encoder = CLIPTextEncoder(
            embed_dim=args.t_feat_dim,
            context_length=args.max_q_l,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12
        )
        if hasattr(args, 'clip_weight_path') and args.clip_weight_path:
            logger.info(f"Loading CLIP weights from {args.clip_weight_path}")
            try:
                loaded_obj = torch.load(args.clip_weight_path, map_location='cpu')
            except Exception:
                loaded_obj = torch.jit.load(args.clip_weight_path, map_location='cpu')
            
            if hasattr(loaded_obj, 'state_dict'):
                state_dict = loaded_obj.state_dict()
            else:
                state_dict = loaded_obj

            if 'positional_embedding' in state_dict:
                ckpt_len = state_dict['positional_embedding'].shape[0]
                model_len = text_encoder.positional_embedding.shape[0]
                if ckpt_len > model_len:
                    state_dict['positional_embedding'] = state_dict['positional_embedding'][:model_len, :]
            text_encoder.load_state_dict(state_dict, strict=False)

    elif args.text_encoder_type == 'glove':
        text_encoder = GloveTextEncoder(dataset_train.vocab, args.glove_path)
    
    if text_encoder is not None:
        text_encoder.to(device)
    elif args.text_encoder_type != 'precomputed': 
        raise ValueError("Text Encoder failed to initialize.")

    # -----------------------------------------------------------
    # 4. 构建模型
    # -----------------------------------------------------------
    logger.info("Building Model...")
    model = build_model(args)
    if hasattr(model, 'text_encoder') and model.text_encoder is None:
        model.text_encoder = text_encoder
    model.to(device)

    # -----------------------------------------------------------
    # 5. 匹配器和损失
    # -----------------------------------------------------------
    matcher = HungarianMatcher(cost_class=2, cost_span=5, cost_giou=2)
    
    weight_dict = {
        'loss_labels': args.label_loss_coef, 
        'loss_span': args.span_loss_coef, 
        'loss_giou': args.giou_loss_coef, 
        'loss_quality': args.quality_loss_coef, 
        'loss_saliency': args.lw_saliency, 
        'loss_contrastive': args.eos_coef, 
        'loss_recfw': args.recfw_loss_coef,
        'loss_isp': getattr(args, 'isp_loss_coef', 1.0)
    }

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'spans', 'quality', 'recfw', 'saliency', 'isp'] 
    criterion = SetCriterion(matcher, weight_dict, losses=losses, eos_coef=args.eos_coef)
    criterion.to(device)

    # -----------------------------------------------------------
    # 6. 优化器与调度器 [核心改进]
    # -----------------------------------------------------------
    
    # [改进1] 分离 bbox_embed 参数，使用更高学习率
    bbox_embed_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "text_encoder" in name:
            continue
        if "bbox_embed" in name:
            bbox_embed_params.append(param)
            logger.info(f"  [bbox_embed 5x LR] {name}")
        else:
            other_params.append(param)
    
    logger.info(f"bbox_embed params: {len(bbox_embed_params)}, other params: {len(other_params)}")
    
    # bbox_embed 使用 5 倍学习率
    param_dicts = [
        {"params": other_params, "lr": args.lr},
        {"params": bbox_embed_params, "lr": args.lr * 5},
    ]
    
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # [改进2] 使用 CosineAnnealingWarmRestarts
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=30,
        T_mult=2,
        eta_min=1e-6
    )
    
    logger.info(f"Using CosineAnnealingWarmRestarts scheduler (T_0=30, T_mult=2)")

    # Resume 逻辑
    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_dict = checkpoint['model_state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters.")
        
        if args.start_epoch > 0 and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            args.start_epoch = checkpoint['epoch'] + 1

    # -----------------------------------------------------------
    # 7. 训练循环
    # -----------------------------------------------------------
    logger.info(f"Start training for {args.epochs} epochs.")
    
    best_r1_07 = 0.0 
    best_r1_05 = 0.0
    best_combined = 0.0
    
    for epoch in range(args.start_epoch, args.epochs):
        
        # [打印当前学习率]
        current_lr = optimizer.param_groups[0]['lr']
        bbox_lr = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else current_lr
        logger.info(f"Epoch {epoch} LR: {current_lr:.2e} | bbox_embed LR: {bbox_lr:.2e}")
        
        # Loss Decay: 训练后期关闭辅助任务
        if epoch >= args.epochs * 0.5 and criterion.weight_dict['loss_recfw'] > 0:
            logger.info(f"Epoch {epoch}: Dropping RecFW Loss weight to 0.0!")
            criterion.weight_dict['loss_recfw'] = 0.0
            for k in criterion.weight_dict.keys():
                if 'recfw' in k:
                    criterion.weight_dict[k] = 0.0

        train_one_epoch(model, criterion, dataloader_train, optimizer, device, epoch)
        lr_scheduler.step()
        
        if dataloader_val is not None:
            metrics = evaluate(model, dataloader_val, device)
            
            curr_r1_05 = metrics.get('R1@0.5', 0)
            curr_r1_07 = metrics.get('R1@0.7', 0)
            curr_comb = curr_r1_05 + curr_r1_07

            if curr_r1_07 > best_r1_07:
                best_r1_07 = curr_r1_07
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'metrics': metrics}, os.path.join(args.save_dir, "checkpoint_best_r1_07.pth"))
                logger.info(f"⭐  New Best R1@0.7: {best_r1_07:.2f}%")

            if curr_r1_05 > best_r1_05:
                best_r1_05 = curr_r1_05
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'metrics': metrics}, os.path.join(args.save_dir, "checkpoint_best_r1_05.pth"))
                logger.info(f"⭐  New Best R1@0.5: {best_r1_05:.2f}%")
                
            if curr_comb > best_combined:
                best_combined = curr_comb
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'metrics': metrics}, os.path.join(args.save_dir, "checkpoint_best_combined.pth"))
                logger.info(f"⭐  New Best Combined: {best_combined:.2f}")

        # 保存最新模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args
        }, os.path.join(args.save_dir, "checkpoint_last.pth"))
    
    # 训练结束
    logger.info("=" * 60)
    logger.info(f"Training finished!")
    logger.info(f"   Best R1@0.5: {best_r1_05:.2f}%")
    logger.info(f"   Best R1@0.7: {best_r1_07:.2f}%")
    logger.info(f"   Best Combined: {best_combined:.2f}")
    logger.info("=" * 60)

if __name__ == '__main__':
    parser = get_args_parser()
    
    if not any(action.dest == 'text_encoder_type' for action in parser._actions):
        parser.add_argument('--text_encoder_type', default='clip', choices=['clip', 'glove'], help='Type of text encoder')
    if not any(action.dest == 'glove_path' for action in parser._actions):
        parser.add_argument('--glove_path', default='', type=str, help='Path to glove vectors')
    if not any(action.dest == 'clip_weight_path' for action in parser._actions):
        parser.add_argument('--clip_weight_path', default='', type=str, help='Path to pretrained CLIP weights')
    
    if not any(action.dest == 'resume' for action in parser._actions):
        parser.add_argument('--resume', default='', help='resume from checkpoint')
    if not any(action.dest == 'start_epoch' for action in parser._actions):
        parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')

    if not any(action.dest == 'isp_loss_coef' for action in parser._actions):
        parser.add_argument('--isp_loss_coef', default=1.0, type=float, help='Coefficient for ISP loss')
    if not any(action.dest == 'use_vcc' for action in parser._actions):
        parser.add_argument('--use_vcc', action='store_true', default=True)
    if not any(action.dest == 'use_kwd' for action in parser._actions):
        parser.add_argument('--use_kwd', action='store_true', default=True)
    if not any(action.dest == 'use_csm' for action in parser._actions):
        parser.add_argument('--use_csm', action='store_true', default=True)

    args = parser.parse_args()
    main(args)
