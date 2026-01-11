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
    # [统计变量初始化]
    total_span_loss = 0
    total_giou_loss = 0
    total_quality_loss = 0
    total_saliency_loss = 0
    total_cont_loss = 0
    total_recfw_loss = 0   
    total_label_loss = 0   
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch} Train")
    
    for i, batch in pbar:
        video_feat = batch['video_feat'].to(device)
        video_mask = batch['video_mask'].to(device)
        words_id = batch['words_id'].to(device)
        txt_ids = batch['txt_ids'].to(device)
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
        
        # 提取各个子 Loss
        l_span = loss_dict.get('loss_span', torch.tensor(0.0)).item()
        l_giou = loss_dict.get('loss_giou', torch.tensor(0.0)).item()
        l_qual = loss_dict.get('loss_quality', torch.tensor(0.0)).item()
        l_cont = loss_dict.get('loss_contrastive', torch.tensor(0.0)).item()
        l_sal = loss_dict.get('loss_saliency', torch.tensor(0.0)).item()
        l_rec = loss_dict.get('loss_recfw', torch.tensor(0.0)).item()    
        l_label = loss_dict.get('loss_labels', torch.tensor(0.0)).item() 
        
        # 累加统计
        total_span_loss += l_span
        total_giou_loss += l_giou
        total_quality_loss += l_qual
        total_cont_loss += l_cont
        total_saliency_loss += l_sal
        total_recfw_loss += l_rec    
        total_label_loss += l_label  
        
        # 实时更新进度条
        pbar.set_postfix({
            'L': f"{losses.item():.2f}",     
            'Span': f"{l_span:.3f}",         
            'GIoU': f"{l_giou:.3f}",
            'Sal': f"{l_sal:.2f}",           
            'Rec': f"{l_rec:.3f}"            
        })
    
    # 计算 Epoch 平均值
    avg_loss = total_loss / len(data_loader)
    avg_span = total_span_loss / len(data_loader)
    avg_giou = total_giou_loss / len(data_loader)
    avg_cont = total_cont_loss / len(data_loader)
    avg_sal = total_saliency_loss / len(data_loader)
    avg_rec = total_recfw_loss / len(data_loader)   
    avg_label = total_label_loss / len(data_loader) 
    
    # 最终日志打印所有 Loss
    logger.info(
        f"Epoch [{epoch}] Avg Loss: {avg_loss:.4f} | "
        f"Cls: {avg_label:.4f} | "
        f"Span: {avg_span:.4f} | "
        f"GIoU: {avg_giou:.4f} | "
        f"Sal: {avg_sal:.4f} | "
        f"Cont: {avg_cont:.4f} | "
        f"Rec: {avg_rec:.4f}"
    )
    
    return avg_loss

def main(args):
    device = torch.device(args.device)
    set_seed(args.seed)
    
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
    
    logger.info(f"✅ Log file created at: {log_path}")
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
    # 5. 匹配器和损失 (配置初始权重)
    # -----------------------------------------------------------
    matcher = HungarianMatcher(cost_class=2, cost_span=5, cost_giou=2)
    
    # [关键修改] 初始权重配置
    weight_dict = {
        'loss_labels': 2.0, 
        'loss_span': 5.0, 
        'loss_giou': 2.0, 
        'loss_quality': 2.0, 
        'loss_saliency': 0.4, 
        'loss_contrastive': 0.5, # [新增] 提升对比学习权重，辅助 Saliency 学习
        'loss_recfw': 0.1        # 初始开启辅助任务
    }

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'spans', 'quality', 'recfw', 'saliency'] 
    criterion = SetCriterion(matcher, weight_dict, losses=losses, eos_coef=args.eos_coef)
    criterion.to(device)

    # -----------------------------------------------------------
    # 6. 优化器与调度器
    # -----------------------------------------------------------
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "text_encoder" not in n and p.requires_grad], "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,      
        eta_min=args.lr * 0.01 
    )

    # Resume 逻辑
    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_dict = checkpoint['model_state_dict']
        # 过滤不匹配的键 (如 saliency_proj)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info(f"✅ Loaded {len(pretrained_dict)}/{len(model_dict)} parameters.")
        
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
        
        # =======================================================
        # [核心逻辑] Loss Decay: 训练后期关闭辅助任务
        # =======================================================
        # 假设 100 个 Epoch，前 40 个 Epoch 用于热身和特征对齐
        # 40 个 Epoch 后关闭重构 Loss，专注回归
        if epoch >= 40 and criterion.weight_dict['loss_recfw'] > 0:
            logger.info(f"📉 Epoch {epoch}: Dropping RecFW Loss weight to 0.0!")
            criterion.weight_dict['loss_recfw'] = 0.0
            # 同时更新 aux_loss 中的 recfw
            for k in criterion.weight_dict.keys():
                if 'recfw' in k:
                    criterion.weight_dict[k] = 0.0
        # =======================================================

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
                logger.info(f"⭐ New Best R1@0.7: {best_r1_07:.2f}%")

            if curr_r1_05 > best_r1_05:
                best_r1_05 = curr_r1_05
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'metrics': metrics}, os.path.join(args.save_dir, "checkpoint_best_r1_05.pth"))
                logger.info(f"⭐ New Best R1@0.5: {best_r1_05:.2f}%")
                
            if curr_comb > best_combined:
                best_combined = curr_comb
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'metrics': metrics}, os.path.join(args.save_dir, "checkpoint_best_combined.pth"))
                logger.info(f"⭐ New Best Combined: {best_combined:.2f}")

        # 保存最新模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args
        }, os.path.join(args.save_dir, "checkpoint_last.pth"))

if __name__ == '__main__':
    parser = get_args_parser()
    # 补全可能缺失的 args
    if not any(action.dest == 'text_encoder_type' for action in parser._actions):
        parser.add_argument('--text_encoder_type', default='clip', choices=['clip', 'glove'], help='Type of text encoder')
    if not any(action.dest == 'glove_path' for action in parser._actions):
        parser.add_argument('--glove_path', default='', type=str, help='Path to glove vectors')
    if not any(action.dest == 'clip_weight_path' for action in parser._actions):
        parser.add_argument('--clip_weight_path', default='', type=str, help='Path to pretrained CLIP weights')
    
    # [新增] Resume 参数
    if not any(action.dest == 'resume' for action in parser._actions):
        parser.add_argument('--resume', default='', help='resume from checkpoint')
    if not any(action.dest == 'start_epoch' for action in parser._actions):
        parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')

    args = parser.parse_args()
    
    if not hasattr(args, 'vocab_size'): args.vocab_size = 49408
    if not hasattr(args, 'rec_fw'): args.rec_fw = True
    
    main(args)