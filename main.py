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

# main.py

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
    total_recfw_loss = 0   # [新增] 文本重构损失统计
    total_label_loss = 0   # [新增] 分类损失统计
    
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
        
        # [修改] 提取各个子 Loss (使用 .get 安全获取，防止某些 loss 未开启时报错)
        l_span = loss_dict.get('loss_span', torch.tensor(0.0)).item()
        l_giou = loss_dict.get('loss_giou', torch.tensor(0.0)).item()
        l_qual = loss_dict.get('loss_quality', torch.tensor(0.0)).item()
        l_cont = loss_dict.get('loss_contrastive', torch.tensor(0.0)).item()
        l_sal = loss_dict.get('loss_saliency', torch.tensor(0.0)).item()
        l_rec = loss_dict.get('loss_recfw', torch.tensor(0.0)).item()    # [新增]
        l_label = loss_dict.get('loss_labels', torch.tensor(0.0)).item() # [新增]
        
        # [累加统计]
        total_span_loss += l_span
        total_giou_loss += l_giou
        total_quality_loss += l_qual
        total_cont_loss += l_cont
        total_saliency_loss += l_sal
        total_recfw_loss += l_rec    # [新增]
        total_label_loss += l_label  # [新增]
        
        # [修改] 实时更新进度条
        pbar.set_postfix({
            'L': f"{losses.item():.2f}",     # 总 Weighted Loss
            'Cls': f"{l_label:.3f}",         # Label/Classification
            'Span': f"{l_span:.3f}",         # Span L1
            'GIoU': f"{l_giou:.3f}",
            'Qual': f"{l_qual:.3f}",
            'Sal': f"{l_sal:.2f}",           # Saliency (注意这里保留2位小数因为数值较大)
            'Cont': f"{l_cont:.3f}",         # Contrastive
            'Rec': f"{l_rec:.3f}"            # RecFW
        })
    
    # 计算 Epoch 平均值
    avg_loss = total_loss / len(data_loader)
    avg_span = total_span_loss / len(data_loader)
    avg_giou = total_giou_loss / len(data_loader)
    avg_cont = total_cont_loss / len(data_loader)
    avg_sal = total_saliency_loss / len(data_loader)
    avg_rec = total_recfw_loss / len(data_loader)   # [新增]
    avg_label = total_label_loss / len(data_loader) # [新增]
    
    # [修改] 最终日志打印所有 Loss
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

    # 日志分离逻辑
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
    logger.info(f"✅ Checkpoints will be saved to: {args.save_dir}")
    logger.info(f"Initializing Dataset: {args.dataset_name}")
    
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
    else:
        logger.warning(f"Validation file not found at {test_anno_path}")

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
                    logger.info(f"⚠️ Truncating positional embedding from {ckpt_len} to {model_len}")
                    state_dict['positional_embedding'] = state_dict['positional_embedding'][:model_len, :]

            text_encoder.load_state_dict(state_dict, strict=False)

    elif args.text_encoder_type == 'glove':
        logger.info("Building GloVe Text Encoder...")
        if not hasattr(dataset_train, 'vocab'):
            raise AttributeError("Dataset needs 'vocab' attribute for GloVe mode.")
        
        text_encoder = GloveTextEncoder(
            vocab_list=dataset_train.vocab, 
            glove_path=args.glove_path
        )
    
    if text_encoder is not None:
        text_encoder.to(device)
    elif args.text_encoder_type != 'precomputed': 
        raise ValueError("Text Encoder failed to initialize.")

    # -----------------------------------------------------------
    # 4. 构建模型
    # -----------------------------------------------------------
    logger.info("Building Model...")
    model = build_model(args)
    # 如果 build_model 内部没有赋值 text_encoder，这里手动赋一次
    if hasattr(model, 'text_encoder') and model.text_encoder is None:
        model.text_encoder = text_encoder
    model.to(device)

    # -----------------------------------------------------------
    # 5. 匹配器和损失
    # -----------------------------------------------------------
    matcher = HungarianMatcher(cost_class=2, 
                               cost_span=5, 
                               cost_giou=2)
    
    # [修改 A] 重构 Loss 权重配置
    weight_dict = {
        # Labels: 分类任务通常收敛很快，权重从 5.0 降为 2.0，防止过拟合
        'loss_labels': 2.0, 
        
        # Span & GIoU: 核心回归任务，保持高权重，主导梯度
        'loss_span': 5.0, 
        'loss_giou': 2.0, 
        
        # Quality: 保持不变，用于辅助排序
        'loss_quality': 2.0, 

        # [关键修改] Saliency: 从 4.0 降至 0.4
        # 理由: Raw Loss 约为 1.1，乘以 0.4 后为 0.44，既能提供辅助信息，又不会掩盖回归任务的梯度
        'loss_saliency': 0.4, 

        # Contrastive: 保持较低权重
        'loss_contrastive': 0.2, 

        # RecFW: 如果不是核心任务，建议进一步降低或保持 0.1
        'loss_recfw': 0.1 
    }

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # 注册所有需要的 loss
    losses = ['labels', 'spans', 'quality', 'recfw', 'saliency'] 
    
    
    criterion = SetCriterion(matcher, weight_dict, losses=losses, eos_coef=args.eos_coef)
    criterion.to(device)

    # -----------------------------------------------------------
    # 6. 优化器
    # -----------------------------------------------------------
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "text_encoder" not in n and p.requires_grad], "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # [修改 C] 使用 CosineAnnealingLR 替换 MultiStepLR
    # Cosine 调度器在微调后期能更平滑地降低学习率，有助于模型在局部极小值附近稳定下来
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,      # 设为总 Epoch 数
        eta_min=args.lr * 0.01  # 最小学习率设为初始 LR 的 1%
    )
    
    # 将 quality_proj 和 masked_token 等新参数加入优化器
    #param_dicts = [
    #    {"params": [p for n, p in model.named_parameters() if "text_encoder" not in n and p.requires_grad], "lr": args.lr},
    #]
    #optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # 将 StepLR 替换为 MultiStepLR
    # main.py
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #    optimizer,
    #    milestones=[60, 80],  
    #    gamma=0.5             
    #)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #    optimizer,
    #    step_size=args.lr_drop, # 30
    #    gamma=0.1
    #)
    
    # ===========================================================
    # [新增] Resume / Fine-tuning 逻辑
    # ===========================================================
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            logger.info(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu') # 先加载到 CPU

        # 1. 加载模型权重
        model_dict = model.state_dict()
        pretrained_dict = checkpoint['model_state_dict']
        
        # 过滤掉不匹配的键 (以防万一你修改了模型结构)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        # 更新权重
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        logger.info(f"✅ Loaded {len(pretrained_dict)}/{len(model_dict)} parameters from checkpoint.")

        # 2. [关键] 关于 Optimizer 和 Epoch 的处理
        # 情况 A: 如果是【断点续训】(比如训练了一半崩了)，你需要恢复 optimizer 和 start_epoch
        # 情况 B: 如果是【Fine-tuning / 第二阶段】(如你现在的情况)，我们通常只加载模型权重，
        #         使用新的 LR 和新的 Scheduler 从头开始优化，所以不要加载 optimizer。
        
        # 这里我写了一个自动判断：如果命令行指定的 start_epoch > 0，则认为是断点续训，加载优化器状态
        if args.start_epoch > 0 and 'optimizer_state_dict' in checkpoint and 'epoch' in checkpoint:
             logger.info(f"Resuming optimizer and scheduler states from epoch {checkpoint['epoch']}...")
             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
             # 如果 scheduler 也在 checkpoint 里，也可以加载
             # lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict']) 
             args.start_epoch = checkpoint['epoch'] + 1
        else:
             logger.info("🚀 Starting Fine-tuning: Resetting Optimizer and Epoch count.")
    # -----------------------------------------------------------
    # 7. 训练循环
    # -----------------------------------------------------------
    logger.info(f"Start training for {args.epochs} epochs.")
    
    best_r1_07 = 0.0 
    
   # [修改 1] 初始化三个最佳指标变量
    best_r1_07 = 0.0 
    best_r1_05 = 0.0       # 新增
    best_combined = 0.0    # 新增
    
    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, dataloader_train, optimizer, device, epoch)
        
        # 验证与最佳模型保存
        if dataloader_val is not None:
            metrics = evaluate(model, dataloader_val, device)
            
            # 获取当前指标
            current_r1_05 = metrics.get('R1@0.5', 0)
            current_r1_07 = metrics.get('R1@0.7', 0)
            current_combined = current_r1_05 + current_r1_07

            # -----------------------------------------------------------
            # 策略 A: 记录 R1@0.7 最高的模型 (保持原有逻辑)
            # -----------------------------------------------------------
            if current_r1_07 > best_r1_07:
                best_r1_07 = current_r1_07
                # 保存为 checkpoint_best.pth 以保持兼容，或者改为 checkpoint_best_r1_07.pth
                best_path = os.path.join(args.save_dir, "checkpoint_best_r1_07.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': metrics
                }, best_path)
                logger.info(f"⭐ New Best R1@0.7 Model! Score: {best_r1_07:.2f}%")

            # -----------------------------------------------------------
            # 策略 B: 记录 R1@0.5 最高的模型 [新增]
            # -----------------------------------------------------------
            if current_r1_05 > best_r1_05:
                best_r1_05 = current_r1_05
                best_path_05 = os.path.join(args.save_dir, "checkpoint_best_r1_05.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': metrics
                }, best_path_05)
                logger.info(f"⭐ New Best R1@0.5 Model! Score: {best_r1_05:.2f}%")

            # -----------------------------------------------------------
            # 策略 C: 记录 (R1@0.5 + R1@0.7) 综合最高的模型 [新增]
            # -----------------------------------------------------------
            if current_combined > best_combined:
                best_combined = current_combined
                best_path_combined = os.path.join(args.save_dir, "checkpoint_best_combined.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': metrics
                }, best_path_combined)
                logger.info(f"⭐ New Best Combined Model! Score: {best_combined:.2f} (R1@0.5={current_r1_05:.2f}, R1@0.7={current_r1_07:.2f})")

        # 保存最新的 Checkpoint (覆盖式，用于恢复训练)
        ckpt_path = os.path.join(args.save_dir, "checkpoint_last.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args
        }, ckpt_path)

if __name__ == '__main__':
    parser = get_args_parser()
    if not any(action.dest == 'text_encoder_type' for action in parser._actions):
        parser.add_argument('--text_encoder_type', default='clip', choices=['clip', 'glove'], help='Type of text encoder')
    if not any(action.dest == 'glove_path' for action in parser._actions):
        parser.add_argument('--glove_path', default='', type=str, help='Path to glove vectors')
    if not any(action.dest == 'clip_weight_path' for action in parser._actions):
        parser.add_argument('--clip_weight_path', default='', type=str, help='Path to pretrained CLIP weights')
    
    args = parser.parse_args()
    
    # [关键修复] 在 args 定义后，main 执行前设置默认参数
    if not hasattr(args, 'vocab_size'):
        args.vocab_size = 49408
    if not hasattr(args, 'rec_fw'):
        args.rec_fw = True
        
    main(args)