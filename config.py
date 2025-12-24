import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    # 路径设置
    parser.add_argument('--dataset_name', default='charades', type=str)
    parser.add_argument('--feature_dir', default='/path/to/features', type=str, help="视频特征存放目录")
    parser.add_argument('--clip_visual_path', default=None, type=str, help="辅助特征目录 (例如 CLIP 视觉特征)，进行拼接")# [新增] 第二个视觉特征目录
    parser.add_argument('--annotation_path', default='/path/to/train.json', type=str, help="标注文件路径")
    parser.add_argument('--save_dir', default='./checkpoints', type=str)
    parser.add_argument('--glove_path', default='', type=str, help='Path to glove vectors')
    parser.add_argument('--clip_weight_path', default='', type=str, help='Path to pretrained CLIP weights')
    parser.add_argument('--cached_text_path', default='./data/charades/charades_clip_text_features.pt', type=str)

    # 模型参数
    parser.add_argument('--text_encoder_type', default='clip', choices=['clip', 'glove', 'precomputed'], help='Type of text encoder')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--enc_layers', default=2, type=int) # MESM Encoder 层数
    parser.add_argument('--dec_layers', default=3, type=int) # BAM Decoder 层数
    parser.add_argument('--num_queries', default=10, type=int)
    parser.add_argument('--t_feat_dim', default=512, type=int) # text dim
    parser.add_argument('--v_feat_dim', default=1024, type=int) # video dim
    parser.add_argument('--max_v_l', default=75, type=int)     # 最大视频序列长度
    parser.add_argument('--max_q_l', default=32, type=int)     # 最大文本长度
    parser.add_argument('--feature_fps', default=1, type=int)
    parser.add_argument('--feature_stride', default=1, type=int)

    # 训练参数
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=30, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)

    # Loss 系数
    parser.add_argument('--span_loss_coef', default=10, type=float)
    parser.add_argument('--giou_loss_coef', default=4, type=float)
    parser.add_argument('--label_loss_coef', default=1, type=float)
    parser.add_argument('--aux_loss', default=True, type=bool)
    parser.add_argument('--span_loss_type', default="l1", type=str)
    parser.add_argument('--eos_coef', default=0.1, type=float, help="Relative classification weight of the no-object class")
    parser.add_argument('--lw_saliency', default=0.2, type=float, help="Weight for saliency loss")
    
    # MESM 重构任务参数
    parser.add_argument('--rec_fw', action='store_true', default=True, help="是否开启 Frame-Word 重构增强")
    parser.add_argument('--vocab_size', default=49408, type=int, help="CLIP 词表大小")
    parser.add_argument('--recfw_loss_coef', default=0.1, type=float, help="MESM 重构损失权重")
    
    parser.add_argument('--quality_loss_coef', default=1.0, type=float, help="IoU 质量预测损失权重 (BAM-DETR 核心)")

    return parser