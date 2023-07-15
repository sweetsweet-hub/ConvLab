import sys
sys.path.insert(0, '..')

import argparse
import os
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.backends.cudnn
import torch.utils.data.distributed

from utils.data_utils import get_loader
from networks.ConvLab_model import ConvLabModel
from optimizers.optimizer import build_optimizer
from optimizers.lr_scheduler import build_scheduler
from trainer import run_training

from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete
from monai.utils.enums import MetricReduction

parser = argparse.ArgumentParser()
# 检查点和目录设置
parser.add_argument("--logdir", type=str, default="logname_brats18")
parser.add_argument("--data_dir", type=str, default="/hy-tmp/BraTS2018/")  # 必须以分隔符"/"来结尾
parser.add_argument("--fold", type=int, default=4)
parser.add_argument("--json_list", type=str, default="BraTS2018_trainingdata_5folds.json")
parser.add_argument("--resume_ckpt", action="store_true")
parser.add_argument("--pretrained_dir", type=str, default="./runs/logname_brats18/")
parser.add_argument("--pretrained_model_name", type=str, default="model_best3.pt")
parser.add_argument("--save_checkpoint", type=bool, default=True)

# 批次大小
parser.add_argument("--max_epochs", type=int, default=800)
parser.add_argument("--val_every", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=3)  # in V100-32G GPU it's 3

# 学习率调度器
parser.add_argument('--lr_scheduler_name', type=str, default="CosineLRScheduler")
parser.add_argument('--warmup_lr', type=float, default=1e-6)
parser.add_argument('--min_lr', type=float, default=3e-6)
parser.add_argument('--warmup_epochs', type=int, default=100)
parser.add_argument('--lr_scheduler_warmup_prefix', type=bool, default=False)
parser.add_argument('--t_initial_multiplier', type=float, default=0.51)

# 优化器
parser.add_argument("--optimizer_name", type=str, default="AdamW")
parser.add_argument("--base_lr", type=int, default=3e-4)  # 根据batchsize调整 3->3e-4
parser.add_argument('--weight_decay', type=float, default=0.0015)
parser.add_argument('--optimizer_eps', type=float, default="1e-8")
parser.add_argument('--optimizer_beats', type=tuple, default=(0.9, 0.999))

# 数据转换
parser.add_argument("--a_min", type=float, default=-175.0)
parser.add_argument("--a_max", type=float, default=250.0)
parser.add_argument("--b_min", type=float, default=0.0)
parser.add_argument("--b_max", type=float, default=1.0)
parser.add_argument("--space_x", type=float, default=1.5)
parser.add_argument("--space_y", type=float, default=1.5)
parser.add_argument("--space_z", type=float, default=1.5)
parser.add_argument("--roi_x", type=int, default=128)
parser.add_argument("--roi_y", type=int, default=128)
parser.add_argument("--roi_z", type=int, default=128)
parser.add_argument("--RandFlipd_prob", type=float, default=0.3)
parser.add_argument("--RandRotate90d_prob", type=float, default=0.3)
parser.add_argument("--RandScaleIntensityd_prob", type=float, default=0.3)
parser.add_argument("--RandShiftIntensityd_prob", type=float, default=0.3)

# 模型设置
parser.add_argument("--in_channels", type=int, default=4)
parser.add_argument("--out_channels", type=int, default=3)
parser.add_argument('--drop_path_rate', type=float, default=0.3)
parser.add_argument("--norm_name", type=str, default="instance")
parser.add_argument("--smooth_dr", type=float, default=1e-6)
parser.add_argument("--smooth_nr", type=float, default=0.0)

# 推理设置
parser.add_argument("--sw_batch_size", type=int, default=8)
parser.add_argument("--infer_overlap", type=float, default=0.5)

# 分布式训练设置
parser.add_argument("--workers", type=int, default=6)
parser.add_argument("--distributed", action="store_true")
parser.add_argument("--world_size", type=int, default=1)
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--dist-url", type=str, default="tcp://127.0.0.1:23456")
parser.add_argument("--dist-backend", type=str, default="nccl")

seed = 4492
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def main():  # 起始行107
    args = parser.parse_args()
    args.amp = True
    args.logdir = "./runs/" + args.logdir
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False

    # ------------------------ utils/data_utils.py ------------------------ #
    # ------------------------ 数据获取、预处理和生成加载器 ------------------------ #
    loader = get_loader(args)

    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    print("len(loader[0]=)", len(loader[0]), "\tlen(loader[1])=", len(loader[1]))
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    pretrained_dir = args.pretrained_dir  # 预训练检查点目录

    # ------------------------ 定义模型 ------------------------ #
    model = ConvLabModel(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        norm_name=args.norm_name,
        dropout_path_rate=args.drop_path_rate,
        depths=[3, 3, 5, 3],
        dims=[48, 96, 192, 384]
    )

    if args.resume_ckpt:  # 从微调好的权重加载模型
        model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))["state_dict"]
        model.load_state_dict(model_dict)
        print("Use pretrained weights")

    # ------------------------ 定义训练/验证时系列函数 ------------------------ #
    # 计算Dice损失和交叉熵损失，并返回这两个损失的加权和。
    # to_onehot_y：是否使用从“input”（“input.shape[1]”）推断的类数将“target”转换为单热格式。默认值为False。
    # softmax：如果为True，则将softmax函数应用于预测，仅由“DiceLoss”使用，无需为“CrossEntropyLoss”指定激活函数。
    # squared_pred：分母中是否使用目标和预测的平方版本。
    # smooth_nr：添加到分子以避免为零的小常数。0.0
    # smooth_dr：添加到分母以避免nan的小常数。1e-6
    dice_loss = DiceLoss(
        to_onehot_y=False, sigmoid=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
    )
    #
    #
    post_sigmoid = Activations(sigmoid=True)
    #
    post_pred = AsDiscrete(argmax=False, logit_thresh=0.5)
    # 计算两个张量之间的平均Dice分数。它可以支持多类和多标签任务。
    # include_background：是否跳过预测输出的第一信道上的Dice计算。
    # reduction：定义度量的缩减模式，将仅对“非nan”值应用缩减
    # 是否返回“not_nans”计数，如果为True，则aggregate（）返回（metric，not_nans）。这里“not_nans”计算度量的非nan数，因此其形状等于度量的形状。
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    # partial：部分应用给定参数和关键字的新函数。
    # sliding_window_inference：使用“预测器”对“输入”进行滑动窗口推断。
    # roi_size：用于推断的空间窗口大小。[96, 96, 96]
    # sw_batch_size:运行窗口切片的批处理大小。1
    # predictor：给定NCHW[D]形状的输入张量“patch_data”，函数调用“predictor（patch_data）”的输出应为张量、元组或具有张量值的字典。
    # 元组或dict值中的每个输出应具有相同的batch_size，即NM'H'W'[D']；其中，H'W'[D']表示输出patch的空间大小，M是输出通道的数量，N是“sw_batch_size”。
    # overlap：扫描之间的重叠量。0.5
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    start_epoch = 0

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

    # /-------- optimizer.py --------/ #
    # /-------- 使用set_weight_decay()将需要权重衰减和不需要权重衰减的参数分开保存获得parameters --------/ #
    # /-------- 使用optim.AdamW()获得optimizer --------/ #
    optimizer = build_optimizer(args, model)

    # /-------- lr_scheduler.py --------/ #
    # /-------- 使用timm.scheduler.cosine_lr.CosineLRScheduler()获得具有warmup的lr_scheduler --------/ #
    lr_scheduler = build_scheduler(args, optimizer, len(loader[0]))

    # ------------------------ trainer.py ------------------------ #
    # ------------------------ 开始训练 ------------------------ #
    run_training(args=args, model=model, train_loader=loader[0], val_loader=loader[1],
                 optimizer=optimizer, scheduler=lr_scheduler, loss_func=dice_loss, acc_func=dice_acc,
                 start_epoch=start_epoch, model_inferer=model_inferer,
                 post_sigmoid=post_sigmoid, post_pred=post_pred, semantic_classes=["Dice_Val_TC", "Dice_Val_WT", "Dice_Val_ET"])  # 结束行223


if __name__ == "__main__":
    main()
