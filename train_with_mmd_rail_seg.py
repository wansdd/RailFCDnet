# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train a YOLOv5 model on a custom dataset. Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""



import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from utils.loss import mmd
try:
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm
from itertools import cycle

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_amp,
    check_dataset,
    check_file,
    check_git_info,
    check_git_status,
    check_img_size,
    check_requirements,
    check_suffix,
    check_yaml,
    colorstr,
    get_latest_run,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    methods,
    one_cycle,
    print_args,
    print_mutation,
    strip_optimizer,
    yaml_save,
    # generate_mask_from_labels,
)
from utils.loggers import LOGGERS, Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import (
    ComputeLoss,
    # ImageLevelLoss,
    # InstanceLevelLoss,
    # ConsensusLoss,
    # ConsistencyLoss
)
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    select_device,
    smart_DDP,
    smart_optimizer,
    smart_resume,
    torch_distributed_zero_first,
)

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
GIT_INFO = check_git_info()




def smooth_mask(mask, device):
    """对mask进行高斯平滑处理
    Args:
        mask: 输入的mask张量
        device: 计算设备
    Returns:
        smoothed_mask: 平滑处理后的mask
    """
    mask = mask[:,0,].squeeze(0)
    
    # 高斯滤波参数
    kernel_size = 15
    sigma = 5
    mask = mask.float()

    # 创建高斯核
    gaussian_kernel = torch.zeros((kernel_size, kernel_size), device=device)
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            gaussian_kernel[i, j] = torch.exp(torch.tensor(-(x*x + y*y)/(2*sigma*sigma), device=device))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    # 形态学闭运算参数
    kernel_size_morph = 5
    kernel_morph = torch.ones(kernel_size_morph, kernel_size_morph, device=device)
    kernel_morph = kernel_morph.view(1, 1, kernel_size_morph, kernel_size_morph)

    # 处理维度并移到设备
    mask = mask.to(device)
    gaussian_kernel = gaussian_kernel.to(device)

    # 执行形态学闭运算
    mask = torch.nn.functional.max_pool2d(mask.unsqueeze(0), kernel_size_morph, stride=1, padding=kernel_size_morph//2)
    mask = torch.nn.functional.conv2d(mask, kernel_morph/kernel_morph.sum(), padding=kernel_size_morph//2)
    
    # 执行高斯平滑
    smoothed_mask = torch.nn.functional.conv2d(mask, gaussian_kernel, padding=kernel_size//2)

    # 归一化处理
    smoothed_mask = (smoothed_mask - smoothed_mask.min()) / (smoothed_mask.max() - smoothed_mask.min() + 1e-6)

    return smoothed_mask

def train(hyp, opt, device, callbacks):
    """
    Train a YOLOv5 model on a custom dataset using specified hyperparameters, options, and device, managing datasets,
    model architecture, loss computation, and optimizer steps.

    Args:
        hyp (str | dict): Path to the hyperparameters YAML file or a dictionary of hyperparameters.
        opt (argparse.Namespace): Parsed command-line arguments containing training options.
        device (torch.device): Device on which training occurs, e.g., 'cuda' or 'cpu'.
        callbacks (Callbacks): Callback functions for various training events.

    Returns:
        None

    Models and datasets download automatically from the latest YOLOv5 release.

    Example:
        Single-GPU training:
        ```bash
        $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
        $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch
        ```

        Multi-GPU DDP training:
        ```bash
        $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights
        yolov5s.pt --img 640 --device 0,1,2,3
        ```

        For more usage details, refer to:
        - Models: https://github.com/ultralytics/yolov5/tree/master/models
        - Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        - Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.weights,
        opt.single_cls,
        opt.evolve,
        opt.data,
        opt.cfg,
        opt.resume,
        opt.noval,
        opt.nosave,
        opt.workers,
        opt.freeze,
    )
    callbacks.run("on_pretrain_routine_start")

    # ===== 消融: 掩码类型/ROI 经环境变量传给 dataloader (读取 _mask 后缀) =====
    _mask_suffix = {"both": "_both_mask", "object": "_obj_mask", "perimeter": "_mask", "none": "_mask"}
    _mask_dirname = {"perimeter-only": "p_masks", "object-only": "o_masks"}
    os.environ["RCD_MASK_SUFFIX"] = _mask_suffix.get(getattr(opt, "mask_type", "perimeter"), "_mask")
    os.environ["RCD_MASK_DIRNAME"] = _mask_dirname.get(getattr(opt, "mask_type", "perimeter"), "")
    os.environ["RCD_ROI_CROP"] = "1" if getattr(opt, "roi_crop", False) else "0"

    use_middle_domain = getattr(opt, "use_middle_domain", False)
    merge_middle_into_source = getattr(opt, "merge_middle_into_source", False)
    if use_middle_domain and merge_middle_into_source:
        raise ValueError("--use-middle-domain and --merge-middle-into-source are mutually exclusive")
    batch_size_t = 1
    batch_size_m = 1 if use_middle_domain else 0
    batch_size_s = batch_size - batch_size_t - batch_size_m
    if batch_size_s < 1:
        raise ValueError(f"batch-size must be at least {2 + batch_size_m} for the configured domains")


    # Directories
    w = save_dir / "weights"  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / "last.pt", w / "best.pt"

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings
    # if not evolve:
    # 保存所有设置参数
    yaml_save(save_dir / "hyp.yaml", hyp)
    yaml_save(save_dir / "opt.yaml", vars(opt))

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        include_loggers = list(LOGGERS)
        if getattr(opt, "ndjson_console", False):
            include_loggers.append("ndjson_console")
        if getattr(opt, "ndjson_file", False):
            include_loggers.append("ndjson_file")

        loggers = Loggers(
            save_dir=save_dir,
            weights=weights,
            opt=opt,
            hyp=hyp,
            logger=LOGGER,
            include=tuple(include_loggers),
        )

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Config
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != "cpu"
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    train_path = []
    val_path = []
    data_dict = []
    count = 0
    with torch_distributed_zero_first(LOCAL_RANK):
        for p in data if isinstance(data, list) else [data]:
            data_dict.append(check_dataset(p))  # check if None
            train_path.append(data_dict[count]["train"])
            val_path.append(data_dict[count]["val"])
            count += 1
    if len(data_dict) != 2:
        raise ValueError(f"expected exactly two --data files (source and target), got {len(data_dict)}")
    if merge_middle_into_source:
        middle_path = getattr(opt, "middle_data", None)
        if not middle_path:
            raise ValueError("--middle-data is required when --merge-middle-into-source is enabled")
        train_path[0] = [train_path[0], middle_path]
        LOGGER.info(f"Middle-domain source-pool mode: source training paths={train_path[0]}")
    nc = 1 if single_cls else int(data_dict[0]["nc"])  # number of classes
    names = (
        {0: "item"}
        if single_cls and len(data_dict[0]["names"]) != 1
        else data_dict[0]["names"]
    )  # class names


    names = {0: "item"} if single_cls and len(data_dict[0]["names"]) != 1 else data_dict[0]["names"]  # class names
    # is_coco = isinstance(val_path, str) and val_path.endswith("coco/val2017.txt")  # COCO dataset
    is_coco = False

    # Model
    check_suffix(weights, ".pt")  # check weights
    pretrained = weights.endswith(".pt")
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak
        model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
        exclude = ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []  # exclude keys
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
    amp = check_amp(model)  # check AMP

    # Freeze
    # freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    # Freeze backbone
    # freeze= [f"model.{x}." for x in (range(24))] 
    freeze = [f"model.{x}." for x in []]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            LOGGER.info(f"freezing {k}")
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)            # accumulate loss before optimizing
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])

    opt.cos_lr = True
    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
    else:
        def lf(x):
            """Linear learning rate scheduler function with decay calculated by epoch proportion."""
            return (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume
    best_fitness, start_epoch = 0.0, 0
    best_fitness0 = 0.0
    fi = 0.0  # 初始化, 防止 --noval 时非最终轮跳过验证导致保存块引用未定义的 fi
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning(
            "WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n"
            "See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started."
        )
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info("Using SyncBatchNorm()")

    # Trainloader  train_loader0 是源域数据集 train_loader1 是小样本目标域数据集
    train_loader = []
    datasets = []
    labels = []
    val_loader = []
    batch_size_list = [batch_size_s, batch_size_t]
    # 创建source domain data loader
    
    for i in range(len(data_dict)):
        mask_label = True
        train_loader0, dataset0 = create_dataloader(
            train_path[i],
            imgsz,
            batch_size_list[i] // WORLD_SIZE,
            gs,
            single_cls=single_cls,
            hyp=hyp,
            augment=True,
            cache=None if opt.cache == "val" else opt.cache,
            rect=opt.rect,
            rank=LOCAL_RANK,
            workers=workers,
            image_weights=opt.image_weights,
            quad=opt.quad,
            prefix=colorstr("train: "),
            shuffle=True,
            seed=opt.seed,
            mask=mask_label,            
            roi_crop=opt.roi_crop,
            drop_last=(i == 0 and getattr(opt, "drop_incomplete_source_batch", False)),
        )
    

        labels0 = np.concatenate(dataset0.labels, 0)####给定轴上进行拼接
        mlc0 = int(labels0[:, 0].max())  # max label class
        assert (
                mlc0 < nc
        ), f"Label class {mlc0} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"
        train_loader.append(train_loader0)
        datasets.append(dataset0)
        labels.append(labels0)

        # Process 0
        if RANK in {-1, 0}:
            val_loader.append(
                create_dataloader(
                    val_path[i],
                    imgsz,
                    batch_size // WORLD_SIZE * 2,
                    gs,
                    single_cls,
                    hyp=hyp,
                    cache=None if noval else opt.cache,
                    rect=True,
                    rank=-1,
                    pad=0.5,
                    workers=workers * 2,
                    prefix=colorstr("val: "),
                    quad=False,
                    mask=True,            
                    roi_crop=opt.roi_crop,
                    # mask=False if i == 0 else True,            
                )[0]
            )

    middle_loader = None
    middle_dataset = None
    if use_middle_domain:
        middle_path = getattr(opt, "middle_data", None)
        if not middle_path:
            raise ValueError("--middle-data is required when --use-middle-domain is enabled")
        middle_loader, middle_dataset = create_dataloader(
            middle_path,
            imgsz,
            batch_size_m // WORLD_SIZE,
            gs,
            single_cls=single_cls,
            hyp=hyp,
            augment=True,
            cache=None if opt.cache == "val" else opt.cache,
            rect=opt.rect,
            rank=LOCAL_RANK,
            workers=workers,
            image_weights=False,
            quad=False,
            prefix=colorstr("middle train: "),
            shuffle=False,
            mask=True,
            roi_crop=opt.roi_crop,
        )
        middle_label_sets = [label for label in middle_dataset.labels if len(label)]
        if not middle_label_sets:
            raise ValueError(f"middle-domain dataset has no detection labels: {middle_path}")
        middle_labels = np.concatenate(middle_label_sets, 0)
        middle_mlc = int(middle_labels[:, 0].max())
        assert middle_mlc < nc, f"Middle-domain label class {middle_mlc} exceeds nc={nc}"
        LOGGER.info(
            f"Middle-domain training enabled: source={batch_size_s}, middle={batch_size_m}, "
            f"target={batch_size_t} per full batch; {len(middle_dataset)} fixed-order middle samples"
        )

        if not resume:
            if not opt.noautoanchor:
                # for j in range(len(datasets)):
                for j in range(1):
                    check_anchors(
                        datasets[j], model=model, thr=hyp["anchor_t"], imgsz=imgsz
                    )  # run AutoAnchor
            model.half().float()  # pre-reduce anchor precision

        callbacks.run("on_pretrain_routine_end", labels0, names)


    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp["box"] *= 3 / nl  # scale to layers
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp["label_smoothing"] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    # vt使用目标域的类别均衡，我使用源域的类别均衡
    # model.class_weights = labels_to_class_weights(datasets[0].labels, nc).to(device) # * nc  # attach class weights
    # model.class_weights = torch.tensor([0.4, 0.6], device=device)
    # model.class_weights = torch.tensor([0.8, 0.2], device=device)
    model.names = names

    # Start training
    t0 = time.time()
    nb = len(train_loader[0])  # number of batches
    nw = max(round(hyp["warmup_epochs"] * nb), 1000)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    # 消融配置传给损失: 对齐层子集 / 是否启用 AAM / 是否启用 PGM分割损失
    de_parallel(model).ablation = {
        "align_levels": getattr(opt, "align_levels", [0, 1, 2]),
        "use_aam": not getattr(opt, "no_aam", False),
        "use_pgm": getattr(opt, "mask_type", "perimeter") != "none" and not getattr(opt, "roi_crop", False),
        "alpha": getattr(opt, "alpha", 0.1),
        "beta": getattr(opt, "beta", 0.1),
        "align_mode": getattr(opt, "align_mode", "intra"),
        "seg_dice": getattr(opt, "seg_dice", False),
        "aam_roi_size": getattr(opt, "aam_roi_size", 7),
        "aam_detach_target": getattr(opt, "aam_detach_target", False),
        "align_method": getattr(opt, "align_method", "mmd"),
        "proto_weight": getattr(opt, "proto_weight", 0.1),
        "adv_margin": getattr(opt, "adv_margin", 1.0),
        "ema_momentum": getattr(opt, "ema_momentum", 0.9),
    }
    compute_loss = ComputeLoss(model)  # init loss class
    base_use_aam = compute_loss.use_aam  # AAM warmup: 记录初始开关, 逐 epoch 据此与 warmup 阈值切换
    callbacks.run("on_train_start")
    LOGGER.info(
        f"Image sizes {imgsz} train, {imgsz} val\n"
        f"Using {train_loader[0].num_workers * WORLD_SIZE} dataloader workers\n"
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f"Starting training for {epochs} epochs..."
    )
    middle_iter = iter(middle_loader) if middle_loader is not None else None
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        # AAM warmup: 前 opt.aam_warmup_epochs 轮关闭域对齐(纯检测), 之后开启多层实例级按类别跨域对齐
        aam_on = base_use_aam and (epoch >= getattr(opt, "aam_warmup_epochs", 0))
        if aam_on != compute_loss.use_aam:
            LOGGER.info(f"[AAM] epoch {epoch}: 域对齐 {'开启' if aam_on else '关闭'}")
        compute_loss.use_aam = aam_on
        # adv 对齐: GRL λ 随对抗阶段进度 p 渐增 (DANN: 2/(1+e^{-10p})-1)
        if getattr(opt, "align_method", "mmd") == "adv":
            _wu = getattr(opt, "aam_warmup_epochs", 0)
            _p = min(1.0, max(0.0, (epoch - _wu) / max(1, epochs - _wu)))
            compute_loss.adv_lambda = 2.0 / (1.0 + math.exp(-10.0 * _p)) - 1.0
###########################################之前的版本#####################################################
        # # 计算所有的target的 backbone_feature,每轮epoch计算一次
        # model.eval()
        # with torch.no_grad():
        #     pbar_target = enumerate(train_loader[1]) # 这里把少量的样本当成目标域吧
        #     target_feature = torch.tensor([]).to(device) # 不确定要不要.to(device)
        #     imgs_t = torch.tensor([]).to(device)
        #     for i, (imgs, targets, paths, space, mask) in pbar_target: # 这里的imgs是torch.tensor  这段代码没有问题
        #         imgs_t_i = imgs.to(device, non_blocking=True).float() / 255.0
        #         imgs_feature = model(imgs_t_i)[1].detach().mean(3).mean(2) # imgs_feature [B, 1280]
        #         imgs_t = torch.cat((imgs_t, imgs_t_i))
        #         target_feature = torch.cat((target_feature, imgs_feature))
        #                 # 初始化三个空 tensor 来存储三个不同 feature 层的累积结果
        #     feature_t = target_feature.clone() # detach #target域特征的备份
        #     torch.cuda.empty_cache()

#############################################最终版本####################################
        model.eval()
        with torch.no_grad():
            # 使用列表存储每个特征层的批次结果
            features_list = [[] for _ in range(3)]
            t_boxes_list = []   # D2: 同步累计目标域目标框(全局图像下标), 用于实例级 MMD
            t_offset = 0
            # 遍历目标域数据
            for i, (imgs, targets_t, paths, space, mask) in enumerate(train_loader[1]):
                imgs_t_i = imgs.to(device, non_blocking=True).float() / 255.0
                # 获取特征 - 不需要 .detach()
                imgs_features = model(imgs_t_i)[1]
                # 将每个特征层的批次结果添加到列表中
                for j in range(3):
                    features_list[j].append(imgs_features[j])
                if targets_t is not None and len(targets_t):
                    tb = targets_t.clone()
                    tb[:, 0] = tb[:, 0] + t_offset      # 批内下标 -> 全局下标
                    t_boxes_list.append(tb)
                t_offset += imgs_t_i.shape[0]
            # 一次性拼接所有批次
            accumulated_features = [torch.cat(feats, dim=0) for feats in features_list]
        feature_t = accumulated_features  # 保存目标域特征(no-grad, 作 fallback)
        # 取消停梯度模式: 缓存目标图张量, 供每 batch 带梯度重算目标特征(源/目标对称)
        target_imgs_cache = None
        if getattr(opt, "aam_target_grad", False):
            with torch.no_grad():
                _t_imgs = []
                for _i, (_imgs, *_rest) in enumerate(train_loader[1]):
                    _t_imgs.append(_imgs.to(device, non_blocking=True).float() / 255.0)
                target_imgs_cache = torch.cat(_t_imgs, 0)
        # D2: 把累计的目标框写入损失器(整个 epoch 复用)
        compute_loss.t_boxes = torch.cat(t_boxes_list, 0) if t_boxes_list else None
        # 释放缓存
        torch.cuda.empty_cache()

        
        callbacks.run("on_train_epoch_start")
        model.train()
        mloss = torch.zeros(5, device=device)  # mean losses: box, obj, cls, mmd, fg(分割)

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(datasets[0].labels, nc=nc, class_weights=cw)  # image weights
            datasets[0].indices = random.choices(range(datasets[0].n), weights=iw, k=datasets[0].n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader[0]) # pbar 最终整合了整个dataloader
        few_shot = list(enumerate(train_loader[1]))

        # LOGGER.info(("\n" + "%11s" * 10) % ("Epoch", "GPU_mem", "box_loss", "obj_loss", "cls_loss", "img_loss", "insta_loss", "conse_loss", "Instances", "Size"))
        # LOGGER.info(("\n" + "%11s" * 8) % ("Epoch", "GPU_mem", "box_loss", "obj_loss", "cls_loss", "lmmd_loss", "Instances", "Size"))
        LOGGER.info(("\n" + "%11s" * 9) % ("Epoch", "GPU_mem", "box_loss", "obj_loss", "cls_loss", "lmmd_loss", "fg_loss","Instances", "Size"))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        optimizer.zero_grad()
        # for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
        for i, data in pbar:  # batch -------------------------------------------------------------
            loss = 0
            callbacks.run("on_train_batch_start")
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs0, targets, paths, scale0, ms = data
            i_tar, (imgs1, targets_tar, paths_tar, scale1, mask_tar) = random.sample(few_shot ,1)[0]
            middle_count = 0

            supervised_imgs = [imgs0]
            supervised_targets = [targets]
            supervised_masks = [ms]
            supervised_paths = list(paths)
            target_offset = imgs0.shape[0]

            if middle_iter is not None:
                try:
                    imgs_mid, targets_mid, paths_mid, scale_mid, mask_mid = next(middle_iter)
                except StopIteration:
                    middle_iter = iter(middle_loader)
                    imgs_mid, targets_mid, paths_mid, scale_mid, mask_mid = next(middle_iter)
                targets_mid = targets_mid.clone()
                targets_mid[:, 0] += target_offset
                supervised_imgs.append(imgs_mid)
                supervised_targets.append(targets_mid)
                supervised_masks.append(mask_mid)
                supervised_paths.extend(paths_mid)
                target_offset += imgs_mid.shape[0]
                middle_count = imgs_mid.shape[0]

            targets_tar = targets_tar.clone()
            targets_tar[:, 0] += target_offset
            supervised_imgs.append(imgs1)
            supervised_targets.append(targets_tar)
            supervised_masks.append(mask_tar)
            supervised_paths.extend(paths_tar)

            # AAM treats the supervised middle domain as source-side data; target remains the final image.
            compute_loss.bs_s = target_offset
            imgs = torch.cat(supervised_imgs, 0)
            targets = torch.cat(supervised_targets, 0)
            ms = torch.cat(supervised_masks, 0).to(device, non_blocking=True).float()
            paths = supervised_paths

            if getattr(opt, "drop_incomplete_source_batch", False):
                source_count = imgs0.shape[0]
                target_count = imgs1.shape[0]
                expected_middle = 1 if use_middle_domain else 0
                expected_total = batch_size_s + expected_middle + batch_size_t
                actual_total = source_count + middle_count + target_count
                if (source_count, middle_count, target_count, actual_total) != (
                    batch_size_s, expected_middle, batch_size_t, expected_total
                ):
                    raise RuntimeError(
                        "domain batch composition mismatch: "
                        f"source={source_count}/{batch_size_s}, "
                        f"middle={middle_count}/{expected_middle}, "
                        f"target={target_count}/{batch_size_t}, "
                        f"total={actual_total}/{expected_total}"
                    )
                if epoch == start_epoch and i == 0 and RANK in {-1, 0}:
                    LOGGER.info(
                        f"Strict domain batch verified: source_or_pool={source_count}, "
                        f"middle={middle_count}, target={target_count}, total={actual_total}"
                    )
         
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
            # batch_size = batch_size_t + batch_size_s

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])


            # Forward
            with torch.cuda.amp.autocast(amp):
                # 输入数据mask 与输入 images 一一对应
                pred, feature_s, mask = model(imgs)  # forward (mask=PGM预测前景概率图)
                # 取消目标域停梯度: 每 batch 对目标图带梯度重新前向, 与源域相同操作(对称对齐)
                if getattr(opt, "aam_target_grad", False) and compute_loss.use_aam and target_imgs_cache is not None:
                    feat_t_in = model(target_imgs_cache)[1]
                else:
                    feat_t_in = feature_t
                loss, loss_items = compute_loss(pred, targets.to(device), feature_s, feat_t_in,(mask, ms))  # loss scaled by batch_size
                # mask 输出 推理图 save_pred_mask(mask,path)

                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.0

            # Backward
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f"{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * 7) %(f"{epoch}/{epochs - 1}", mem, *mloss, targets.shape[0], imgs.shape[-1])
                )
                # pbar.total = len(train_loader[0])  # Ensure the progress bar shows the total number of batches
                callbacks.run("on_train_batch_end", model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------
        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in {-1, 0}:
            # mAP
            callbacks.run("on_train_epoch_end", epoch=epoch)
            ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            
            if not noval or final_epoch:  # Calculate mAP
                # 跳过源域(domain0)验证: 源域 val 不参与选最优/早停, 仅曾用于日志, 去掉省每epoch一次源域推理
                for domain in range(1, len(val_loader)):
                    mask = True
                    results, maps, _ = validate.run(
                        data_dict[domain],
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        half=amp,
                        model=ema.ema,
                        single_cls=single_cls,
                        dataloader=val_loader[domain],
                        save_dir=save_dir,
                        plots=False,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                        mask = True,
                    )
                
                    if domain == 0:
                        # Update best mAP
                        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                        # stop = stopper(epoch=epoch, fitness=fi)  # early stop check
                        if fi > best_fitness0:
                            best_fitness0 = fi
                        log_vals = list(loss_items) + list(results) + lr
                        callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness0, fi, domain=0)
                    if domain == 1:
                        # Update best mAP
                        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                        stop = stopper(epoch=epoch, fitness=fi)  # early stop check
                        # adv warmup 期内不早停: 冻结 patience 时钟到 warmup 结束, 保证对齐确实被训练到
                        if epoch < getattr(opt, "aam_warmup_epochs", 0):
                            stopper.best_epoch = epoch
                            stop = False
                        if fi > best_fitness:
                            best_fitness = fi
                        log_vals = list(loss_items) + list(results) + lr

                        callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness, fi, domain=1)
            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(de_parallel(model)).half(),
                    "ema": deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "opt": vars(opt),
                    "git": GIT_INFO,  # {remote, branch, commit} if a git repo
                    "date": datetime.now().isoformat(),
                }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f"epoch{epoch}.pt")
                del ckpt
                callbacks.run("on_model_save", last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        LOGGER.info(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f"\nValidating {f}...")
                    results, _, _ = validate.run(
                        data_dict[1],
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader[-1],
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                        mask=True,
                    )  # val best model with plots
                    if is_coco:
                        callbacks.run("on_fit_epoch_end", list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run("on_train_end", last, best, epoch, results)

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    """
    Parse command-line arguments for YOLOv5 training, validation, and testing.

    Args:
        known (bool, optional): If True, parses known arguments, ignoring the unknown. Defaults to False.

    Returns:
        (argparse.Namespace): Parsed command-line arguments containing options for YOLOv5 execution.

    Example:
        ```python
        from ultralytics.yolo import parse_opt
        opt = parse_opt()
        print(opt)
        ```

    Links:
        - Models: https://github.com/ultralytics/yolov5/tree/master/models
        - Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        - Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="./yolov5x.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="models/yolov5x_seg.yaml", help="model.yaml path")
    parser.add_argument("--data", help="dataset.yaml path", action="append")
    parser.add_argument("--use-middle-domain", action="store_true",
                        help="add one supervised middle-domain image to each training batch")
    parser.add_argument("--merge-middle-into-source", action="store_true",
                        help="merge middle-domain images into the shuffled source training pool")
    parser.add_argument("--middle-data", type=str, default=None,
                        help="middle-domain training image directory or image-list txt; never used for val/test")
    parser.add_argument("--drop-incomplete-source-batch", action="store_true",
                        help="drop the final short source/source-pool batch to keep the requested domain ratio exact")
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-high.yaml", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--roi-crop", action="store_true", help="crop images to per-image ROI rectangles")
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    parser.add_argument("--noautoanchor", action="store_true", help="disable AutoAnchor")
    parser.add_argument("--noplots", action="store_true", help="save no plot files")
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")
    parser.add_argument(
        "--evolve_population", type=str, default=ROOT / "data/hyps", help="location for loading population"
    )
    parser.add_argument("--resume_evolve", type=str, default=None, help="resume evolve from last generation")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT / "runs/train", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--patience", type=int, default=10, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")

    # Logger arguments
    parser.add_argument("--entity", default=None, help="Entity")
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")
    parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")

    # NDJSON logging
    parser.add_argument("--ndjson-console", action="store_true", help="Log ndjson to console")
    parser.add_argument("--ndjson-file", action="store_true", help="Log ndjson to file")

    # ===== 消融实验旋钮 (Ablation) =====
    parser.add_argument("--align-levels", type=int, nargs="+", default=[0, 1, 2],
                        help="AAM 对齐使用的特征层子集: 0=shallow(l4) 1=mid(l6) 2=deep(l9) (论文 Table 4)")
    parser.add_argument("--no-aam", action="store_true", help="关闭 AAM 域对齐(β=0) (论文 Table 2)")
    parser.add_argument("--aam-warmup-epochs", type=int, default=0,
                        help="AAM warmup: 前 N 轮不启用域对齐(纯检测), 第 N 轮起开启多层实例级按类别跨域对齐")
    parser.add_argument("--aam-target-grad", action="store_true",
                        help="取消目标域实例停梯度: 每 batch 对目标域重新带梯度前向, 源/目标对称对齐")
    parser.add_argument("--mask-type", type=str, default="perimeter",
                        choices=["both", "object", "perimeter", "perimeter-only", "object-only", "none"],
                        help="PGM 掩码类型(L_bce 的GT): perimeter-only/object-only 读取同级 p_masks/o_masks; "
                             "perimeter/both/object/none 保留原路径规则")
    # 注: --roi-crop 已在上方定义(复用现有 ROI 裁剪机制, 已接入 create_dataloader)
    parser.add_argument("--alpha", type=float, default=0.1, help="L_bce(分割) 权重 α (论文默认0.1; Table 5 搜索)")
    parser.add_argument("--beta", type=float, default=0.1, help="L_adaptive_MMD 权重 β (论文默认0.1; Table 5 搜索)")
    parser.add_argument("--align-mode", type=str, default="intra", choices=["intra", "can"],
                        help="D2 对齐模式: intra=原类内MMD(默认) / can=CAN风格(跨域同类拉近+同域紧致+异类推远)")
    parser.add_argument("--aam-roi-size", type=int, default=7, help="AAM 实例 ROIAlign output_size(默认7)")
    parser.add_argument("--aam-detach-target", action="store_true", help="AAM 对齐时关闭目标域特征梯度(非对称)")
    parser.add_argument("--align-method", type=str, default="mmd", choices=["mmd", "adv"],
                        help="对齐方法: mmd=原 MMD(默认) / adv=实例级域判别器+GRL非对称对抗+EMA类原型")
    parser.add_argument("--proto-weight", type=float, default=0.1, help="adv: EMA 类原型约束 L_proto 权重")
    parser.add_argument("--adv-margin", type=float, default=1.0, help="adv: 异类原型排斥 hinge margin")
    parser.add_argument("--ema-momentum", type=float, default=0.9, help="adv: 目标类原型 EMA 动量")
    parser.add_argument("--seg-dice", action="store_true", help="PGM 分割损失加 Dice(BCE+Dice), 锐化周界区域边界")

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    """
    Runs the main entry point for training or hyperparameter evolution with specified options and optional callbacks.

    Args:
        opt (argparse.Namespace): The command-line arguments parsed for YOLOv5 training and evolution.
        callbacks (ultralytics.utils.callbacks.Callbacks, optional): Callback functions for various training stages.
            Defaults to Callbacks().

    Returns:
        None

    Note:
        For detailed usage, refer to:
        https://github.com/ultralytics/yolov5/tree/master/models
    """
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements(ROOT / "requirements.txt")

    # Load hyperparameters from opt.hyp YAML file
    
    hyp = check_yaml(opt.hyp)
    with open(opt.hyp, errors="ignore") as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    # Extract learning rate and other hyperparameters
    # lr = hyp.get('lr0', 0.01)  # default learning rate
    # lrf = hyp.get('lrf', 0.1)  # default final learning rate factor
    # momentum = hyp.get('momentum', 0.937)  # default momentum
    # weight_decay = hyp.get('weight_decay', 0.0005)  # default weight decay
    # warmup_epochs = hyp.get('warmup_epochs', 3.0)  # default warmup epochs
    # warmup_momentum = hyp.get('warmup_momentum', 0.8)  # default warmup initial momentum
    # warmup_bias_lr = hyp.get('warmup_bias_lr', 0.1)  # default warmup initial bias learning rate
    
    # Resume (from specified or most recent last.pt)
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / "opt.yaml"  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors="ignore") as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location="cpu")["opt"]
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = "", str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        opt.cfg, opt.hyp, opt.weights, opt.project = (
            check_yaml(opt.cfg),
            check_yaml(opt.hyp),
            str(opt.weights),
            str(opt.project),
        )  # checks
        opt.data = [check_file(opt.data[i]) for i in range(len(opt.data))]  # check all ,
        if getattr(opt, "use_middle_domain", False) or getattr(opt, "merge_middle_into_source", False):
            middle_path = Path(opt.middle_data).resolve() if opt.middle_data else None
            if middle_path is None or not middle_path.exists():
                raise FileNotFoundError(f"middle-domain training data not found: {opt.middle_data}")
            opt.middle_data = str(middle_path)
        assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified"
        if opt.evolve:
            if opt.project == str(ROOT / "runs/train"):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / "runs/evolve")
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == "cfg":
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        # 创建保存路径

        # opt.save_dir = str(increment_path(Path(opt.project) / str(opt.name + "_lr"+str(lr)+ "_"), exist_ok=opt.exist_ok))
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = "is not compatible with YOLOv5 Multi-GPU DDP training"
        assert not opt.image_weights, f"--image-weights {msg}"
        assert not opt.evolve, f"--evolve {msg}"
        assert opt.batch_size != -1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(seconds=10800)
        )

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (including this hyperparameter True-False, lower_limit, upper_limit)
        meta = {
            "lr0": (False, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            "lrf": (False, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (False, 0.6, 0.98),  # SGD momentum/Adam beta1
            "weight_decay": (False, 0.0, 0.001),  # optimizer weight decay
            "warmup_epochs": (False, 0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (False, 0.0, 0.95),  # warmup initial momentum
            "warmup_bias_lr": (False, 0.0, 0.2),  # warmup initial bias lr
            "box": (False, 0.02, 0.2),  # box loss gain
            "cls": (False, 0.2, 4.0),  # cls loss gain
            "cls_pw": (False, 0.5, 2.0),  # cls BCELoss positive_weight
            "obj": (False, 0.2, 4.0),  # obj loss gain (scale with pixels)
            "obj_pw": (False, 0.5, 2.0),  # obj BCELoss positive_weight
            "iou_t": (False, 0.1, 0.7),  # IoU training threshold
            "anchor_t": (False, 2.0, 8.0),  # anchor-multiple threshold
            "anchors": (False, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            "fl_gamma": (False, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            "hsv_h": (True, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (True, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (True, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": (True, 0.0, 45.0),  # image rotation (+/- deg)
            "translate": (True, 0.0, 0.9),  # image translation (+/- fraction)
            "scale": (True, 0.0, 0.9),  # image scale (+/- gain)
            "shear": (True, 0.0, 10.0),  # image shear (+/- deg)
            "perspective": (True, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            "flipud": (True, 0.0, 1.0),  # image flip up-down (probability)
            "fliplr": (True, 0.0, 1.0),  # image flip left-right (probability)
            "mosaic": (True, 0.0, 1.0),  # image mosaic (probability)
            "mixup": (True, 0.0, 1.0),  # image mixup (probability)
            "copy_paste": (True, 0.0, 1.0),  # segment copy-paste (probability)
        }

        # GA configs
        pop_size = 50
        mutation_rate_min = 0.01
        mutation_rate_max = 0.5
        crossover_rate_min = 0.5
        crossover_rate_max = 1
        min_elite_size = 2
        max_elite_size = 5
        tournament_size_min = 2
        tournament_size_max = 10

        with open(opt.hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if "anchors" not in hyp:  # anchors commented in hyp.yaml
                hyp["anchors"] = 3
        if opt.noautoanchor:
            del hyp["anchors"], meta["anchors"]
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / "hyp_evolve.yaml", save_dir / "evolve.csv"
        if opt.bucket:
            # download evolve.csv if exists
            subprocess.run(
                [
                    "gsutil",
                    "cp",
                    f"gs://{opt.bucket}/evolve.csv",
                    str(evolve_csv),
                ]
            )

        # Delete the items in meta dictionary whose first value is False
        del_ = [item for item, value_ in meta.items() if value_[0] is False]
        hyp_GA = hyp.copy()  # Make a copy of hyp dictionary
        for item in del_:
            del meta[item]  # Remove the item from meta dictionary
            del hyp_GA[item]  # Remove the item from hyp_GA dictionary

        # Set lower_limit and upper_limit arrays to hold the search space boundaries
        lower_limit = np.array([meta[k][1] for k in hyp_GA.keys()])
        upper_limit = np.array([meta[k][2] for k in hyp_GA.keys()])

        # Create gene_ranges list to hold the range of values for each gene in the population
        gene_ranges = [(lower_limit[i], upper_limit[i]) for i in range(len(upper_limit))]

        # Initialize the population with initial_values or random values
        initial_values = []

        # If resuming evolution from a previous checkpoint
        if opt.resume_evolve is not None:
            assert os.path.isfile(ROOT / opt.resume_evolve), "evolve population path is wrong!"
            with open(ROOT / opt.resume_evolve, errors="ignore") as f:
                evolve_population = yaml.safe_load(f)
                for value in evolve_population.values():
                    value = np.array([value[k] for k in hyp_GA.keys()])
                    initial_values.append(list(value))

        # If not resuming from a previous checkpoint, generate initial values from .yaml files in opt.evolve_population
        else:
            yaml_files = [f for f in os.listdir(opt.evolve_population) if f.endswith(".yaml")]
            for file_name in yaml_files:
                with open(os.path.join(opt.evolve_population, file_name)) as yaml_file:
                    value = yaml.safe_load(yaml_file)
                    value = np.array([value[k] for k in hyp_GA.keys()])
                    initial_values.append(list(value))

        # Generate random values within the search space for the rest of the population
        if initial_values is None:
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size)]
        elif pop_size > 1:
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size - len(initial_values))]
            for initial_value in initial_values:
                population = [initial_value] + population

        # Run the genetic algorithm for a fixed number of generations
        list_keys = list(hyp_GA.keys())
        for generation in range(opt.evolve):
            if generation >= 1:
                save_dict = {}
                for i in range(len(population)):
                    little_dict = {list_keys[j]: float(population[i][j]) for j in range(len(population[i]))}
                    save_dict[f"gen{str(generation)}number{str(i)}"] = little_dict

                with open(save_dir / "evolve_population.yaml", "w") as outfile:
                    yaml.dump(save_dict, outfile, default_flow_style=False)

            # Adaptive elite size
            elite_size = min_elite_size + int((max_elite_size - min_elite_size) * (generation / opt.evolve))
            # Evaluate the fitness of each individual in the population
            fitness_scores = []
            for individual in population:
                for key, value in zip(hyp_GA.keys(), individual):
                    hyp_GA[key] = value
                hyp.update(hyp_GA)
                results = train(hyp.copy(), opt, device, callbacks)
                callbacks = Callbacks()
                # Write mutation results
                keys = (
                    "metrics/precision",
                    "metrics/recall",
                    "metrics/mAP_0.5",
                    "metrics/mAP_0.5:0.95",
                    "val/box_loss",
                    "val/obj_loss",
                    "val/cls_loss",
                )
                print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)
                fitness_scores.append(results[2])

            # Select the fittest individuals for reproduction using adaptive tournament selection
            selected_indices = []
            for _ in range(pop_size - elite_size):
                # Adaptive tournament size
                tournament_size = max(
                    max(2, tournament_size_min),
                    int(min(tournament_size_max, pop_size) - (generation / (opt.evolve / 10))),
                )
                # Perform tournament selection to choose the best individual
                tournament_indices = random.sample(range(pop_size), tournament_size)
                tournament_fitness = [fitness_scores[j] for j in tournament_indices]
                winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
                selected_indices.append(winner_index)

            # Add the elite individuals to the selected indices
            elite_indices = [i for i in range(pop_size) if fitness_scores[i] in sorted(fitness_scores)[-elite_size:]]
            selected_indices.extend(elite_indices)
            # Create the next generation through crossover and mutation
            next_generation = []
            for _ in range(pop_size):
                parent1_index = selected_indices[random.randint(0, pop_size - 1)]
                parent2_index = selected_indices[random.randint(0, pop_size - 1)]
                # Adaptive crossover rate
                crossover_rate = max(
                    crossover_rate_min, min(crossover_rate_max, crossover_rate_max - (generation / opt.evolve))
                )
                if random.uniform(0, 1) < crossover_rate:
                    crossover_point = random.randint(1, len(hyp_GA) - 1)
                    child = population[parent1_index][:crossover_point] + population[parent2_index][crossover_point:]
                else:
                    child = population[parent1_index]
                # Adaptive mutation rate
                mutation_rate = max(
                    mutation_rate_min, min(mutation_rate_max, mutation_rate_max - (generation / opt.evolve))
                )
                for j in range(len(hyp_GA)):
                    if random.uniform(0, 1) < mutation_rate:
                        child[j] += random.uniform(-0.1, 0.1)
                        child[j] = min(max(child[j], gene_ranges[j][0]), gene_ranges[j][1])
                next_generation.append(child)
            # Replace the old population with the new generation
            population = next_generation
        # Print the best solution found
        best_index = fitness_scores.index(max(fitness_scores))
        best_individual = population[best_index]
        print("Best solution found:", best_individual)
        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(
            f"Hyperparameter evolution finished {opt.evolve} generations\n"
            f"Results saved to {colorstr('bold', save_dir)}\n"
            f"Usage example: $ python train.py --hyp {evolve_yaml}"
        )


def generate_individual(input_ranges, individual_length):
    """
    Generate an individual with random hyperparameters within specified ranges.

    Args:
        input_ranges (list[tuple[float, float]]): List of tuples where each tuple contains the lower and upper bounds
            for the corresponding gene (hyperparameter).
        individual_length (int): The number of genes (hyperparameters) in the individual.

    Returns:
        list[float]: A list representing a generated individual with random gene values within the specified ranges.

    Example:
        ```python
        input_ranges = [(0.01, 0.1), (0.1, 1.0), (0.9, 2.0)]
        individual_length = 3
        individual = generate_individual(input_ranges, individual_length)
        print(individual)  # Output: [0.035, 0.678, 1.456] (example output)
        ```

    Note:
        The individual returned will have a length equal to `individual_length`, with each gene value being a floating-point
        number within its specified range in `input_ranges`.
    """
    individual = []
    for i in range(individual_length):
        lower_bound, upper_bound = input_ranges[i]
        individual.append(random.uniform(lower_bound, upper_bound))
    return individual


def run(**kwargs):
    """
    Execute YOLOv5 training with specified options, allowing optional overrides through keyword arguments.

    Args:
        weights (str, optional): Path to initial weights. Defaults to ROOT / 'yolov5s.pt'.
        cfg (str, optional): Path to model YAML configuration. Defaults to an empty string.
        data (str, optional): Path to dataset YAML configuration. Defaults to ROOT / 'data/coco128.yaml'.
        hyp (str, optional): Path to hyperparameters YAML configuration. Defaults to ROOT / 'data/hyps/hyp.scratch-high.yaml'.
        epochs (int, optional): Total number of training epochs. Defaults to 100.
        batch_size (int, optional): Total batch size for all GPUs. Use -1 for automatic batch size determination. Defaults to 16.
        imgsz (int, optional): Image size (pixels) for training and validation. Defaults to 640.
        rect (bool, optional): Use rectangular training. Defaults to False.
        resume (bool | str, optional): Resume most recent training with an optional path. Defaults to False.
        nosave (bool, optional): Only save the final checkpoint. Defaults to False.
        noval (bool, optional): Only validate at the final epoch. Defaults to False.
        noautoanchor (bool, optional): Disable AutoAnchor. Defaults to False.
        noplots (bool, optional): Do not save plot files. Defaults to False.
        evolve (int, optional): Evolve hyperparameters for a specified number of generations. Use 300 if provided without a
            value.
        evolve_population (str, optional): Directory for loading population during evolution. Defaults to ROOT / 'data/ hyps'.
        resume_evolve (str, optional): Resume hyperparameter evolution from the last generation. Defaults to None.
        bucket (str, optional): gsutil bucket for saving checkpoints. Defaults to an empty string.
        cache (str, optional): Cache image data in 'ram' or 'disk'. Defaults to None.
        image_weights (bool, optional): Use weighted image selection for training. Defaults to False.
        device (str, optional): CUDA device identifier, e.g., '0', '0,1,2,3', or 'cpu'. Defaults to an empty string.
        multi_scale (bool, optional): Use multi-scale training, varying image size by ±50%. Defaults to False.
        single_cls (bool, optional): Train with multi-class data as single-class. Defaults to False.
        optimizer (str, optional): Optimizer type, choices are ['SGD', 'Adam', 'AdamW']. Defaults to 'SGD'.
        sync_bn (bool, optional): Use synchronized BatchNorm, only available in DDP mode. Defaults to False.
        workers (int, optional): Maximum dataloader workers per rank in DDP mode. Defaults to 8.
        project (str, optional): Directory for saving training runs. Defaults to ROOT / 'runs/train'.
        name (str, optional): Name for saving the training run. Defaults to 'exp'.
        exist_ok (bool, optional): Allow existing project/name without incrementing. Defaults to False.
        quad (bool, optional): Use quad dataloader. Defaults to False.
        cos_lr (bool, optional): Use cosine learning rate scheduler. Defaults to False.
        label_smoothing (float, optional): Label smoothing epsilon value. Defaults to 0.0.
        patience (int, optional): Patience for early stopping, measured in epochs without improvement. Defaults to 100.
        freeze (list, optional): Layers to freeze, e.g., backbone=10, first 3 layers = [0, 1, 2]. Defaults to [0].
        save_period (int, optional): Frequency in epochs to save checkpoints. Disabled if < 1. Defaults to -1.
        seed (int, optional): Global training random seed. Defaults to 0.
        local_rank (int, optional): Automatic DDP Multi-GPU argument. Do not modify. Defaults to -1.

    Returns:
        None: The function initiates YOLOv5 training or hyperparameter evolution based on the provided options.

    Examples:
        ```python
        import train
        train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
        ```

    Notes:
        - Models: https://github.com/ultralytics/yolov5/tree/master/models
        - Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        - Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)




def save_pred_mask(pred_masks, save_path, threshold=0.5):
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    from PIL import Image
    """
    保存推理得到的前景mask图
    Args:
        pred_masks: (Tensor) shape: (1, 1, H, W)，网络原始输出
        save_path: (str) 保存的文件路径
        threshold: (float) 二值化阈值，默认0.5
    """
    # 1. 做sigmoid
    pred_probs = torch.sigmoid(pred_masks)
    # 2. 二值化
    pred_binary = (pred_probs > threshold).float()
    # 3. 转成 [0,255]
    pred_binary = pred_binary.squeeze().cpu() * 255  # (H,W)
    # 4. 转成PIL图再保存
    pred_img = TF.to_pil_image(pred_binary.byte())  # 必须转成uint8
    pred_img.save(save_path)
