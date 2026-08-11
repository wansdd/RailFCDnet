# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO-specific modules.

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import math
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from torch.nn import functional as F
from conv2former import ConvModulOperationSpatialAttention, Ranker_dropout, LayerAttentionFusion
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import (
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Contract,
    Conv,
    CrossConv,
    DetectMultiBackend,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    Proto,
)
from models.experimental import MixConv2d, Sum, RailStructureFusion, RailRegionFusion, SegHeadLogit, PerimeterGuidedRefine
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, colorstr, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (
    fuse_conv_and_bn,
    initialize_weights,
    model_info,
    profile,
    scale_img,
    select_device,
    time_sync,
)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max = torch.max(x, dim=1, keepdim=True)[0]
        x_cat = torch.cat([avg, max], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 替换为标准 MaxPool2d
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        
        # 手动实现最大池化
        batch, channel, _, _ = x.size()
        max_val = torch.max(torch.max(x, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
        
        max_val = self.fc(max_val)
        out = avg + max_val
        return x * self.sigmoid(out)

class ChannelSelfAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape
        # 平均池化获得通道特征 [B, C]
        x_pool = x.view(B, C, -1).mean(-1)  # [B, C]

        # QKV projection
        Q = self.q_proj(x_pool)  # [B, C]
        K = self.k_proj(x_pool)  # [B, C]
        V = self.v_proj(x_pool)  # [B, C]

        # 注意力打分 [B, C] × [B, C] → [B, 1, C]
        attn_score = torch.bmm(Q.unsqueeze(1), K.unsqueeze(2)).squeeze(2) / (C ** 0.5)  # [B, 1]
        attn_weight = self.softmax(attn_score)  # [B, 1]

        # 用注意力权重加权 V
        out = attn_weight * V.sum(dim=-1, keepdim=True)  # [B, 1] × [B, 1] → [B, 1]
        out = out.repeat(1, C)  # [B, C] 让输出维度对齐
        return out  # [B, C]


class SimpleSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.fc_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, N, D]
        B, N, D = x.shape
        H = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: [B, N, D]
        q, k, v = map(lambda t: t.view(B, N, H, D // H).transpose(1, 2), qkv)  # [B, H, N, D//H]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.fc_out(out)

class Detect(nn.Module):
    """YOLOv5 Detect head for processing input tensors and generating detection outputs in object detection models."""

    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """Initializes YOLOv5 detection layer with specified classes, anchors, channels, and inplace operations."""
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`."""
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    """YOLOv5 Segment head for segmentation models, extending Detect with mask and prototype layers."""

    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        """Initializes YOLOv5 Segment head with options for mask count, protos, and channel adjustments."""
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        """Processes input through the network, returning detections and prototypes; adjusts output based on
        training/export mode.
        """
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class AAMProjection(nn.Module):
    """Lightweight projection head for instance alignment: channel reduction plus local spatial context."""

    def __init__(self, c1, c2=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.GroupNorm(16, c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.GroupNorm(16, c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c2, 1, bias=True),
        )

    def forward(self, x):
        return self.proj(x)


class Ranker(nn.Module):  # batch 1280, 4, 4
    def __init__(self, in_dim=1280):
        self.in_dim = in_dim
        #print(self.in_dim)
        super().__init__()
        self.ranker = nn.Sequential(  # input [B, C, H, W] output [B, 3, H, W]
            nn.Conv2d(
                in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_dim),
            )
    def forward(self, x):
        x = self.ranker(x)
        return x

class Ranker2(nn.Module):
    def __init__(self, in_dim=1280):
        super().__init__()
        self.ranker = nn.Sequential(
            # Depthwise Conv
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            # Pointwise Conv
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),

            # 再加一次DWConv + Pointwise
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            nn.BatchNorm2d(in_dim),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)  # 输出 B, 1280, 1, 1

    def forward(self, x):
        x = self.ranker(x)
        x = self.pool(x)
        return x

class Ranker3(nn.Module):
    def __init__(self, in_dim=1280):
        super().__init__()
        self.in_dim = in_dim
        self.ranker = nn.Sequential(
            nn.Conv2d(self.in_dim, self.in_dim, 3, 1, 1),
            nn.BatchNorm2d(self.in_dim),
            nn.ReLU(),
            nn.Conv2d(self.in_dim, self.in_dim, 3, 1, 1),
            nn.BatchNorm2d(self.in_dim),
            ChannelAttention(self.in_dim),
            SpatialAttention()
        )



    def forward(self, x):
        x = self.ranker(x)
        return x

class Ranker4(nn.Module):
    def __init__(self, in_dim=1280):
        super().__init__()
        self.in_dim = in_dim
        self.ranker = nn.Sequential(
            nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_dim)
        )

        self.self_attn = SimpleSelfAttention(dim=in_dim, heads=8)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):  # x: [B, 1280, 4, 4]
        x = self.ranker(x)  # [B, 1280, 4, 4]

        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, 16, 1280]

        attn_out = self.self_attn(x)  # [B, 16, 1280]
        x = self.norm(attn_out + x)   # 加残差连接与 LayerNorm

        x = x.permute(0, 2, 1).view(B, C, H, W)  # [B, 1280, 4, 4]

        return x

class Ranker5(nn.Module):
    def __init__(self, in_dim=1280):
        super().__init__()
        self.in_dim = in_dim
        self.ranker = nn.Sequential(
            nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_dim),
        )
        self.channel_attn = ChannelSelfAttention(channels=self.in_dim)

    def forward(self, x):  # x: [B, 1280, 4, 4]
        x = self.ranker(x)  # [B, 1280, 4, 4]
        x_attn = self.channel_attn(x)  # [B, 1280] → 注意力加权后的通道特征
        return x_attn  # 用这个送入 MMD


class FeatureCycleConsistency(nn.Module):
    def __init__(self, dim=320, loss_fn=nn.L1Loss()):
        super(FeatureCycleConsistency, self).__init__()
        self.G = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )
        self.F = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )
        self.loss_fn = loss_fn

    def forward(self, features):
        B = features.size(0)
        split_idx = int(B - 1)  # ✅ 修复这里

        feat_src = features[:split_idx]
        feat_tgt = features[split_idx:]

        # 源域: G -> F -> src
        feat_tgt_hat = self.G(feat_src)
        feat_src_recon = self.F(feat_tgt_hat)

        # 目标域: F -> G -> tgt
        feat_src_hat = self.F(feat_tgt)
        feat_tgt_recon = self.G(feat_src_hat)

        # 拼接：前80%为G(src)，后20%为F(tgt)
        transformed_features = torch.cat([feat_tgt_hat, feat_src_hat], dim=0)

        # 循环一致性损失
        loss_src_cycle = self.loss_fn(feat_src_recon, feat_src)
        loss_tgt_cycle = self.loss_fn(feat_tgt_recon, feat_tgt)

        total_cycle_loss = loss_src_cycle + loss_tgt_cycle  # ⚠️ 初期建议权重相同

        return transformed_features, total_cycle_loss

class BaseModel(nn.Module):
    """YOLOv5 base model."""

    def forward(self, x, mask=None, profile=False, visualize=False):
        """Executes a single-scale inference or training pass on the YOLOv5 base model, with options for profiling and
        visualization.
        """
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        """Performs a forward pass on the YOLOv5 model, enabling profiling and feature visualization options."""
        y, dt = [], []  # outputs
        backbone_feature = []
        mask_output = None
        # self.save = [1, 2, 3, 4, 5, 6, 7, 8,9 ,10, 11, 12, 13,14 ,15 ,16, 17, 18, 19, 20, 21, 22, 23, 24]  # save layers
        for m in self.model:
            # if m.i in [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]:
            #     continue
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)

            x = m(x)  # run
            # 语义分割分支(PGM): 分割头输出单通道前景概率图, 按通道数==1 鲁棒捕获(兼容不同 cfg 的层号)
            if isinstance(x, torch.Tensor) and x.dim() == 4 and x.shape[1] == 1:
                mask_output = x
            # 多层实例特征(浅/中/深, 供 AAM 对齐): backbone 第 4/6/9 层
            if m.i in [4, 6, 9]:
                backbone_feature.append(x)
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        if mask_output is not None:
            return x, backbone_feature, mask_output
            # return x, self.ranker(backbone_feature), mask_output
        else:
            return x, backbone_feature 
            # return x 


    def _forward_once2(self, x, profile=False, visualize=False):
        """Performs a forward pass on the YOLOv5 model, enabling profiling and feature visualization options."""
        y, dt = [], []  # outputs
        backbone_feature = []
        mask_output = None
        # self.save = [1, 2, 3, 4, 5, 6, 7, 8,9 ,10, 11, 12, 13,14 ,15 ,16, 17, 18, 19, 20, 21, 22, 23, 24]  # save layers
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            # 根据不同层数使用对应尺寸的mask进行特征增强
            if m.i ==  36:
                # 语义分割网络分支
                mask_output = x
            if m.i == 15:
                # 循环一致性操作
                x, clc_loss = self.cyc(x)
            # if m.i == 16:
            #     # 特征提取分支
            #     backbone_feature.append(x)
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        if mask_output is not None:
            return x, clc_loss, mask_output
        else:
            return x, clc_loss


    def _profile_one_layer(self, m, x, dt):
        """Profiles a single layer's performance by computing GFLOPs, execution time, and parameters."""
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):
        """Fuses Conv2d() and BatchNorm2d() layers in the model to improve inference speed."""
        LOGGER.info("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, (Conv)) and hasattr(m, "bn"):
            # if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        """Prints model information given verbosity and image size, e.g., `info(verbose=True, img_size=640)`."""
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        """Applies transformations like to(), cpu(), cuda(), half() to model tensors excluding parameters or registered
        buffers.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    """YOLOv5 detection model class for object detection tasks, supporting custom configurations and anchors."""

    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):
        """Initializes YOLOv5 model with configuration file, input channels, number of classes, and custom anchors."""
        super().__init__()

        self.ranker = Ranker(in_dim=320)  # 特征增强模块
        # self.fusion = LayerAttentionFusion(
        #     in_dims=[320, 640, 1280], out_dims=[320, 640, 1280]
        # )
        # self.cyc = FeatureCycleConsistency()


#######################################################
        # 初始化最后一个卷积层权重为0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.in_channels == 4:
                nn.init.zeros_(m.weight)
                nn.init.constant_(m.bias, 0)
########################################################


        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)

        # AAM 多尺度自适应对齐权重 (论文 Eq.6): 3 级可学习 logits, 初始化为 0,
        # 经 softmax -> A_i, 随网络联合优化 (在 ComputeLoss 中用于加权多层实例 MMD)
        self.aam_s = nn.Parameter(torch.zeros(3))

        # D2(论文2.3/Eq.2): 实例级对齐的投影网络 —— 对 3 级骨干特征(层4/6/9)降维到统一维度,
        # 并用轻量 depthwise 3x3 融合局部空间上下文。
        # 之后在 ComputeLoss 中按目标框切片+区域平均池化得到实例嵌入再做 MMD。
        # 必须在 __init__ 内创建(早于优化器), 否则参数不会被训练。通道数由一次 dummy 前向探测。
        try:
            with torch.no_grad():
                _bb = self._forward_once(torch.zeros(1, ch, 256, 256))[1]
            self.aam_proj_dim = int(os.environ.get("RCD_AAM_DIM", 256))  # 投影头维度, 可由环境变量设(默认256)
            self.aam_proj = nn.ModuleList([AAMProjection(int(f.shape[1]), self.aam_proj_dim) for f in _bb])
        except Exception as _e:
            LOGGER.warning(f"aam_proj 初始化探测失败, D2实例对齐将退化: {_e}")
            self.aam_proj = None

        # 新对齐模块: 实例级域判别器 + GRL 非对称对抗 + EMA 类原型。
        # dom_proj: 对 C4/C5(backbone 层6/9 = backbone_feature[1],[2]) 做 1×1 投影到 256;
        # dom_disc: 每类一个判别器(256→128→1, 跨层共享); dom_proto: 目标各层各类 EMA 原型(buffer)。
        try:
            with torch.no_grad():
                _bb2 = self._forward_once(torch.zeros(1, ch, 256, 256))[1]
            self.dom_proj_dim = 256
            self.dom_levels = [1, 2]  # C4=层6, C5=层9
            self.dom_proj = nn.ModuleList(
                [nn.Conv2d(int(_bb2[li].shape[1]), self.dom_proj_dim, 1) for li in self.dom_levels])
            _ncls = self.yaml["nc"]
            self.dom_disc = nn.ModuleList([
                nn.Sequential(nn.Linear(self.dom_proj_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
                for _ in range(_ncls)])
            self.register_buffer("dom_proto", torch.zeros(len(self.dom_levels), _ncls, self.dom_proj_dim))
            self.register_buffer("dom_proto_ready", torch.zeros(len(self.dom_levels), _ncls))
        except Exception as _e:
            LOGGER.warning(f"dom_proj/dom_disc 初始化失败, 新对抗对齐将退化: {_e}")
            self.dom_proj = None
            self.dom_disc = None

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):

            def _forward(x):
                """Passes the input 'x' through the model and returns the processed output."""
                return self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)

            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))[0]])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info("")

    def forward(self, x, mask=None, augment=False, profile=False, visualize=False):
        """Performs single-scale or augmented inference and may include profiling or visualization."""
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile=profile, visualize=visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        """Performs augmented inference across different scales and flips, returning combined detections."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        """De-scales predictions from augmented inference, adjusting for flips and image size."""
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        """Clips augmented inference tails for YOLOv5 models, affecting first and last tensors based on grid points and
        layer counts.
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):
        """
        Initializes biases for YOLOv5's Detect() module, optionally using class frequencies (cf).

        For details see https://arxiv.org/abs/1708.02002 section 3.3.
        """
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    """YOLOv5 segmentation model for object detection and segmentation tasks with configurable parameters."""

    def __init__(self, cfg="yolov5s-seg.yaml", ch=3, nc=None, anchors=None):
        """Initializes a YOLOv5 segmentation model with configurable params: cfg (str) for configuration, ch (int) for channels, nc (int) for num classes, anchors (list)."""
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    """YOLOv5 classification model for image classification tasks, initialized with a config file or detection model."""

    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):
        """Initializes YOLOv5 model with config file `cfg`, input channels `ch`, number of classes `nc`, and `cuttoff`
        index.
        """
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        """Creates a classification model from a YOLOv5 detection model, slicing at `cutoff` and adding a classification
        layer.
        """
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, "models.common.Classify"  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        """Creates a YOLOv5 classification model from a specified *.yaml configuration file."""
        self.model = None


def parse_model(d, ch):
    """Parses a YOLOv5 model from a dict `d`, configuring layers based on input channels `ch` and model architecture."""
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act, ch_mul = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
        d.get("activation"),
        d.get("channel_multiple"),
    )
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    if not ch_mul:
        ch_mul = 8
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            # DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3TR,
            C3SPP,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,

        }:  

            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                if m is Conv and len(args)==5:
                    args = args[2:]
                    pass
                else:
                    c2 = make_divisible(c2 * gw, ch_mul)
    
            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m in {RailStructureFusion, RailRegionFusion}:
            args = [make_divisible(x * gw, ch_mul) for x in args]
            c2 = args[-1]
        elif m is SegHeadLogit:  # 纯 logit 分割头: 1 通道输出, 免 gw 缩放
            c1 = ch[f]
            c2 = 1
            args = [c1]
        elif m is PerimeterGuidedRefine:  # [feat, mask] -> 调制后特征(通道不变)
            c1 = ch[f[0]]
            c2 = ch[f[0]]
            args = [c1]
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is nn.Conv2d:
            c1, c2 = args[0], args[1]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is DWConv:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, ch_mul)
            args = [c1, c2, *args[1:]]
            pass
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}: #多个from
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, ch_mul)
        elif m is Sum: #多个from
            c1 = ch[f[0]]
            c2 = ch[f[0]]
            pass
        # elif m is RailStructureFusion:
        #     c2 = args[-1]
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="model.yaml")
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--profile", action="store_true", help="profile model speed")
    parser.add_argument("--line-profile", action="store_true", help="profile model speed layer by layer")
    parser.add_argument("--test", action="store_true", help="test all yolo*.yaml")
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / "models").rglob("yolo*.yaml"):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f"Error in {cfg}: {e}")

    else:  # report fused model summary
        model.fuse()
