# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Experimental modules."""

import math

import numpy as np
import torch
import torch.nn as nn

from utils.downloads import attempt_download
from torch.nn import functional as F

class Sum(nn.Module):
    """Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070."""

    def __init__(self, n, weight=False):
        """Initializes a module to sum outputs of layers with number of inputs `n` and optional weighting, supporting 2+
        inputs.
        """
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        """Processes input through a customizable weighted sum of `n` inputs, optionally applying learned weights."""
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                assert x[i + 1].shape == y.shape
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    """Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595."""

    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        """Initializes MixConv2d with mixed depth-wise convolutional layers, taking input and output channels (c1, c2),
        kernel sizes (k), stride (s), and channel distribution strategy (equal_ch).
        """
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1e-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)]
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        """Performs forward pass by applying SiLU activation on batch-normalized concatenated convolutional layer
        outputs.
        """
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))

class RailStructureFusion(nn.Module):
    def __init__(self, c_low, c_high, c_out):
        super().__init__()

        # 通道对齐
        self.low_proj = nn.Conv2d(c_low, c_out, 1)
        self.high_proj = nn.Conv2d(c_high, c_out, 1)

        # 方向结构增强（针对轨道细长结构）
        self.dir_conv1 = nn.Conv2d(c_out, c_out, kernel_size=(1,7), padding=(0,3))
        self.dir_conv2 = nn.Conv2d(c_out, c_out, kernel_size=(7,1), padding=(3,0))

        # 深层语义通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_out, c_out // 4, 1),
            nn.ReLU(),
            nn.Conv2d(c_out // 4, c_out, 1),
            nn.Sigmoid()
        )

        # 跨层空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(c_out * 2, c_out, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c_out, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        f_high, f_low = x
        f_low = self.low_proj(f_low)
        f_high = self.high_proj(f_high)

        # 上采样高层特征到低层尺寸
        f_high = F.interpolate(f_high, size=f_low.shape[-2:], mode='nearest')

        # 方向结构增强（浅层）
        structure = self.dir_conv1(f_low) + self.dir_conv2(f_low)
        f_low_enhanced = f_low * torch.sigmoid(structure)

        # 深层语义增强
        f_high_sem = f_high * self.channel_att(f_high)

        # 空间融合权重
        m = self.spatial_att(torch.cat([f_low_enhanced, f_high_sem], dim=1))

        return m * f_low_enhanced + (1 - m) * f_high_sem


class RailRegionFusion(nn.Module):
    """面向"周界区域(面)"的跨层融合: 取代 RailStructureFusion 的线状方向自门控。

    与 RSF 的差异(线→面):
      1) 1×7/7×1 条状方向卷积 → 各向同性多尺度空洞卷积(d=1,3,6), 聚合面状上下文;
      2) 乘性 sigmoid 自门控(掏空区域内部) → 加性残差(只增不减, 保持区域实心)。
    上采样保持 nearest(bilinear backward 无确定性实现, 与 seed42 确定性协议冲突);
    通道注意力与末端空间凸组合融合与 RSF 保持一致。
    """

    def __init__(self, c_low, c_high, c_out):
        super().__init__()

        # 通道对齐
        self.low_proj = nn.Conv2d(c_low, c_out, 1)
        self.high_proj = nn.Conv2d(c_high, c_out, 1)

        # 面状多尺度上下文(depthwise 空洞卷积 + 1×1 融合), 加性残差
        self.context = nn.ModuleList([
            nn.Conv2d(c_out, c_out, 3, padding=d, dilation=d, groups=c_out)
            for d in (1, 3, 6)
        ])
        self.ctx_fuse = nn.Sequential(
            nn.Conv2d(c_out * 3, c_out, 1),
            nn.BatchNorm2d(c_out),
            nn.SiLU(),
        )

        # 深层语义通道注意力(同 RSF)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_out, c_out // 4, 1),
            nn.ReLU(),
            nn.Conv2d(c_out // 4, c_out, 1),
            nn.Sigmoid()
        )

        # 跨层空间注意力(同 RSF)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(c_out * 2, c_out, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c_out, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        f_high, f_low = x
        f_low = self.low_proj(f_low)
        f_high = self.high_proj(f_high)

        f_high = F.interpolate(f_high, size=f_low.shape[-2:], mode='nearest')

        # 面状上下文增强: 加性残差, 不掏空区域内部
        ctx = torch.cat([conv(f_low) for conv in self.context], dim=1)
        f_low_enhanced = f_low + self.ctx_fuse(ctx)

        # 深层语义增强
        f_high_sem = f_high * self.channel_att(f_high)

        # 空间融合权重
        m = self.spatial_att(torch.cat([f_low_enhanced, f_high_sem], dim=1))

        return m * f_low_enhanced + (1 - m) * f_high_sem


class SegHeadLogit(nn.Module):
    """纯 logit 分割头(重启优化#1): 无 BN 无激活, 供 BCEWithLogits/Dice 直接消费。

    取代原 yaml 末层 Conv(=Conv2d+BN+SiLU): SiLU 会把 logit 下界卡在 ≈-0.278,
    前景概率地板 ≈0.43, 无法自信预测背景。
    """

    def __init__(self, c1):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 3, padding=1, bias=True)

    def forward(self, x):
        return self.conv(x)


class PerimeterGuidedRefine(nn.Module):
    """D1(论文2.2): 用 PGM 前景概率图作为软注意力, 经深度可分卷积 + 残差逐元素相加,
    调制检测特征, 把周界先验注入检测主路 (而非仅做辅助分割损失)。
    输入 x=[feat, mask]: feat 为某尺度检测特征[B,C,h,w], mask 为分割头输出[B,1,H,W](logit)。
    """
    def __init__(self, c):
        super().__init__()
        # 瓶颈式卷积精炼(规则卷积, 确定性 cuDNN 友好且省显存): c->c/4 (3x3) ->c
        cr = max(16, c // 4)
        self.reduce = nn.Conv2d(c, cr, 1, bias=False)
        self.conv = nn.Conv2d(cr, cr, 3, padding=1, bias=False)
        self.expand = nn.Conv2d(cr, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()
        # 优化#2: 可学习残差缩放(ReZero/LayerScale), 初始化 0 -> 开局恒等(保住预训练检测特征), 再逐步注入精炼
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        feat, mask = x[0], x[1]
        attn = torch.sigmoid(mask)                                     # 前景概率图 -> [0,1] (mask 现为纯logit, 对比强)
        if attn.shape[-2:] != feat.shape[-2:]:
            # 优化#3: 平均池化下采样注意力(每格=周界覆盖率), 比 nearest 平滑; 用 avg_pool2d(反向确定性, adaptive版不确定)
            sh, sw = attn.shape[-2] // feat.shape[-2], attn.shape[-1] // feat.shape[-1]
            if sh >= 1 and sw >= 1 and attn.shape[-2] % feat.shape[-2] == 0 and attn.shape[-1] % feat.shape[-1] == 0:
                attn = F.avg_pool2d(attn, kernel_size=(sh, sw), stride=(sh, sw))
            else:
                attn = F.interpolate(attn, size=feat.shape[-2:], mode="nearest")
        refined = self.act(self.bn(self.expand(self.conv(self.reduce(feat)))))
        return feat + self.gamma * refined * attn                      # 优化#2: 残差 + γ缩放 + 周界软注意力调制



class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initializes an ensemble of models to be used for aggregated predictions."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Performs forward pass aggregating outputs from an ensemble of models.."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, device=None, inplace=True, fuse=True):
    """
    Loads and fuses an ensemble or single YOLOv5 model from weights, handling device placement and model adjustments.

    Example inputs: weights=[a,b,c] or a single model weights=[a] or weights=a.
    """
    from models.yolo import Detect, Model

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location="cpu")  # load
        ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, "stride"):
            ckpt.stride = torch.tensor([32.0])
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval())  # model in eval mode

    # Module updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f"Models have different class counts: {[m.nc for m in model]}"
    return model
