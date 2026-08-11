# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441."""
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    """Modified BCEWithLogitsLoss to reduce missing label effects in YOLOv5 training with optional alpha smoothing."""

    def __init__(self, alpha=0.05):
        """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
        parameter.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
        returns mean loss.
        """
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class DynamicFocalLoss(nn.Module):
    """Applies focal loss to address class imbalance by modifying BCEWithLogitsLoss with gamma and alpha parameters."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        # self.class_weights11 = torch.tensor([0.18, 0.81]).to(true.device)
        self.class_weights11 = torch.tensor([0.81, 0.18]).to(true.device)
        # alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        alpha_factor = true * self.class_weights11[1] + (1 - true) * self.class_weights11[0]
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss



class FocalLoss(nn.Module):
    """Applies focal loss to address class imbalance by modifying BCEWithLogitsLoss with gamma and alpha parameters."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss






class QFocalLoss(nn.Module):
    """Implements Quality Focal Loss to address class imbalance by modulating loss based on prediction confidence."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算Gram核矩阵
    source: sample_size_1 * feature_size 的数据
    target: sample_size_2 * feature_size 的数据
    kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
    kernel_num: 表示的是多核的数量
    fix_sigma: 表示是否使用固定的标准差
        return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
                        矩阵，表达形式:
                        [   K_ss K_st
                            K_ts K_tt ]
    """
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0) # 合并在一起

    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) # 计算高斯核中的|x-y|

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        # bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth = torch.sum(L2_distance.detach()) / (n_samples**2 - n_samples)
        bandwidth += 1e-5  # 防止除零
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val) # 将多个核合并在一起

def coral(source, target):
    n_s, d = source.size()
    n_t, _ = target.size()

    source_c = source - source.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)

    cov_s = (source_c.T @ source_c) / (n_s - 1)
    cov_t = (target_c.T @ target_c) / (n_t - 1)

    loss = torch.mean((cov_s - cov_t) ** 2)
    return loss.unsqueeze(0)

def euclidean(source, target):
    mean_s = torch.mean(source, dim=0)
    mean_t = torch.mean(target, dim=0)
    loss = ((mean_s - mean_t) ** 2).sum()
    return loss.unsqueeze(0)




def mmd_multi(feature_st, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    # print("source:", source.size())
    # print("target:", target.size())
    source = feature_st.mean(3).mean(2)
    target = target.mean(3).mean(2)
    n = int(source.size()[0])
    m = int(target.size()[0])

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1,-1)  # K_ss矩阵，Source<->Source
    XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1) # K_st矩阵，Source<->Target

    YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1) # K_ts矩阵,Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1,-1)  # K_tt矩阵,Target<->Target

    loss = (XX + XY).sum() + (YX + YY).sum() # torch.tensor([loss])
    return loss.unsqueeze(0)  # 返回标量损失

def roi_pool_instances(feat, boxes_xywh, img_idx, cls=None):
    """D2(论文Eq.2): 把投影特征图按目标框切片 S(z,b) 后区域平均池化, 得到实例级嵌入。
    feat: [B,C,h,w]; boxes_xywh: [N,4] 归一化(cx,cy,w,h); img_idx: [N] 对应图像下标; cls: [N] 类别。
    返回 (emb[N_valid,C], cls[N_valid]) —— 同步返回每个实例的类别, 供类内对齐使用。
    """
    B, C, h, w = feat.shape
    out, oc = [], []
    for k in range(boxes_xywh.shape[0]):
        bi = int(img_idx[k])
        if bi < 0 or bi >= B:
            continue
        cx, cy, bw, bh = [float(v) for v in boxes_xywh[k]]
        x1 = int(max(0, min(w - 1, (cx - bw / 2) * w)));  x2 = int(max(1, min(w, round((cx + bw / 2) * w))))
        y1 = int(max(0, min(h - 1, (cy - bh / 2) * h)));  y2 = int(max(1, min(h, round((cy + bh / 2) * h))))
        if x2 <= x1: x2 = x1 + 1
        if y2 <= y1: y2 = y1 + 1
        region = feat[bi, :, y1:y2, x1:x2]            # [C, rh, rw]
        out.append(region.mean(dim=(1, 2)))           # [C]
        oc.append(int(cls[k]) if cls is not None else 0)
    if not out:
        return None, None
    emb = torch.stack(out, 0)                         # [N,C]
    cl = torch.tensor(oc, device=feat.device, dtype=torch.long)
    return emb, cl

class _GradReverse(torch.autograd.Function):
    """梯度反转层(DANN): 前向恒等, 反向梯度乘 -lambda。"""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


def roi_align_instances(feat, boxes_xywh, img_idx, cls=None, out_size=1, sampling_ratio=2, pool="avgmax"):
    """ROIAlign(亚像素)按 GT 框抠实例嵌入。
    feat[B,C,h,w]; boxes_xywh[N,4] 归一化(cx,cy,w,h); img_idx[N] 图像下标; cls[N] 类别。
    返回 (emb[N_valid,C], cls[N_valid]); 无有效框返回 (None,None)。
    """
    from torchvision.ops import roi_align as _roi_align
    if boxes_xywh is None or boxes_xywh.numel() == 0:
        return None, None
    B, C, h, w = feat.shape
    img_idx = img_idx.to(feat.device)
    valid = (img_idx >= 0) & (img_idx < B)
    if valid.sum() == 0:
        return None, None
    b = boxes_xywh.to(feat.device).float()[valid]
    cx, cy, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    x1 = (cx - bw / 2) * w; y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w; y2 = (cy + bh / 2) * h
    rois = torch.stack([img_idx[valid].float(), x1, y1, x2, y2], dim=1)  # [N,5]
    roi = _roi_align(feat.float(), rois, output_size=(out_size, out_size),
                     spatial_scale=1.0, sampling_ratio=sampling_ratio, aligned=True)
    if pool == "avgmax":
        emb = torch.cat([roi.mean(dim=(2, 3)), roi.amax(dim=(2, 3))], dim=1)
    elif pool == "max":
        emb = roi.amax(dim=(2, 3))
    else:
        emb = roi.mean(dim=(2, 3))
    cl = cls.to(feat.device).long()[valid] if cls is not None else None
    return emb, cl


def mmd_inst(zs, zt, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """实例级多核 MMD: zs[Ns,C], zt[Nt,C] 已是实例嵌入(无需再池化)。"""
    n, m = zs.size(0), zt.size(0)
    kernels = guassian_kernel(zs, zt, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]
    return (XX.mean() + YY.mean() - XY.mean() - YX.mean()).unsqueeze(0)

def class_align_can(si, sc, ti, tc):
    """CAN/CDD 风格类别对齐(align_mode="can"), 单层实例嵌入上计算:
      intra_cross: 跨域同类 MMD 最小化(不同场景同类别对齐);
      compact:     同域同类紧致(中心损失式, L2 归一化嵌入; 同场景同类别对齐);
      inter:       跨域异类 MMD 取负最大化(不同类别拉远; 多核高斯核值≤1 ⇒ MMD 有界, 负项安全)。
    si[Ns,C]/ti[Nt,C] 实例嵌入, sc/tc 对应类别。返回标量 [1] 张量。
    """
    device = si.device
    # --- 跨域同类 MMD(沿用原类内对齐) ---
    intra_cross, n_intra = si.new_zeros(1), 0
    classes = torch.unique(torch.cat([sc, tc]))
    for c in classes:
        sic, tic = si[sc == c], ti[tc == c]
        if sic.size(0) > 1 and tic.size(0) > 1:
            intra_cross = intra_cross + mmd_inst(sic, tic)
            n_intra += 1
    if n_intra > 0:
        intra_cross = intra_cross / n_intra

    # --- 同域同类紧致: ||ẑ_i − μ_{d,c}||², 嵌入 L2 归一化保证尺度稳定 ---
    compact, n_comp = si.new_zeros(1), 0
    for emb, cl in ((si, sc), (ti, tc)):
        z = F.normalize(emb, dim=1)
        for c in torch.unique(cl):
            zc = z[cl == c]
            if zc.size(0) > 1:
                compact = compact + (zc - zc.mean(0, keepdim=True)).pow(2).sum(1).mean().unsqueeze(0)
                n_comp += 1
    if n_comp > 0:
        compact = compact / n_comp

    # --- 跨域异类 MMD 取负(CAN: 推大不同类的分布距离) ---
    inter, n_inter = si.new_zeros(1), 0
    for c in torch.unique(sc):
        for c2 in torch.unique(tc):
            if int(c) == int(c2):
                continue
            sic, tic = si[sc == c], ti[tc == c2]
            if sic.size(0) > 1 and tic.size(0) > 1:
                inter = inter + mmd_inst(sic, tic)
                n_inter += 1
    if n_inter > 0:
        inter = inter / n_inter

    return intra_cross + compact - inter

def dice_loss(pred_logits, target01, eps=1.0):
    """软 Dice 损失: pred_logits[N,1,H,W] 纯 logit, target01[N,1,H,W] ∈{0,1}。逐样本计算后取均值。"""
    p = torch.sigmoid(pred_logits).flatten(1)
    t = target01.flatten(1)
    inter = (p * t).sum(1)
    dice = (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)
    return (1 - dice).mean().unsqueeze(0)

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    # print("source:", source.size())
    # print("target:", target.size())
    n = int(source.size()[0])
    m = int(target.size()[0])

    kernels = guassian_kernel(source.mean(3).mean(2), target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n] 
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1,-1)  # K_ss矩阵，Source<->Source
    XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1) # K_st矩阵，Source<->Target

    YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1) # K_ts矩阵,Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1,-1)  # K_tt矩阵,Target<->Target

    loss = (XX + XY).sum() + (YX + YY).sum() # torch.tensor([loss])
    return loss.unsqueeze(0)  # 返回标量损失

import torch
import torch.nn.functional as F

def foreground_loss(pred_masks, target_masks):
    """
    计算前景分割损失（Binary Cross Entropy with Logits Loss）
    
    Args:
        pred_masks: (Tensor) 模型输出，未经激活的预测值，shape: (N, 1, H, W)
        target_masks: (Tensor) 真实标签，像素值为0或255，shape: (N, 1, H, W)
    
    Returns:
        loss: 前景分割损失
    """
    # 将 target_masks 从 0/255 归一化到 0/1
    target_masks = target_masks.float() / 255.0

    # 使用 BCEWithLogitsLoss，不需要手动做 sigmoid
    loss = F.binary_cross_entropy_with_logits(pred_masks, target_masks[:, 0:1, :, :])
    
    return loss.unsqueeze(0)  # 返回标量损失

def downsample_bg_mask(mask_bg, target_hw):
    """
    mask_bg: [B, 1, H_img, W_img]
    return : [B, H_feat, W_feat] (bool)
    """
    mask_bg = F.interpolate(
        mask_bg.float(),
        size=target_hw,
        mode='nearest'
    )
    return mask_bg.squeeze(1).bool()

def extract_bg_feature_set(feat, mask_bg, max_samples=1024):
    """
    feat   : [B, C, H, W]
    mask_bg: [B, H, W] (bool)
    return : [N, C]
    """
    B, C, H, W = feat.shape
    bg_feats = []

    for b in range(B):
        mask = mask_bg[b]                          # [H, W]
        f = feat[b].permute(1, 2, 0)[mask]         # [N_bg, C]

        if f.numel() == 0:
            continue

        if f.shape[0] > max_samples:
            idx = torch.randperm(
                f.shape[0], device=f.device
            )[:max_samples]
            f = f[idx]

        bg_feats.append(f)

    return torch.cat(bg_feats, dim=0)               # [N, C]


def background_mean_std_alignment(bg_s, bg_t, eps=1e-6):
    """
    bg_s, bg_t: [N, C]
    """
    mu_s = bg_s.mean(dim=0)
    mu_t = bg_t.mean(dim=0)

    std_s = torch.sqrt(bg_s.var(dim=0, unbiased=False) + eps)
    std_t = torch.sqrt(bg_t.var(dim=0, unbiased=False) + eps)

    loss_mean = F.mse_loss(mu_s, mu_t)
    loss_std  = F.mse_loss(std_s, std_t)

    return loss_mean + loss_std



class ComputeLoss:
    """Computes the total loss for YOLOv5 model predictions, including classification, box, and objectness losses."""

    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            
            # BCEcls, BCEobj = DynamicFocalLoss(BCEcls, g), DynamicFocalLoss(BCEobj, g)
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        # AAM 自适应权重(论文 Eq.6): 引用模型上的可学习 logits, softmax 后加权多层实例 MMD。
        self.aam_s = getattr(de_parallel(model), "aam_s", None)
        if self.aam_s is None:  # 向后兼容: 模型未定义则用本地可学习参数
            import torch.nn as _nn
            self.aam_s = _nn.Parameter(torch.zeros(3, device=device))
        # D2: 实例级对齐的投影网络(论文Eq.2), 由模型持有以纳入优化器
        self.aam_proj = getattr(de_parallel(model), "aam_proj", None)
        # 目标域实例信息(每个 epoch 由训练脚本写入): 累计的目标特征对应的框 + 源域图像数
        self.t_boxes = None
        self.bs_s = None
        # 消融配置(默认全开): 对齐层子集 / AAM 开关 / PGM 分割损失开关
        abl = getattr(de_parallel(model), "ablation", {}) or {}
        self.align_levels = abl.get("align_levels", [0, 1, 2])
        self.use_aam = abl.get("use_aam", True)
        self.use_pgm = abl.get("use_pgm", True)
        self.alpha = abl.get("alpha", 0.1)   # L_bce 权重 (论文 Table 5 搜索)
        self.beta = abl.get("beta", 0.1)     # L_adaptive_MMD 权重
        # 新旋钮: 对齐模式("intra"=原类内MMD / "can"=CAN风格 同类拉近+紧致+异类推远); 分割损失加 Dice
        self.align_mode = abl.get("align_mode", "intra")
        self.seg_dice = abl.get("seg_dice", False)
        # AAM 实例提取: ROIAlign output_size(默认7); 是否对齐时关闭目标域特征梯度(默认否)
        self.aam_roi_size = int(abl.get("aam_roi_size", 7))
        self.aam_detach_target = bool(abl.get("aam_detach_target", False))
        # 新对齐模块: 实例级域判别器 + GRL 非对称对抗 + EMA 类原型(由模型持有以纳入优化器/buffer)
        self.dom_proj = getattr(de_parallel(model), "dom_proj", None)
        self.dom_disc = getattr(de_parallel(model), "dom_disc", None)
        self.dom_proto = getattr(de_parallel(model), "dom_proto", None)
        self.dom_proto_ready = getattr(de_parallel(model), "dom_proto_ready", None)
        self.dom_levels = getattr(de_parallel(model), "dom_levels", [1, 2])
        self.align_method = abl.get("align_method", "mmd")   # mmd / adv
        self.proto_weight = abl.get("proto_weight", 0.1)      # L_proto 权重
        self.adv_margin = abl.get("adv_margin", 1.0)          # 异类排斥 hinge margin
        self.ema_momentum = abl.get("ema_momentum", 0.9)      # 目标类原型 EMA 动量
        self.adv_lambda = 1.0                                 # GRL λ, 由训练脚本每 epoch 设置

    def __call__(self, p, targets, feature_s=None, feature_t=None, mask=None, ):  # predictions, targets
        """总损失(论文): L_total = L_det + α·L_bce + β·L_adaptive_MMD, α=β=0.1。"""
        # ----- PGM 分割损失 L_bce: BCE(预测前景概率图, 掩码GT); 消融可关闭(use_pgm) -----
        fg_loss = torch.zeros(1, device=self.device)
        if self.use_pgm and mask is not None and mask[0] is not None and mask[1] is not None:
            mask1_ds = F.interpolate(mask[1], size=mask[0].shape[-2:], mode="nearest")
            fg_loss = foreground_loss(mask[0], mask1_ds).to(self.device)
            if self.seg_dice:  # BCE + Dice(锐化边界, 抗前景占比小)
                fg_loss = fg_loss + dice_loss(mask[0], mask1_ds[:, 0:1].float() / 255.0).to(self.device)

        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        lmmd = torch.zeros(1, device=self.device)  # mmd loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            if n := b.shape[0]:  #海象运算符,赋值
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        # ----- AAM 多尺度域对齐 L_adaptive_MMD (论文 Eq.2/5/6) -----
        # D2: 投影网络降维 -> 按目标框切片+区域平均池化得到实例嵌入 -> 实例级 MMD, 权重 A_i=softmax(可学习 s)。
        # 消融: use_aam 关闭则 β=0; align_levels 选择对齐的层子集(Table 4)。
        if self.use_aam and self.align_method != "adv" and feature_s is not None:
            levels = [i for i in self.align_levels if 0 <= i < len(feature_s)]
            use_inst = (self.aam_proj is not None and self.bs_s is not None and targets is not None
                        and len(feature_s) > 0 and feature_s[0].shape[0] > self.bs_s)
            if levels and use_inst:
                A = torch.softmax(self.aam_s.to(self.device)[levels], dim=0)
                # 当前 batch: targets 中图像下标 < 源域图像数为源域, 其余为本 batch 目标域。
                src_sel = targets[targets[:, 0] < self.bs_s]
                tgt_sel = targets[targets[:, 0] >= self.bs_s].clone()
                tgt_sel[:, 0] -= self.bs_s  # 目标域特征 feature_s[i][bs_s:] 内部下标从 0 开始
                src_boxes, src_idx, src_cls = src_sel[:, 2:6], src_sel[:, 0], src_sel[:, 1]
                tgt_boxes, tgt_idx, tgt_cls = tgt_sel[:, 2:6], tgt_sel[:, 0], tgt_sel[:, 1]
                if src_sel.numel() > 0 and tgt_sel.numel() > 0:
                    for k, i in enumerate(levels):
                        # zs = self.aam_proj[i](feature_s[i][: self.bs_s])     # 源域投影特征
                        zs = feature_s[i][: self.bs_s]
                        _ft = feature_s[i][self.bs_s:]                       # 当前 batch 目标域特征
                        if self.aam_detach_target:                           # 关闭目标域特征梯度(非对称对齐)
                            _ft = _ft.detach()
                        zt = _ft
                        # zt = self.aam_proj[i](_ft)                           # 目标域投影特征
                        si, sc = roi_align_instances(zs, src_boxes, src_idx, src_cls, out_size=self.aam_roi_size)
                        ti, tc = roi_align_instances(zt, tgt_boxes, tgt_idx, tgt_cls, out_size=self.aam_roi_size)
                        if si is None or ti is None:
                            continue
                        si = F.normalize(si, dim=1)
                        ti = F.normalize(ti, dim=1)
                        if self.align_mode == "can":
                            # CAN 风格: 跨域同类拉近 + 同域同类紧致 + 跨域异类推远
                            lmmd += A[k] * class_align_can(si, sc, ti, tc).to(self.device)
                            continue
                        # 类内对齐(论文 class-consistent): 仅同类实例间算 MMD, 跨类不混; 各类取平均
                        classes = torch.unique(torch.cat([sc, tc]))
                        mmd_c, n_used = 0.0, 0
                        for c in classes:
                            sic, tic = si[sc == c], ti[tc == c]
                            if sic.size(0) > 1 and tic.size(0) > 1:
                                mmd_c = mmd_c + mmd_inst(sic, tic).to(self.device)
                                n_used += 1
                        if n_used > 0:
                            lmmd += A[k] * (mmd_c / n_used)
            elif levels and feature_t is not None:  # 回退: 无实例信息时退化为整图级 MMD(旧行为)
                A = torch.softmax(self.aam_s.to(self.device)[levels], dim=0)
                for k, i in enumerate(levels):
                    src_feat = feature_s[i][: self.bs_s] if self.bs_s is not None else feature_s[i]
                    lmmd += A[k] * mmd_multi(src_feat, feature_t[i]).to(self.device)

        # ----- 新对齐模块: 实例级域判别器 + GRL 非对称对抗 + EMA 类原型 (替代 MMD) -----
        ladv = torch.zeros(1, device=self.device)
        lproto = torch.zeros(1, device=self.device)
        if (self.use_aam and self.align_method == "adv" and feature_s is not None and feature_t is not None
                and self.dom_proj is not None and self.dom_disc is not None
                and self.t_boxes is not None and self.bs_s is not None and targets is not None):
            src_sel = targets[targets[:, 0] < self.bs_s]
            src_boxes, src_idx, src_cls = src_sel[:, 2:6], src_sel[:, 0], src_sel[:, 1]
            tb = self.t_boxes.to(self.device)
            tgt_boxes, tgt_idx, tgt_cls = tb[:, 2:6], tb[:, 0], tb[:, 1]
            lam = float(self.adv_lambda)
            mmt = float(self.ema_momentum)
            n_used = 0
            for k, li in enumerate(self.dom_levels):
                if li >= len(feature_s):
                    continue
                zs = self.dom_proj[k](feature_s[li][: self.bs_s].float())   # 源域投影(带梯度)
                zt = self.dom_proj[k](feature_t[li].float())                # 目标域投影
                si, sc = roi_align_instances(zs, src_boxes, src_idx, src_cls, out_size=7)
                ti, tc = roi_align_instances(zt, tgt_boxes, tgt_idx, tgt_cls, out_size=7)
                if si is None or ti is None:
                    continue
                si = F.normalize(si, dim=1)
                ti = F.normalize(ti, dim=1)
                for c in range(self.nc):
                    s_c = si[sc == c]
                    t_c = ti[tc == c]
                    if s_c.numel() == 0 or t_c.numel() == 0:   # 两域均非空才计(对齐 domadv 文档)
                        continue
                    disc = self.dom_disc[c]
                    # ① 条件域对抗: source 过 GRL 标 0(域混淆), target detach 作 anchor 标 1
                    ds = disc(grad_reverse(s_c, lam))
                    dt = disc(t_c.detach())
                    ladv = ladv + F.binary_cross_entropy_with_logits(ds, torch.zeros_like(ds)) \
                                + F.binary_cross_entropy_with_logits(dt, torch.ones_like(dt))
                    # ② EMA 目标类原型更新(detach)
                    tc_mean = t_c.detach().mean(0)
                    if float(self.dom_proto_ready[k, c]) < 1:
                        self.dom_proto[k, c] = tc_mean
                        self.dom_proto_ready[k, c] = 1.0
                    else:
                        self.dom_proto[k, c] = mmt * self.dom_proto[k, c] + (1 - mmt) * tc_mean
                    # ②(续) 原型约束: 原型先 L2 归一化; 同类吸引 + 异类 hinge 排斥
                    proto_c = F.normalize(self.dom_proto[k, c], dim=0)
                    s_mean = s_c.mean(0)
                    lproto = lproto + ((s_mean - proto_c) ** 2).sum()
                    for c2 in range(self.nc):
                        if c2 == c or float(self.dom_proto_ready[k, c2]) < 1:
                            continue
                        proto_o = F.normalize(self.dom_proto[k, c2], dim=0)
                        lproto = lproto + torch.clamp(self.adv_margin - ((s_mean - proto_o) ** 2).sum(), min=0.0)
                    n_used += 1
            if n_used > 0:
                ladv = ladv / n_used
                lproto = lproto / n_used

        bs = tobj.shape[0]  # batch size
        # 总损失: L_det + α·L_bce + (β·L_adaptive_MMD  或  β·L_adv + w·L_proto)
        loss = lbox + lobj + lcls + self.alpha * fg_loss
        if self.align_method == "adv":
            loss = loss + self.beta * ladv + self.proto_weight * lproto
            align_report = (self.beta * ladv + self.proto_weight * lproto).detach()
        else:
            loss = loss + self.beta * lmmd
            align_report = lmmd.detach()
        loss = loss + 0.0 * self.aam_s.sum()  # 确保 aam_s 始终有梯度路径(DDP 安全)
        return loss * bs, torch.cat((lbox, lobj, lcls, align_report.view(1), fg_loss)).detach()

    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch




class ComputeLoss2:
    """Computes the total loss for YOLOv5 model predictions, including classification, box, and objectness losses."""

    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets, feature_s, feature_t):  # predictions, targets
        """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""


        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        lmmd = torch.zeros(1, device=self.device)  # mmd loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            if n := b.shape[0]:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        if feature_s is not None:

            lmmd = mmd(feature_s, feature_t).to(self.device)
            
        lmmd *= self.hyp["mmd"]
        bs = tobj.shape[0]  # batch size
        
        return (lbox + lobj + lcls + lmmd) * bs, torch.cat((lbox, lobj, lcls, lmmd)).detach()

    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
