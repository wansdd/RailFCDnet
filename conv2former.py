
import torch
import torch.nn as nn
from thop import profile
from thop import clever_format

# 论文题目：Conv2Former: A Simple Transformer-Style ConvNet for Visual Recognition
# 中文题目：Conv2Former: 一种简单的视觉识别用的Transformer风格卷积网络
# 论文链接：https://arxiv.org/pdf/2211.11943
# 官方github：https://github.com/HVision-NKU/Conv2Former
# 所属机构：天津南开大学计算机科学学院，字节跳动（新加坡）
# 代码整理：微信公众号《AI缝合术》
# 全部即插即用模块代码：https://github.com/AIFengheshu/Plug-play-modules
import torch.nn.functional as F
class LayerAttentionFusion2(nn.Module):
    def __init__(self, in_dims=[320, 640, 1280], out_dim=1280):
        super(LayerAttentionFusion, self).__init__()
        self.num_layers = len(in_dims)
        self.out_dim = out_dim

        # 用 1x1 conv 把每层映射到统一维度
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ) for in_dim in in_dims
        ])

        # 可学习权重（初始化为1）
        self.weights = nn.Parameter(torch.ones(self.num_layers))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, feats):  # feats 是长度为3的 list: [b, 320,32,32], [b,640,16,16], [b,1280,8,8]
        assert len(feats) == self.num_layers
        weights = self.softmax(self.weights)  # shape: [3]
        fused = 0

        # Resize 所有层到目标分辨率（例如最大那层的大小：32x32）
        target_size = feats[0].size()[2:]  # (H, W) from layer 4

        for i in range(self.num_layers):
            f = self.proj_layers[i](feats[i])  # [b, out_dim, H_i, W_i]
            if f.shape[2:] != target_size:
                f = F.interpolate(f, size=target_size, mode='nearest')
            fused += weights[i] * f  # 加权求和

        return fused  # [b, out_dim, H, W]


import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerAttentionFusion(nn.Module):
    def __init__(self, in_dims=[320, 640, 1280], out_dims=[320, 640, 1280]):
        super(LayerAttentionFusion, self).__init__()
        assert len(in_dims) == len(out_dims), "in_dims 和 out_dims 长度必须一致"
        self.num_layers = len(in_dims)
        self.out_dims = out_dims

        # 每层一个独立的1x1 conv，仅用于保持结构一致和可能的非线性变化（不改变通道）
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1),
                # nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
                # nn.Conv2d(in_dim, out_dim, kernel_size=1),
                # nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_dim),
            ) for in_dim, out_dim in zip(in_dims, out_dims)
        ])

        # 每层一个可学习融合权重
        self.weights = nn.Parameter(torch.ones(self.num_layers))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, feats):  # feats: list of [B, C_i, H_i, W_i]
        assert len(feats) == self.num_layers
        weights = self.softmax(self.weights)  # shape: [num_layers]

        projected_feats = []
        for i in range(self.num_layers):
            f = self.proj_layers[i](feats[i])  # [B, C_i, H_i, W_i]
            projected_feats.append(f)

        return projected_feats, weights


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class ConvModulOperationSpatialAttention(nn.Module):
    def __init__(self, dim, kernel_size, expand_ratio=2):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.att = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)        
        x = self.att(x) * self.v(x)
        x = self.proj(x)
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

        # self.pool = nn.AdaptiveAvgPool2d(1)  # 输出 B, 1280, 1, 1
    def forward(self, x):
        x = self.ranker(x)
        # x = self.pool(x)
        return x


class Ranker_dropout(nn.Module):  # input shape: [B, 1280, 4, 4]
    def __init__(self, in_dim=1280, drop_prob=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.ranker = nn.Sequential(
            nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_prob)  # ✅ dropout along channel dimension
        )

    def forward(self, x):
        x = self.ranker(x)
        return x

if __name__ == "__main__":
    # 将模块移动到 GPU（如果可用）
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # 创建测试输入张量 (batch_size, channels, height, width)
    x = torch.randn(1, 1280, 4, 4).to(device)
    # 初始化 ConvSAtt 模块
    ConvSAtt = ConvModulOperationSpatialAttention(dim=1280, kernel_size=3)
    print("微信公众号:AI缝合术")
    ConvSAtt = ConvSAtt.to(device)
    # 前向传播
    output = ConvSAtt(x)
    # 打印输入和输出张量的形状
    print("输入张量形状:", x.shape)
    print("输出张量形状:", output.shape)
    model = Ranker2()
    model.train()
    model2 = ConvModulOperationSpatialAttention(dim=1280, kernel_size=3)
    x = torch.randn(4, 1280, 4, 4)
    y = model(x)
    print(y.shape)  # torch.Size([4, 1280, 4, 4])
    flops, params = profile(model, inputs=(x, ))
    flops2, params2 = profile(model2, inputs=(x, ))

    # 格式化输出
    flops, params = clever_format([flops, params], "%.3f")
    flops2, params2 = clever_format([flops2, params2], "%.3f")

    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
    print(f"FLOPs: {flops2}")
    print(f"Params: {params2}")