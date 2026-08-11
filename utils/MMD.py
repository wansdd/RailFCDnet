import torch
from models.experimental import attempt_load
from utils.torch_utils import select_device
# from models.yolo_feature import Model_feature
import random
import torch.nn as nn

# this feature extraction method exists bug !!!
"""
def get_ins_feature(feature, targets, device, number=100):
    feat_w, feat_h = feature.shape[2], feature.shape[3]
    ins_set = torch.tensor([]).to(device)
    targets_ins = random_sample_ins(targets, number)
    for ins in targets_ins:
        img_idx = int(ins[0])
        img_cls = ins[1] # for multi class api
        x,y,w,h = ins[2], ins[3], ins[4], ins[5]
        # x1 < x2; y1 < y2
        # now exsting bugs !!!
        x1, x2 = max(int(feat_w*(x-w)/2), 0), max(int(feat_w*(x+w)/2), int(feat_w*(x-w)/2)+1)
        y1, y2 = max(int(feat_h*(y-h)/2), 0), max(int(feat_h*(y+h)/2), int(feat_h*(y-h)/2)+1)
        #print(img_idx)
        #print(x1, x2, y1, y2)
        #print(feature.shape)
        o_ins = feature[img_idx, :, x1:x2, y1:y2].mean(2).mean(1).unsqueeze(0)
        #print(o_ins.shape)
        ins_set=torch.cat((ins_set, o_ins))
    return ins_set
"""
def get_ins_feature(feature, targets, device, number=100, feature_level=1):
    """
    从YOLO检测输出中提取实例特征
    
    Args:
        feature: YOLO检测头的输出，包含3个不同尺度的特征图的列表/元组
        targets: 目标标注信息
        device: 设备
        number: 随机采样的实例数量
        feature_level: 选择哪个特征层 (0, 1, 2)，0是最大尺度，2是最小尺度
    """
    # 检查feature是否为列表或元组
    if isinstance(feature, (list, tuple)):
        if len(feature) == 0:
            raise ValueError("Feature list is empty")
        # 选择指定的特征层
        selected_feature = feature[feature_level]
    else:
        selected_feature = feature
    
    # 检查特征维度
    if len(selected_feature.shape) == 5:
        # YOLO检测头输出: [batch, anchors, height, width, predictions]
        batch_size, num_anchors, feat_h, feat_w, predictions = selected_feature.shape
        
        # 对于YOLO检测头输出，我们需要重新组织数据
        # 方法1: 直接使用前4个预测值作为特征 (x, y, w, h)
        selected_feature = selected_feature[:, :, :, :, :4]  # 只取前4个通道
        
        # 重新调整形状: [batch, anchors*4, height, width]
        selected_feature = selected_feature.permute(0, 4, 1, 2, 3).contiguous()
        selected_feature = selected_feature.view(batch_size, 4 * num_anchors, feat_h, feat_w)
        
        # 或者方法2: 平均所有anchor的特征
        # selected_feature = selected_feature.mean(dim=1)  # 平均anchor维度
        # selected_feature = selected_feature.permute(0, 3, 1, 2).contiguous()  # [B, predictions, H, W]
        
    elif len(selected_feature.shape) == 4:
        # 标准特征图格式: [batch, channels, height, width]
        feat_h, feat_w = selected_feature.shape[2], selected_feature.shape[3]
    else:
        raise ValueError(f"Unexpected feature shape: {selected_feature.shape}")
    
    # 更新特征图尺寸
    feat_h, feat_w = selected_feature.shape[2], selected_feature.shape[3]
    
    ins_set = torch.tensor([]).to(device)
    targets_ins = random_sample_ins(targets, number)
    
    for ins in targets_ins:
        img_idx = int(ins[0])
        img_cls = ins[1]  # for multi class api
        x, y, w, h = ins[2], ins[3], ins[4], ins[5]
        
        # 计算边界框在特征图上的坐标
        # 注意：这里假设坐标是归一化的（0-1之间）
        x1 = max(int(feat_w * (x - 0.5 * w)), 0)
        x2 = min(int(feat_w * (x + 0.5 * w)), feat_w - 1)
        y1 = max(int(feat_h * (y - 0.5 * h)), 0)
        y2 = min(int(feat_h * (y + 0.5 * h)), feat_h - 1)
        
        # 确保x2 > x1, y2 > y1
        x2 = max(x2, x1 + 1)
        y2 = max(y2, y1 + 1)
        
        # 提取实例特征 (注意维度顺序: [batch, channels, height, width])
        o_ins = selected_feature[img_idx, :, y1:y2, x1:x2].mean(2).mean(1).unsqueeze(0)
        ins_set = torch.cat((ins_set, o_ins))
    
    return ins_set

def get_ins_feature_by_class(feature, targets, device, number=100, feature_level=1):
    """
    按照类别提取实例特征，并返回每个类别的特征集合
    
    Args:
        feature: YOLO检测头的输出
        targets: 目标标注信息
        device: 设备
        number: 每个类别随机采样的实例数量
        feature_level: 选择哪个特征层 (0, 1, 2)
        
    Returns:
        tuple: (class0_features, class1_features) - 类别0和类别1的特征集合
    """
    # 检查feature是否为列表或元组
    if isinstance(feature, (list, tuple)):
        if len(feature) == 0:
            raise ValueError("Feature list is empty")
        # 选择指定的特征层
        selected_feature = feature[feature_level]
    else:
        selected_feature = feature
    
    # 检查特征维度并处理
    if len(selected_feature.shape) == 5:
        batch_size, num_anchors, feat_h, feat_w, predictions = selected_feature.shape
        selected_feature = selected_feature[:, :, :, :, :4]
        selected_feature = selected_feature.permute(0, 4, 1, 2, 3).contiguous()
        selected_feature = selected_feature.view(batch_size, 4 * num_anchors, feat_h, feat_w)
    elif len(selected_feature.shape) == 4:
        feat_h, feat_w = selected_feature.shape[2], selected_feature.shape[3]
    else:
        raise ValueError(f"Unexpected feature shape: {selected_feature.shape}")
    
    # 更新特征图尺寸
    feat_h, feat_w = selected_feature.shape[2], selected_feature.shape[3]
    
    # 创建每个类别的特征集合
    class0_features = torch.tensor([]).to(device)
    class1_features = torch.tensor([]).to(device)
    
    # 按类别分组targets
    class0_targets = targets[targets[:, 1] == 0]
    class1_targets = targets[targets[:, 1] == 1]
    
    # 对每个类别随机采样实例
    class0_samples = random_sample_ins(class0_targets, number)
    class1_samples = random_sample_ins(class1_targets, number)
    
    # 处理类别0的实例
    for ins in class0_samples:
        img_idx = int(ins[0])
        x, y, w, h = ins[2], ins[3], ins[4], ins[5]
        
        # 计算边界框在特征图上的坐标
        x1 = max(int(feat_w * (x - 0.5 * w)), 0)
        x2 = min(int(feat_w * (x + 0.5 * w)), feat_w - 1)
        y1 = max(int(feat_h * (y - 0.5 * h)), 0)
        y2 = min(int(feat_h * (y + 0.5 * h)), feat_h - 1)
        
        # 确保x2 > x1, y2 > y1
        x2 = max(x2, x1 + 1)
        y2 = max(y2, y1 + 1)
        
        # 提取实例特征
        o_ins = selected_feature[img_idx, :, y1:y2, x1:x2].mean(2).mean(1).unsqueeze(0)
        class0_features = torch.cat((class0_features, o_ins))
    
    # 处理类别1的实例
    for ins in class1_samples:
        img_idx = int(ins[0])
        x, y, w, h = ins[2], ins[3], ins[4], ins[5]
        
        # 计算边界框在特征图上的坐标
        x1 = max(int(feat_w * (x - 0.5 * w)), 0)
        x2 = min(int(feat_w * (x + 0.5 * w)), feat_w - 1)
        y1 = max(int(feat_h * (y - 0.5 * h)), 0)
        y2 = min(int(feat_h * (y + 0.5 * h)), feat_h - 1)
        
        # 确保x2 > x1, y2 > y1
        x2 = max(x2, x1 + 1)
        y2 = max(y2, y1 + 1)
        
        # 提取实例特征
        o_ins = selected_feature[img_idx, :, y1:y2, x1:x2].mean(2).mean(1).unsqueeze(0)
        class1_features = torch.cat((class1_features, o_ins))
    
    return class0_features, class1_features



def random_sample_ins(targets, number):
    n = targets.shape[0]
    k = number
    # print("number of targets is", n)
    # print("considered number is", k)
    if n <= k:
        #print("continue")
        return targets
    else:
        # print("seleted")
        indices = torch.tensor(random.sample(range(n), k))
        targets_ = targets[indices]
        return targets_

def get_feature(img, model): 
    model.eval()
    img_feature = model(img)[1] # torch.tensor [B 1280 H/32 W/32]
    img_feature = img_feature.mean(3).mean(2)
    return img_feature  # torch.tensor [B, 1280] feature

def get_feature_train(img, model): 
    img_feature = model(img)[1] # torch.tensor [B 1280 H/32 W/32]
    img_feature = img_feature.mean(3).mean(2)
    return img_feature  # torch.tensor [B, 1280] feature

# To get the weight for source and target samples in task-oriented supervised trainng
def MMD_weight(feature_s, feature_t, k): # S: [B1 1280] T: [B2 1280] 
    feature_s = feature_s - feature_t.mean(0) # boardcast
    feature_s = feature_s.mul(feature_s) # multiply by every position

    feature_s = feature_s.sum(dim=1)

    topk_index = feature_s.topk(k=int(feature_s.shape[0]*k), largest = False)[1]
    #if feature_s.shape[0] > k:
    #    topk_index = feature_s.topk(k=k, largest = False)[1]
    #else: 
    #    topk_index = torch.arange(0, feature_s.shape[0])
    
    batch_size = feature_s.shape[0]
    weight_ini = torch.zeros(batch_size)
    weight_ini[topk_index] = 1.0
    weight = weight_ini.clone()
    
    return weight # return sample's weight defined by MMD distance

def MMD_distance(f_S, f_T, k): # S: [B1 1280] T: [B2 1280] 这里的f_T其实代表源域特征 返回源域当中的topk个
    f_T = f_T - f_S.mean(0) # 广播 每行都会减小
    f_T = f_T.mul(f_T) #对位相乘

    f_T = f_T.sum(dim=1)

    if f_T.shape[0] > k:
        T_topk = f_T.topk(k=k, largest = False)
    else: 
        return torch.arange(0, f_T.shape[0])
    return T_topk[1] # return topk_idx

def MMD_distance_v2(f_t, f_s, k): # S: [B1 1280] T: [B2 1280] 这里的f_T其实代表源域特征 返回源域当中的topk个
    time = 0
    for i in range(f_t.shape[0]):
    # print(i)
        f_i = f_s - f_t[i]
        f_i = f_i.mul(f_i).sum(dim=1)

        if f_i.shape[0] > k:
            i_topk = f_i.topk(k=k, largest = False)[1]
        else: 
            return torch.arange(0, f_i.shape[0])
        
        if time == 0:
            f_topk = i_topk 
        else:
            f_topk = torch.cat((f_topk, i_topk), 0)
        time += 1
    idx, counts = f_topk.unique(sorted=False, return_counts=True)
        
    return idx[counts.topk(k=k, largest = True)[1]] # return topk_idx

def cosine_distance(f_S, f_T, k): # f_S: [B1 1280] 所有target图片的特征 T: [B2 1280]
    dist_tensor = []
    for i in f_T:
        i = i.unsqueeze(0)
        dist = (1-torch.cosine_similarity(f_S,i,dim=1)).sum()
        dist_tensor.append(dist)
        
    f_T = torch.tensor(dist_tensor)

    if f_T.shape[0] > k:
        T_topk = f_T.topk(k=k, largest = False)
    else: 
        return torch.arange(0, f_T.shape[0])
    return T_topk[1] # T中和 

def choice_topk(imgs, targets, paths, topk_index):
    paths_refine = []
    labels_refine = torch.tensor([])

    for i in list(topk_index):  #  这里可能遇到一种情况  就是要补充上的样本数目小于筛选后的样本
        paths_refine.append(paths[i])
        
    for index in topk_index:
        t = targets[targets[:, 0] == index, :]
        if t.shape[0] > 0:
            number = torch.nonzero((topk_index == t[0][0]))[0][0]
            t[:, 0] = number
        labels_refine = torch.cat((labels_refine, t), dim = 0)
    
    imgs_refine = imgs[topk_index, :, :, :]
    return imgs_refine, labels_refine, paths_refine

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

    # 计算多核中每个核的bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val) # 将多个核合并在一起
