import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import config

class MultiScaleSRMLayer(nn.Module):
    """
    Multi-Scale Spatial Rich Model (MS-SRM) High-Pass Frequency Residual Extractor.
    Extracts high-frequency noise residuals across 3x3, 5x5, and 7x7 spatial filter kernels.
    """
    def __init__(self):
        super().__init__()
        # 1. Standard 3x3 SRM High-Pass Kernels
        f1_3 = [[0, 0, 0], [0, 1, -1], [0, 0, 0]]
        f2_3 = [[0, 1, 0], [0, -1, 0], [0, 0, 0]]
        f3_3 = [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]
        filters_3 = torch.tensor([f1_3, f2_3, f3_3], dtype=torch.float32)

        # 2. 5x5 SRM High-Pass Filter (Edge & Blending Seams)
        f_5 = [
            [0,  0, -1,  0,  0],
            [0, -1,  2, -1,  0],
            [-1, 2, -4,  2, -1],
            [0, -1,  2, -1,  0],
            [0,  0, -1,  0,  0]
        ]
        filter_5 = torch.tensor([f_5], dtype=torch.float32)

        # 3. 7x7 SRM High-Pass Filter (Broader Texture Artifacts)
        f_7 = [
            [ 0,  0,  0, -1,  0,  0,  0],
            [ 0,  0, -1,  2, -1,  0,  0],
            [ 0, -1,  2, -4,  2, -1,  0],
            [-1,  2, -4,  8, -4,  2, -1],
            [ 0, -1,  2, -4,  2, -1,  0],
            [ 0,  0, -1,  2, -1,  0,  0],
            [ 0,  0,  0, -1,  0,  0,  0]
        ]
        filter_7 = torch.tensor([f_7], dtype=torch.float32)

        # Build depthwise convolutions for RGB channels
        self.conv_3x3 = nn.Conv2d(3, 9, kernel_size=3, padding=1, bias=False, groups=3)
        self.conv_5x5 = nn.Conv2d(3, 3, kernel_size=5, padding=2, bias=False, groups=3)
        self.conv_7x7 = nn.Conv2d(3, 3, kernel_size=7, padding=3, bias=False, groups=3)

        # Weights assignment
        w3 = torch.zeros(9, 1, 3, 3)
        for i in range(3):
            for j in range(3):
                w3[i * 3 + j, 0, :, :] = filters_3[j] / (4.0 if j == 2 else 1.0)
        self.conv_3x3.weight.data.copy_(w3)
        self.conv_3x3.weight.requires_grad = False

        w5 = torch.zeros(3, 1, 5, 5)
        for i in range(3):
            w5[i, 0, :, :] = filter_5 / 8.0
        self.conv_5x5.weight.data.copy_(w5)
        self.conv_5x5.weight.requires_grad = False

        w7 = torch.zeros(3, 1, 7, 7)
        for i in range(3):
            w7[i, 0, :, :] = filter_7 / 16.0
        self.conv_7x7.weight.data.copy_(w7)
        self.conv_7x7.weight.requires_grad = False

    def forward(self, x):
        res3 = self.conv_3x3(x)
        res5 = self.conv_5x5(x)
        res7 = self.conv_7x7(x)
        return torch.cat([res3, res5, res7], dim=1) # 15 channels output


class SpatialFrequencyCrossAttention(nn.Module):
    """
    Spatial-Frequency Cross-Attention (SFCA) Module.
    Attends spatial RGB feature maps (Queries) to MS-SRM frequency residual maps (Keys & Values).
    """
    def __init__(self, spatial_dim, freq_dim, embed_dim=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.query_proj = nn.Conv2d(spatial_dim, embed_dim, kernel_size=1)
        self.key_proj = nn.Conv2d(freq_dim, embed_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(freq_dim, embed_dim, kernel_size=1)
        self.scale = self.head_dim ** -0.5
        self.out_proj = nn.Conv2d(embed_dim, spatial_dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(spatial_dim)

    def forward(self, rgb_map, freq_map):
        if freq_map.shape[2:] != rgb_map.shape[2:]:
            freq_map = F.interpolate(freq_map, size=rgb_map.shape[2:], mode='bilinear', align_corners=False)

        B, C_rgb, H, W = rgb_map.shape
        N = H * W

        Q = self.query_proj(rgb_map).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2) # [B, heads, N, head_dim]
        K = self.key_proj(freq_map).view(B, self.num_heads, self.head_dim, N)                     # [B, heads, head_dim, N]
        V = self.value_proj(freq_map).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2) # [B, heads, N, head_dim]

        attn = torch.matmul(Q, K) * self.scale                                                    # [B, heads, N, N]
        attn = F.softmax(attn, dim=-1)

        context = torch.matmul(attn, V).permute(0, 1, 3, 2).reshape(B, self.embed_dim, H, W)       # [B, embed_dim, H, W]
        attended = self.out_proj(context)                                                         # [B, C_rgb, H, W]

        return self.norm(rgb_map + attended)

class EnhancedHead(nn.Module):
    def __init__(self, in_features, dropout=0.3):
        super().__init__()
        # Enhanced head: FC(512) -> BN -> ReLU -> Dropout -> FC(1)
        self.fc1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.fc1(x)
        if x.size(0) > 1 or not self.training:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        return out, x # Return features for attention pooling

class DeepfakeModel(nn.Module):
    def __init__(self, model_name, pretrained=True, branch_mode="fusion"):
        super().__init__()
        self.branch_mode = branch_mode # 'rgb', 'freq', 'fusion'
        
        # 1. RGB Backbone
        sd_prob = getattr(config, "STOCHASTIC_DEPTH_PROB", 0.35)
        if model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.rgb_backbone = models.efficientnet_b0(weights=weights, stochastic_depth_prob=sd_prob).features
            rgb_features = 1280
        elif model_name == "efficientnet_b4":
            weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            self.rgb_backbone = models.efficientnet_b4(weights=weights, stochastic_depth_prob=sd_prob).features
            rgb_features = 1792
        elif model_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.rgb_backbone = models.convnext_tiny(weights=weights).features
            rgb_features = 768
        elif model_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.rgb_backbone = models.mobilenet_v3_small(weights=weights).features
            rgb_features = 576
        elif model_name == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.rgb_backbone = nn.Sequential(*list(m.children())[:-2])
            rgb_features = 512
        elif model_name == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.rgb_backbone = nn.Sequential(*list(m.children())[:-2])
            rgb_features = 2048
        elif model_name == "shufflenet_v2":
            m = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None)
            self.rgb_backbone = nn.Sequential(m.conv1, m.maxpool, m.stage2, m.stage3, m.stage4, m.conv5)
            rgb_features = 1024
        elif model_name == "densenet121":
            self.rgb_backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None).features
            rgb_features = 1024
        else:
            raise ValueError(f"Unsupported backbone: {model_name}")

        # Freeze percentage of backbone (only when PROGRESSIVE_UNFREEZE is active)
        num_features = len(self.rgb_backbone)
        num_freeze = int(num_features * config.FREEZE_PERCENT) if getattr(config, "PROGRESSIVE_UNFREEZE", False) else 0
        for layer in self.rgb_backbone[:num_freeze]:
            for param in layer.parameters():
                param.requires_grad = False
                
        # 2. Multi-Scale Frequency Branch (MS-SRM + CNN)
        if self.branch_mode in ["freq", "fusion"]:
            self.srm = MultiScaleSRMLayer()
            # Normalize 15-channel SRM residuals (raw high-pass responses on ImageNet-normalized input)
            # before feeding into freq_convs to prevent early-training instability.
            self.srm_norm = nn.BatchNorm2d(15, affine=True)
            # Lightweight 2D Conv feature extractor taking 15 channels (3x3: 9 + 5x5: 3 + 7x7: 3)
            self.freq_convs = nn.Sequential(
                nn.Conv2d(15, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            freq_features = 128
        else:
            freq_features = 0

        # Spatial-Frequency Cross-Attention Module for Fusion Mode
        if self.branch_mode == "fusion":
            self.sfca = SpatialFrequencyCrossAttention(spatial_dim=rgb_features, freq_dim=freq_features, embed_dim=256)

        # Fusion & Head
        if self.branch_mode == "rgb":
            total_features = rgb_features
        elif self.branch_mode == "freq":
            total_features = freq_features
        else:
            total_features = rgb_features + freq_features

        self.head = EnhancedHead(total_features, config.DROPOUT)

    @property
    def features(self):
        return self.rgb_backbone

    def forward(self, x):
        features = []
        
        if self.branch_mode == "rgb":
            rgb_map = self.rgb_backbone(x)
            rgb_f = F.adaptive_avg_pool2d(rgb_map, 1).flatten(1)
            features.append(rgb_f)
            
        elif self.branch_mode == "freq":
            freq_x = self.srm(x)
            freq_x = self.srm_norm(freq_x)  # Normalize residuals before freq_convs
            freq_map = self.freq_convs(freq_x)
            freq_f = F.adaptive_avg_pool2d(freq_map, 1).flatten(1)
            features.append(freq_f)
            
        elif self.branch_mode == "fusion":
            rgb_map = self.rgb_backbone(x)
            freq_x = self.srm(x)
            freq_x = self.srm_norm(freq_x)  # Normalize residuals before freq_convs
            freq_map = self.freq_convs(freq_x)
            
            # Apply Spatial-Frequency Cross-Attention
            attended_rgb_map = self.sfca(rgb_map, freq_map)
            
            rgb_f = F.adaptive_avg_pool2d(attended_rgb_map, 1).flatten(1)
            freq_f = F.adaptive_avg_pool2d(freq_map, 1).flatten(1)
            
            features.append(rgb_f)
            features.append(freq_f)
            
        fused = torch.cat(features, dim=1) if len(features) > 1 else features[0]
        out, feat = self.head(fused)
        return out, feat

def build_model(model_name, pretrained=None, branch_mode=None):
    if pretrained is None:
        pretrained = getattr(config, 'PRETRAINED', True)
    if branch_mode is None:
        branch_mode = getattr(config, 'BRANCH_MODE', 'fusion')
    return DeepfakeModel(model_name, pretrained, branch_mode=branch_mode)
