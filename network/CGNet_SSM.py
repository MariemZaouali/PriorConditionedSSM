import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ============================================================
# Basic Convolution Block
# ============================================================
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# ============================================================
# Recursive Prior State Space (RPSS) Module
# ============================================================
class RecursivePriorStateSpace(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, use_detach=True):
        super().__init__()
        self.use_detach = use_detach
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, 1)
        self.A = nn.Parameter(torch.randn(hidden_dim))
        self.B = nn.Parameter(torch.randn(hidden_dim))
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.output_proj = nn.Conv2d(hidden_dim, in_channels, 1)

    def forward(self, x, prior):
        Bsz, C, H, W = x.shape
        
        # Injection du Prior
        prior = F.interpolate(prior, size=(H, W), mode="bilinear", align_corners=True)
        prior = torch.clamp(prior, -1.0, 1.0).expand(Bsz, C, H, W)
        F_mod = x + self.alpha * prior
        x_proj = self.input_proj(F_mod)

        A = torch.tanh(self.A).view(1, -1, 1)
        B = self.B.view(1, -1, 1)

        # Balayage horizontal et vertical
        def scan(feat, dim): # dim: 0 for W, 1 for H
            steps = []
            prev = None
            size = feat.shape[3] if dim == 0 else feat.shape[2]
            for i in range(size):
                x_i = feat[:, :, :, i] if dim == 0 else feat[:, :, i, :]
                h = (B * x_i) if prev is None else (A * (prev.detach() if self.use_detach else prev) + B * x_i)
                steps.append(h)
                prev = h
            return torch.stack(steps, dim=-1 if dim == 0 else 2)

        h_hor = scan(x_proj, 0)
        h_ver = scan(x_proj, 1)
        
        out = self.output_proj(h_hor + h_ver)
        return F_mod + self.gamma * out

# ============================================================
# CGNet-SSM Architecture
# ============================================================
class CGNet_SSM(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        self.inc, self.d1, self.d2, self.d3, self.d4 = vgg.features[:5], vgg.features[5:12], vgg.features[12:22], vgg.features[22:32], vgg.features[32:42]
        
        self.red1 = BasicConv2d(256, 128, 3, 1, 1)
        self.red2 = BasicConv2d(512, 256, 3, 1, 1)
        self.red3 = BasicConv2d(1024, 512, 3, 1, 1)
        self.red4 = BasicConv2d(1024, 512, 3, 1, 1)

        self.decoder = nn.Sequential(BasicConv2d(512, 64, 3, 1, 1), nn.Conv2d(64, 1, 3, 1, 1))
        self.ssm1 = RecursivePriorStateSpace(128, 64)
        self.ssm2 = RecursivePriorStateSpace(256, 128)
        self.ssm3 = RecursivePriorStateSpace(512, 256)
        
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_mod4 = BasicConv2d(1024, 512, 3, 1, 1)
        self.dec_mod3 = BasicConv2d(768, 256, 3, 1, 1)
        self.dec_mod2 = BasicConv2d(384, 128, 3, 1, 1)
        self.final = nn.Sequential(BasicConv2d(128, 64, 3, 1, 1), nn.Conv2d(64, 1, 1))

    def forward(self, A, B):
        size = A.shape[2:]
        # Encoder & Fusion
        def enc(img): return self.d4(self.d3(self.d2(self.d1(self.inc(img)))))
        l1A, l1B = self.d1(self.inc(A)), self.d1(self.inc(B))
        l2A, l2B = self.d2(l1A), self.d2(l1B)
        l3A, l3B = self.d3(l2A), self.d3(l2B)
        l4A, l4B = self.d4(l3A), self.d4(l3B)
        
        l1, l2, l3, l4 = self.red1(torch.cat([l1A, l1B], 1)), self.red2(torch.cat([l2A, l2B], 1)), self.red3(torch.cat([l3A, l3B], 1)), self.red4(torch.cat([l4A, l4B], 1))

        coarse = self.decoder(l4)
        l3 = self.ssm3(l3, coarse)
        f4 = self.dec_mod4(torch.cat([self.up(l4), l3], 1))
        l2 = self.ssm2(l2, coarse)
        f3 = self.dec_mod3(torch.cat([self.up(f4), l2], 1))
        l1 = self.ssm1(l1, coarse)
        f2 = self.dec_mod2(torch.cat([self.up(f3), l1], 1))

        return F.interpolate(coarse, size, mode="bilinear"), F.interpolate(self.final(f2), size, mode="bilinear")

# Test
if __name__ == "__main__":
    model = CGNet_SSM().cuda()
    print("Modèle initialisé avec succès.")
