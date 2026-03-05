# ============================================================
# CGNet-SSM v2
# Change Guiding Network with Prior-Conditioned State Space
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# Basic convolution block
# ============================================================

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size,
            stride,
            padding,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


# ============================================================
# Prior Conditioned State Space Module
# ============================================================

class PriorConditionedSSM(nn.Module):

    def __init__(self, in_channels, hidden_dim):

        super().__init__()

        self.input_proj = nn.Conv2d(in_channels, hidden_dim, 1)

        self.A = nn.Parameter(torch.randn(hidden_dim))
        self.B = nn.Parameter(torch.randn(hidden_dim))

        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.gamma = nn.Parameter(torch.tensor(0.1))

        self.output_proj = nn.Conv2d(hidden_dim, in_channels, 1)

        self.fusion = nn.Conv2d(hidden_dim * 4, hidden_dim, 1)

    # --------------------------------------------------------

    def scan_forward(self, x, A, B):

        Bsz, C, H, W = x.shape

        states = []

        prev = None

        for i in range(W):

            x_i = x[:, :, :, i]

            if i == 0:

                h = B * x_i

            else:

                h = A * prev + B * x_i

            states.append(h)

            prev = h

        states = torch.stack(states, dim=-1)

        return states

    # --------------------------------------------------------

    def scan_backward(self, x, A, B):

        Bsz, C, H, W = x.shape

        states = [None] * W

        prev = None

        for idx, i in enumerate(reversed(range(W))):

            x_i = x[:, :, :, i]

            if idx == 0:

                h = B * x_i

            else:

                h = A * prev + B * x_i

            states[i] = h

            prev = h

        states = torch.stack(states, dim=-1)

        return states

    # --------------------------------------------------------

    def forward(self, F_in, prior):

        Bsz, C, H, W = F_in.shape

        if prior.shape[2:] != (H, W):

            prior = F.interpolate(
                prior,
                size=(H, W),
                mode="bilinear",
                align_corners=True
            )

        prior = prior.repeat(1, C, 1, 1)

        F_mod = F_in + self.alpha * prior

        x = self.input_proj(F_mod)

        A = torch.tanh(self.A).view(1, -1, 1)
        Bp = self.B.view(1, -1, 1)

        # horizontal scans

        h_lr = self.scan_forward(x, A, Bp)
        h_rl = self.scan_backward(x, A, Bp)

        # vertical scans

        x_t = x.permute(0, 1, 3, 2)

        h_tb = self.scan_forward(x_t, A, Bp)
        h_bt = self.scan_backward(x_t, A, Bp)

        h_tb = h_tb.permute(0, 1, 3, 2)
        h_bt = h_bt.permute(0, 1, 3, 2)

        h = torch.cat([h_lr, h_rl, h_tb, h_bt], dim=1)

        h = self.fusion(h)

        out = self.output_proj(h)

        F_out = F_mod + self.gamma * out

        return F_out


# ============================================================
# CGNet with SSM
# ============================================================

class CGNet_SSM(nn.Module):

    def __init__(self):

        super().__init__()

        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)

        self.inc = vgg.features[:5]

        self.down1 = vgg.features[5:12]
        self.down2 = vgg.features[12:22]
        self.down3 = vgg.features[22:32]
        self.down4 = vgg.features[32:42]

        self.conv_reduce_1 = BasicConv2d(128 * 2, 128, 3, 1, 1)
        self.conv_reduce_2 = BasicConv2d(256 * 2, 256, 3, 1, 1)
        self.conv_reduce_3 = BasicConv2d(512 * 2, 512, 3, 1, 1)
        self.conv_reduce_4 = BasicConv2d(512 * 2, 512, 3, 1, 1)

        # coarse change decoder

        self.decoder = nn.Sequential(
            BasicConv2d(512, 64, 3, 1, 1),
            nn.Conv2d(64, 1, 3, 1, 1)
        )

        self.decoder_final = nn.Sequential(
            BasicConv2d(128, 64, 3, 1, 1),
            nn.Conv2d(64, 1, 1)
        )

        # SSM modules

        self.ssm1 = PriorConditionedSSM(128, 64)
        self.ssm2 = PriorConditionedSSM(256, 128)
        self.ssm3 = PriorConditionedSSM(512, 256)

        # decoder refinement

        self.decoder_module4 = BasicConv2d(1024, 512, 3, 1, 1)
        self.decoder_module3 = BasicConv2d(768, 256, 3, 1, 1)
        self.decoder_module2 = BasicConv2d(384, 128, 3, 1, 1)

        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)

    # --------------------------------------------------------

    def forward(self, A, B):

        size = A.shape[2:]

        # encoder

        l1A = self.down1(self.inc(A))
        l2A = self.down2(l1A)
        l3A = self.down3(l2A)
        l4A = self.down4(l3A)

        l1B = self.down1(self.inc(B))
        l2B = self.down2(l1B)
        l3B = self.down3(l2B)
        l4B = self.down4(l3B)

        # fusion

        l1 = self.conv_reduce_1(torch.cat([l1A, l1B], 1))
        l2 = self.conv_reduce_2(torch.cat([l2A, l2B], 1))
        l3 = self.conv_reduce_3(torch.cat([l3A, l3B], 1))
        l4 = self.conv_reduce_4(torch.cat([l4A, l4B], 1))

        # coarse change map

        coarse = self.decoder(l4)

        # layer3

        prior3 = F.interpolate(coarse, size=l3.shape[2:], mode="bilinear", align_corners=True)

        l3 = self.ssm3(l3, prior3)

        f4 = self.decoder_module4(torch.cat([self.up2x(l4), l3], 1))

        # layer2

        prior2 = F.interpolate(coarse, size=l2.shape[2:], mode="bilinear", align_corners=True)

        l2 = self.ssm2(l2, prior2)

        f3 = self.decoder_module3(torch.cat([self.up2x(f4), l2], 1))

        # layer1

        prior1 = F.interpolate(coarse, size=l1.shape[2:], mode="bilinear", align_corners=True)

        l1 = self.ssm1(l1, prior1)

        f2 = self.decoder_module2(torch.cat([self.up2x(f3), l1], 1))

        # outputs

        change_map = F.interpolate(coarse, size=size, mode="bilinear", align_corners=True)

        final_map = self.decoder_final(f2)

        final_map = F.interpolate(final_map, size=size, mode="bilinear", align_corners=True)

        return change_map, final_map


# ============================================================
# Test block
# ============================================================

if __name__ == "__main__":

    print("Testing CGNet_SSM...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CGNet_SSM().to(device)

    A = torch.randn(2, 3, 256, 256).to(device)
    B = torch.randn(2, 3, 256, 256).to(device)

    change_map, final_map = model(A, B)

    print("Change map shape:", change_map.shape)
    print("Final map shape:", final_map.shape)

    params = sum(p.numel() for p in model.parameters())

    print("Total parameters:", params)
