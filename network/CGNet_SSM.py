#  Change Guiding Network with Recursive Prior State Space (CGNet_SSM)
#  Incorporating recursive state-space dynamics with prior conditioning
#  for change detection in remote sensing imagery

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BasicConv2d(nn.Module):
    """Basic Conv2d block with BatchNorm and ReLU"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RecursivePriorStateSpace(nn.Module):
    """
    Recursive Prior State Space (RPSS) Module
    
    Applies recursive state-space dynamics with prior conditioning.
    No attention, no softmax, no sigmoid - pure linear complexity O(HW).
    """
    
    def __init__(self, in_channels, hidden_dim=128):
        """
        Args:
            in_channels: Input feature dimension
            hidden_dim: Hidden state dimension
        """
        super(RecursivePriorStateSpace, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        # State parameters
        self.A = nn.Parameter(torch.randn(hidden_dim))
        self.B = nn.Parameter(torch.randn(hidden_dim))
        
        # Gating parameters
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Prior injection strength
        self.gamma = nn.Parameter(torch.tensor(0.1))  # Output residual strength
        
        # Output projection
        self.output_proj = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.normal_(self.A, mean=0.0, std=0.1)
        nn.init.normal_(self.B, mean=0.0, std=0.1)
    
    def forward(self, x, prior):
        """
        Forward pass with recursive state-space dynamics
        
        Args:
            x: Input features [B, C, H, W]
            prior: Prior guidance map [B, 1, H, W] or coarse prior
            
        Returns:
            Output features [B, C, H, W]
        """
        batch_size, channels, height, width = x.shape
        
        # Normalize prior to [B, 1, H, W]
        if prior.shape[1] != 1:
            prior_normalized = prior.mean(dim=1, keepdim=True)
        else:
            prior_normalized = prior
        
        # Upsample prior to match resolution if needed
        if prior_normalized.shape[2:] != x.shape[2:]:
            prior_normalized = F.interpolate(prior_normalized, size=(height, width),
                                            mode='bilinear', align_corners=True)
        
        # Clamp prior to reasonable range
        prior_normalized = torch.clamp(prior_normalized, -1.0, 1.0)
        
        # Inject prior into original feature space (additive)
        # Expand prior from [B, 1, H, W] to [B, C, H, W] by repeating
        prior_expanded = prior_normalized.expand(batch_size, channels, height, width)
        F_mod = x + self.alpha * prior_expanded
        
        # Project to hidden dimension for recursive computation
        x_proj = self.input_proj(F_mod)  # [B, hidden_dim, H, W]
        
        # Constrain A with tanh for stability
        A = torch.tanh(self.A)  # [hidden_dim]
        B = self.B  # [hidden_dim]
        
        # Reshape for recursive computation
        # For broadcasting: [1, hidden_dim, 1] broadcasts with [B, hidden_dim, H/W]
        A_expanded = A.view(1, -1, 1)  # [1, hidden_dim, 1]
        B_expanded = B.view(1, -1, 1)  # [1, hidden_dim, 1]
        
        # Horizontal recursive dynamics along width dimension
        h_horizontal = torch.zeros_like(x_proj)
        for i in range(width):
            if i == 0:
                h_horizontal[:, :, :, i] = B_expanded * x_proj[:, :, :, i]
            else:
                h_horizontal[:, :, :, i] = A_expanded * h_horizontal[:, :, :, i-1] + B_expanded * x_proj[:, :, :, i]
        
        # Vertical recursive dynamics along height dimension
        h_vertical = torch.zeros_like(x_proj)
        for j in range(height):
            if j == 0:
                h_vertical[:, :, j, :] = B_expanded * x_proj[:, :, j, :]
            else:
                h_vertical[:, :, j, :] = A_expanded * h_vertical[:, :, j-1, :] + B_expanded * x_proj[:, :, j, :]
        
        # Fusion of horizontal and vertical paths
        h_fused = h_horizontal + h_vertical
        
        # Output projection
        out = self.output_proj(h_fused)  # [B, in_dim, H, W]
        
        # Final residual connection in original feature space
        F_out = F_mod + self.gamma * out
        
        return F_out


class CGNet_SSM(nn.Module):
    """
    CGNet with Recursive Prior State Space Module
    
    Replaces ChangeGuideModule with RecursivePriorStateSpace for improved
    prior-conditioned feature processing using state-space dynamics.
    """
    
    def __init__(self):
        super(CGNet_SSM, self).__init__()
        
        # Load VGG16-BN backbone
        vgg16_bn = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        
        # Encoder: Feature extraction
        self.inc = vgg16_bn.features[:5]      # 64 channels
        self.down1 = vgg16_bn.features[5:12]  # 128 channels
        self.down2 = vgg16_bn.features[12:22] # 256 channels
        self.down3 = vgg16_bn.features[22:32] # 512 channels
        self.down4 = vgg16_bn.features[32:42] # 512 channels
        
        # Feature fusion layers for bi-temporal fusion
        self.conv_reduce_1 = BasicConv2d(128*2, 128, 3, 1, 1)
        self.conv_reduce_2 = BasicConv2d(256*2, 256, 3, 1, 1)
        self.conv_reduce_3 = BasicConv2d(512*2, 512, 3, 1, 1)
        self.conv_reduce_4 = BasicConv2d(512*2, 512, 3, 1, 1)
        
        # Upsampling layers for coarse-to-fine prior propagation
        self.up_layer4 = BasicConv2d(512, 512, 3, 1, 1)
        self.up_layer3 = BasicConv2d(512, 512, 3, 1, 1)
        self.up_layer2 = BasicConv2d(256, 256, 3, 1, 1)
        
        # Change map decoder (coarse)
        self.decoder = nn.Sequential(
            BasicConv2d(512, 64, 3, 1, 1),
            nn.Conv2d(64, 1, 3, 1, 1)
        )
        
        # Final output decoder
        self.decoder_final = nn.Sequential(
            BasicConv2d(128, 64, 3, 1, 1),
            nn.Conv2d(64, 1, 1)
        )
        
        # Recursive Prior State Space modules for prior-conditioned processing
        self.rpss_2 = RecursivePriorStateSpace(in_channels=256, hidden_dim=128)
        self.rpss_3 = RecursivePriorStateSpace(in_channels=512, hidden_dim=256)
        self.rpss_4 = RecursivePriorStateSpace(in_channels=512, hidden_dim=256)
        
        # Decoder modules for coarse-to-fine refinement
        self.decoder_module4 = BasicConv2d(1024, 512, 3, 1, 1)
        self.decoder_module3 = BasicConv2d(768, 256, 3, 1, 1)
        self.decoder_module2 = BasicConv2d(384, 128, 3, 1, 1)
        
        # Upsampling operator
        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)
    
    def forward(self, A, B):
        """
        Forward pass with dual-temporal inputs and RPSS-based prior conditioning
        
        Args:
            A: First temporal image [B, 3, H, W]
            B: Second temporal image [B, 3, H, W]
            
        Returns:
            change_map: Change map output [B, 1, H, W]
            final_map: Final refined map [B, 1, H, W]
        """
        size = A.size()[2:]
        
        # ===== Encoder: Extract multi-scale features =====
        # Image A features
        layer1_pre_A = self.inc(A)
        layer1_A = self.down1(layer1_pre_A)
        layer2_A = self.down2(layer1_A)
        layer3_A = self.down3(layer2_A)
        layer4_A = self.down4(layer3_A)
        
        # Image B features
        layer1_pre_B = self.inc(B)
        layer1_B = self.down1(layer1_pre_B)
        layer2_B = self.down2(layer1_B)
        layer3_B = self.down3(layer2_B)
        layer4_B = self.down4(layer3_B)
        
        # Bi-temporal fusion: concatenate A and B temporal pairs
        layer1 = torch.cat((layer1_B, layer1_A), dim=1)  # [B, 256, H/4, W/4]
        layer2 = torch.cat((layer2_B, layer2_A), dim=1)  # [B, 512, H/8, W/8]
        layer3 = torch.cat((layer3_B, layer3_A), dim=1)  # [B, 1024, H/16, W/16]
        layer4 = torch.cat((layer4_B, layer4_A), dim=1)  # [B, 1024, H/32, W/32]
        
        # Feature dimension reduction
        layer1 = self.conv_reduce_1(layer1)
        layer2 = self.conv_reduce_2(layer2)
        layer3 = self.conv_reduce_3(layer3)
        layer4 = self.conv_reduce_4(layer4)
        
        # ===== Generate coarse change prior at highest level =====
        change_map_coarse = self.decoder(layer4)  # [B, 1, H/32, W/32]
        
        # ===== Coarse-to-fine propagation with RPSS =====
        # Upsample prior and apply RPSS at layer3
        change_map_up = F.interpolate(change_map_coarse, size=layer3.size()[2:],
                                      mode='bilinear', align_corners=True)
        layer3_enhanced = self.rpss_3(layer3, change_map_up)
        
        # Decoder and refinement at layer3
        feature4 = self.decoder_module4(torch.cat([self.upsample2x(layer4), layer3_enhanced], 1))
        
        # Upsample prior and apply RPSS at layer2
        change_map_up = F.interpolate(change_map_coarse, size=layer2.size()[2:],
                                      mode='bilinear', align_corners=True)
        layer2_enhanced = self.rpss_2(layer2, change_map_up)
        
        # Decoder and refinement at layer2
        feature3 = self.decoder_module3(torch.cat([self.upsample2x(feature4), layer2_enhanced], 1))
        
        # Upsample prior and apply RPSS at layer1
        change_map_up = F.interpolate(change_map_coarse, size=layer1.size()[2:],
                                      mode='bilinear', align_corners=True)
        layer1_enhanced = self.rpss_2(layer1, change_map_up)
        
        # Final decoder
        layer1_final = self.decoder_module2(torch.cat([self.upsample2x(feature3), layer1_enhanced], 1))
        
        # ===== Generate final outputs =====
        change_map = F.interpolate(change_map_coarse, size=size,
                                   mode='bilinear', align_corners=True)
        final_map = self.decoder_final(layer1_final)
        final_map = F.interpolate(final_map, size=size,
                                  mode='bilinear', align_corners=True)
        
        return change_map, final_map


if __name__ == '__main__':
    # Test module
    print("Testing CGNet_SSM...")
    
    model = CGNet_SSM().cuda()
    A = torch.randn(2, 3, 256, 256).cuda()
    B = torch.randn(2, 3, 256, 256).cuda()
    
    change_map, final_map = model(A, B)
    
    print(f"✓ CGNet_SSM test passed!")
    print(f"  Input shapes: A={A.shape}, B={B.shape}")
    print(f"  Output shapes: change_map={change_map.shape}, final_map={final_map.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params:,}")
