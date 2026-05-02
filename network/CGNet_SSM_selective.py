import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ============================================================
# 1. BLOC DE CONVOLUTION DE BASE
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
# 2. MODULE RPSS SELECTIF (Conditioned Selective SSM type Mamba)
# ============================================================
class PriorConditionedSelectiveStateSpace(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, use_detach=True):
        super().__init__()
        self.use_detach = use_detach
        self.hidden_dim = hidden_dim
        
        # Projections
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, 1)
        self.output_proj = nn.Conv2d(hidden_dim, in_channels, 1)
        
        # Projections pour les paramètres dépendants de l'entrée (x_k)
        self.W_delta = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.W_B = nn.Conv2d(hidden_dim, hidden_dim, 1)
        
        # Scalaires d'apprentissage (learnable scalars) comme décrits dans les équations LaTeX
        self.lambda_delta = nn.Parameter(torch.tensor(0.1))
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        # Matrice d'évolution A (statique mais apprise, stabilisée avec log)
        self.A = nn.Parameter(torch.randn(hidden_dim))
        
        # Résiduel
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, prior):
        Bsz, C, H, W = x.shape
        
        # 1. Ajustement du Prior
        prior_up = F.interpolate(prior, size=(H, W), mode="bilinear", align_corners=True)
        prior_up = torch.clamp(prior_up, -1.0, 1.0)
        
        x_proj = self.input_proj(x)
        
        # 2. Génération des paramètres dynamiques conditionnés par le Prior
        # Equation: Delta_k^P = Softplus(W_delta * x_k + lambda_delta * P_seq)
        delta = F.softplus(self.W_delta(x_proj) + self.lambda_delta * prior_up)
        
        # Equation: B_k^P = (W_B * x_k) * (1 + alpha * P_seq)
        B_k = self.W_B(x_proj) * (1.0 + self.alpha * prior_up)
        
        # Préparation de la matrice A (négative pour la stabilité du système)
        A = -torch.exp(self.A) 
        A = A.view(1, -1, 1, 1)
        
        # Discrétisation (Zero-Order Hold)
        # A_bar = exp(Delta * A)
        A_bar = torch.exp(delta * A)
        
        # B_bar = Delta * B_k (approximation standard utilisée dans Mamba pour ZOH)
        B_bar = delta * B_k

        # 3. Parallel Prefix Scan — VECTORISÉ GPU (remplace la boucle Python séquentielle)
        # Principe : h_i = A_i * h_{i-1} + B_i * x_i  avec h_0 = 0
        # Forme fermée : h = cumprod(A) * cumsum(B*x / cumprod(A))
        # → Zéro boucle Python, entièrement parallélisé sur CUDA (O(n) GPU vs O(n) CPU séquentiel)
        def parallel_scan(feat, A_seq, B_seq, seq_dim):
            b = B_seq * feat                                          # b_i = B_i * x_i
            P = torch.cumprod(A_seq, dim=seq_dim)                    # P_i = A_1 * ... * A_i
            P_safe = P.clamp(min=1e-8)                               # éviter division par zéro
            h = P * torch.cumsum(b / P_safe, dim=seq_dim)            # h_i = P_i * Σ(b_j / P_j)
            return h

        # 2-Directional Cross-Scan (horizontal et vertical, forward uniquement)
        # dim=3 → scan horizontal (W), dim=2 → scan vertical (H)
        scan_h_fwd = parallel_scan(x_proj, A_bar, B_bar, seq_dim=3)
        scan_v_fwd = parallel_scan(x_proj, A_bar, B_bar, seq_dim=2)

        out = self.output_proj(scan_h_fwd + scan_v_fwd)
        
        # Retourne l'output. On retourne prior_up comme 2eme argument pour la compatibilité avec train_CGNet.py (gate visualization)
        return x + self.gamma * out, prior_up

# ============================================================
# 3. ARCHITECTURE CGNet_SSM_selective
# ============================================================
class CGNet_SSM(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        self.inc, self.d1, self.d2, self.d3, self.d4 = vgg.features[:5], vgg.features[5:12], vgg.features[12:22], vgg.features[22:32], vgg.features[32:42]
        
        self.red1, self.red2, self.red3, self.red4 = BasicConv2d(256, 128, 3, 1, 1), BasicConv2d(512, 256, 3, 1, 1), BasicConv2d(1024, 512, 3, 1, 1), BasicConv2d(1024, 512, 3, 1, 1)
        self.decoder_coarse = nn.Sequential(BasicConv2d(512, 64, 3, 1, 1), nn.Conv2d(64, 1, 3, 1, 1))
        
        # On utilise le module Selectif Conditionné
        self.ssm3 = PriorConditionedSelectiveStateSpace(512, 256)
        self.ssm2 = PriorConditionedSelectiveStateSpace(256, 128)
        self.ssm1 = PriorConditionedSelectiveStateSpace(128, 64)
        
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_mod4, self.dec_mod3, self.dec_mod2 = BasicConv2d(1024, 512, 3, 1, 1), BasicConv2d(768, 256, 3, 1, 1), BasicConv2d(384, 128, 3, 1, 1)
        self.final_head = nn.Sequential(BasicConv2d(128, 64, 3, 1, 1), nn.Conv2d(64, 1, 1))

    def forward(self, A, B):
        size = A.shape[2:]
        l1A, l1B = self.d1(self.inc(A)), self.d1(self.inc(B))
        l2A, l2B = self.d2(l1A), self.d2(l1B)
        l3A, l3B = self.d3(l2A), self.d3(l2B)
        l4A, l4B = self.d4(l3A), self.d4(l3B)
        
        l1, l2, l3, l4 = self.red1(torch.cat([l1A, l1B], 1)), self.red2(torch.cat([l2A, l2B], 1)), self.red3(torch.cat([l3A, l3B], 1)), self.red4(torch.cat([l4A, l4B], 1))

        coarse = self.decoder_coarse(l4)
        l3_ssm, gate3 = self.ssm3(l3, coarse)
        f4 = self.dec_mod4(torch.cat([self.up(l4), l3_ssm], 1))
        l2_ssm, gate2 = self.ssm2(l2, coarse)
        f3 = self.dec_mod3(torch.cat([self.up(f4), l2_ssm], 1))
        l1_ssm, gate1 = self.ssm1(l1, coarse)
        f2 = self.dec_mod2(torch.cat([self.up(f3), l1_ssm], 1))

        return F.interpolate(coarse, size, mode="bilinear"), F.interpolate(self.final_head(f2), size, mode="bilinear"), (gate1, gate2, gate3)
