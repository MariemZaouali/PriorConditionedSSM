class PriorConditionedSSM(nn.Module):
    """
    Prior-Conditioned Bidirectional State Space Module
    """

    def __init__(self, in_channels, hidden_dim):
        super().__init__()

        self.input_proj = nn.Conv2d(in_channels, hidden_dim, 1)

        self.A = nn.Parameter(torch.randn(hidden_dim))
        self.B = nn.Parameter(torch.randn(hidden_dim))

        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.gamma = nn.Parameter(torch.tensor(0.1))

        self.output_proj = nn.Conv2d(hidden_dim, in_channels, 1)

        self.fusion = nn.Conv2d(hidden_dim * 4, hidden_dim, 1)

    def forward_scan(self, x, A, B):
        B_, C, H, W = x.shape
        h = torch.zeros_like(x)

        for i in range(W):
            if i == 0:
                h[:, :, :, i] = B * x[:, :, :, i]
            else:
                h[:, :, :, i] = A * h[:, :, :, i-1] + B * x[:, :, :, i]

        return h

    def backward_scan(self, x, A, B):
        B_, C, H, W = x.shape
        h = torch.zeros_like(x)

        for i in reversed(range(W)):
            if i == W-1:
                h[:, :, :, i] = B * x[:, :, :, i]
            else:
                h[:, :, :, i] = A * h[:, :, :, i+1] + B * x[:, :, :, i]

        return h

    def forward(self, F, prior):

        B, C, H, W = F.shape

        if prior.shape[2:] != (H, W):
            prior = F.interpolate(prior, size=(H, W), mode='bilinear')

        prior = prior.repeat(1, C, 1, 1)

        F_mod = F + self.alpha * prior

        x = self.input_proj(F_mod)

        A = torch.tanh(self.A).view(1,-1,1,1)
        Bp = self.B.view(1,-1,1,1)

        h_lr = self.forward_scan(x, A, Bp)
        h_rl = self.backward_scan(x, A, Bp)

        x_t = x.permute(0,1,3,2)

        h_tb = self.forward_scan(x_t, A, Bp).permute(0,1,3,2)
        h_bt = self.backward_scan(x_t, A, Bp).permute(0,1,3,2)

        h = torch.cat([h_lr, h_rl, h_tb, h_bt], dim=1)

        h = self.fusion(h)

        out = self.output_proj(h)

        return F_mod + self.gamma * out
