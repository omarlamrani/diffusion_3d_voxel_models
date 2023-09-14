import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class AttentionLayer(nn.Module):
    ''' Attempted to be used, but due to high dimensionality, completely destroys GPU's V-RAM'''

    def __init__(self, channels, size):
        super(AttentionLayer, self).__init__()
        self.channels = channels
        self.size = size
        self.attention = nn.MultiheadAttention(channels, 2)
        self.layer_norm1 = nn.LayerNorm([channels])
        self.line1 = nn.Linear(channels, channels)
        self.act = nn.GELU(),
        self.line2 = nn.Linear(channels, channels),
        self.layer_norm2 = nn.LayerNorm([channels])

    def forward(self, x):

        x = x.view(-1, self.channels, self.size * self.size * self.size)
        x = x.transpose(1, 2) 
        x_ln_1 = self.layer_norm1(x)
        attention_value, _ = self.self_attention(x_ln_1, x_ln_1, x_ln_1)
        attention_value = attention_value + x
        x_ff = self.line1(attention_value)
        x_ff = self.act(x_ff)
        x_ff = self.line2(x_ff)
        x_ln_2 = self.layer_norm2(x_ff + attention_value)

        return x_ln_2.transpose(1, 2).view(-1, self.channels, self.size, self.size, self.size)


class DoubleAdjConv(nn.Module): 
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
                mid_channels = out_channels
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.grp_norm1 = nn.GroupNorm(1, mid_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.grp_norm2 = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.grp_norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.grp_norm2(out)
        if self.residual:
            return F.gelu(x + out)
        else:
            return out

class DownsampleBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, time_embed_dim=256):
        super(DownsampleBlock, self).__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.d_conv1 = DoubleAdjConv(in_channels, in_channels, residual=True)
        self.d_conv2 = DoubleAdjConv(in_channels, out_channels)
        
        self.silu = nn.SiLU()
        self.linear = nn.Linear(time_embed_dim, out_channels)

    def forward(self, x, t):
        x = self.maxpool(x)
        x = self.d_conv1(x)
        x = self.d_conv2(x)

        emb = self.linear(self.silu(t))
        emb = emb.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        
        return x + emb

class UpsampleBlock(nn.Module): # gud
    def __init__(self, in_channels, out_channels, time_embed_dim=256):
        super(UpsampleBlock, self).__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.double_conv1 = DoubleAdjConv(in_channels, in_channels, residual=True)
        self.double_conv2 = DoubleAdjConv(in_channels, out_channels, in_channels // 2)
        
        self.silu = nn.SiLU()
        self.linear = nn.Linear(time_embed_dim, out_channels)

    def forward(self, x, skip_connect_x, t):

        x = self.up(x)

        x = torch.cat([skip_connect_x, x], dim=1)
        x = self.double_conv1(x)
        x = self.double_conv2(x)
        
        emb = self.linear(self.silu(t))
        emb = emb.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        
        return x + emb


class UNetModel(nn.Module): #gud
    def __init__(self, channels_in=1, channels_out=1, time_embed_dim=256, device="cuda"):

        super(UNetModel, self).__init__()
        self.device = device
        self.time_embed_dim = time_embed_dim
        
        self.in_layer = DoubleAdjConv(channels_in, 64)
        self.down_blocks = nn.ModuleList([
            DownsampleBlock(64, 128),
            DownsampleBlock(128, 256),
            DownsampleBlock(256, 256),
        ])
        
        self.middle_blocks = nn.ModuleList([
            DoubleAdjConv(256, 512),
            DoubleAdjConv(512, 512),
            DoubleAdjConv(512, 256),
        ])
        
        self.up_blocks = nn.ModuleList([
            UpsampleBlock(512, 128),
            UpsampleBlock(256, 64),
            UpsampleBlock(128, 64),
        ])
        
        self.out_layer = nn.Conv3d(64, channels_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        num_timescales = channels // 2
        inv_freq = 1 / (10000** (torch.arange(0, channels, 2, device=self.device) / channels))
        in_pos = t.repeat(1, num_timescales) * inv_freq
        return torch.cat([torch.sin(in_pos), torch.cos(in_pos)], dim=-1)
    
    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_embed_dim).to(self.device)

        in1 = self.in_layer(x)

        down1 = self.down_blocks[0](in1,t)
        down2 = self.down_blocks[1](down1,t)
        down3 = self.down_blocks[2](down2,t)

        mid = self.middle_blocks[0](down3)
        mid = self.middle_blocks[1](mid)
        mid = self.middle_blocks[2](mid)

        up = self.up_blocks[0](mid,down2,t)
        up = self.up_blocks[1](up,down1,t)
        up = self.up_blocks[2](up,in1,t)

        out1 = self.out_layer(up)

        return out1

class UNetModel_conditional(nn.Module):
    def __init__(self, channels_in=1, channels_out=1, time_embed_dim=256, num_classes=-1, device="cuda"):
        super(UNetModel_conditional, self).__init__()
        self.device = device
        self.time_embed_dim = time_embed_dim
        
        self.in_layer = DoubleAdjConv(channels_in, 64)
        self.down_blocks = nn.ModuleList([
            DownsampleBlock(64, 128),
            DownsampleBlock(128, 256),
            DownsampleBlock(256, 256),
        ])
        
        self.middle_blocks = nn.ModuleList([
            DoubleAdjConv(256, 512),
            DoubleAdjConv(512, 512),
            DoubleAdjConv(512, 256),
        ])
        
        self.up_blocks = nn.ModuleList([
            UpsampleBlock(512, 128),
            UpsampleBlock(256, 64),
            UpsampleBlock(128, 64),
        ])
        
        self.out_layer = nn.Conv3d(64, channels_out, kernel_size=1)

        if not num_classes == (torch.tensor([2310]).to(device=self.device)):
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

    def pos_encoding(self, t, channels):
            num_timescales = channels // 2
            inv_freq = 1 / (10000** (torch.arange(0, channels, 2, device=self.device) / channels))
            in_pos = t.repeat(1, num_timescales) * inv_freq
            return torch.cat([torch.sin(in_pos), torch.cos(in_pos)], dim=-1)
    

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_embed_dim)

        if not y == (torch.tensor([2310]).to(device=self.device)):
            t += self.label_emb(y)

        in1 = self.in_layer(x)

        down1 = self.down_blocks[0](in1,t)
        down2 = self.down_blocks[1](down1,t)
        down3 = self.down_blocks[2](down2,t)

        mid = self.middle_blocks[0](down3)
        mid = self.middle_blocks[1](mid)
        mid = self.middle_blocks[2](mid)

        up = self.up_blocks[0](mid,down2,t)
        up = self.up_blocks[1](up,down1,t)
        up = self.up_blocks[2](up,in1,t)

        out1 = self.out_layer(up)

        return out1


if __name__ == '__main__':
    net = UNetModel_conditional(device="cpu",channels_in=1,channels_out=1,num_classes=6)
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(1, 1, 64, 64, 64)
    t = x.new_tensor([500] * x.shape[0])
    y = x.new_tensor([4] * x.shape[0]).int()
    # print(y)
    print(net(x,t,y).shape)