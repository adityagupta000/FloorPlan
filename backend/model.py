import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# =========================
#   ATTENTION BLOCK
# =========================
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# =========================
#   DOUBLE CONV BLOCK
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout2d(0.3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =========================
#   MAIN MODEL
# =========================
class ResNet50_UNet_Attention(nn.Module):
    """
    EXACT SAME ARCHITECTURE AS USED DURING TRAINING
    """

    def __init__(self, num_classes=5):
        super(ResNet50_UNet_Attention, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Encoder
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool0 = resnet.maxpool
        self.encoder1 = resnet.layer1  # 256 channels
        self.encoder2 = resnet.layer2  # 512 channels
        self.encoder3 = resnet.layer3  # 1024 channels
        self.encoder4 = resnet.layer4  # 2048 channels

        # Attention gates
        self.att4 = AttentionBlock(F_g=1024, F_l=1024, F_int=512)
        self.att3 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.att2 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)

        # Decoder
        self.up4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(2048, 1024, dropout=True)

        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(1024, 512, dropout=True)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(512, 256, dropout=False)

        self.up1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64, dropout=False)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e0 = self.encoder0(x)
        p0 = self.pool0(e0)

        e1 = self.encoder1(p0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with attention
        d4 = self.up4(e4)
        e3_att = self.att4(d4, e3)      # <-- MUST MATCH TRAINING
        d4 = self.dec4(torch.cat([d4, e3_att], dim=1))

        d3 = self.up3(d4)
        e2_att = self.att3(d3, e2)
        d3 = self.dec3(torch.cat([d3, e2_att], dim=1))

        d2 = self.up2(d3)
        e1_att = self.att2(d2, e1)
        d2 = self.dec2(torch.cat([d2, e1_att], dim=1))

        d1 = self.up1(d2)
        e0_att = self.att1(d1, e0)
        d1 = self.dec1(torch.cat([d1, e0_att], dim=1))

        out = self.final_conv(d1)
        return out
