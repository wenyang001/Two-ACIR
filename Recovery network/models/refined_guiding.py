import torch
import torch.nn as nn
import numpy as np
import math
from PIL import Image
from torch.nn.functional import interpolate

def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


class conv_block(nn.Module):
    def __init__(self):
        super(conv_block, self).__init__()
        # self.num = 64 # for 512 or 1024  # 32 for 2048
        self.num = 64
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self.num, out_channels=self.num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.num, out_channels=self.num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.num, out_channels=self.num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.num, out_channels=self.num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.num, out_channels=self.num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.num, out_channels=self.num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.num, out_channels=self.num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.num, out_channels=self.num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.num, out_channels=self.num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.num, out_channels=self.num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self.num, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        )


    def forward(self, x):
        output = self.block(x)
        return output

class upsample_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(upsample_conv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class refined(nn.Module):
    def __init__(self, scen):
        super(refined, self).__init__()

        self.scen = scen

        self.conv_first = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv_F1 = self.make_layer(conv_block)

        self.conv_I1 = upsample_conv(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv_F2 = self.make_layer(conv_block)

        self.conv_I2 = upsample_conv(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv_F3 = self.make_layer(conv_block)

        if self.scen > 2:
            self.conv_I3 = upsample_conv(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
            self.conv_F4 = self.make_layer(conv_block)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, LR_input, HR_fake, HR_real):

        # 1x 128*256
        # 2x 256*512
        # 4x 512*1024
        # 8x 1024*2048

        LR_8x, LR_4x, LR_2x, LR_1x = None, None, None, None
        GT_8x, GT_4x, GT_2x, GT_1x = None, None, None, None

        LR_input = self.conv_first(LR_input)

        # 512*256
        if self.scen == 1:
            GT_2x = HR_real
            GT_1x = interpolate(GT_2x, scale_factor=1 / 2, mode='bicubic')
            HR_fake_2x = HR_fake
            HR_fake_1x = interpolate(HR_fake_2x, scale_factor=1 / 2, mode='bicubic')

            HR_fake_1 = self.conv_F1(HR_fake_1x)
            LR_1x = HR_fake_1 + LR_input
            LR_1x_Up = self.conv_I1(LR_1x)

            HR_fake_2 = self.conv_F2(HR_fake_2x)
            LR_2x = HR_fake_2 + LR_1x_Up


        # 1024*512
        elif self.scen == 2:
            GT_4x = HR_real
            GT_2x = interpolate(GT_4x, scale_factor=1 / 2, mode='bicubic')
            GT_1x = interpolate(GT_2x, scale_factor=1 / 2, mode='bicubic')

            HR_fake_4x = HR_fake
            HR_fake_2x = interpolate(HR_fake_4x, scale_factor=1 / 2, mode='bicubic')
            HR_fake_1x = interpolate(HR_fake_2x, scale_factor=1 / 2, mode='bicubic')

            HR_fake_1 = self.conv_F1(HR_fake_1x)
            LR_1x = HR_fake_1 + LR_input
            LR_1x_Up = self.conv_I1(LR_1x)

            HR_fake_2 = self.conv_F2(HR_fake_2x)
            LR_2x = HR_fake_2 + LR_1x_Up
            LR_2x_Up = self.conv_I2(LR_2x)

            HR_fake_3 = self.conv_F3(HR_fake_4x)
            LR_4x = HR_fake_3 + LR_2x_Up


        # 2048x1024
        else:
            GT_8x = HR_real
            GT_4x = interpolate(GT_8x, scale_factor=1 / 2, mode='bicubic')
            GT_2x = interpolate(GT_4x, scale_factor=1 / 2, mode='bicubic')
            GT_1x = interpolate(GT_2x, scale_factor=1 / 2, mode='bicubic')

            HR_fake_8x = HR_fake
            HR_fake_4x = interpolate(HR_fake_8x, scale_factor=1 / 2, mode='bicubic')
            HR_fake_2x = interpolate(HR_fake_4x, scale_factor=1 / 2, mode='bicubic')
            HR_fake_1x = interpolate(HR_fake_2x, scale_factor=1 / 2, mode='bicubic')

            HR_fake_1 = self.conv_F1(HR_fake_1x)
            LR_1x = HR_fake_1 + LR_input
            LR_1x_Up = self.conv_I1(LR_1x)

            HR_fake_2 = self.conv_F2(HR_fake_2x)
            LR_2x = HR_fake_2 + LR_1x_Up
            LR_2x_Up = self.conv_I2(LR_2x)

            HR_fake_3 = self.conv_F3(HR_fake_4x)
            LR_4x = HR_fake_3 + LR_2x_Up
            LR_4x_Up = self.conv_I3(LR_4x)

            HR_fake_4 = self.conv_F4(HR_fake_8x)
            LR_8x = HR_fake_4 + LR_4x_Up

        return self.scen, [LR_1x, LR_2x, LR_4x, LR_8x], [GT_1x, GT_2x, GT_4x, GT_8x]


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        #loss = torch.sum(error)
        loss = torch.mean(error)
        return loss


if __name__ == '__main__':
    from PIL import Image
    import torchvision.transforms as transform
    import matplotlib.pyplot as plt

    filter = get_upsample_filter(16)
    kim = transform.functional.to_pil_image(filter)
    plt.imshow(kim)
    plt.show()