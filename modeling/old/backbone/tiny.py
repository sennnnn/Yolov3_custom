import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        o = self.conv(x)

        return o


class pool2d(nn.Module):
    def __init__(self):
        super(pool2d, self).__init__()

    def forward(self, x):
        o = F.max_pool2d(F.pad(x, (0, 1, 0, 1), mode='replicate'), 2, stride=1)

        return o


class backbone(nn.Module):
    def __init__(self, ch_in, width_in, height_in, num_class, ch_base=16):
        super(backbone, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool_ = pool2d()
        self.ReLU = nn.ReLU(inplace=True)
        self.up_sample = nn.Upsample(scale_factor=2)

        self.conv_1 = conv_block(ch_in, ch_base)
        self.conv_2 = conv_block(ch_base, ch_base*2)
        self.conv_3 = conv_block(ch_base*2, ch_base*4)
        self.conv_4 = conv_block(ch_base*4, ch_base*8)
        self.conv_5 = conv_block(ch_base*8, ch_base*16)
        self.conv_6 = conv_block(ch_base*16, ch_base*32)
        self.conv_7 = conv_block(ch_base*32, ch_base*64)

        self.conv_sub1 = nn.Conv2d(ch_base*64, ch_base*16, kernel_size=1, stride=1)
        self.conv_sub1_1 = conv_block(ch_base*16, ch_base*32)
        self.conv_sub1_out = nn.Conv2d(ch_base*32, 3*(num_class+1+4), kernel_size=1)

        self.conv_sub2 = nn.Conv2d(ch_base*16, ch_base*8, kernel_size=1, stride=1)
        self.conv_sub2_1 = conv_block(ch_base*(8+16), ch_base*16)
        

    def forward(self, x):
        o = self.conv_1(x)        # 416x416x3  -> 416x416x16
        o = self.maxpool(o)       # 416x416x16 -> 208x208x16

        o = self.conv_2(o)        # 208x208x16 -> 208x208x32
        o = self.maxpool(o)       # 208x208x32 -> 104x104x32

        o = self.conv_3(o)        # 104x104x32 -> 104x104x64
        o = self.maxpool(o)       # 104x104x64 -> 52x52x64

        o = self.conv_4(o)        # 52x52x64   -> 52x52x128
        o = self.maxpool(o)       # 52x52x128  -> 26x26x128

        o = self.conv_5(o)        # 26x26x128  -> 26x26x256
        sub2 = o

        # sub1
        o = self.maxpool(o)       # 26x26x256  -> 13x13x256
        o = self.conv_6(o)        # 13x13x256  -> 13x13x512
        o = self.maxpool_(o)      # 13x13x512  -> 13x13x512
        o = self.conv_7(o)        # 13x13x512  -> 13x13x1024

        ## sub1 out
        o = self.conv_sub1(o)     # 13x13x1024 -> 13x13x256
        route = o
        o = self.conv_sub1_1(o)   # 13x13x256  -> 13x13x512
        sub1_out = o

        # sub2
        o = sub2                  # 26x26x256
        
        ## sub2_out
        r = self.conv_sub2(route) # 13x13x256  -> 13x13x128
        r = self.up_sample(r)     # 13x13x128  -> 26x26x128
        o = torch.cat([r, o], dim=1) # 26x26x384
        o = self.conv_sub2_1(o)   # 26x26x384  -> 26x26x256
        sub2_out = o

        return sub2_out, sub1_out


if __name__ == "__main__":
    img = torch.randn(2, 3, 416, 416)

    model = backbone(3, 416, 416, 1)

    out = model(img)

    print(out[0].shape, out[1].shape)