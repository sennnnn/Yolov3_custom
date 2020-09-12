import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import *
from modeling.darknet import *


class Yolov3_tiny(nn.Module):
    def __init__(self, in_channels, num_classes, all_anchors, img_size):
        super(Yolov3_tiny, self).__init__()
        self.block_list = nn.ModuleList()
        self.block_index = 0
        self.num_classes = num_classes
        self.all_anchors = all_anchors
        self.img_size = img_size

        # backbone
        # (conv 3x3 1 + maxpool 2x2 2) * 5 + (conv 3x3 1 + maxpool 2x2 1) * 1
        base_channels = 16
        self.register_block(self.conv(self.block_index, in_channels, base_channels))
        self.register_block(self.maxpool(self.block_index))
        self.register_block(self.conv(self.block_index, base_channels, base_channels*2))
        self.register_block(self.maxpool(self.block_index))
        self.register_block(self.conv(self.block_index, base_channels*2, base_channels*4))
        self.register_block(self.maxpool(self.block_index))
        self.register_block(self.conv(self.block_index, base_channels*4, base_channels*8))
        self.register_block(self.maxpool(self.block_index))
        self.register_block(self.conv(self.block_index, base_channels*8, base_channels*16))
        self.register_block(self.maxpool(self.block_index))
        self.register_block(self.conv(self.block_index, base_channels*16, base_channels*32))
        self.register_block(self.maxpool(self.block_index, stride=1))
        self.register_block(self.conv(self.block_index, base_channels*32, base_channels*64))
        
        # middle
        self.register_block(self.conv(self.block_index, base_channels*64, base_channels*16, kernel_size=1, padding=0))
        
        # path 1
        self.register_block(self.conv(self.block_index, base_channels*16, base_channels*32))
        self.register_block(self.conv(self.block_index, base_channels*32, (num_classes + 1 + 4)*3, kernel_size=1, padding=0, bias=1, bn=0))
        self.register_block(self.yolo(self.block_index, [3, 4, 5]))

        # path 2
        self.register_block(EmptyLayer()) # route layer 1024
        self.register_block(self.conv(self.block_index, base_channels*16, base_channels*8, kernel_size=1, padding=0))
        self.register_block(self.upsample(self.block_index, stride=2))
        self.register_block(EmptyLayer()) # route layer 256+128
        self.register_block(self.conv(self.block_index, base_channels*16 + base_channels*8, base_channels*16))
        self.register_block(self.conv(self.block_index, base_channels*16, (num_classes + 1 + 4)*3, kernel_size=1, padding=0, bias=1, bn=0))
        self.register_block(self.yolo(self.block_index, [0, 1, 2]))

    def forward(self, x):
        # backbone
        o = self.block_list[0](x)               # convolution 416x416x3 -> 416x416x16
        o = self.block_list[1](o)               # maxpool     416x416x16 -> 208x208x16
        o = self.block_list[2](o)               # convolution 208x208x16 -> 208x208x32
        o = self.block_list[3](o)               # maxpool     208x208x32 -> 104x104x32
        o = self.block_list[4](o)               # convolution 104x104x32 -> 104x104x64
        o = self.block_list[5](o)               # maxpool     104x104x64 -> 52x52x64
        o = self.block_list[6](o)               # convolution 52x52x64 -> 52x52x128
        o = self.block_list[7](o)               # maxpool     52x52x128 -> 26x26x128
        o = self.block_list[8](o)               # convolution 26x26x128 -> 26x26x256
        path_2_route = o
        o = self.block_list[9](o)               # maxpool     26x26x256 -> 13x13x256
        o = self.block_list[10](o)              # convolution 13x13x256 -> 13x13x512
        o = self.block_list[11](o)              # maxpool     13x13x512 -> 13x13x512
        o = self.block_list[12](o)              # convolution 13x13x512 -> 13x13x1024
        backbone_out = o

        # middle
        middle_out = self.block_list[13](o)     # convolution 13x13x1024 -> 13x13x256

        # path 1
        o = middle_out
        o = self.block_list[14](o)              # convolution 13x13x256 -> 13x13x512
        o = self.block_list[15](o)              # convolution 13x13x512 -> 13x13x(num_classes + 1 + 4)*3
        path_1_out, _ = self.block_list[16](o)  # yolo head 13x13x(num_classes + 1 + 4)*3 -> 3*13*13 x (num_classes + 1 + 4)

        # path 2
        o = middle_out                          # route -4
        o = self.block_list[18](o)              # convolution 13x13x256 -> 13x13x128
        o = self.block_list[19](o)              # upsample 13x13x128 -> 26x26x128
        o = torch.cat([o, path_2_route], dim=1) # route -1,-8 concatenate 26x26x128 + 26x26x256 -> 26x26x384
        o = self.block_list[21](o)              # convolution 26x26x384 -> 26x26x256
        o = self.block_list[22](o)              # convolution 26x26x256 -> 26x26x(num_classes + 1 + 4)*3
        path_2_out, _ = self.block_list[23](o)  # yolo head 26x26x(num_classes + 1 + 4)*3 -> 3*26*26 x (num_classes + 1 + 4)*3

        # (num_classes + 1 + 4) x (3*13*13 + 3*26*26)
        network_out = torch.cat([path_1_out, path_2_out], dim=1)
        network_out = to_cpu(network_out)

        return network_out

    def register_block(self, block):
        self.block_list.append(block)
        self.block_index += 1

    def conv(self, index, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=0, bn=1):
        block = nn.Sequential()

        block.add_module(
            f"conv_{index}",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
                )
            )

        if bn:
            block.add_module(
                f"batch_norm_{index}",
                nn.BatchNorm2d(
                    num_features=out_channels,
                    momentum=0.9, 
                    eps=1e-5,
                )
            )

        block.add_module(
            f"leaky_{index}",
            nn.LeakyReLU(0.1),
        )
        
        return block

    def maxpool(self, index, kernel_size=2, stride=2):
        block = nn.Sequential()

        if kernel_size == 2 and stride == 1:
            block.add_module(
                f"fix_pad_{index}",
                nn.ZeroPad2d((0, 1, 0, 1))
            )

        block.add_module(
            f"maxpool_{index}",
            nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=int(kernel_size - 1) // 2
            )
        )

        return block

    def upsample(self, index, stride, mode="nearest"):
        block = nn.Sequential()

        block.add_module(
            f"upsample_{index}",
            Upsample(
                scale_factor=stride,
                mode=mode,
            )
        )

        return block

    def yolo(self, index, selected_anchor_indexs):
        block = nn.Sequential()
        # selected_anchor_indexs = [int(x) for x in module_def["mask"].split(",")]
        selected_anchors = [(self.all_anchors[2*i], self.all_anchors[2*i+1]) for i in selected_anchor_indexs]

        block.add_module(
            f"yolo_{index}",
            YOLOLayer(
                selected_anchors,
                self.num_classes,
                self.img_size,
            )
        )

        return block

