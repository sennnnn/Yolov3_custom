import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import *
from modeling.darknet import *
    

class Yolov3(nn.Module):
    def __init__(self, in_channels, num_classes, all_anchors, img_size):
        super(Yolov3, self).__init__()
        self.block_list = nn.ModuleList()
        self.block_index = 0
        self.num_classes = num_classes
        self.all_anchors = all_anchors
        self.img_size = img_size

        # backbone
        base_channels = 32
        self.register_block(self.conv(self.block_index, in_channels, base_channels))
        
        # Downsample
        self.register_block(self.conv(self.block_index, base_channels, base_channels*2, stride=2))
        # Residual Block x1
        self.register_block(self.conv(self.block_index, base_channels*2, base_channels, kernel_size=1))
        self.register_block(self.conv(self.block_index, base_channels, base_channels*2))
        self.register_block(EmptyLayer())
        
        # Downsample
        self.register_block(self.conv(self.block_index, base_channels*2, base_channels*4, stride=2))
        # Residual Block x2
        for i in range(2):
            self.register_block(self.conv(self.block_index, base_channels*4, base_channels*2, kernel_size=1))
            self.register_block(self.conv(self.block_index, base_channels*2, base_channels*4))
            self.register_block(EmptyLayer())

        # Downsample
        self.register_block(self.conv(self.block_index, base_channels*4, base_channels*8, stride=2))
        # Residual Block x8
        for i in range(8):
            self.register_block(self.conv(self.block_index, base_channels*8, base_channels*4, kernel_size=1))
            self.register_block(self.conv(self.block_index, base_channels*4, base_channels*8))
            self.register_block(EmptyLayer())

        # Downsample
        self.register_block(self.conv(self.block_index, base_channels*8, base_channels*16, stride=2))
        # Residual Block x8
        for i in range(8):
            self.register_block(self.conv(self.block_index, base_channels*16, base_channels*8, kernel_size=1))
            self.register_block(self.conv(self.block_index, base_channels*8, base_channels*16))
            self.register_block(EmptyLayer())

        # Downsample
        self.register_block(self.conv(self.block_index, base_channels*16, base_channels*32, stride=2))
        # Residual Block x4
        for i in range(4):
            self.register_block(self.conv(self.block_index, base_channels*32, base_channels*16, kernel_size=1))
            self.register_block(self.conv(self.block_index, base_channels*16, base_channels*32))
            self.register_block(EmptyLayer())
        
        # path 1
        self.register_block(self.conv(self.block_index, base_channels*32, base_channels*16, kernel_size=1))
        self.register_block(self.conv(self.block_index, base_channels*16, base_channels*32))
        self.register_block(self.conv(self.block_index, base_channels*32, base_channels*16, kernel_size=1))
        self.register_block(self.conv(self.block_index, base_channels*16, base_channels*32))
        self.register_block(self.conv(self.block_index, base_channels*32, base_channels*16, kernel_size=1))
        self.register_block(self.conv(self.block_index, base_channels*16, base_channels*32))
        self.register_block(self.conv(self.block_index, base_channels*32, (num_classes + 1 + 4)*3, kernel_size=1, bn=0, bias=1))
        self.register_block(self.yolo(self.block_index, [6, 7, 8]))

        # path 2
        self.register_block(EmptyLayer()) # route layer 1024
        self.register_block(self.conv(self.block_index, base_channels*16, base_channels*8, kernel_size=1))
        self.register_block(self.upsample(self.block_index, stride=2))
        self.register_block(EmptyLayer()) # route layer 256+128
        self.register_block(self.conv(self.block_index, base_channels*8 + base_channels*16, base_channels*8, kernel_size=1))
        self.register_block(self.conv(self.block_index, base_channels*8, base_channels*16))
        self.register_block(self.conv(self.block_index, base_channels*16, base_channels*8, kernel_size=1))
        self.register_block(self.conv(self.block_index, base_channels*8, base_channels*16))
        self.register_block(self.conv(self.block_index, base_channels*16, base_channels*8, kernel_size=1))
        self.register_block(self.conv(self.block_index, base_channels*8, base_channels*16))
        self.register_block(self.conv(self.block_index, base_channels*16, (num_classes + 1 + 4)*3, kernel_size=1, bn=0, bias=1))
        self.register_block(self.yolo(self.block_index, [3, 4, 5]))

        # path 3
        self.register_block(EmptyLayer())
        self.register_block(self.conv(self.block_index, base_channels*8, base_channels*4, kernel_size=1))
        self.register_block(self.upsample(self.block_index, stride=2))
        self.register_block(EmptyLayer())
        self.register_block(self.conv(self.block_index, base_channels*4 + base_channels*8, base_channels*4, kernel_size=1))
        self.register_block(self.conv(self.block_index, base_channels*4, base_channels*8))
        self.register_block(self.conv(self.block_index, base_channels*8, base_channels*4, kernel_size=1))
        self.register_block(self.conv(self.block_index, base_channels*4, base_channels*8))
        self.register_block(self.conv(self.block_index, base_channels*8, base_channels*4, kernel_size=1))
        self.register_block(self.conv(self.block_index, base_channels*4, base_channels*8))
        self.register_block(self.conv(self.block_index, base_channels*8, (num_classes + 1 + 4)*3, kernel_size=1, bn=0, bias=1))
        self.register_block(self.yolo(self.block_index, [0, 1, 2]))

    def forward(self, x):
        index = 0
        # backbone
        ## input convolution
        o = self.block_list[index](x); index += 1

        ## Downsample 1
        o = self.block_list[index](o); index += 1
        ### Residual Block x1
        shortcut = o
        o = self.block_list[index](o); index += 1
        o = self.block_list[index](o); index += 1
        o = o + shortcut; index += 1

        ## Downsample 2
        o = self.block_list[index](o); index += 1
        ### Residual Block x2
        for i in range(2):
            shortcut = o
            o = self.block_list[index](o); index += 1
            o = self.block_list[index](o); index += 1
            o = o + shortcut; index += 1
        
        ## Dwonsample 3
        o = self.block_list[index](o); index += 1
        ### Residual Block x8
        for i in range(8):
            shortcut = o
            o = self.block_list[index](o); index += 1
            o = self.block_list[index](o); index += 1
            o = o + shortcut; index += 1
        
        route_path_3 = o

        ## Downsample 4
        o = self.block_list[index](o); index += 1
        ### Residual Block x8
        for i in range(8):
            shortcut = o
            o = self.block_list[index](o); index += 1
            o = self.block_list[index](o); index += 1
            o = o + shortcut; index += 1
        
        route_path_2 = o
        
        ## Downsample 5
        o = self.block_list[index](o); index += 1
        ### Residual Block x4
        for i in range(4):
            shortcut = o
            o = self.block_list[index](o); index += 1
            o = self.block_list[index](o); index += 1
            o = o + shortcut; index += 1

        # path 1
        for i in range(5):
            o = self.block_list[index](o); index += 1
        middle_path_1 = o
        for i in range(3):
            o = self.block_list[index](o); index += 1
        path_1_out, _ = o

        # path 2
        o = middle_path_1; index += 1
        o = self.block_list[index](o); index += 1
        o = self.block_list[index](o); index += 1
        o = torch.cat([o, route_path_2], dim=1); index += 1
        for i in range(5):
            o = self.block_list[index](o); index += 1
        middle_path_2 = o
        for i in range(3):
            o = self.block_list[index](o); index += 1
        path_2_out, _ = o

        # path 3
        o = middle_path_2; index += 1
        o = self.block_list[index](o); index += 1
        o = self.block_list[index](o); index += 1
        o = torch.cat([o, route_path_3], dim=1); index += 1
        for i in range(5):
            o = self.block_list[index](o); index += 1
        middle_path_3 = o
        for i in range(3):
            o = self.block_list[index](o); index += 1
        path_3_out, _ = o

        # (num_classes + 1 + 4) x (3*13*13 + 3*26*26 + 3*52*52)
        network_out = torch.cat([path_1_out, path_2_out, path_3_out], dim=1)
        network_out = to_cpu(network_out)
        
        return network_out

    def register_block(self, block):
        self.block_list.append(block)
        self.block_index += 1

    def conv(self, index, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=0, bn=1):
        block = nn.Sequential()

        pad_length = int(kernel_size - 1) // 2 if padding == 1 else 0

        block.add_module(
            f"conv_{index}",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad_length,
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
