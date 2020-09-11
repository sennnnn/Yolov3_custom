import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import *


class Upsample(nn.Module):
    '''
        上采样层, 通过 torch.nn.functional 这个库中的插值函数 interpolate 来实现, 对于 init 函数
    有两个参数:
        scale_factor: 缩放因子, 是可以大于 1 也可以小于 1 的浮点数。
        mode: 插值方式, 即选择不同的插值算法来实现, 一般采用的是最近邻插值。
    对于 forward 函数, 则是直接对输入的特征图进行插值处理然后再输出即可。
    '''
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        
        return x


class Mish(nn.Module):
    '''
        Mish 激活函数, 是最近提出的一种新型激活函数, 比起 ReLU 来说有着更加柔和的梯度, 并且
    在多个数据集上测试发现比起 ReLU 都有一定的效果提升。
        论文题目: Mish: A Self Regularized Non-Monotonic Neural Activation Function
        论文链接: https://arxiv.org/pdf/1908.08681.pdf
    '''
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")

    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class EmptyLayer(nn.Module):
    '''
        空层, 这主要是为了让 route 层和 shortcut 层也有着相似的层定义。
    '''
    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    '''
        yolo head 层, 主要作用为将经过 backbone 处理得到的特征图, 转化为 bounding box, 输出 bounding box 的同时
    还可以根据处于训练阶段还是推理阶段来控制是否计算 loss, 而后来解释一下 yolo head 层的原理和工作流。
        以 yolov3_wider 举例, 最终输出的特征图有三种尺寸, [batch_size, (num_classes + 1 + 4)*3, 13, 13], [batc-
    h_size, (num_classes + 1 + 4)*3, 26, 26], [batch_size, (num_classes + 1 + 4)*3, 52, 52], 首先解释 3, 4 维
    度, 13, 26, 52 实际上是经过 backbone 处理之后得到的三种尺度下的特征图的尺寸, 不同尺度的特征图用于适应不同大小的检
    测目标, 在 yolo 网络中特征图中的一个小格被称为 grid, 一个 grid 统管了图像对应检测区域, 每一个尺寸都会分配三个 an-
    chors, 即每一个尺寸下的 grid 对应的 anchors 是相同的, 这就是为什么第二个维度为什么要 *3 的原因, 而后解释 2 维, 2
    维中的 *3 已经解释完毕, 而后是 (num_classes + 1 + 4), num_classes 对应的通道主要是用于对于预测得到的 bounding box
    需要对其进行一个分类, 需要明确其是检测目标是什么, 所以才需要 num_classes 个通道, 而后 1 为 confidence score, 这一
    项为置信度, 用于体现这一个检测框检测的对象为物体的正确度, 这一项的主要作用为辅助提高检测的准确率, 实际上是可以去掉的, 
    而后 4 为 (tx, ty, tw, th), 这四个值为神经网络推理得到的 bounding box 的位置和大小参数, 但是这 4 个值需要作如下变
    换才能作为真实的图像域 bounding box 定位信息:
            bx = σ(tx) + cx
            by = σ(ty) + cy
            bw = pw*exp(tw)
            bh = ph*exp(th)
        bx, by, bw, bh 只是 grid 域内的坐标, 而 cx 和 cy 为 grid 域的偏移量, 例如 (0, 0) 即 cx = 0, cy = 0 代表着第
    一行第一列的 grid, 要想将其转化为图像域的坐标, 需要乘一个比例因子 stride = img_size/grid_size, 由此可得, 真正的图
    像域的 (x, y, w, h):
            x = bx * stride
            y = by * stride
            w = bw * stride
            h = bh * stride
        需要注意的是, 这里的 x 为 x_center, y 为 y_center, 即 bounding box 的中心坐标。
    '''
    def __init__(self, anchors, num_classes, img_size=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.metrics = {}

        self.img_size = img_size
        self.grid_size = 0  # grid size

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        self.num_anchors = len(anchors)
        self.num_classes = num_classes

        self.ignore_thres = 0.5

        self.obj_scale = 1
        self.noobj_scale = 100

    def calculate_grid_offsets(self, grid_size, cuda=True):
        # torch tensor creating function.
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        
        self.grid_size = grid_size
        self.stride = self.img_size / self.grid_size
        # Calculate offsets for each grid.
        self.grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).type(FloatTensor)
        self.grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).type(FloatTensor)
        
        self.scaled_anchors = FloatTensor([(anchor_w / self.stride, anchor_h / self.stride) for anchor_w, anchor_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_size=None):
        # Tensors for cuda support.
        # Tensor creating functions.
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_size = img_size if img_size != None else self.img_size
        batch_size = x.size(0)
        grid_size = x.size(2)

        # view don't change the matrix meta information and it just change the read pattern. 
        # view require that the matrix meta information is same as the matrix memory layout.
        # permute and transpose change the matrix meta information, but permute and transpose still use the same underlying data.
        # contiguous will let the operating system assign a new block of memory to storage the matrix whose memory layout is same as meta information.
        prediction = (
            x.view(batch_size, self.num_anchors, (self.num_classes + 1 + 4), grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.calculate_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = torch.sigmoid(prediction[..., 0]) + self.grid_x # bx = σ(tx) + cx
        pred_boxes[..., 1] = torch.sigmoid(prediction[..., 1]) + self.grid_y # by = σ(ty) + cy
        pred_boxes[..., 2] = torch.exp(prediction[..., 2]) * self.anchor_w   # pw * exp(tw)       
        pred_boxes[..., 3] = torch.exp(prediction[..., 3]) * self.anchor_h   # ph * exp(th)
        pred_conf = torch.sigmoid(prediction[..., 4])                        # confidence score
        pred_cls = torch.sigmoid(prediction[..., 5:])                        # class score

        # (bx, by, bw, bh, conf, cls1, cls2, ..., clsn) concatenate
        output = torch.cat(
            (
                pred_boxes.view(batch_size, -1, 4) * self.stride,
                pred_conf.view(batch_size, -1, 1),
                pred_cls.view(batch_size, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


def _create_conv_block(index, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=0, bn=1):
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
        nn.LeakyReLU(0.1)
    )

    return block


def _create_maxpool_block(index, kernel_size=2, stride=2):
        block = nn.Sequential()
        block.add_module(
            f"maxpool_{index}",
            nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
            )
        )
        if kernel_size == 2 and stride == 1:
            block.add_module(
                f"fix_pad_{index}",
                nn.ZeroPad2d(0, 1, 0, 1)
            )

        return block


def _create_upsample_block(index, stride, mode="nearest"):
    block = nn.Sequential()
    block.add_module(
        f"upsample_{index}",
        Upsample(
            scale_factor=stride,
            mode=mode,
        )
    )

    return block


def _create_tiny_backbone(in_channels):
    block_list = nn.ModuleList()
    out_channels_per_block = [16]
    block_index = 0
    # (conv + maxpool) * 5
    for iter_index in range(6):
        block = _create_conv_block(
            index=block_index,
            in_channels=in_channels,
            out_channels=out_channels_per_block[-1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=0,
            bn=1,
        )
        block_index += 1
        block_list.append(block)
        if iter_index != 5:
            block = _create_maxpool_block(
                index=block_index,
                kernel_size=2,
                stride=2,
            )
        else:
            block = _create_maxpool_block(
                index=block_index,
                kernel_size=2,
                stride=2,
            )
        block_index += 1
        block_list.append(block)
        in_channels = out_channels_per_block[-1]
        out_channels_per_block.append(out_channels_per_block[-1]*2)
    
    block_list.append(
        _create_conv_block(
            index=block_index,
            in_channels=in_channels,
            out_channels=out_channels_per_block[-1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=0,
            bn=1,
        )
    )
    
    return block_list, block_index, out_channels_per_block[-1]
    

class Yolov3_tiny(nn.Module):
    def __init__(self, in_channels, num_classes, all_anchors, img_size):
        super(Yolov3_tiny, self).__init__()
        self.block_list = nn.ModuleList()
        self.block_index = 0
        self.num_classes = num_classes
        self.all_anchors = all_anchors
        self.img_size = img_size

        # backbone
        # (conv 3x3 1 + maxpool 2x2 2) * 5 + (conv 3x3 1 + maxpool 2x2 1) * 1 + conv 3x3 1
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
        o = self.block_list[13](o)              # convolution 13x13x1024 -> 13x13x256

        # path 1
        backbone_out = o
        o = self.block_list[14](o)              # convolution 13x13x256 -> 13x13x512
        o = self.block_list[15](o)              # convolution 13x13x512 -> 13x13x(num_classes + 1 + 4)*3
        path_1_out, _ = self.block_list[16](o)  # yolo head 13x13x(num_classes + 1 + 4)*3 -> 3*13*13 x (num_classes + 1 + 4)

        # path 2
        o = backbone_out                        # route -4
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

