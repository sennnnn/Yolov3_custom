import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np

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
    """
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
    """
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


def Consturct_Model_from_Module_define(block_defs):
    '''
        解析了 .cfg 文件之后会获得一个 block 列表定义, 每一个 block 定义对应 .cfg 文件中 [xxx] 开头的一段内容,
    而每一个 block 定义中带有很多 module 定义, module 是 pytorch 的概念, 一般为一个 module 就为一层, 将 block 列表定义
    转化为 pytorch 中的层, 便是这个函数的意义, 通过解析 block 列表定义, 构建相应的 pytorch module 然后再通过 nn.Sequential 打包成
    pytorch block 而后再注册到一个 ModuleList 里面以便前向传播时使用, 同时会记录各个 block 的输出通道数是多少, 以便构建 pytorch module。
    输入:
        block_defs: 解析了 .cfg 文件得到的 block 列表定义。
    输出:
        hyper_params: 模型的超参数。
        block_list: 解析 block 列表定义, 得到的 pytorch block 列表, pytorch block 为由多个 pytorch module 构成的 nn.Sequential 模型。
    ''' 
    hyper_params = block_defs.pop(0)
    out_channels_per_block = [int(hyper_params["channels"])]
    block_list = nn.ModuleList()
    # Model consists of many blocks, and block consits of some modules.
    for index, module_def in enumerate(block_defs):
        block = nn.Sequential()

        if module_def["type"] == "convolutional":
            batch_norm_flag = int(module_def["batch_normalize"])
            out_channels = int(module_def["filters"])
            # in channels is the out channels of last block.    
            in_channels = out_channels_per_block[-1]
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            pad_length = int(kernel_size - 1)//2
            activation = module_def["activation"]
            block.add_module(
                f"conv_{index}", 
                nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = pad_length,
                    bias = not batch_norm_flag,
                )
            )
            if batch_norm_flag == True:
                block.add_module(f"batch_norm_{index}", nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5))
            if activation == "leaky":
                block.add_module(f"leaky_{index}", nn.LeakyReLU(0.1))
            elif activation == "mish":
                block.add_module(f"mish_{index}", Mish())
        
        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            out_channels = out_channels_per_block[-1]
            pad_length = int(kernel_size - 1) // 2
            if kernel_size == 2 and stride == 1:
                # The Maxpool2d with stride == 1 will remove 1 pixel length on right and bottom respectively.
                block.add_module(f"_padding_{index}", nn.ZeroPad2d((0, 1, 0, 1)))
            block.add_module(
                f"maxpool_{index}",
                nn.MaxPool2d(
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = pad_length,
                )
            )
        
        elif module_def["type"] == "upsample":
            stride = int(module_def["stride"])
            out_channels = out_channels_per_block[-1]
            block.add_module(
                f"upsample_{index}", 
                Upsample(
                    scale_factor = stride,
                    mode = "nearest",
                    ),
            )

        elif module_def["type"] == "route":
            layer_indexs = [int(layer_index) for layer_index in module_def["layers"].split(",")]
            out_channels = sum([out_channels_per_block[1:][layer_index] for layer_index in layer_indexs])
            block.add_module(
                f"route_{index}",
                EmptyLayer(),
            )

        elif module_def["type"] == "shortcut":
            shortcut_from_index = int(module_def["from"])
            out_channels = out_channels_per_block[1:][shortcut_from_index]
            block.add_module(
                f"shortcut_{index}",
                EmptyLayer(),
            )

        elif module_def["type"] == "yolo":
            selected_anchor_indexs = [int(x) for x in module_def["mask"].split(",")]
            all_anchors = [int(x) for x in module_def["anchors"].split(",")]
            selected_anchors = [(all_anchors[2*i], all_anchors[2*i+1]) for i in selected_anchor_indexs]
            num_classes = int(module_def["classes"])
            img_size = (int(hyper_params["width"]), int(hyper_params["height"]))
            out_channels = out_channels_per_block[-1]
            block.add_module(
                f"yolo_{index}",
                YOLOLayer(
                    selected_anchors,
                    num_classes,
                    img_size[0],
                )
            )
        # Register block in the model and Record the number of output channel of each block.
        block_list.append(block)
        out_channels_per_block.append(out_channels)

    return hyper_params, block_list


def Parse_Model_Config(cfg_path):
    '''
        解析 darknet 框架下的 .cfg 文件, 构建一个模型配置列表, 列表中的每一项代表了神经网络的一层
    或者模型的全局配置。
    输入:
        cfg_path: darknet 框架下的 .cfg 文件的路径。
    输出:
        block_defs: 存放有模型配置的列表。
    '''
    f = open(cfg_path, 'r')
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != '' and not line.startswith('#')]
    block_defs = []
    for line in lines:
        # Module starts with "[xxx]"
        if line[0] == ('['): 
            block_defs.append({})
            block_defs[-1]['type'] = line[1:-1].strip()
            if block_defs[-1]['type'] == 'convolutional':
                block_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            key = key.strip()
            block_defs[-1][key] = value

    return block_defs


class Darknet(nn.Module):
    '''
        Darknet 框架的简易版本, 可以通过解析 .cfg 文件构建对应的 pytorch 形式的神经网络, 可以加载 .weights 文件来
    为 pytorch 形式的神经网络赋予参数, 同时还可以将 pytorch 形式的神经网络的参数保存为 .weights 文件, 这里有几个概
    念需要说明, 首先使用 .cfg 文件构建得到的 pytorch 形式的神经网络, 主要的单元为 block, block 中会有多个 layer, 
    .cfg 文件解析之后会得到一个 block 定义列表, 一个 block 就对应了 .cfg 文件中 [xxx] 开头的一段内容, 解析 .cfg 的
    函数为 Parse_Model_Config, 根据 block_defs 即 block 定义列表, 构建 pytorch 形式的神经网络的函数为 Construct_
    Model_from_Module_define。
        对于 load_darknet_weights 和 save_darknet_weights 有几个概念也需要说明, 首先由于 darknet 框架下可以将所有
    的网络都表达成顺序网络, 所以每一个 block 可以用 index 来表示, cutoff 则代表着保存或者加载网络参数时的截断层的下标,
    而后需要解释一下 .weights 文件的存储内容, .weights 文件中分为头部信息和网络参数, 前 5 个 32 进制数为头部信息, 这
    五个数为 32 进制整型数, 而剩下的都为网络参数, 为 32 进制浮点数, 网络参数实际上只有 conv layer 和 batch_norm layer
    两个层的参数, conv layer 的参数分为 bias, weights, 其中 bias 在前, weights 在后, (conv layer 如果后面有 batch_-
    norm layer 那么添加 bias 是无意义的, 所以一般在后有 batch_norm layer, conv layer 不需要 bias), batch_norm layer
    的参数分为 bias, weights, mean, var (按参数在文件中的排列顺序)。
    '''
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.block_defs = Parse_Model_Config(config_path)
        self.hyper_params, self.block_list = Consturct_Model_from_Module_define(self.block_defs)
        self.yolo_layers = [block[0] for block in self.block_list if hasattr(block[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        img_size = x.shape[2]
        loss = 0
        block_outputs, yolo_outputs = [], []
        for i, (module_def, block) in enumerate(zip(self.block_defs, self.block_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                out = block(x)
            elif module_def["type"] == "route":
                # route layer is actually concatenate operation.
                # route layer can use a sequential way to express multi path network architecture.
                out = torch.cat([block_outputs[int(block_index)] for block_index in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                # shortcut layer is actually residual addition operation.
                layer_index = int(module_def["from"])
                out = block_outputs[-1] + block_outputs[layer_index]
            elif module_def["type"] == "yolo":
                out, layer_loss = block[0](x, targets, img_size)
                loss += layer_loss
                yolo_outputs.append(out)
            block_outputs.append(out)
            x = out
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))

        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        # Open the weights file
        with open(weights_path, "rb") as f:
            # Intel x86 system use little endian to storage data.
            # First five are header values, [major, minor, revision, batch_num_low, batch_num_high]
            header = np.fromfile(f, dtype=np.int32, count=5)  
            # Saving header info to write in weights when saving weights.
            self.header_info = header   
            # Number of images seen during training.
            self.seen = header[3]
            # Rest of the weights are parameters of neural network.
            parameters = np.fromfile(f, dtype=np.float32) 

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for index, (module_def, block) in enumerate(zip(self.block_defs, self.block_list)):
            if index == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = block[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    batch_norm_layer = block[1]
                    num_b_params = batch_norm_layer.bias.numel()  # Number of biases
                    # Bias
                    batch_norm_b = torch.from_numpy(parameters[ptr : ptr + num_b_params]).view_as(batch_norm_layer.bias)
                    batch_norm_layer.bias.data.copy_(batch_norm_b)
                    ptr += num_b_params
                    # Weight
                    batch_norm_w = torch.from_numpy(parameters[ptr : ptr + num_b_params]).view_as(batch_norm_layer.weight)
                    batch_norm_layer.weight.data.copy_(batch_norm_w)
                    ptr += num_b_params
                    # Running Mean
                    batch_norm_rm = torch.from_numpy(parameters[ptr : ptr + num_b_params]).view_as(batch_norm_layer.running_mean)
                    batch_norm_layer.running_mean.data.copy_(batch_norm_rm)
                    ptr += num_b_params
                    # Running Var
                    batch_norm_rv = torch.from_numpy(parameters[ptr : ptr + num_b_params]).view_as(batch_norm_layer.running_var)
                    batch_norm_layer.running_var.data.copy_(batch_norm_rv)
                    ptr += num_b_params
                else:
                    # Load conv bias
                    num_b_params = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(parameters[ptr : ptr + num_b_params]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b_params
                # Load conv weights
                num_w_params = conv_layer.weight.numel()
                conv_w = torch.from_numpy(parameters[ptr : ptr + num_w_params]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w_params

    def save_darknet_weights(self, save_path, cutoff=-1):
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        for i, (module_def, block) in enumerate(zip(self.block_defs[:cutoff], self.block_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = block[0]
                # If batch norm, load batch norm first
                if module_def["batch_normalize"]:
                    batch_norm_layer = block[1]
                    batch_norm_layer.bias.data.cpu().numpy().tofile(fp)
                    batch_norm_layer.weight.data.cpu().numpy().tofile(fp)
                    batch_norm_layer.running_mean.data.cpu().numpy().tofile(fp)
                    batch_norm_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()


def Save_Torch_State_Dict(model, save_path):
    save_suffix = os.path.splitext(save_path)[1]
    
    assert save_suffix in [".pt", ".pth", ".pkl"], \
        f"filename suffix {save_suffix} not right, it should be one of [\".pt\", \".pth\", \".pkl\"]."
    
    torch.save(model.state_dict(), save_path)


def Load_Torch_State_Dict(model, load_path):
    load_suffix = os.path.splitext(load_path)[1]
    
    assert load_suffix in [".pt", ".pth", ".pkl"], \
        f"filename suffix {load_suffix} not right, it should be one of [\".pt\", \".pth\", \".pkl\"]."
    
    model.load_state_dict(torch.load(load_path))