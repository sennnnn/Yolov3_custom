import cv2
import torch
import numpy as np
import torch.nn.functional as F


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    
    return names


def Convert_Img_to_Tensor(img):
    '''
        将图像转化为 pytorch 中的 tensor 类型, 由于这里读入图像使用的是 opencv 的 imread, 
    所以读入为 numpy array 形式, 不考虑 PIL 库的格式读入之后, 做归一化。

    输入:
        img: 使用 opencv 库读入的 numpy array 形式的图片, (r, c, 3) 的形状。
    
    输出:
        tensor: 将 numpy array 形式的图片转化为 pytorch tensor 形式得到的结果。
    '''
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)

    if img.type() == "torch.ByteTensor":
        return img.float().div(255)
    else:
        return img


def Convert_String_to_Int(string):
    '''
        将字符串转换为整型数, 如果转换失败, 则直接返回原字符串, 如果转换成功, 则返回转换得到
    的整型数。

    输入:
        string: 待处理的字符串。
    
    输出:
        out: 转换成功得到的整型数或者转换失败的原字符串。
    '''
    try:
        out = int(string)
        return out
    except ValueError:
        print(f"{string} can't be converted to int type.")
        out = string
        return out


def Convert_xywh_to_xyxy(x):
    '''
        将 [x_center, y_center, bbox_w, bbox_h] 这种 bounding box 描述方式, 转化为
    [xmin, ymin, xmax, ymax] 这种 bounding box 描述方式。
    '''
    if type(x) is np.ndarray:
        y = x.copy()
    elif type(x) is torch.Tensor:
        y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    
    return y


def Non_Max_Suppression(prediction, conf_thres=0.5, nms_IoU_thres=0.4):
    '''
    极大值抑制:
        1. 首先滤除 object confidence 低于 conf_thres 的 bounding box。
        2. 计算 bounding box 得分, (score = class confidence * object confidence) 从大到小排序。
        3. 选出得分最高的框 bbox_max, 然后遍历其他的框, 分别与 bbox_max 计算 IoU, 大于 nms_IoU_thres 则保留。
        4. 而后从剩余的框中重复 2, 3 步, 直至没有框剩余。

    输入:
        prediction: 神经网络推理图像之后得到的输出, 格式为: [batch_size, bbox_count, num_classes + 1 + 4], 
                    最后一个维度为 (x_center, y_center, bbox_width, bbox_height, obj_conf, cls1_conf, cls2_conf, ..., clsn_conf)
        conf_thres: 第 1 步中滤除 bounding box 时的阈值参数。
        nms_IoU_thres: 第 3 步中滤除 bounding box 时的阈值参数。

    输出:
        output: 经过非极大值抑制处理之后得到的检测结果, 形状为: [valid_bbox_count, 4 + 1 + 1 + 1], 最后一个维
                度为 (xmin, ymin, xmax, ymax, obj_conf, cls_score, cls_pred)。
    '''
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = Convert_xywh_to_xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    # One iteration process one image in the batch.
    for image_i, bboxes_pred in enumerate(prediction):
        # Filtering out bounding boxes whose confidence scores is below threshold.
        bboxes_pred = bboxes_pred[bboxes_pred[:, 4] >= conf_thres]
        # If none of the bounding boxes are remaining, processing next image in one batch.
        if bboxes_pred.shape[0] == 0:
            continue
        # Object confidence times class confidence.
        score = bboxes_pred[:, 4] * bboxes_pred[:, 5:].max(1)[0]
        # Sorting the bounding boxes by score from small to large.
        bboxes_pred = bboxes_pred[(-score).argsort()]
        # torch.Tensor.max(dim, keepdim=True) function will return (max value, max value index).
        class_confs, class_preds = bboxes_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((bboxes_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_IoU_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def bbox_iou(bbox1, bbox2, xyxy=True):
    '''
        计算两个 bounding box 之间的交并比 (IoU, Intersection over Union), 交并比:
    
    原始公式为:
                                    A∩B
                            IoU = --------
                                    A∪B
    |-----------|
    |           |
    |     |-------------|            C
    |  A  |     |       |   IoU = -------
    |     |  C  |       |          A+B-C
    |     |     |  B    |
    |------------       |
          |             |
          |-------------|

    输入: 
        bbox1, bbox2: 需要进行 IoU 计算的两个 bounding box。
        xyxy: 说明 bounding box 的表达格式是否为 [xmin, ymin, xmax, ymax]。
    
    输出:
        IoU: 交并比, Intersection over Union。
    '''
    if not xyxy:
        # Transform from center and width to exact coordinates
        bbox1 = Convert_xywh_to_xyxy(bbox1)
        bbox2 = Convert_xywh_to_xyxy(bbox2)

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[:, 0], bbox1[:, 1], bbox1[:, 2], bbox1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[:, 0], bbox2[:, 1], bbox2[:, 2], bbox2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    IoU = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return IoU


def Pad_Tensor_to_Square(tensor, pad_value=0):
    '''
        将 pytorch tensor 通过垫值, 使得其长宽相等, 且 new_w or new_h = max(raw_w, raw_h),
    这主要是考虑到 yolo 系列网络要求输入的图像为长宽相等的形式, 然而并不是所有的图像都是长宽相等
    的。
    
    输入:
        tensor: 待处理的 pytorch tensor。
        pad_value: 用于垫边的值, 一般为 0。
    
    输出:
        processed_tensor: 处理之后的 pytorch tensor, 长宽是相同的, 为一正方形张量。
    '''
    c, h, w = tensor.shape
    size_diff = abs(h - w)
    # up, bottom padding or left, right padding.
    pad1, pad2 = size_diff // 2, size_diff - size_diff // 2
    # Determine up, bottom or left, right padding.
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding by functional tools.
    processed_tensor = F.pad(tensor, pad, "constant", value=pad_value)

    return processed_tensor


def Tensor_Resize(tensor, size, mode="nearest"):
    tensor = tensor.unsqueeze(0)
    tensor = F.interpolate(tensor, size=size, mode=mode)
    processed_tensor = tensor.squeeze(0)
    
    return processed_tensor

  
def Rescale_Boxes(bboxes, current_size, original_shape):
    '''
        以输入张量尺寸为 416x416, 原始图像尺寸为 480x640 举例说明:
        bounding box 的表达形式为 [xmin, xmax, ymin, ymax], 而最初 bounding box 的坐标是对应于 416x416 
    的输入张量, 而后分析原始图像尺寸, 其中第一个维度的维数小于第二个维度的维数 (480 > 640), 所以需要对第一个
    维度进行 pad, 经过 pad 之后, 图像尺寸变化为 640x640 而后经过 resize 变为输入张量, 尺寸变化为 416x416,
    由此可得 416x416 中, 原始图像占有的尺寸为 312x416, 由于原始图像被 pad 时, 是被放置在正中间, 所以输入张量
    的第一个维度中 0~47 维与 368~416 维都不属于原始图像, 原始图像只占有第一个维度的 48~367 维, 而对于第二个维
    度原始图像完全占有, 将 bounding box 转化到对应于原始图像的 480x640 的尺寸, 需要完成以下几步:
        1. 将对应于输入张量 416x416 的坐标转化为对应于第一个维度为输入张量的 48~367 维, 第二个维度为输入张量的
     0~416 维的 312x416 尺寸的输入张量, 由于第一个维度对应于 y 坐标, 第二个维度对应于 x 坐标, 所以只需要将 y 坐
    标都减去 48 就好了, 这样就对应于 312x416 尺寸了。
        2. bounding box 已经对应于 312x416 尺寸, 需要将其转化为 480x640 尺寸, 这是只需要乘比例因子: scale_factor = 
    origin_size / current_size (480/312 or 640/416) 即可。

    输入:
        bboxes: 待处理, 需要进行坐标转换的 bounding boxes。
        current_size: 当前的输入张量的尺寸, 输入张量的长宽一定是相等的。
        original_shape: 原始图像的尺寸。
    
    输出:
        bboxes: 对应于原始图像尺寸的 bounding boxes。 
    '''

    orig_h, orig_w = original_shape
    orig_size = max(orig_h, orig_w)
    size_diff = abs(orig_h - orig_w)
    # Just the dimension which has shorter channels needs to pad.
    # For instance, 480x640, So the first dimension needs to pad.
    pad_x, pad_y = (0, size_diff // 2) if orig_h <= orig_w else (size_diff // 2, 0)

    bboxes[:, 0] = ((bboxes[:, 0] - pad_x) / (current_size - 2*pad_x)) * orig_w 
    bboxes[:, 1] = ((bboxes[:, 1] - pad_y) / (current_size - 2*pad_y)) * orig_h
    bboxes[:, 2] = ((bboxes[:, 2] - pad_x) / (current_size - 2*pad_x)) * orig_w
    bboxes[:, 3] = ((bboxes[:, 3] - pad_y) / (current_size - 2*pad_y)) * orig_h
    
    return bboxes