import torch
import torch.nn as nn

from backbone.tiny import *
from YOLO_head import *

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                        (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                        (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)

class YOLOv3_tiny(nn.Module):
    def __init__(self, config):
        super(YOLOv3_tiny, self).__init__()
        self.ch_in = config['INPUT_CHANNEL']
        self.ch_base = config['BASE_CHANNEL']
        self.width_in = config['INPUT_WIDTH']
        self.height_in = config['INPUT_HEIGHT']
        self.num_class = config['N_CLASSES']
        self.backbone = backbone(self.ch_in, self.width_in, self.height_in, self.num_class, self.ch_base)
        self.head_0 = Yololayer(config, 0, 256)
        self.head_1 = Yololayer(config, 1, 512)

    def forward(self, x, labels=None):
        f0, f1 = self.backbone(x)

        if labels is None:
            boxes0 = self.head_0(f0)
            boxes1 = self.head_1(f1)

            out = torch.cat([boxes0, boxes1], dim=1)
            print(out.shape)
        else:


if __name__ == "__main__":
    from config import config
    model = YOLOv3_tiny(config)

    x = torch.rand(2, 3, 416, 416)

    print(model(x).shape)