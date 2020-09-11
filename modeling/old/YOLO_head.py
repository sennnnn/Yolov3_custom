import torch
import torch.nn as nn

import numpy as np

class Yololayer(nn.Module):
    strides = [16, 8]
    def __init__(self, config, layer_num, ch_in, ignore_thre=0.7):
        super(Yololayer, self).__init__()
        self.num_class = config['N_CLASSES']

        self.anchors = config['ANCHORS']
        self.anchors_index = config['ANCHORS_INDEX'][layer_num]
        self.all_anchors_grid = [(w / self.stride, h / self.stride)
                                 for w, h in self.anchors]
        self.anchors_selected = [self.all_anchors_grid[i] for i in self.anchors_index]
        self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
        self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)
        self.ref_anchors = torch.FloatTensor(self.ref_anchors)
        self.num_anchors = len(self.anchors_index)

        self.stride = self.strides[layer_num]
        
        self.conv = nn.Conv2d(ch_in, 3*(self.num_class+1+4), kernel_size=1)

        self.l2_loss = nn.MSELoss(size_average=False)
        self.bce_loss = nn.BCELoss(size_average=False)

        self.ignore_thre = ignore_thre

    def forward(self, x, labels=None):
        output = self.conv(x)

        batch_size = output.shape[0]
        fsize = output.shape[2]
        ch_box = (4+1) + self.num_class
        dtype = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor

        output = output.view(batch_size, self.num_anchors, ch_box, fsize, fsize)
        output = output.permute(0, 1, 3, 4, 2)

        # 对 tx, ty, to 以及分类概率使用 sigmoid 处理
        output[..., np.array([0, 1] + list(range(4, ch_box)))] = \
        torch.sigmoid(output[..., np.array([0, 1] + list(range(4, ch_box)))])

        # 构建格点的坐标
        x_grid = dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32), output.shape[:4]))
        y_grid = dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32).reshape(fsize, 1), output.shape[:4]))

        masked_anchors = np.array(self.anchors_selected)

        # 构建 anchor 的 mask
        w_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 0], (1, self.num_anchors, 1, 1)), output.shape[:4]))
        h_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 1], (1, self.num_anchors, 1, 1)), output.shape[:4]))

        pred = output.clone()
        pred[..., 0] += x_grid
        pred[..., 1] += y_grid
        pred[..., 2] += torch.exp(pred[..., 2]) * w_anchors
        pred[..., 3] += torch.exp(pred[..., 3]) * h_anchors

        if labels is None:
            pred[..., 4] *= self.stride
            return pred.view(batch_size, -1, ch_box).data

        # target assignment

        tgt_mask = torch.zeros(batchsize, self.num_anchors,
                               fsize, fsize, 4 + self.num_class).type(dtype)
        obj_mask = torch.ones(batchsize, self.num_anchors,
                              fsize, fsize).type(dtype)
        tgt_scale = torch.zeros(batchsize, self.num_anchors,
                                fsize, fsize, 2).type(dtype)

        target = torch.zeros(batchsize, self.num_anchors,
                             fsize, fsize, ch_box).type(dtype)

        labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = labels[:, :, 1] * fsize
        truth_y_all = labels[:, :, 2] * fsize
        truth_w_all = labels[:, :, 3] * fsize
        truth_h_all = labels[:, :, 4] * fsize
        truth_i_all = truth_x_all.to(torch.int16).numpy()
        truth_j_all = truth_y_all.to(torch.int16).numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = dtype(np.zeros((n, 4)))
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors)
            best_n_all = np.argmax(anchor_ious_all, axis=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anchors_index[0]) | (
                best_n_all == self.anchors_index[1]) | (best_n_all == self.anchors_index[2]))

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(
                pred[b].view(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = 1 - pred_best_iou

            if sum(best_n_mask) == 0:
                continue

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - \
                        truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - \
                        truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.anchors_selected)[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.anchors_selected)[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti,
                                                  0].to(torch.int16).numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(
                        2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)

        # loss calculation

        output[..., 4] *= obj_mask
        output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        output[..., 2:4] *= tgt_scale

        target[..., 4] *= obj_mask
        target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        target[..., 2:4] *= tgt_scale

        bceloss = nn.BCELoss(weight=tgt_scale*tgt_scale,
                             size_average=False)  # weighted BCEloss
        loss_xy = bceloss(output[..., :2], target[..., :2])
        loss_wh = self.l2_loss(output[..., 2:4], target[..., 2:4]) / 2
        loss_obj = self.bce_loss(output[..., 4], target[..., 4])
        loss_cls = self.bce_loss(output[..., 5:], target[..., 5:])
        loss_l2 = self.l2_loss(output, target)

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


if __name__ == "__main__":
    from config import config

    model = Yololayer(config, 0, 256)

    x = torch.rand(2, 256, 26, 26)

    print(model(x).shape)