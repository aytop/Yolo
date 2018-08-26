import torch
from torch.nn import functional as f
from torch import nn
import math
import numpy as np


class MyLoss(nn.Module):
    def __init__(self, average=True):
        super(MyLoss, self).__init__()
        self.average = average
        self.print_counter = 0

    def forward(self, prediction, target, coord_coef=5, noobj_coef=0.5, obj_coef=5, cls_coef=1., PRINT_STEP=10):
        loss_xy = \
            coord_coef * \
            target[:, 14, :, :] * \
            torch.pow(f.sigmoid(prediction[:, 10:12, :, :]) - f.sigmoid(target[:, 10:12, :, :]), 2)

        loss_xy = loss_xy.sum()
     

        loss_wh = \
            coord_coef * \
            target[:, 14, :, :] * \
            torch.pow(f.sigmoid(prediction[:, 12:14, :, :]) - f.sigmoid(target[:, 12:14, :, :]), 2)

        loss_wh = loss_wh.sum()

        loss_cls = \
            cls_coef * \
            target[:, 14, :, :] * \
            TensorCrossEntropy(f.softmax(prediction[:, :10, :, :], dim=1), target[:, :10, :, :])

        loss_cls = loss_cls.sum()

        loss_conf_obj = \
            obj_coef * \
            target[:, 14, :, :] * \
            torch.pow(f.sigmoid(target[:, 14, :, :]) - f.sigmoid(prediction[:, 14, :, :]), 2)

        loss_conf_obj = loss_conf_obj.sum()

        loss_conf_noobj = \
            noobj_coef * \
            (1 - target[:, 14, :, :]) * \
            torch.pow(target[:, 14, :, :] - prediction[:, 14, :, :], 2)

        loss_conf_noobj = loss_conf_noobj.sum()

        if self.print_counter % PRINT_STEP == 0:
            print('Loss xy:', loss_xy)
            print('Loss wh:', loss_wh)
            print('Loss cls:', loss_cls)
            print('Loss conf obj:', loss_conf_obj)
            print('Loss conf noobj:', loss_conf_noobj)

        self.print_counter += 1

        return loss_xy + loss_wh + loss_cls + loss_conf_obj + loss_conf_noobj


class ConfidenceLoss(nn.Module):
    def __init__(self):
        super(ConfidenceLoss, self).__init__()
        self.print_counter = 0

    def forward(self, prediction, target, coord_coef=5, noobj_coef=0.5, obj_coef=5, cls_coef=1., PRINT_STEP=10):
        loss_conf_obj = \
            obj_coef * \
            target[:, 14, :, :] * \
            torch.pow(f.sigmoid(target[:, 14, :, :]) - f.sigmoid(prediction[:, 14, :, :]), 2)

        loss_conf_obj = loss_conf_obj.sum()

        loss_conf_noobj = \
            noobj_coef * \
            (1 - target[:, 14, :, :]) * \
            torch.pow(target[:, 14, :, :] - prediction[:, 14, :, :], 2)

        loss_conf_noobj = loss_conf_noobj.sum()

        if self.print_counter % PRINT_STEP == 0:
            print('Loss conf obj:', loss_conf_obj)
            print('Loss conf noobj:', loss_conf_noobj)

        self.print_counter += 1

        return loss_conf_obj+loss_conf_noobj


class AnchorLoss(nn.Module):
    def __init__(self):
        super(AnchorLoss, self).__init__()
        self.print_counter = 0

    def forward(self, prediction, target, coord_coef=5, noobj_coef=0.5, obj_coef=5, cls_coef=1., PRINT_STEP=10):
        loss = 0
        for anchor in range(0, 31, 15):
            loss_xy = \
                coord_coef * \
                target[:, anchor + 14, :, :] * \
                torch.pow(f.sigmoid(prediction[:, anchor + 10:anchor + 12, :, :]) - target[:, anchor + 10:anchor + 12, :, :], 2)

            loss_xy = loss_xy.sum()

            loss_wh = \
                coord_coef * \
                target[:, anchor + 14, :, :] * \
                torch.pow(f.sigmoid(prediction[:, anchor + 12:anchor + 14, :, :]) - target[:, anchor + 12:anchor + 14, :, :], 2)

            loss_wh = loss_wh.sum()

            loss_cls = \
                cls_coef * \
                target[:, anchor + 14, :, :] * \
                torch.pow(f.softmax(prediction[:, anchor:anchor + 10, :, :], dim=1) - target[:, anchor:anchor + 10, :, :], 2)

            loss_cls = loss_cls.sum()

            loss_conf_obj = \
                obj_coef * \
                target[:, anchor + 14, :, :] * \
                torch.pow(f.sigmoid(target[:, anchor + 14, :, :]) - f.sigmoid(prediction[:, anchor + 14, :, :]), 2)

            loss_conf_obj = loss_conf_obj.sum()

            loss_conf_noobj = \
                noobj_coef * \
                (1 - target[:, anchor + 14, :, :]) * \
                torch.pow(target[:, anchor + 14, :, :] - prediction[:, anchor + 14, :, :], 2)

            loss_conf_noobj = loss_conf_noobj.sum()
            loss += loss_xy + loss_wh + loss_cls + loss_conf_obj + loss_conf_noobj

        if self.print_counter % PRINT_STEP == 0:
                print('Loss:', loss)

        self.print_counter += 1

        return loss


def mse(a, b, average):
    x = torch.pow(torch.add(a, torch.neg(b)), 2)
    if average:
        return torch.mean(x)
    return torch.sum(x)


def tensor_iou(prediction, target, epsilon=1e-5):
    t = target[0, 10:14, :, :]
    p = prediction[0, 10:14, :, :]
    tx = t[0]
    ty = t[1]
    tw = t[2]
    th = t[3]
    px = p[0]
    py = p[1]
    pw = p[2]
    ph = p[3]
    tx1, ty1 = tx - tw / 2, ty - th / 2
    bx1, by1 = tx + tw / 2, ty + th / 2
    tx2, ty2 = px - pw / 2, py - ph / 2
    bx2, by2 = px + pw / 2, py + ph / 2
    x1, y1 = torch.max(tx1, tx2), torch.max(ty1, ty2)
    x2, y2 = torch.min(bx1, bx2), torch.min(by1, by2)
    i = f.relu(x2 - x1) * f.relu(y2 - y1)
    p1 = torch.abs(bx1 - tx1 * by1 - ty1)
    p2 = torch.abs(bx2 - tx2 * by2 - ty2)
    ret = i / (p1 + p2 - i + epsilon)
    # ret[ret != ret] = 0
    return ret


def TensorCrossEntropy(t1, t2):
        print(t2*torch.log(t1))
        print('-'*20)
        print((torch.ones_like(t2)-t2)*torch.log((torch.ones_like(t1)-t1)))
        input()
        nlog = t2*torch.log(t1) + (torch.ones_like(t2)-t2)*torch.log((torch.ones_like(t1)-t1))
        return -nlog.sum()
