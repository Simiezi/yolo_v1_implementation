import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable
import torchvision.models as models

IMG_WIDTH = 448
IMG_HEIGHT = 448
SIZE = 7
CLASS = 20
BBOX = 2
LAMBDA_COORD = 5
LAMBDA_NOOBJ = .5
device = torch.device('cpu')


def convert2img(tenzor):
    for box in range(BBOX):
        cell_index = tenzor[:, :, box * 5 - 1].nonzero()
        for i in range(cell_index.size(0)):
            (m, n) = cell_index[i]
            m = int(m)
            n = int(n)
            tenzor[m, n, box * 5] = n * (1. / SIZE) + tenzor[m, n, box * 5].clone() * (1. / SIZE)
            tenzor[m, n, box * 5 + 1] = m * (1. / SIZE) + tenzor[m, n, box * 5 + 1].clone() * (1. / SIZE)
        return tenzor


def IOU(gt_box, pred_box):
    gtx_min1 = (gt_box[0] - gt_box[2] / 2).to(device)
    gtx_max1 = (gt_box[0] + gt_box[2] / 2).to(device)
    gty_min1 = (gt_box[1] - gt_box[3] / 2).to(device)
    gty_max1 = (gt_box[1] + gt_box[3] / 2).to(device)

    predx_min2 = (pred_box[0] - pred_box[2] / 2).to(device)
    predx_max2 = (pred_box[0] + pred_box[2] / 2).to(device)
    predy_min2 = (pred_box[1] - pred_box[3] / 2).to(device)
    predy_max2 = (pred_box[1] + pred_box[3] / 2).to(device)

    zero = torch.tensor(0.).to(device)
    point_a = torch.min(gtx_max1, predx_max2)
    point_b = torch.max(gtx_min1, predx_min2)
    point_c = torch.min(gty_max1, predy_max2)
    point_d = torch.max(gty_min1, predy_min2)

    width = torch.max(point_a - point_b, zero)
    height = torch.max(point_c - point_d, zero)

    square = width * height

    union = (gtx_max1 - gtx_min1) * (gty_max1 - gty_min1) + (predx_max2 - predx_min2) * (predy_max2 - predy_min2) - square

    return square / union


def box_prediction(exit_tensor, gt_tensor):
    out_tensor = convert2img(exit_tensor.clone())
    predict_tensor = torch.zeros(SIZE, SIZE, 5 + CLASS)
    best_box_index = 0
    for i in range(SIZE):
        for j in range(SIZE):
            pred_boxes = torch.empty(2, 5).to(device)

            for box in range(BBOX):
                pred_boxes[box] = out_tensor[i, j, box * 5:box * 5 + 5]
            if len(gt_tensor[i, j, :].nonzero()) > 1:
                max_iou_coefficient = torch.tensor([0.]).to(device)
                gt_box = torch.clone(gt_tensor[i, j, : 4])

                for box in range(BBOX):
                    current_iou = IOU(gt_box, pred_boxes[box])
                    if current_iou > max_iou_coefficient:
                        max_iou_coefficient = current_iou
                        best_box_index = box

            else:
                max_confidence = 0
                for box in range(BBOX):
                    confidence = pred_boxes[box][-1]
                    if max_confidence < confidence:
                        max_confidence = confidence
                        best_box_index = box

            predict_tensor[i, j, :5] = pred_boxes[best_box_index]
            predict_tensor[i, j, 5:] = out_tensor[i, j, -CLASS:]

    return predict_tensor


def loss_func(pred_tensor, gt_tensor, device=torch.device('cpu')):
    loss = torch.zeros(1).to(device)
    for i in range(SIZE):
        for j in range(SIZE):
            if len(gt_tensor[i, j, :].nonzero()) > 1:

                label_class = gt_tensor[i, j, -1].type(torch.int64)

                one_hot_gt = torch.zeros(CLASS)

                one_hot_gt[label_class] = torch.tensor(1.)
                class_probs = pred_tensor[i, j, -CLASS:]

                loss = loss + LAMBDA_COORD * (torch.pow(pred_tensor[i, j, 0] - gt_tensor[i, j, 0], 2)
                                              + torch.pow(pred_tensor[i, j, 1] - gt_tensor[i, j, 1], 2)) \
                       + LAMBDA_COORD * (torch.pow(
                    torch.sqrt(torch.abs(pred_tensor[i, j, 2])) - torch.sqrt(torch.abs(gt_tensor[i, j, 2])), 2)
                                         + torch.pow(
                            torch.sqrt(torch.abs(pred_tensor[i, j, 3])) - torch.sqrt(torch.abs(gt_tensor[i, j, 3])), 2)) \
                       + torch.pow(pred_tensor[i, j, 4] - 1, 2) + torch.sum(torch.pow(class_probs - one_hot_gt, 2))
            else:
                loss = loss + LAMBDA_NOOBJ * torch.pow(pred_tensor[i, j, 4] - 0, 2)

    return loss


def full_loss(pred_tensor, gt_tensor):
    total_loss = torch.tensor(0.0)
    for i in range(pred_tensor.size(0)):
        # print(pred_tensor.size())
        # print(pred_tensor)
        # print('\n\n\n')
        pt = pred_tensor[i]
        # print(pt.size())
        # print(pt)
        # print('\n\n\n')
        gt = gt_tensor[i]
        # print(gt_tensor.size())
        # print(gt_tensor)
        # print('\n\n\n')
        # print(gt.size())
        # print(gt)
        pred_box = box_prediction(pt, gt)
        total_loss = total_loss + loss_func(pred_box, gt)
    total_loss = total_loss / pred_tensor.size(0)
    return total_loss