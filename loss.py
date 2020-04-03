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



def IOU(gt_box, pred_box, device = torch.device("cuda")):
    gtx_min = (gt_box[0] - gt_box[2] / 2).to(device)
    gtx_max = (gt_box[0] + gt_box[2] / 2).to(device)
    gty_min = (gt_box[1] - gt_box[3] / 2).to(device)
    gty_max = (gt_box[1] + gt_box[3] / 2).to(device)
    predx_min = (pred_box[0] - pred_box[2] / 2).to(device)
    predx_max = (pred_box[0] + pred_box[2] / 2).to(device)
    predy_min = (pred_box[1] - pred_box[3] / 2).to(device)
    predy_max = (pred_box[1] + pred_box[3] / 2).to(device)

    zero = torch.tensor(0.).to(device)
    point_a = torch.min(gtx_max, predx_max)
    point_b = torch.max(gtx_min, predx_min)
    point_c = torch.min(gty_max, predy_max)
    point_d = torch.max(gty_min, predy_min)
    width = torch.max(point_a - point_b, zero)
    height = torch.max(point_c - point_d, zero)
    square = width * height
    union = (gtx_max - gtx_min) * (gty_max - gty_min) + (predx_max - predx_min) * (predy_max - predy_min) - square

    return square / union


def box_prediction(exit_tensor, gt_tensor, device = torch.device("cuda")):
    out_tensor = convert2img(exit_tensor.clone())
    predict_tensor = torch.zeros(SIZE, SIZE, 5 + CLASS)
    best_box_index = 0
    for i in range(SIZE):
        for j in range(SIZE):
            pred_boxes = torch.zeros(2, 5).to(device)
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


def loss_func(out_tensor, gt_tensor):
    pred_tensor = box_prediction(out_tensor, gt_tensor)
    loss = torch.zeros(1)
    for i in range(SIZE):
        for j in range(SIZE):
            if len(gt_tensor[i, j, :].nonzero()) > 1:
                label_class = gt_tensor[i, j, -1].type(torch.int64)
                one_hot_gt = torch.zeros(CLASS)
                one_hot_gt[label_class] = 1
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
        pt = pred_tensor[i]
        gt = gt_tensor[i]
        total_loss = total_loss + loss_func(pt, gt)
    total_loss = total_loss / pred_tensor.size(0)
    return total_loss