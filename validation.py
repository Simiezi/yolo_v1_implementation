import torch
import numpy as np


SIZE = 7
CLASS = 20
BBOX = 2


def predict_boxes_for_classes(out_tensors, image_names):
    conf_threshold = 0.3
    out_tensors = [convert_coord_cell2img(out_tensors[i]) for i in range(out_tensors.size(0))]
    Detections = dict()
    for classes in range(CLASS):
        Detections[classes] = []

    for k in range(len(out_tensors)):
        temp_tensor = out_tensors[k]
        img_name = image_names[k]
        for i in range(SIZE):
            for j in range(SIZE):
                _, cls = torch.max(temp_tensor[i, j, :][-C:], 0)

                best_conf = 0
                for b in range(BBOX):
                    box = (img_name,)
                    box = box + tuple(T[i, j, 5 * b: 5 * b + 5])

                    if b == 0:
                        best_box = box

                    if T[i, j, 5 * b + 4] > best_conf:
                        best_box = box

                if best_box[-1] > conf_threshold:
                    Detections[cls.item()].append(best_box)
    return Detections


def gt_boxes_for_classes(csv, image_names):
    Detections = dict()
    for classes in range(CLASS):
        Detections[classes] = []

    return Detections

def eval_IOU(Detections, Ground_truth):
    Results = {}
    for c in range(CLASS):
        Det = Detections[classes]
        GT = Ground_truth[classes]
        Results[c] = []
        for det in Det:
            img_ground_truth = list(filter(lambda x: x[0] == det[0].split('.')[0], GT))
            if len(img_ground_truth) > 0:
                inter_over_unions = []
                for gt in img_ground_truth:
                    curr_iou = calc_IOU(det[1:5], torch.tensor(gt[1:5]))
                    inter_over_unions.append(curr_iou.item())

                iou = max(inter_over_unions)
                img_ground_truth.pop(np.argmax(inter_over_unions))
            else:
                iou = 0.0
            Results[c].append(list(det) + [iou])
    return Results