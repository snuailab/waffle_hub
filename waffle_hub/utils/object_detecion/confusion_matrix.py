from math import sqrt
from typing import Union

from waffle_hub.schema.evaluate import ObjectDetectionMetric
from waffle_hub.schema.fields import Annotation


def near_box_idx(label, pred, label_idx, format="xywh"):
    """
    For the two tensor inputs, the label_idx index box of label (correct answer) is sorted in order of the closest index in pred.
    However, priority is given to those in the same category.

    args
        pred: dictionary each containing the key-values (each dictionary corresponds to a single image)
            boxes (Tensor): float tensor of shape (num_boxes, 4) containing num_boxes detection boxes of the format specified in the constructor.
                By default, this method expects (xmin, ymin, xmax, ymax) in absolute image coordinates.
            scores (Tensor): float tensor of shape (num_boxes) containing detection scores for the boxes.
            labels (Tensor): integer tensor of shape (num_boxes) containing 0-indexed detection classes for the boxes.
        label: dictionary each containing the key-values (each dictionary corresponds to a single image)
            boxes (Tensor): float tensor of shape (num_boxes, 4) containing num_boxes ground truth boxes of the format specified in the constructor
                By default, this method expects (xmin, ymin, xmax, ymax) in absolute image coordinates.
            labels (Tensor): integer tensor of shape (num_boxes) containing 0-indexed ground truth classes for the boxes.
        label_idx: target number of class
        format(str): xywh, x1y1x2y2, cxcywh...

    return
        result(list): A list sorted in order of label being closest to the box specified in label_idx.
                    The internal element is the index of pred, and if the classes are the same, the priority increases.
    """

    distance_result = []
    result = []
    pred_center_list = []

    class_num = label["labels"][label_idx]

    if format == "xywh":
        xywh_label_bbox = label["boxes"][label_idx]
        label_cx = (xywh_label_bbox[0] + xywh_label_bbox[2]) / 2
        label_cy = (xywh_label_bbox[1] + xywh_label_bbox[3]) / 2

        for index, num_class in enumerate(pred["labels"]):
            pred_center_list.append(
                (
                    (pred["boxes"][index][0] + pred["boxes"][index][2]) / 2,
                    (pred["boxes"][index][1] + pred["boxes"][index][3]) / 2,
                    num_class,
                )
            )
    else:
        raise ValueError("not support box format.")

    for pred_info in pred_center_list:
        distance = 0
        if pred_info[2] != class_num:
            distance += 1e8  # bias

        distance += sqrt(abs(pred_info[0] - label_cx) ** 2 + abs(pred_info[1] - label_cy) ** 2)
        distance_result.append(distance)

    for _ in range(len(distance_result)):
        min_index = distance_result.index(min(distance_result))
        result.append(min_index)
        distance_result[min_index] = float("inf")

    return result


def bbox_iou(label_box: list, pred_box: list, format="xywh"):
    """
    Find the intersection over union(Iou) using two bounding box information.
    Args:
        label_box (list): bbox point
        pred_box (list): bbox point
        format (str): bbox format. ex)xywh

    Returns:
        iou (float): 0~1 float value
    """
    if format == "xywh":
        pred_x1, pred_y1 = pred_box[0:2]
        pred_x2 = pred_box[0] + pred_box[2]
        pred_y2 = pred_box[1] + pred_box[3]
        label_x1, label_y1 = label_box[0:2]
        label_x2 = label_box[0] + label_box[2]
        label_y2 = label_box[1] + label_box[3]

    else:
        raise ValueError("not support bbox format.")

    inter_x1 = max(pred_x1, label_x1)
    inter_y1 = max(pred_y1, label_y1)
    inter_x2 = min(pred_x2, label_x2)
    inter_y2 = min(pred_y2, label_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    label_area = (label_x2 - label_x1) * (label_y2 - label_y1)
    union = pred_area + label_area - inter_area + 1e-7  # Add epsilon for not allowing divide/0

    iou = inter_area / union

    return iou


def getConfusionMatrix(
    iou_threshold: float = 0.5,
    preds: list[Annotation] = None,
    labels: list[Annotation] = None,
    num_classes: int = None,
) -> dict[Union[set, list]]:
    """
    It can find confusion matrix for object detection model analysis.

    Args:
        iou_threshold (float): Threshold value for iou calculation
        preds (list): A list consisting of pred (see explanation above)
        labels (list): A list consisting of label (see explanation above)
        num_classes (int): number of classes

    Retruns:
        result (dict): Details about the confusion matrix
            confusion_matrix (list): Confusion matrix made up of square matrices
            tpfpfn (list): Each index has a dictionary with keys "tp", "fp", and "fn".
            fp (set): fp Set index of images. The image can be tracked through this information.
            fn (set): fn Set index of images
    """

    result = dict()
    confusion_list = list()
    for _ in range(num_classes + 1):
        content = [0] * (num_classes + 1)
        confusion_list.append(content)
    background_idx = num_classes

    table_list = list()
    for _ in range(num_classes):
        content = {"tp": 0, "fp": 0, "fn": 0, "bbox_overlap": 0}
        table_list.append(content)

    fn_images_set = set()
    fp_images_set = set()

    # 속도 향상 필요 (brute force -> near_box_idx 활용 or serach algorithm)
    for img_idx, (pred_list, label_list) in enumerate(zip(preds, labels)):
        # label_list = list(map(int, label["labels"]))
        fp_list = list(map(int, pred_list["labels"]))  # 이미 찾은 것은 -1로 처리
        fn_list = list(map(int, label_list["labels"]))  # 이미 찾은 것은 -1로 처리
        for label_idx, label in enumerate(label_list["labels"]):
            label = int(label)
            for pred_idx, pred in enumerate(pred_list["labels"]):
                pred = int(pred)
                iou_score = bbox_iou(
                    pred_list["boxes"][pred_idx], label_list["boxes"][label_idx], format="xywh"
                )
                if (iou_score >= iou_threshold) and (
                    label_list["labels"][label_idx] == fp_list[pred_idx]
                ):  # 겹치고 같은 라벨
                    table_list[label]["tp"] += 1  # TP
                    confusion_list[label][label] += 1  # TP
                    fp_list[pred_idx] = -1
                    fn_list[label_idx] = -1
                    break

                    # confusion_list[int(label["labels"][label_idx])][
                    #     classnum_background
                    # ] += 1  # FP(overlap)
                    # table_list[int(label["labels"][label_idx])]["bbox_overlap"] += 1  # Overlap
                    # table_list[int(label["labels"][label_idx])]["fp"] += 1  # Overlap ???
                    # fp_images_set.add(img_idx)

        for fn_label in fn_list:  # FN 처리
            if fn_label == -1:
                continue
            confusion_list[background_idx][fn_label] += 1  # FN
            table_list[fn_label]["fn"] += 1  # FN 미탐
            fn_images_set.add(img_idx)

        for pred_idx, fp_pred in enumerate(fp_list):  ## FP over 처리 안됨
            if fp_pred == -1:
                continue
            confusion_list[fp_pred][background_idx] += 1
            table_list[fp_pred]["fp"] += 1  # FP
            for label_idx, label in enumerate(fn_list):
                if label != -1:  # TP 처리된 것 중에 겹치는 것이 있으면
                    continue
                iou_score = bbox_iou(
                    pred_list["boxes"][pred_idx], label_list["boxes"][label_idx], format="xywh"
                )
                if iou_score >= iou_threshold and fp_pred == label_list["labels"][label_idx]:
                    table_list[fp_pred]["bbox_overlap"] += 1
            fp_images_set.add(img_idx)

    result["confusion_matrix"] = confusion_list
    result["tpfpfn"] = table_list
    result["fp"] = fp_images_set
    result["fn"] = fn_images_set

    return result


def getf1(
    TPFPFN: list[dict],
):
    """
    Calculate indicators related to f1.

    Args:
        TPFPFN (list[dict]): Each index has a dictionary with keys "tp", "fp", and "fn".

    Returns:
        (dict): Computed f1 dictionary
            f1_scores (dict) : f1 number calculated for each class. Harmonic mean of precision and recall
            macro_f1_score (float): macro average f1 score, the sum of f1 values ​​divided by the total number of classes.
                                If all labels have similar importance, refer to the macro average value.
            micro_f1_score (float): micro average f1 score, It is called F1.
                                Calculate metrics globally by counting the total true positives, false negatives and false positives.
                                Micro-average is a more effective evaluation indicator in datasets with class imbalance problems.
            weighted_f1_score (float): weighted_f1_score,
                                Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
                                This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
    """
    f1_scores = []
    weighted_f1_score = 0

    cnt_true = []
    ratio = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    eps = 1e-7

    for conf in TPFPFN:
        total_tp += conf["tp"]
        total_fp += conf["fp"]
        total_fn += conf["fn"]
        precision = conf["tp"] / (conf["tp"] + conf["fp"] + eps)
        recall = conf["tp"] / (conf["tp"] + conf["fn"] + eps)
        f1_scores.append(2 * (precision * recall) / (precision + recall + eps))
        cnt_true.append(conf["tp"] + conf["fn"])
    macro_f1_score = sum(f1_scores) / (len(f1_scores) + eps)
    micro_f1_score = total_tp / (total_tp + 0.5 * (total_fp + total_fn) + eps)

    for num in range(len(TPFPFN)):
        ratio.append(cnt_true[num] / (sum(cnt_true) + eps))
        weighted_f1_score += ratio[num] * f1_scores[num]

    return {
        "f1_scores": f1_scores,
        "macro_f1_score": macro_f1_score,
        "micro_f1_score": micro_f1_score,
        "weighted_f1_score": weighted_f1_score,
    }
