from math import sqrt

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
) -> ObjectDetectionMetric:
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

    table_list = list()
    for _ in range(num_classes):
        content = {"tp": 0, "fp": 0, "fn": 0, "bbox_overlap": 0}
        table_list.append(content)

    classnum_background = num_classes
    fn_images_set = set()
    fp_images_set = set()

    for img_num, label in enumerate(labels):
        pred_list = list(map(int, preds[img_num]["labels"]))
        for label_idx in range(len(label["boxes"])):
            near_idx_list = near_box_idx(label, preds[img_num], label_idx, format="xywh")
            for cnt, near_idx in enumerate(near_idx_list):
                iou_score = bbox_iou(
                    preds[img_num]["boxes"][near_idx], label["boxes"][label_idx], format="xywh"
                )
                if (iou_score >= iou_threshold) & (
                    label["labels"][label_idx] == preds[img_num]["labels"][near_idx]
                ):
                    table_list[int(label["labels"][label_idx])]["tp"] += 1  # TP
                    confusion_list[int(label["labels"][label_idx])][
                        int(label["labels"][label_idx])
                    ] += 1  # TP
                    if label["labels"][label_idx] in pred_list:
                        pred_list.remove(label["labels"][label_idx])
                    else:
                        confusion_list[int(label["labels"][label_idx])][
                            classnum_background
                        ] += 1  # FP(overlap)
                        table_list[int(label["labels"][label_idx])]["bbox_overlap"] += 1  # Overlap
                        fp_images_set.add(img_num)
                    break
                elif iou_score < iou_threshold:
                    if len(near_idx_list) - 1 == cnt:
                        confusion_list[classnum_background][
                            int(label["labels"][label_idx])
                        ] += 1  # FN
                        table_list[int(label["labels"][label_idx])]["fn"] += 1  # FN
                        fn_images_set.add(img_num)
                    else:
                        continue

        for fp_pred in pred_list:
            confusion_list[fp_pred][classnum_background] += 1  # FP
            table_list[fp_pred]["fp"] += 1  # FP
            fp_images_set.add(img_num)

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
