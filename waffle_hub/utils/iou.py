from math import sqrt


def near_box_idx(label, pred, label_idx, format = "xywh"):
    """
    For the two tensor inputs, the label_idx index box of label (correct answer) is sorted in order of the closest index in pred.
    However, priority is given to those in the same category.
    
    args
        pred: dictionary each containing the key-values (each dictionary corresponds to a single image). Parameters that should be provided per dict
            boxes (Tensor): float tensor of shape (num_boxes, 4) containing num_boxes detection boxes of the format specified in the constructor.
                By default, this method expects (xmin, ymin, xmax, ymax) in absolute image coordinates, but can be changed using the box_format parameter.
                Only required when iou_type=”bbox”.
            scores (Tensor): float tensor of shape (num_boxes) containing detection scores for the boxes.
            labels (Tensor): integer tensor of shape (num_boxes) containing 0-indexed detection classes for the boxes.
        label: dictionary each containing the key-values (each dictionary corresponds to a single image). Parameters that should be provided per dict:
            boxes (Tensor): float tensor of shape (num_boxes, 4) containing num_boxes ground truth boxes of the format specified in the constructor. only required when iou_type=”bbox”.
                By default, this method expects (xmin, ymin, xmax, ymax) in absolute image coordinates.
            labels (Tensor): integer tensor of shape (num_boxes) containing 0-indexed ground truth classes for the boxes.
        label_idx: target number of class
        format(str): xywh, x1y1x2y2, cxcywh...
            
    return
        result(list): A list sorted in order of label being closest to the box specified in label_idx. The internal element is the index of pred, and if the classes are the same, the priority increases.
    """
    
    distance_result = []
    result = []
    pred_center_list = []
    
    class_num = label['labels'][label_idx]
    
    if format == "xywh":
        xywh_label_bbox = label['boxes'][label_idx]
        label_cx = (xywh_label_bbox[0] + xywh_label_bbox[2]) / 2
        label_cy = (xywh_label_bbox[1] + xywh_label_bbox[3]) / 2
        
        for index, num_class in enumerate(pred['labels']):
            pred_center_list.append((
                (pred['boxes'][index][0] + pred['boxes'][index][2]) / 2,
                (pred['boxes'][index][1] + pred['boxes'][index][3]) / 2,
                num_class,
            ))
    else:
        raise ValueError(
            "not support box format."
        )
        
    for pred_info in pred_center_list:
        distance = 0
        if pred_info[2] != class_num:
            distance += 1e8             # bias
        
        distance += sqrt(abs(pred_info[0] - label_cx)**2 +abs(pred_info[1] - label_cy)**2)
        distance_result.append(distance)
    
    for _ in range(len(distance_result)):
        min_index = distance_result.index(min(distance_result))
        result.append(min_index)
        distance_result[min_index] = float('inf')
        
    return result

def bbox_iou(label_box, pred_box, format = "xywh"):
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
        raise ValueError(
            "not support box format."
        )
        
    inter_x1 = max(pred_x1, label_x1)
    inter_y1 = max(pred_y1, label_y1)
    inter_x2 = min(pred_x2, label_x2)
    inter_y2 = min(pred_y2, label_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    label_area = (label_x2 - label_x1) * (label_y2 - label_y1)
    union = pred_area + label_area - inter_area + 1e-7          # Add epsilon for not allowing divide/0
    
    iou = inter_area / union
    
    return iou