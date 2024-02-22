from math import sqrt


def near_box_idx(label, pred, label_idx, format = "xywh", ):
    """
    두 텐서입력에 대해서 label(정답)의 label_idx 인덱스 박스에 대해 pred에서 가장 가까운 인덱스 순으로 정렬한다.
    단 카테고리가 같은 것이 우선이다.
    args
        pred: dictionary each containing the key-values (each dictionary corresponds to a single image). Parameters that should be provided per dict
            boxes (Tensor): float tensor of shape (num_boxes, 4) containing num_boxes detection boxes of the format specified in the constructor. By default, this method expects (xmin, ymin, xmax, ymax) in absolute image coordinates, but can be changed using the box_format parameter. Only required when iou_type=”bbox”.
            scores (Tensor): float tensor of shape (num_boxes) containing detection scores for the boxes.
            labels (Tensor): integer tensor of shape (num_boxes) containing 0-indexed detection classes for the boxes.
        label: dictionary each containing the key-values (each dictionary corresponds to a single image). Parameters that should be provided per dict:
            boxes (Tensor): float tensor of shape (num_boxes, 4) containing num_boxes ground truth boxes of the format specified in the constructor. only required when iou_type=”bbox”. By default, this method expects (xmin, ymin, xmax, ymax) in absolute image coordinates.
            labels (Tensor): integer tensor of shape (num_boxes) containing 0-indexed ground truth classes for the boxes.
        label_idx: target number of class
        format(str): xywh, x1y1x2y2, cxcywh...\
            
    return
        result(list) label에서 label_idx에 지정한 박스와 가장 가까운 순으로 정렬된 리스트. 내부 요소는 pred의 인덱스이며 클래스가 동일할 시 우선순위 올라감.
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
        pass #TODO other form
        
    for pred_info in pred_center_list:
        distance = 0
        if pred_info[2] != class_num:   # 클래스가 다를시 후순위로 넘김
            distance += 1e8         # bias
        
        distance += sqrt(abs(pred_info[0] - label_cx)**2 +abs(pred_info[1] - label_cy)**2)
        distance_result.append(distance)
    
    for _ in range(len(distance_result)):
        min_index = distance_result.index(min(distance_result))
        result.append(min_index)
        distance_result[min_index] = float('inf')  # 중복 방지
        
    return result

def bbox_iou(label_box, pred_box, format = None):
    """
    2개의 경계박스 정보를 이용하여 Intersection over union의 비를 구합니다.
    Args:
        label_box (list): bbox point
        pred_box (list): bbox point 
        format (str): bbox format. ex)xywh

    Returns:
        iou (tensor(float)): 0~1 float value
    """
    pred_x1, pred_y1 = pred_box[0:2]
    pred_x2 = pred_box[0] + pred_box[2]
    pred_y2 = pred_box[1] + pred_box[3] 
    label_x1, label_y1 = label_box[0:2]
    label_x2 = label_box[0] + label_box[2]
    label_y2 = label_box[1] + label_box[3]
    
    inter_x1 = max(pred_x1, label_x1)
    inter_y1 = max(pred_y1, label_y1)
    inter_x2 = min(pred_x2, label_x2)
    inter_y2 = min(pred_y2, label_y2)
    
    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)
    
    pred_area = (pred_x2 - pred_x1 + 1) * (pred_y2 - pred_y1 + 1)
    label_area = (label_x2 - label_x1 + 1) * (label_y2 - label_y1 + 1)
    
    iou = inter_area / float(pred_area + label_area - inter_area)
    
    return iou