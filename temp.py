from ultralytics import YOLO

model = YOLO(model = "/home/ljj/ljj/waffle/hubs/HanlimDet_v1.1.0/weights/best_ckpt.pt")
model.val(data = "/home/ljj/ljj/waffle/datasets/HanlimDataset_v1.0.0/exports/ULTRALYTICS/data.yaml",split='train',iou = 0.25)