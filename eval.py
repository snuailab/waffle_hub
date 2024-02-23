from waffle_hub.dataset.dataset import Dataset
from waffle_hub.hub.hub import Hub

dataset = Dataset.load(name = "HanlimDataset_v1.0.0")
hub = Hub.load(name ="HanlimDet_v1.1.0")

print(hub.evaluate(dataset, set_name = "train", iou_threshold=0.25, confidence_threshold=0.25, extended_summary= True))