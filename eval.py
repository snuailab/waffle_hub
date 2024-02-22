from waffle_hub.dataset.dataset import Dataset
from waffle_hub.hub.hub import Hub

dataset = Dataset.load(name = "PeopleDataset_KISA_HADR_v1.0.0")
hub = Hub.load(name ="PeopleDet_v1.6.3")

#hub.evaluate(dataset, set_name = "val",draw = True)

print(dataset.category_dict)