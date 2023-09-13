from waffle_hub.dataset import Dataset
from waffle_hub.hub import Hub

if __name__ == "__main__":

    hub = Hub.new(
        name="test",
        task="classification",
        model_type="yolov8",
        model_size="n",
    )

    dataset = Dataset.load(name="mnist_classification")

    hub.hpo(dataset=dataset)
