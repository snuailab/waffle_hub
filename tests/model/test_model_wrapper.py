import torch
from torch import nn

from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.schema.fields.category import Category
from waffle_hub.type import TaskType


def test_model_wrapper():
    class simple_module(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x))

    module = simple_module()

    model_wrapper = ModelWrapper(
        model=module,
        preprocess=lambda x: x,
        postprocess=(lambda x, image_size: x),
        task=TaskType.CLASSIFICATION,
        categories=["1", "2", "3"],
    )

    assert model_wrapper(torch.randn(1, 3, 224, 224)).shape == (1, 3, 222, 222)
    assert model_wrapper.task == "classification"
    assert model_wrapper.task != "object_detection"
    assert len(model_wrapper.categories) == 3
    assert isinstance(model_wrapper.categories[0], Category)
    assert len(model_wrapper.get_layer_names()) == 3
    assert len(model_wrapper.get_layers(["conv", "relu"])) == 2
