import inspect
from typing import Union

import fire

from waffle_hub.hub import Hub
from waffle_hub.utils.base_cli import BaseCLI


class HubCLI(BaseCLI):
    def __init__(
        self,
        name: str = None,
        backend: str = None,
        task: str = None,
        model_type: str = None,
        model_size: str = None,
        categories: Union[list[dict], list] = None,
        root_dir: str = None,
    ):

        self.hub = None
        if name is None:
            pass
        elif name in Hub.get_hub_list():
            if all([backend, task, model_type, model_size, categories]):
                raise ValueError(
                    "You can't specify any arguments except name when loading existing hub.\n"
                    + "If you are trying to create new hub, please specify another name."
                )
            self.hub = Hub.load(name, root_dir=root_dir)
        else:
            self.hub = Hub.new(
                name=name,
                backend=backend,
                task=task,
                model_type=model_type,
                model_size=model_size,
                categories=categories,
                root_dir=root_dir,
            )

        super().__init__()

    def get_class(self):
        return Hub

    def get_instance(self):
        return self.hub


def main():
    fire.Fire(HubCLI, serialize=str)


if __name__ == "__main__":
    main()
