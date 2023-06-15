import fire

from waffle_hub.dataset import Dataset
from waffle_hub.utils.base_cli import BaseCLI


class DatasetCLI(BaseCLI):
    def __init__(
        self,
        name: str = None,
        task: str = None,
        root_dir: str = None,
    ):

        self.dataset = None
        if name is None:
            pass
        elif name in Dataset.get_dataset_list():
            if task is not None:
                raise ValueError(
                    "You can't specify any arguments except name when loading existing dataset.\n"
                    + "If you are trying to create new dataset, please specify another name."
                )
            self.dataset = Dataset.load(name, root_dir=root_dir)
        else:
            self.dataset = Dataset.new(
                name=name,
                task=task,
                root_dir=root_dir,
            )

        super().__init__()

    def get_class(self):
        return Dataset

    def get_instance(self):
        return self.dataset


if __name__ == "__main__":
    fire.Fire(DatasetCLI, serialize=str)
