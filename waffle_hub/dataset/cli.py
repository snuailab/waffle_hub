import inspect

import fire

from waffle_hub.dataset import Dataset
from waffle_hub.utils.base_cli import BaseCLI, cli


class DatasetInstance(BaseCLI):
    def __init__(self, name: str, root_dir: str = None):
        self.dataset = None
        if name in Dataset.get_dataset_list(root_dir):
            self.dataset = Dataset.load(name, root_dir=root_dir)
        else:
            raise ValueError(f"Dataset {name} does not exist.")

        super().__init__()

    def get_object(self):
        return self.dataset


def main():
    fire.Fire(cli(Dataset, DatasetInstance), serialize=str)


if __name__ == "__main__":
    main()
